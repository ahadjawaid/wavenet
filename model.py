import torch
import torch.nn as nn
from preprocess import quantize_and_onehot_waveform, decodeMuLaw
from math import ceil

class WaveNet(nn.Module):
    def __init__(self, num_residual_layers, num_blocks, num_casual_layers, residual_channels=32, 
                 gate_channels=32, skip_channels=512, quantize_channels=256, local_channels=0, 
                 global_channels=0, device=None):
        super(WaveNet, self).__init__()
        
        self.casual_layers = [CasualConv1D(quantize_channels, residual_channels, device=device)]
        for i in range(num_casual_layers-1): 
            self.casual_layers.append(CasualConv1D(residual_channels, residual_channels, device=device))
        self.casual_layers = nn.Sequential(*self.casual_layers)
        
        residual_layers = [
            ResidualLayer(2**i, residual_channels, gate_channels, skip_channels,
                          local_channels, global_channels, device=device) for i in range(num_residual_layers)
        ]
        self.residual_blocks = nn.ModuleList(residual_layers * num_blocks)
        
        self.head = Head(skip_channels, quantize_channels)
        
    
    def forward(self, inputs, local_inputs=None, global_inputs=None):
        processed_inputs = quantize_and_onehot_waveform(inputs)
        casual_out = self.casual_layers(processed_inputs)
        
        residual_out = casual_out
        skip_connections = []
        for residual_layer in self.residual_blocks:            
            residual_out, skip_out = residual_layer(residual_out, local_inputs, global_inputs)
            skip_connections.append(skip_out)
        
        skip_connections = list(map(lambda skip: skip[:,:,-skip_connections[-1].size(2)], skip_connections))
        skip_connections = torch.stack(skip_connections).transpose(0,1)

        head_out = self.head(skip_connections)
        return head_out
    
    def generate(self, inputs, time_steps):
        x = inputs
        for _ in range(time_steps):
            prob = self.forward(inputs)
            category = torch.argmax(prob)
            pred = decodeMuLaw(category)
            x = torch.cat((x, pred.view(1,1,1)), dim=2)
        
        return x

class ResidualLayer(nn.Module):
    def __init__(self, dilation, residual_channels=32, gate_channels=32, skip_channels=512,
                 local_channels=0, global_channels=0,  **kwargs):
        '''
        Args:
            dilation: (int)
            residual_channels: (int)
            gate_channels: (int)
            skip_channels: (int)
            local_channels: (int) 0 if no local conditional inputs
            global_channels: (int) 0 if no global conditional inputs
        '''
        super(ResidualLayer, self).__init__()
        self.dilated_conv = DialatedConv1d(residual_channels, gate_channels, dilation, **kwargs)
        
        self.local_1x1 = self.global_1x1 = None
        if local_channels > 0:
            self.local_1x1 = Conv1d1x1(local_channels, gate_channels, bias=False, **kwargs)
        
        if global_channels > 0:
            self.global_1x1 = Conv1d1x1(global_channels, gate_channels, bias=False, **kwargs)
        
        self.residual_1x1 = Conv1d1x1(gate_channels, residual_channels, **kwargs)
        self.skip_1x1 = Conv1d1x1(gate_channels, skip_channels, **kwargs)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs, local_inputs=None, global_inputs=None):
        conv_out = self.dilated_conv(inputs)
        time_steps = conv_out.size(2)
            
        if local_inputs != None:
            assert self.local_1x1 != None and local_inputs.dim() == 3

            upsampling_factor = ceil(time_steps / local_inputs.size(2))
            upsampled_local_inputs = local_inputs.repeat(1, 1, upsampling_factor)[:,:,:time_steps]

            local_out = self.local_1x1(upsampled_local_inputs)
            conv_out += local_out

        if global_inputs != None:
            assert self.global_1x1 != None
            assert global_inputs.dim() == 2 or global_inputs.size(2) == time_steps

            global_out = self.global_1x1(global_inputs)   
            conv_out += global_out
        
        z = self.tanh(conv_out) * self.sigmoid(conv_out)
        residual_out, skip_out  = self.residual_1x1(z), self.skip_1x1(z)
        
        assert inputs.dim() == 3
        clipped_inputs = inputs[:,:, -residual_out.size(2):]
        residual_out += clipped_inputs
        
        return residual_out, skip_out
    
class Head(nn.Sequential):
    def __init__(self, in_channels, out_channels,  **kwargs):
        super(Head, self).__init__(
            nn.ReLU(),
            Conv1d1x1(in_channels, in_channels, **kwargs),
            nn.ReLU(),
            Conv1d1x1(in_channels, out_channels, **kwargs),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        '''
        inputs: (tensor) stacked skip connections
        '''
        batch_size = inputs.size(0)
        summed_inputs = inputs.sum(dim=1).view(batch_size,-1,1)
        return super().forward(summed_inputs)
    
class CasualConv1D(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super(CasualConv1D, self).__init__(in_channels, out_channels, kernel_size, 
                                           padding=kernel_size-1, **kwargs)
    def forward(self, inputs):
        assert inputs.dim() == 3 # To shift output

        activations = super(CasualConv1D, self).forward(inputs)
        return activations[:,:,:activations.shape[-1]-self.kernel_size[0]]

class DialatedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3, **kwargs):
        super(DialatedConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                             dilation=dilation,  **kwargs)
        
    def forward(self, inputs):
        return super(DialatedConv1d, self).forward(inputs)

class Conv1d1x1(nn.Conv1d):
    def __init__(self, in_channels, out_channels,  **kwargs):
        super(Conv1d1x1, self).__init__(in_channels, out_channels, kernel_size=1,  **kwargs)
        
    def forward(self, inputs):
        return super(Conv1d1x1, self).forward(inputs)