import torch
import torch.nn.functional as f

def quantize_and_onehot_waveform(inputs, quantatize_categories=256):
    assert inputs.dim() == 3 # To get correct outputs
    
    quantized_inputs = muLaw(inputs, quantatize_categories).squeeze(1)
    return f.one_hot(quantized_inputs, quantatize_categories).transpose(1, 2).to(torch.float)

def muLaw(inputs, quantatize_categories=256):
    mu = quantatize_categories -  1
    out = torch.log(1 + (mu *  torch.abs_(inputs))) / torch.log(torch.tensor(1 + mu)) * torch.sign(inputs)
    return torch.floor(((out + 1) * (mu + 0.5)) / 2).to(torch.int64)

def decodeMuLaw(inputs, quantatize_categories=256):
    mu = quantatize_categories - 1
    out = (2 * (inputs - 0.5) / mu) - 1
    return torch.sign(out) / mu * ((1 + mu) ** torch.abs(out) - 1)
