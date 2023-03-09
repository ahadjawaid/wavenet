from torch.utils.data import Dataset
from torch.nn.functional import pad
from pathlib import Path
import librosa
import os
import torch

class AudioDataset(Dataset):
    def __init__(self, root_dir, format="wav", transforms=None, sample_rate=16000):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.transforms = transforms
        self.music_files = list(filter(lambda x: f".{format}" in x, os.listdir(root_dir)))
        
    def __len__(self):
        return len(self.music_files)
    
    def __getitem__(self, i):
        assert type(i) == int
        
        waveform, sample_rate = librosa.load(self.root_dir/self.music_files[i], sr=self.sample_rate)
        waveform = torch.tensor(waveform)
        
        if self.transforms:
            waveform = self.transforms(waveform)
        
        return waveform, sample_rate


def collate_fn(batch, max_len=50000):
    split_waveform = []
    for waveform, _ in batch:
        for i in range(0, waveform.size(-1), max_len):
            split_waveform.append(waveform[i:i+max_len])

        split_waveform[-1] = pad(split_waveform[-1], (0, max_len - split_waveform[-1].size(-1)))
    split_waveform = torch.stack(split_waveform).unsqueeze(1)
    
    return split_waveform