import librosa
import os
from math import ceil
import soundfile as sf
import csv

def processAudio(path, out_path, sample_rate, crop_length):
    audio_files = os.listdir(path)
    audio_file_embedding = {audio_files[i]: i for i in range(len(audio_files))}
    step = crop_length*sample_rate
    
    for file in audio_files:
        waveform, sr = librosa.load(path/file, sr=sample_rate)
        for i in range(1, ceil(waveform.size/step) + 1):
            crop_index = i*step
            save_path = out_path/f"{audio_file_embedding[file]}{i-1}.wav"
            cropAndSaveWaveForm(waveform, sr, crop_index, save_path)

    saveAudioMetaData(out_path, audio_files)

def cropAndSaveWaveForm(waveform, sample_rate, crop_index, step, save_path):
    cropped_waveform = waveform[crop_index-step:crop_index]
    sf.write(save_path, cropped_waveform, sample_rate)

def saveAudioMetaData(path, audio_files):
    with open("meta.csv", 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["index", "name"])
        for i in range(len(audio_files)):
            csv_writer.writerow([i, audio_files[i]])