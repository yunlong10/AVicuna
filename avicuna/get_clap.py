import os
import torch
import numpy as np
import librosa
import laion_clap
from test_clap import *
from tqdm import tqdm
import random

def process_audio_files(source_dir, target_dir, model):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    wav_files = [f for f in os.listdir(source_dir) if f.endswith(".wav")]
    wav_files = wav_files[::-1]

    cnt=0
    for filename in tqdm(wav_files, desc="Processing audio files"):
        target_file = os.path.join(target_dir, filename.replace(".wav", ".npy"))
        cnt += 1
        
        if os.path.exists(target_file):
            continue

        audio_path = os.path.join(source_dir, filename)
        try:
            feature_tensor = audio_feat_extraction(audio_path, model)
            np.save(target_file, feature_tensor.detach().cpu().numpy())
        except Exception as e:
            print(e)
            print(audio_path)
            continue


if __name__ == "__main__":
    source_directory = "/path/to/audio/wav/folder"
    target_directory = "/path/to/audio/features/folder"

    model = laion_clap.CLAP_Module(enable_fusion=True).to("cuda:0")
    model.load_ckpt("/path/to/CLAP/checkpoint/630k-fusion-best.pt")

    process_audio_files(source_directory, target_directory, model)
