import numpy as np
import librosa
import torch
import laion_clap


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def audio_feat_extraction(audio_path, model, sr=48000, stride=4):
    audio_data, _ = librosa.load(audio_path, sr=sr)

    segment_length = 10 * sr
    stride_length = stride * sr
    remaining_samples = len(audio_data) % segment_length

    if 0 < remaining_samples <= sr:
        audio_data = audio_data[:-(remaining_samples)]
    elif remaining_samples > sr:
        padding_length = segment_length - remaining_samples
        audio_data = np.pad(audio_data, (0, padding_length), mode='constant')
    
    num_segments = (len(audio_data) - segment_length) // stride_length + 1
    num_segments = 1 if num_segments < 1 else num_segments
    audio_segments = np.array([audio_data[i * stride_length:i * stride_length + segment_length] for i in range(num_segments)])
    audio_segments = torch.from_numpy(int16_to_float32(float32_to_int16(audio_segments))).float()
    audio_embed = model.get_audio_embedding_from_data(x=audio_segments, use_tensor=True)

    return audio_embed

