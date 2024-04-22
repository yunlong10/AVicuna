import torch
import clip
from avicuna.mm_utils import VideoExtractor
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import random

if __name__ == "__main__":
    clip_model, _ = clip.load("/path/to/CLIP/checkpoint/ViT-L-14.pt")
    clip_model.eval()
    clip_model = clip_model.to('cuda:0')

    video_loader = VideoExtractor(N=100)

    transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    video_folder = "/path/to/video/mp4/folder"
    output_folder = "/path/to/video/features/folder"
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    random.shuffle(video_files)

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_folder, video_file)
        output_path = os.path.join(output_folder, video_file.replace(".mp4", ".npy"))
        if os.path.exists(output_path):
            print(f'{video_file.replace(".mp4", ".npy")} exists!')
            continue
        try:
            _, images = video_loader.extract({'id': None, 'video': video_path})

            images = transform(images / 255.0)
            images = images.to(torch.float16)

            with torch.no_grad():
                features = clip_model.encode_image(images.to('cuda:0'))

            np.save(output_path, features.cpu().numpy())

        except Exception as e:
            print(f"errors in {video_file}: {e}")