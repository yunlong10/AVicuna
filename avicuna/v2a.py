import os
import subprocess
from tqdm import tqdm

def convert_mp4_to_wav(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    mp4_files = [f for f in os.listdir(source_dir) if f.endswith(".mp4")]

    for filename in tqdm(mp4_files, desc="Converting .mp4 to .wav"):
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename.replace(".mp4", ".wav"))
        
        if os.path.exists(target_file):
            continue

        command = f'ffmpeg -i "{source_file}" -acodec pcm_s16le -ar 44100 "{target_file}"'
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting {source_file}: {e}")
            continue


if __name__ == "__main__":
    source_directory = "/p61/ytang37/newav/VTimeLLM/images"#"/p61/ytang37/newav/data/raw_data/video"
    target_directory = "/p61/ytang37/newav/VTimeLLM/images"#"/p61/ytang37/newav/data/raw_data/audio"

    convert_mp4_to_wav(source_directory, target_directory)
