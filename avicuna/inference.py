# Adopted from https://github.com/huangb23/VTimeLLM
import os
import sys
import argparse
import torch
from avicuna.constants import IMAGE_TOKEN_INDEX
from avicuna.conversation import conv_templates, SeparatorStyle
from avicuna.model.builder import load_pretrained_model, load_lora
from avicuna.utils import disable_torch_init
from avicuna.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from easydict import EasyDict as edict
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import numpy as np
import clip


def inference(model, image, query, tokenizer):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image,
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/avicuna-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--pretrain_mm_mlp_adapter_a", type=str, default="checkpoints/avicuna-vicuna-v1-5-7b-stage2/mm_projector_a.bin")
    parser.add_argument("--stage3", type=str, default="checkpoints/avicuna-vicuna-v1-5-7b-stage3-12")
    parser.add_argument("--stage4", type=str, default="checkpoints/avicuna-vicuna-v1-5-7b-stage4-insunav-12")
    parser.add_argument("--model_base", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--video_path", type=str, default="images/jump.mp4")
    parser.add_argument("--av_ratio", type=float, default=0.25)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage3, args.stage4)
    model = model.cuda()
    model.to(torch.bfloat16)
    av_ratio = args.av_ratio
    n_audio_feats = int(100 * av_ratio)
    n_image_feats = int(100 - n_audio_feats)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()

    video_loader = VideoExtractor(N=n_image_feats)
    _, images = video_loader.extract({'id': None, 'video': args.video_path})

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # print(images.shape) # <N, 3, H, W>
    images = transform(images / 255.0)
    images = images.to(torch.bfloat16)
    audio = torch.tensor(np.load('demo/jump.npy'), dtype=torch.bfloat16)
    with torch.no_grad():
        v_features = clip_model.encode_image(images.to('cuda'))
        a_features = audio.cuda()

        tmp_len = len(a_features)
        if tmp_len != n_audio_feats:
            repeat_factor = n_audio_feats // tmp_len
            remainder = n_audio_feats % tmp_len
            a_features = torch.cat([a_features[i].unsqueeze(0).repeat(repeat_factor + (1 if i < remainder else 0), 1) for i in range(tmp_len)], dim=0)
            print(v_features.shape, a_features.shape)
        features = [v_features.unsqueeze(0), a_features.unsqueeze(0)]

    query = "Is the person dancing from 43 to 83? And What is he doing after that and when?"# what can you hear from 20 to 90?What is the person doing from 17 to 42?
    print("query: ", query)
    print("answer: ", inference(model, features, "<video>\n " + query, tokenizer))


