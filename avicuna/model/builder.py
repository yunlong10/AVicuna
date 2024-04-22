# Adopted from https://github.com/huangb23/VTimeLLM
import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from avicuna.model import *
from peft import PeftModel

def load_lora(model, lora_path):
    non_lora_trainables_path = os.path.join(lora_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_trainables_path):
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, lora_path)
    return model

def load_pretrained_model(args, stage3=None, stage4=None):
    kwargs = {'torch_dtype': torch.float16}

    # model_path = os.path.expanduser(args.model_path)
    model_base = args.model_base


    # lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    print('Loading AVicuna from base model...')
    
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    model = AVicunaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))


    # load stage1/2:
    model.get_model().initialize_vision_modules(args)

    if stage3 is not None:
        print('Loading stage3 weights...')
        model = load_lora(model, stage3)
        print('Merging stage3 weights...')
        model = model.merge_and_unload()
        if stage4 is not None:
            print('Loading stage4 weights...')
            model = load_lora(model, stage4)
            print('Merging stage4 weights...')
            model = model.merge_and_unload()


    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
