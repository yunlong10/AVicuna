import random
import copy
import json
import torch
import os
import transformers
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from avicuna.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from avicuna import conversation as conversation_lib
from avicuna.mm_utils import tokenizer_image_token

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    feat_folder: Optional[str] = field(default=None)
    feat_folder_a: Optional[str] = field(default=None)
    pseudo_feat_folder: Optional[str] = field(default=None)
    pseudo_feat_folder_a: Optional[str] = field(default=None)
    av_ratio: float = 0.10



def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(conversations)
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n'
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)



class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = json.load(open(data_path, "r"))
        self.data_args = data_args
        self.av_ratio = self.data_args.av_ratio
        self.n_audio_feats = int(100 * self.av_ratio)
        self.n_image_feats = int(100 - self.n_audio_feats)
        self.datasets_w_audio = ["pseudo-valor", "unav100", "avsd", "musicqa", "audiocaps", "clotho"]

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = copy.deepcopy(self.list_data_dict[i])

        data_type = 'video'
        has_audio = False
        has_pseudo = False
        if '<image>' in source['conversations'][0]['value']:
            source['conversations'][0]['value'] = source['conversations'][0]['value'].replace('<image>', '<video>')
            data_type = 'image_only'
            has_audio = False
        
        if '<audio>' in source['conversations'][0]['value']:
            source['conversations'][0]['value'] = source['conversations'][0]['value'].replace('<audio>', '<video>')
            data_type = 'audio_only'
            has_audio = True

        if 'meta' in source:
            def convert(duration, x):
                x = x / duration * 100
                x = str(min(round(x), 99))
                if len(x) == 1:
                    x = "0" + x
                return x

            replace_set = []
            for k, v in source['meta']['token'].items():
                replace_set.append((k, convert(source['meta']['duration'], v)))
            for l in range(len(source['conversations'])):
                for x1, x2 in replace_set:
                    source['conversations'][l]['value'] = source['conversations'][l]['value'].replace(x1, x2)
        
        if 'source' in source:
            data_source = source['source']
            if data_source in self.datasets_w_audio:
                has_audio = True
            if data_source == 'pseudo-valor':
                has_pseudo = True

        
        image = torch.zeros((self.n_image_feats if data_type == 'video' else 1, 768), dtype=torch.float16)
        audio = torch.zeros((self.n_audio_feats if data_type == 'video' else 1, 512), dtype=torch.float16)
        
        if data_type != 'audio_only':#可能是image也可能是video
            try:
                if not has_pseudo:
                    feat_path = '{}/{}.npy'.format(self.data_args.feat_folder, source['id'])
                    tmp_image = np.load(feat_path) # <N, 768> float16
                    tmp_image = torch.from_numpy(tmp_image).to(torch.float16)
                else:
                    scales = source['meta']['scale']
                    tmp_image = []
                    for i, sub_id in enumerate(source['meta']['sub_videos']):
                        len_sub_video = int(scales[i] * 10)
                        feat_path = os.path.join(self.data_args.pseudo_feat_folder, f"{sub_id}.npy")
                        if os.path.isfile(feat_path):
                            sub_video = torch.from_numpy(np.load(feat_path)).to(torch.float16)
                            # print(sub_video.dtype)
                        else:
                            print(f"sub video loss: {sub_id}")
                            sub_video = torch.zeros((len_sub_video, 768), dtype=torch.float16)
                        indices = torch.linspace(0, len(sub_video) - 1, len_sub_video).long()
                        sub_video = sub_video[indices].to(torch.float16)
                        tmp_image.append(sub_video)
                    tmp_image = torch.cat(tmp_image, 0)

                tmp_len = len(tmp_image)
                if data_type == 'video':
                    if tmp_len >= self.n_image_feats:
                        sampled = []
                        for j in range(self.n_image_feats):
                            sampled.append(tmp_image[(j * tmp_len) // self.n_image_feats])
                        image = torch.stack(sampled)
                    else:
                        image = torch.cat(
                            [tmp_image, torch.zeros(self.n_image_feats - tmp_len, 768)], 0
                        )
                else:
                    image = tmp_image

                if data_type == 'image_only' and len(image.shape) == 1: # <768>
                    image = image.unsqueeze(0)
                    
            except Exception as e:
                print(e)
                return random.choice(self)

        if has_audio: 
            try:
                if not has_pseudo:
                    feat_path = '{}/{}.npy'.format(self.data_args.feat_folder_a, source['id'])
                    tmp_audio = np.load(feat_path) # <N, 512> float16
                    tmp_audio = torch.from_numpy(tmp_audio).to(torch.float16)
                else:
                    tmp_audio = []
                    for i, sub_id in enumerate(source['meta']['sub_videos']):
                        feat_path = os.path.join(self.data_args.pseudo_feat_folder_a, f"{sub_id}.npy")
                        if os.path.isfile(feat_path):
                            sub_audio = torch.from_numpy(np.load(feat_path)).to(torch.float16)
                        else:
                            print(f"sub video loss: {sub_id}")
                            sub_audio = torch.zeros((1, 512), dtype=torch.float16)
                    
                        tmp_audio.append(sub_audio)
                    tmp_audio = torch.cat(tmp_audio, 0)
                
                tmp_len = len(tmp_audio)
                if data_type == 'video':
                    if tmp_len == self.n_audio_feats:
                        audio = tmp_audio
                    else:
                        repeat_factor = self.n_audio_feats // tmp_len  # Calculate repeat factor
                        remainder = self.n_audio_feats % tmp_len
                        audio = torch.cat([tmp_audio[i].unsqueeze(0).repeat(repeat_factor + (1 if i < remainder else 0), 1) for i in range(tmp_len)], dim=0)
                else:
                    audio = tmp_audio
                
                if data_type == 'audio_only' and len(audio.shape) == 1: # <512>
                    audio = audio.unsqueeze(0)
            except Exception as e:
                print(e)
                return random.choice(self)

        data_dict = preprocess(
                [source["conversations"]],
                self.tokenizer,
                has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        
        if data_type == 'image_only':
            data_dict['image'] = image
        elif data_type == 'audio_only':
            data_dict['audio'] = audio
        else:#video
            data_dict['image'] = image
            data_dict['audio'] = audio

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        tmp_images = None
        tmp_audios = None
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                tmp_images = torch.stack(images)
            else:
                tmp_images = images
        
        if 'audio' in instances[0]:
            audios = [instance['audio'] for instance in instances]
            if all(x is not None and x.shape == audios[0].shape for x in audios):
                tmp_audios = torch.stack(audios)
            else:
                tmp_audios = audios
        
        if tmp_images is not None and tmp_audios is not None:
            batch['images'] = [tmp_images, tmp_audios]
        elif tmp_images is not None and tmp_audios is None:
            batch['images'] = tmp_images
        elif tmp_images is None and tmp_audios is not None:
            batch['images'] = tmp_audios

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args: DataArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

