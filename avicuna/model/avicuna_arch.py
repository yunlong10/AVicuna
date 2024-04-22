# Adopted from https://github.com/huangb23/VTimeLLM
import torch
import torch.nn as nn
from avicuna.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from abc import ABC, abstractmethod

class AVicunaMetaModel:

    def initialize_vision_modules(self, model_args):
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_mm_mlp_adapter_a = model_args.pretrain_mm_mlp_adapter_a

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(768, self.config.hidden_size)
        
        if not hasattr(self, 'mm_projector_a'):
            self.mm_projector_a = nn.Linear(512, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print("load mlp:", pretrain_mm_mlp_adapter)

        
        if pretrain_mm_mlp_adapter_a is not None:
            mm_projector_weights_a = torch.load(pretrain_mm_mlp_adapter_a, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector_a.load_state_dict(get_w(mm_projector_weights_a, 'mm_projector_a'))
            print("load mlp:", pretrain_mm_mlp_adapter_a)


class AVicunaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        if images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list:
            if len(images) == 2:
                tmp_images, tmp_audios = images
                tmp_image_features = self.get_model().mm_projector(tmp_images.to(torch.bfloat16))
                
                audio_features = self.get_model().mm_projector_a(tmp_audios.to(torch.bfloat16))
                if not hasattr(self, 'aud_mask') or self.aud_mask is None:
                    print("generate aud_mask")

                    self.bs, self.image_len, self.dim = tmp_image_features.shape
                    _, self.audio_len, _ = audio_features.shape
                    self.total_len = self.image_len + self.audio_len

                    self.insert_freq = int(round(self.image_len / (self.audio_len + 1)))

                    audio_positions = torch.arange(0, self.total_len, self.insert_freq + 1)[:self.audio_len]

                    mask = torch.ones(self.total_len, dtype=torch.bool)
                    mask[audio_positions] = False

                    self.img_mask = mask.unsqueeze(0).unsqueeze(-1).expand(self.bs, -1, self.dim)

                    self.aud_mask = ~self.img_mask

                # image_features = torch.cat([tmp_image_features, audio_features], dim=1)# non-interleave
                image_features = torch.zeros(self.bs, self.total_len, self.dim, device=tmp_image_features.device, dtype=tmp_image_features.dtype)
                try:
                    image_features[self.aud_mask] = audio_features.reshape(-1)
                except Exception as e:
                    print("feat shape mismatch with aud_mask")
                try:
                    image_features[self.img_mask] = tmp_image_features.reshape(-1)
                except Exception as e:
                    print("feat shape mismatch in img_mask")
        else:
            if images.shape[-1] == 768:
                image_features = self.get_model().mm_projector(images)
            elif images.shape[-1] == 512:
                image_features = self.get_model().mm_projector_a(images)
        
        # print([image.shape for image in image_features])
        
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

       
        fake_input_ids = None
        # print(position_ids, attention_mask)

        return fake_input_ids, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
