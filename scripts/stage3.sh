#!/bin/bash

MODEL_VERSION=vicuna-v1-5-7b
gpu_vis=0
MASTER_PORT=29570


deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT avicuna/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./data/stage3.json \
    --feat_folder /path/to/stage3/video/feat \
    --feat_folder_a /path/to/stage3/audio/feat \
    --pseudo_feat_folder /path/to/pseudo/valor/video/feat \
    --pseudo_feat_folder_a /path/to/pseudo/valor/audio/feat \
    --tune_mm_mlp_adapter False \
    --tune_mm_mlp_adapter_a False \
    --freeze_mm_mlp_adapter True \
    --freeze_mm_mlp_adapter_a True \
    --pretrain_mm_mlp_adapter ./checkpoints/avicuna-$MODEL_VERSION-stage1/mm_projector.bin \
    --pretrain_mm_mlp_adapter_a ./checkpoints/avicuna-$MODEL_VERSION-stage2/mm_projector_a.bin \
    --output_dir ./checkpoints/avicuna-$MODEL_VERSION-stage3 \
    --av_ratio 0.25 \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
