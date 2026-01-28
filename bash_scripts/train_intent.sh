#!/bin/bash

# Training script for intent LoRA adapter

CMD="import sys; from llamafactory.cli import main; sys.argv=['llamafactory-cli', 'train'] + sys.argv[1:]; main()"

CUDA_VISIBLE_DEVICES=0 python -c "$CMD" \
    --stage sft \
    --do_train \
    --model_name_or_path /opt/Data_Copilot/models/Qwen2.5-7B-Instruct \
    --dataset_dir ./data/training \
    --dataset intent_train \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ./output/intent_lora \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 20 \
    --learning_rate 5e-05 \
    --num_train_epochs 3 \
    --max_length 512 \
    --plot_loss \
    --bf16 true
