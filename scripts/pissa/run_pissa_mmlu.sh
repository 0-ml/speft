#!/bin/bash
lora_r=(
    # 4
    # 8
    # 16
    # 32
    64
    # 128
)

for r in "${lora_r[@]}"; do
CUDA_VISIBLE_DEVICES=0 python run_mmlu.py \
    --do_train True \
    --do_lm_eval True \
    --model_name_or_path="Qwen/Qwen2-7B" \
    --trust_remote_code \
    --output_dir="results/pissa_oasst2_qwen2_mmlu_${r}" \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --learning_rate=5e-5 \
    --warmup_ratio=0.03 \
    --lora_enable \
    --pissa_enable \
    --lora_r=${r} \
    --seed=242 \
    --max_steps=2000 \
    --run_name="pissa_oasst2_qwen2_mmlu_${r}" \
    --per_device_eval_batch_size=1 \
    --num_fewshot=5 \
    --tasks="mmlu" \
    --do_eval True \
    --eval_dataset_size=0.01 \
    --eval_strategy="steps" \
    --eval_steps=500 \
    --dataset="oasst2" \
    --mmlu True
done