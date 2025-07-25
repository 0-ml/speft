#!/bin/bash
sparse_methods=(
    # "fisher_info"
    "gradient"
    "synflow"
    # "grasp"
    "magnitude"
    "snip"
    "force"
    # "Taylor-FO"
)

for s in "${sparse_methods[@]}"; do
    echo "Queueing ${s}..."
    CUDA_VISIBLE_DEVICES=0 python run_mmlu.py \
        --do_train True \
        --do_lm_eval True \
        --model_name_or_path="google/gemma-2-2b" \
        --trust_remote_code \
        --output_dir="results/${s}_alpaca_gemma2_mmlu" \
        --per_device_train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --learning_rate=1e-5 \
        --warmup_ratio=0.03 \
        --sparse_enable --sparse_method="${s}" \
        --sparse_global_topk --sparse_k=25559040 \
        --sparse_mask_update_steps=0 \
        --seed=242 \
        --max_steps=2000 \
        --run_name="${s}_alpaca_gemma2_mmlu" \
        --per_device_eval_batch_size=1 \
        --num_fewshot=5 \
        --tasks="mmlu" \
        --save_strategy="no" \
        --do_eval True \
        --eval_dataset_size=0.01 \
        --eval_strategy="steps" \
        --eval_steps=200 \
        --dataset="alpaca"
done