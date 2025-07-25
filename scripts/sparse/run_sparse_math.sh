MODEL_PATH="Qwen/Qwen2-7B"
DATA_PATH="meta-math/MetaMathQA"

sparse_methods=(
    "fisher_info"
    "gradient"
    "synflow"
    "grasp"
    "magnitude"
    "snip"
    "force"
    "Taylor-FO"
)

sparse_k=(
    # 2523136
    # 5046272
    # 10092544
    # 20185088
    40370176
    # 80740352
)

for s in "${sparse_methods[@]}"; do
    for k in "${sparse_k[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python run_math.py \
            --model_name_or_path $MODEL_PATH \
            --data_path $DATA_PATH \
            --dataset_field query response \
            --dataset_split "train[:100000]"\
            --output_dir "results/qwen_math_${s}_${k}" \
            --num_train_epochs 1 \
            --model_max_length 1024 \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 4 \
            --save_strategy "no" \
            --save_total_limit 2 \
            --learning_rate 2e-5 \
            --weight_decay 0.0 \
            --warmup_ratio 0.03 \
            --logging_steps 100 \
            --report_to "wandb" \
            --lora_enable False \
            --run_name "qwen_math_${s}_${k}" \
            --sparse_enable --sparse_method="${s}" \
            --sparse_global_topk --sparse_k="${k}" \
            --sparse_mask_update_steps=0 \
            --do_train
    done
done