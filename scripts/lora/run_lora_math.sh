MODEL_PATH="Qwen/Qwen2-7B"
DATA_PATH="meta-math/MetaMathQA"
lora_r=(
    # 4
    # 8
    # 16
    # 32
    64
    # 128
)

for r in "${lora_r[@]}"; do
CUDA_VISIBLE_DEVICES=0 python run_math.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --dataset_field query response \
    --dataset_split "train[:100000]"\
    --output_dir "lora_qwen_math_${r}" \
    --num_train_epochs 1 \
    --model_max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --logging_steps 100 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --merge True \
    --lora_enable True \
    --run_name "lora_qwen_math_${r}" \
    --do_train True \