MODEL_PATH="meta-llama/Meta-Llama-3-8B"
OUTPUT_PATH="results/llama_code_lora"
DATA_PATH="fxmeng/pissa-dataset"

CUDA_VISIBLE_DEVICES=0 python run_code.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --sub_task python \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "no" \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --lora_enable True \
    --pissa_enable False \
    --run_name "llama_code_lora" \
    --do_train True \