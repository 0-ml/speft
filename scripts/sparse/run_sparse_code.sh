MODEL_PATH="google/gemma-2-2b"
OUTPUT_PATH="results/gemma_code_sparse"
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
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --lora_enable False \
    --run_name "gemma_code_sparse" \
    --do_train True \
    --sparse_enable True \
    --sparse_method="gradient" \
    --sparse_global_topk \
    --sparse_k=25559040 \
    --sparse_mask_update_steps=0 \
    --do_train