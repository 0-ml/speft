models=(
    "FacebookAI/roberta-base"
    # "google-bert/bert-base-uncased"
    # "facebook/opt-125m"
    # "facebook/opt-350m"
)

tasks=(
    # "mrpc"
    # "stsb"
    "qnli"
    # "cola"
    # "rte"
    "sst2"
    "mnli"
    "qqp"
)

for m in "${models[@]}"; do
    for t in "${tasks[@]}"; do
        echo "Queueing ${m}..."
        CUDA_VISIBLE_DEVICES=0 python run_glue.py \
           --per_device_train_batch_size=64 \
           --lora_enable \
           --pissa_enable \
           --lora_r=8 \
           --lora_alpha=8 \
           --max_length=196 \
           --task_name="${t}" \
           --model_name_or_path="${m}" \
           --seed=42 \
           --num_train_epochs=30 \
           --learning_rate=5e-5 \
           --run_name="lora_${t}"
    done
done