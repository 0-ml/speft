tasks=(
    "mrpc"
    "qnli"
    "stsb"
    "cola"
    "rte"
    "sst2"
    "qqp"
    "mnli"
)

sparse_methods=(
    # "fisher_info"
    "gradient"
    "synflow"
    # "grasp"
    "magnitude"
    "snip"
    "force"
    "Taylor-FO"
)

for t in "${tasks[@]}"; do
    echo "Queueing ${t}..."
    for s in "${sparse_methods[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python run_glue.py \
           --sparse_enable --sparse_method="${s}" \
           --per_device_train_batch_size=32 \
           --gradient_accumulation_steps=2 \
           --seed=42 \
           --task_name="${t}" \
           --model_name_or_path="facebook/opt-350m" \
           --sparse_global_topk --sparse_k=1179648 \
           --sparse_mask_update_steps=0 \
           --max_length=196 \
           --learning_rate=5e-5 \
           --run_name="${s}_${t}" \
           --num_train_epochs=30 \
           --output_dir="results/${s}_opt350m_${t}"
    done
done