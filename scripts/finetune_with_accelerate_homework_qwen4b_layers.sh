export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=4B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

cd /data/250010186/SFT_code/open-instruct-main

model_path=/data/250010186/hf/hub/Qwen3-4B-Base

pip install beaker

export HF_HOME=/data/250010186/hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# case_name -> trainable_layers ("" 表示全参/baseline，不传 --trainable_layers)
declare -A CASES=(
  ["baseline"]=""
  ["layers_0_1_2"]="0,1,2"
  ["layers_16_17_18"]="16,17,18"
  ["layers_33_34_35"]="33,34,35"
)
CASE_ORDER=(baseline layers_0_1_2 layers_16_17_18 layers_33_34_35)

for case_name in "${CASE_ORDER[@]}"; do
    trainable_layers="${CASES[$case_name]}"
    out_path=/data/250010186/SFT_model/homework/Qwen4B_SFT_${case_name}
    rm -rf $out_path
    mkdir -p $out_path
    touch $out_path/log.log

    echo "=== Training case: $case_name (trainable_layers=${trainable_layers:-ALL}) ==="

    extra_args=()
    if [ -n "$trainable_layers" ]; then
        extra_args+=(--trainable_layers "$trainable_layers")
    fi

    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage1_accelerate.conf \
        open_instruct/finetune.py \
        --model_name_or_path $model_path \
        --use_flash_attn \
        --tokenizer_name /data/250010186/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
        --use_slow_tokenizer \
        --dataset_mixer_list allenai/tulu-3-sft-personas-instruction-following 1.0 /data/250010186/SFT_code/open-instruct-main/data/ifeval_like/ifeval_like_train_dataset.jsonl 0.5 allenai/tulu-3-sft-mixture 0.1 \
        --max_seq_length 2048 \
        --preprocessing_num_workers 128 \
        --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
        --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
        --learning_rate 1e-5 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --clean_checkpoints_at_end False \
        --add_seed_and_date_to_exp_name False \
        --weight_decay 0. \
        --num_train_epochs 2 \
        --output_dir $out_path \
        --checkpointing_steps 600 \
        --report_to tensorboard \
        --push_to_hub False \
        --logging_steps 10 \
        "${extra_args[@]}" &> >(tee -a $out_path/log.log)

    echo "=== Finished training $case_name, running ifeval eval ==="
    bash /data/250010186/SFT_code/open-instruct-main/scripts/eval/eval_homework_ifeval.sh "$out_path"
done
