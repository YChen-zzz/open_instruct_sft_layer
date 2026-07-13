export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=4B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


cd /data/250010186/SFT_code/open-instruct-main

model_path=/data/250010186/SFT_model/homework/Qwen4B_SFT_MoFO020/finetune

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.
out_path=/data/250010186/SFT_model/homework/Qwen4B_DPO_MoFO020_from_Qwen4B_SFT_MoFO020
rm -rf $out_path
mkdir -p $out_path
touch $out_path/log.log

pip install beaker

export HF_HOME=/data/250010186/hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1



accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage1_accelerate.conf \
    open_instruct/dpo_tune_cache.py \
    --model_name_or_path $model_path \
    --model_revision main \
    --tokenizer_name $model_path \
    --tokenizer_revision main \
    --use_slow_tokenizer \
    --dataset_mixer_list allenai/tulu-3-pref-personas-instruction-following 1.0 allenai/llama-3.1-tulu-3-8b-preference-mixture 0.2 \
    --max_seq_length 2048 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-06 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --use_AdamW_MoFO True \
    --MoFO_fraction 0.2 \
    --use_flash_attn \
    --push_to_hub False \
    --output_dir $out_path \
    --try_launch_beaker_eval_jobs False \
    --checkpointing_steps 150 \
    --keep_last_n_checkpoints -1 \
    --report_to tensorboard \
    --logging_steps 5 \
    --seed 8 &> >(tee -a $out_path/log.log)

#allenai/llama-3.1-tulu-3-8b-preference-mixture 0.2