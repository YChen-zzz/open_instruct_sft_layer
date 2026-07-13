export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


pip install beaker -i https://pypi.tuna.tsinghua.edu.cn/simple

cd /chenyupeng/open-instruct-main


model_name=/chenyupeng/hf/hub/models--allenai--OLMoE-1B-7B-0924-SFT/snapshots/215cc4f73147dd68bd11e7a7dcc56bac397f4221

base_output_path=/chenyupeng/model_files/moe_sft/metamathqa

out_path=$base_output_path/metamathqa_2e
rm -rf $out_path
mkdir $out_path
touch $out_path/log.log
# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage0_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path $model_name \
    --use_flash_attn \
    --tokenizer_name $model_name \
    --use_slow_tokenizer \
    --dataset_mixer_list data/metamathqa/metamathqa_train_dataset.jsonl 0.2 \
    --max_seq_length 1024 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --load_balancing_loss \
    --load_balancing_weight 0.001 \
    --output_dir $out_path \
    --checkpointing_steps 120 \
    --reduce_loss sum \
    --report_to tensorboard \
    --logging_steps 10 \
    --keep_last_n_checkpoints -1 \
    --clean_checkpoints_at_end False &> >(tee -a $out_path/log.log)
