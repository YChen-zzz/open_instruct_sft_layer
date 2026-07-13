export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


pip install beaker -i https://pypi.tuna.tsinghua.edu.cn/simple


script_dir=$(dirname "$(realpath "$0")")

cd $script_dir
cd .. 



model_name=/chenyupeng/hf/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062

tokenizer_name=/L00120230003/models/meta-llama/Llama-3.2-3B-Instruct

base_output_path=/chenyupeng/model_files/openinstruct/llama32_3B_2e_2e-5

out_path=$base_output_path
rm -rf $out_path
mkdir -p $out_path
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
    --tokenizer_name_or_path $tokenizer_name \
    --use_slow_tokenizer \
    --dataset_mixer_list data/metamathqa/metamathqa_train_dataset_20par.jsonl 1.0 \
    --max_seq_length 1024 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir $out_path \
    --checkpointing_steps 60 \
    --report_to tensorboard \
    --logging_steps 10 \
    --keep_last_n_checkpoints -1 \
    --clean_checkpoints_at_end False &> >(tee -a $out_path/log.log)
