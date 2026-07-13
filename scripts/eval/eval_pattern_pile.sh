# -*- coding: utf-8 -*-
# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0

export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME=/chenyupeng/hf



###############################################################################################################################SFT Eval #######################################################
model_path="/chenyupeng/model_files/openinstruct/llama32_3B_2e_2e-5_mofo015/finetune"

model_list=()

# 类似 Python 的 append 函数：将元素添加到 model_list
append_to_model_list() {
    model_list+=("$1")  # 将参数 $1 添加到数组
}

# 使用 seq 生成 step_60 到 step_360，并添加到 model_list
for step in $(seq 120 120 1200); do
   filename="step_${step}"
   append_to_model_list "$model_path/$filename"
done


append_to_model_list "$model_path"

script_dir=$(dirname "$(realpath "$0")")

cd $script_dir

cd ..
cd ..



# 打印 model_list 确认内容
echo "model_list 内容："
for item in "${model_list[@]}"; do

    echo "$item"
    #export CUDA_VISIBLE_DEVICES=0,1
    #torchrun --nproc-per-node=2 --master_port=25902 --no-python lm_eval --model hf \
    #    --model_args pretrained=$item \
    #    --tasks hellaswag,arc_challenge,arc_easy,piqa,openbookqa  \
    #    --output_path $item/metrics/commonsense \
    #    --batch_size 32 
#
    export CUDA_VISIBLE_DEVICES=0
    #python ./eval/mmlu/run_eval.py \
    #    --ntrain 5 \
    #    --data_dir ./data/eval_data/data/eval/mmlu \
    #    --save_dir $item/metrics/mmlu-5shot-openinstruct \
    #    --model_name_or_path $item \
    #    --tokenizer_name_or_path $item \
    #    --eval_batch_size 16  
        
        
     #python ./eval/pajama_ppl.py \
     #   --model_name_or_path $item \
     #   --max_iter 200 \
     #   --output_path $item \

    python ./eval/pile_ppl.py \
        --model_name_or_path $item \
        --max_iter 200 \
        --output_path $item \
    
    #for i in {1..3}
    #do
    #    export CUDA_VISIBLE_DEVICES=0,1
    #    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    #    
    #    lm_eval --model vllm \
    #        --model_args pretrained=$item,dtype=auto,tensor_parallel_size=2 \
    #        --tasks squadv2 \
    #        --num_fewshot 1 \
    #        --output_path $item/metrics/squadv2_lm_eval_1shot_vllm_run$i \
    #        --batch_size auto
    #        
    #        
    #    lm_eval --model vllm \
    #        --model_args pretrained=$item,dtype=auto,tensor_parallel_size=2 \
    #        --tasks gsm8k \
    #        --num_fewshot 5 \
    #        --output_path $item/metrics/gsm8k_lm_eval_5shot_vllm_run$i \
    #        --batch_size auto
    #        
    #        
    #        
    #    export CUDA_VISIBLE_DEVICES=0
    #    export HF_ALLOW_CODE_EVAL=1
    #    export TOKENIZERS_PARALLELISM=false
    #    
    #    python  ./eval/codex_humaneval/run_eval.py \
    #        --data_file ./data/eval_data/data/eval/codex_humaneval/HumanEval.jsonl.gz \
    #        --eval_pass_at_ks 10 \
    #        --unbiased_sampling_size_n 20 \
    #        --temperature 0.8 \
    #        --save_dir $item/metrics/humaneval_10k_8temp-openinstruct_run$i \
    #        --model $item \
    #        --tokenizer $item \
    #        --use_vllm
    #        
    #    done

done
