# -*- coding: utf-8 -*-
# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/250010186/hf


cp /data/250010186/nltk/nltk_data-gh-pages/packages/tokenizers/punkt_tab.zip /root/nltk_data/tokenizers
cp /data/250010186/nltk/nltk_data-gh-pages/packages/tokenizers/punkt.zip /root/nltk_data/tokenizers
cd /root/nltk_data/tokenizers
unzip punkt.zip
unzip punkt_tab.zip

model_path="/data/250010186/hf/hub/Qwen3-4B-Base"

script_dir=$(dirname "$(realpath "$0")")

cd $script_dir

cd ..
cd ..

#"/data/250010186/SFT_model/homework/Qwen4B_SFT_MoFO020/finetune"
model_list=(
    /data/250010186/SFT_model/homework/Qwen4B_DPO_MoFO020_from_Qwen4B_SFT_MoFO020/dpo_tune_cache__8__1766755255/step_450
    /data/250010186/SFT_model/homework/Qwen4B_DPO_MoFO020_from_Qwen4B_SFT_MoFO020/dpo_tune_cache__8__1766755255/step_300
    /data/250010186/SFT_model/homework/Qwen4B_DPO_MoFO020_from_Qwen4B_SFT_MoFO020/dpo_tune_cache__8__1766755255/step_150
    /data/250010186/SFT_model/homework/Qwen4B_SFT_MoFO020/finetune/step_1800)

# append_to_model_list() {
#     model_list+=("$1")
# }

# append_to_model_list "$model_path"

# 打印 model_list 确认内容
echo "model_list 内容："
for item in "${model_list[@]}"; do
#torchrun --nproc-per-node=1 --master_port=25902 --no-python lm_eval --model hf \

    echo "$item"
    export CUDA_VISIBLE_DEVICES=0
    lm_eval --model hf \
       --model_args pretrained=$item \
       --tasks hellaswag,arc_challenge,arc_easy,piqa,winogrande,commonsense_qa  \
       --output_path $item/metrics/commensense \
       --batch_size 32

    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    lm_eval --model vllm \
           --model_args pretrained=$item,dtype=auto \
           --tasks gsm8k \
           --num_fewshot 5 \
           --output_path $item/metrics/gsm8k_lm_eval_5shot \
           --batch_size auto

    lm_eval --model vllm \
           --model_args pretrained=$item,dtype=auto \
           --tasks ifeval \
           --output_path $item/metrics/ifeval_lm_eval_5shot \
           --batch_size auto


    export CUDA_VISIBLE_DEVICES=0
    python ./eval/mmlu/run_eval.py \
       --ntrain 5 \
       --data_dir ./data/eval_data/data/eval/mmlu \
       --save_dir $item/metrics/mmlu-5shot-openinstruct \
       --model_name_or_path $item \
       --tokenizer_name_or_path $item \
       --eval_batch_size 16 \


        
    
    export CUDA_VISIBLE_DEVICES=0
    export HF_ALLOW_CODE_EVAL=1
    export TOKENIZERS_PARALLELISM=false
    
    python  ./eval/codex_humaneval/run_eval.py \
        --data_file ./data/eval_data/data/eval/codex_humaneval/HumanEval.jsonl.gz \
        --eval_pass_at_ks 10 \
        --unbiased_sampling_size_n 20 \
        --temperature 0.8 \
        --save_dir $item/metrics/humaneval_10k_8 \
        --model $item \
        --tokenizer $item \
        --use_vllm
    
done

