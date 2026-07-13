#!/bin/bash
# 用法: bash eval_homework_ifeval.sh <model_path>
set -e
item="$1"
if [ -z "$item" ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/250010186/hf
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ifeval 依赖 nltk 的 punkt/punkt_tab 分词数据；若只存在 zip 未解压，
# nltk 会把 zip 当目录拼接路径读取，报 NotADirectoryError。这里检测并解压。
NLTK_TOKENIZERS_DIR="/root/nltk_data/tokenizers"
for pkg in punkt punkt_tab; do
    zip_path="$NLTK_TOKENIZERS_DIR/$pkg.zip"
    dir_path="$NLTK_TOKENIZERS_DIR/$pkg"
    if [ -f "$zip_path" ] && [ ! -d "$dir_path" ]; then
        echo "Unzipping $zip_path ..."
        unzip -o -q "$zip_path" -d "$NLTK_TOKENIZERS_DIR"
    fi
done

echo "Running ifeval eval on: $item"
lm_eval --model vllm \
   --model_args pretrained=$item,dtype=auto \
   --tasks ifeval \
   --output_path $item/metrics/ifeval_lm_eval \
   --batch_size auto
