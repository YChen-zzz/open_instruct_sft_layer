import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
    GPTNeoXModel
)

from transformers import GPTNeoXForCausalLM, AutoTokenizer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name_or_path", 
    type=str, 
    default="xx",
    help="Path to the huggingface models."
)    
parser.add_argument(
    "--max_iter", 
    type=int, 
    default=1000,
    help="max_iter for eval ppl"
)
parser.add_argument(
    "--output_path", 
    type=str, 
    default=None, 
    help="If specified, we will save ppl into this path"
)

args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=True)
import torch
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,device_map="auto",trust_remote_code=True,low_cpu_mem_usage=True)
df = pd.read_parquet("/chenyupeng/old_files/yupeng_gpt/lit_gpt/litgpt-main/litgpt/finetune/0000.parquet")

# 假设df是包含文本数据的DataFrame，tokenizer和model已经正确初始化
loss_total = []
#model = model.cuda()  # 将模型移动到GPU上，只需要做一次
model.eval()  # 设置模型为评估模式

for i in range(args.max_iter):
    text = df['text'][i]
    texts = tokenizer(text, return_tensors="pt")
    input_ids = texts["input_ids"][:,:-1].cuda()
    attention_mask = texts["attention_mask"][:,:-1].cuda()
    
    # 假设目标是输入的下一个词的索引
    targets = texts["input_ids"][:,1:].cuda()
    
    # 计算logits
    logit = model(input_ids=input_ids, attention_mask=attention_mask,return_dict=True)
    
    # 使用cross_entropy计算损失
    loss = torch.nn.functional.cross_entropy(logit.logits[0,:,:], targets.view(-1))
    
    loss_total.append(loss.item())

print("average loss: ", np.mean(loss_total))

import json

# 创建一个字典，将 loss 值存储在其中
loss_data = {
    'loss': np.mean(loss_total)
}

# 指定要存储的 JSON 文件路径
file_path = args.output_path + '/pile_loss.json'

# 将字典转换为 JSON 格式并写入文件
with open(file_path, 'w') as json_file:
    json.dump(loss_data, json_file, indent=4)

print(f'Loss value has been saved to {file_path}')


