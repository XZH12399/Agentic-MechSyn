import os
import glob
import sys
from dotenv import load_dotenv

# 1. 先加载 .env 文件中的环境变量
# load_dotenv() 会自动寻找根目录下的 .env 文件，并将内容注入到 os.environ 中
load_dotenv() 

# ================= 1. 身份与网络配置 (在导入 torch 之前) =================
# 此时 os.environ 中已经有了 HF_TOKEN 等变量，您可以直接继续后面的导入
# 如果您想确保某些变量必须存在，可以加一个简单的检查（可选）：
if not os.getenv("HF_TOKEN"):
    print("警告: 未检测到 HF_TOKEN，请检查 .env 文件")

import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    Qwen2VLForConditionalGeneration, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# ================= 🎛️ 核心配置开关 =================
USE_MODEL = "7B" 
DATA_PATH = "dataset_builder/output/balanced_dataset/train_dataset_natural.json"

# --- 🛰️ 自动寻找模型缓存路径逻辑 ---
def get_model_path(model_type="7B"):
    if model_type == "7B":
        cache_base = "/home/XZH/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/*"
        fallback = "/home/XZH/projects/Agentic-MechSyn/Qwen2-VL-7B-Instruct"
    else:
        cache_base = "/home/XZH/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/*"
        fallback = "/home/XZH/projects/Agentic-MechSyn/Qwen2-VL-2B-Instruct"
        
    paths = glob.glob(cache_base)
    if paths:
        # 找到哈希路径，返回第一个
        resolved_path = paths[0]
        print(f"🔍 Found cached model at: {resolved_path}")
        return resolved_path
    else:
        print(f"⚠️ No cache found, using fallback: {fallback}")
        return fallback

# ================= 🛠️ 梯度监控回调 =================
class CheckGradCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "grad_norm" in logs:
            grad = logs["grad_norm"]
            if grad > 10.0:
                print(f"\n⚠️  [WARN] Step {state.global_step}: Grad_norm high ({grad:.2f})")
            if grad > 1000.0:
                print(f"\n🚨 [DANGER] Step {state.global_step}: Gradient Explosion! ({grad:.2e})")

# === 2. 参数自动配置 ===
MODEL_ID = get_model_path(USE_MODEL)

if USE_MODEL == "7B":
    print(f"🚀 [Mode] Qwen2-VL-7B High-Stability Tuning")
    OUTPUT_DIR = "/mnt/sda/xzh/vla_checkpoints_7b"
    PER_DEVICE_BATCH_SIZE = 2      
    GRADIENT_ACCUMULATION = 16      
    LEARNING_RATE = 2e-5            
    WARMUP_RATIO = 0.15             
    MAX_GRAD_NORM = 0.5             
    LORA_R = 64
    LORA_ALPHA = 128
else:
    print(f"🚀 [Mode] Qwen2-VL-2B Standard Tuning")
    OUTPUT_DIR = "/mnt/sda/xzh/vla_checkpoints"
    PER_DEVICE_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 1e-4
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    LORA_R = 16
    LORA_ALPHA = 32

# ================= 3. 数据处理 =================
def process_func(example, tokenizer):
    MAX_LENGTH = 1024 
    instruction = example["instruction"]
    output = example["output"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant for mechanism design."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    model_inputs = tokenizer(text, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = model_inputs.input_ids[0]
    attention_mask = model_inputs.attention_mask[0]
    
    user_messages = messages[:-1]
    user_text = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    user_input_ids = tokenizer(user_text, add_special_tokens=False).input_ids
    
    len_user_prompt = len(user_input_ids)
    labels = input_ids.clone()
    labels[:len_user_prompt] = -100
    labels[input_ids == tokenizer.pad_token_id] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def load_and_process_data(tokenizer):
    print(f"📂 Loading: {DATA_PATH}")
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    return ds.map(lambda x: process_func(x, tokenizer), remove_columns=ds.column_names, num_proc=4)

# ================= 4. 模型加载 =================
def load_model():
    print(f"📦 Loading Tokenizer/Model from: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager", 
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    model.gradient_checkpointing_enable()
    return model, tokenizer

def apply_lora(model):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=target_modules, bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

# ================= 5. 训练执行 =================
def train():
    model, tokenizer = load_model()
    model = apply_lora(model)
    train_dataset = load_and_process_data(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        num_train_epochs=3,
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        log_level="info"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[CheckGradCallback()]
    )
    
    print("🚀 Training Started...")
    trainer.train()
    trainer.save_model(output_dir=OUTPUT_DIR)
    print(f"✅ Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()