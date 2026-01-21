import os

# ================= 1. 核弹级离线配置 (必须放在最前面) =================
# 强制让 huggingface_hub 库认为自己在断网环境，禁止一切 metadata 查询
os.environ["HF_HUB_OFFLINE"] = "1" 
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import torch
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration

# ================= 2. 模型路径 =================
BASE_MODEL_ID = "/home/XZH/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c"

def load_base_model():
    print(f"🚀 Loading Base Model from {BASE_MODEL_ID}...")
    print("   (Mode: Strictly Offline / Force Cache)")
    
    # 1. 加载原始 Tokenizer
    # 注意：有了上面的 HF_HUB_OFFLINE=1，这里的 local_files_only 其实是双保险
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, 
        trust_remote_code=True,
        local_files_only=True 
    )

    # 2. 加载原始模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True 
    )
    
    model.eval()
    print("✅ Base Model Loaded Successfully!")
    return model, tokenizer

def generate_text(model, tokenizer, prompt_text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"\n🤖 User Asking: {prompt_text}")
    print("⏳ Generating...")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return output_text

# ================= 3. 主测试程序 =================
if __name__ == "__main__":
    try:
        model, tokenizer = load_base_model()

        # 测试 1: 通用知识 (考察它原本知不知道 Bennett)
        prompt_1 = "你知道Bennett空间机构是什么吗？"
        response_1 = generate_text(model, tokenizer, prompt_1)
        print(f"\n💬 Base Model Response 1:\n{'-'*50}\n{response_1}\n{'-'*50}")

    except Exception as e:
        print("\n❌ 依然报错？尝试备用方案：")
        print(f"Error: {e}")
        print("\n💡 提示：如果依然报错，请使用 'huggingface-cli scan-cache' 找到模型的真实绝对路径，替换 BASE_MODEL_ID。")