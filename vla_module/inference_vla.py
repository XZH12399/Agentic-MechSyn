import os
import re
import torch
import glob
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration
from peft import PeftModel

# ================= 🎛️ 1. 模型切换与硬件配置 =================
USE_MODEL = "7B" 

# 锁定显卡 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 强制离线模式
os.environ["HF_HUB_OFFLINE"] = "1"

# --- 自动寻找 7B 模型缓存路径 ---
def get_7b_cache_path():
    # 标准 Hugging Face 缓存根目录
    cache_base = "/home/XZH/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/*"
    paths = glob.glob(cache_base)
    if not paths:
        return "/home/XZH/projects/Agentic-MechSyn/Qwen2-VL-7B-Instruct" # 回退到项目目录
    return paths[0] # 返回找到的第一个哈希快照路径

if USE_MODEL == "7B":
    print("🚀 Inference Mode: [7B] High Intelligence")
    # 自动获取类似 /home/XZH/.cache/huggingface/hub/.../snapshots/xxxx 的路径
    BASE_MODEL_ID = get_7b_cache_path()
    ADAPTER_PATH = "/mnt/sda/xzh/vla_checkpoints_7b/checkpoint-2814" 
else:
    print("🚀 Inference Mode: [2B] Standard Speed")
    # 这里请填入你之前 2B 的那个完整长路径
    BASE_MODEL_ID = "/home/XZH/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c"
    ADAPTER_PATH = "./vla_checkpoints/checkpoint-xxx"

# ================= 2. 加载函数 =================
def load_model_and_tokenizer():
    # 预检路径
    abs_base_path = os.path.abspath(BASE_MODEL_ID)
    abs_adapter_path = os.path.abspath(ADAPTER_PATH)

    if not os.path.exists(abs_base_path):
        raise FileNotFoundError(f"❌ 找不到底座模型绝对路径: {abs_base_path}")
    if not os.path.exists(abs_adapter_path):
        raise FileNotFoundError(f"❌ 找不到 Adapter 路径: {abs_adapter_path}")

    print(f"📦 Loading Tokenizer from: {abs_base_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        abs_base_path, 
        trust_remote_code=True,
        local_files_only=True
    )

    print(f"📦 Loading Base Model from: {abs_base_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        abs_base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto", 
        trust_remote_code=True,
        local_files_only=True
    )

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("🔧 Resizing token embeddings...")
        model.resize_token_embeddings(len(tokenizer))

    print(f"🔗 Loading LoRA Adapter: {abs_adapter_path}")
    model = PeftModel.from_pretrained(model, abs_adapter_path)
    
    model.eval()
    print("✅ Model loaded successfully from absolute path!")
    return model, tokenizer

# ================= 3. 推理生成函数 =================
def generate_mechanism(model, tokenizer, prompt_text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant for mechanism design."},
        {"role": "user", "content": prompt_text}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"\n💬 Input Prompt: {prompt_text}")
    print("⏳ Generating...")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text

# ================= 4. 解码可视化报告 =================
def parse_and_print_mechanism(token_string):
    print("\n" + "="*115)
    print("🧩 机构全息解码报告 (Agentic-MechSyn Report)")
    print("="*115)
    
    header = f"{'拓扑动作 (Topology)':<35} | {'公共垂线 (Common Normal)':<24} | {'源轴 (Source Axis)':<24} | {'目标轴 (Target Axis)':<24}"
    print(header)
    print("-" * 115)

    all_actions = list(re.finditer(r"(<Action_[a-zA-Z_]+>)", token_string))
    param_pattern = re.compile(r"<([a-zA-Z]+)_([a-zA-Z0-9]+)>")

    current_source = "Base"

    for i, match in enumerate(all_actions):
        action_tag = match.group(1)
        start_idx = match.end()
        end_idx = all_actions[i+1].start() if i+1 < len(all_actions) else len(token_string)
        content_str = token_string[start_idx:end_idx]
        
        params = param_pattern.findall(content_str)
        p_dict = {'ID': '?', 'Type': '', 'Role': ''}
        geo_vals = {'Len': [], 'Twist': [], 'Off': [], 'State': []}
        
        for key, val in params:
            if key in p_dict: p_dict[key] = val
            elif key in geo_vals: geo_vals[key].append(val)

        a_val = geo_vals['Len'][0] if geo_vals['Len'] else "-"
        alpha_val = geo_vals['Twist'][0] if geo_vals['Twist'] else "-"
        d_src = geo_vals['Off'][0] if len(geo_vals['Off']) > 0 else "-"
        d_tgt = geo_vals['Off'][1] if len(geo_vals['Off']) > 1 else "-"
        theta_src = geo_vals['State'][0] if len(geo_vals['State']) > 0 else "-"
        theta_tgt = geo_vals['State'][1] if len(geo_vals['State']) > 1 else "-"

        common_str = f"a={a_val}, α={alpha_val}"
        src_str = f"ds={d_src}, θs={theta_src}"
        tgt_str = f"dt={d_tgt}, θt={theta_tgt}"

        target_id = p_dict['ID']
        type_info = f"[{p_dict['Type']}{'-' + p_dict['Role'] if p_dict['Role'] else ''}]"
        
        action_desc = ""
        if action_tag == "<Action_New_Node>":
            action_desc = f"🔹 基座 {target_id} {type_info}"
            current_source = target_id
            common_str = src_str = tgt_str = ""
        elif action_tag == "<Action_Link_To_New>":
            action_desc = f" ├── 🔗 {current_source} -> {target_id} {type_info}"
            current_source = target_id
        elif action_tag == "<Action_Link_To_Old>":
            action_desc = f" └── 🔄 闭环 {current_source} -> {target_id}"
        elif action_tag == "<Action_Jump_To>":
            action_desc = f"🚀 跳转焦点 -> {target_id}"
            current_source = target_id
            common_str = src_str = tgt_str = ""

        if action_desc:
            print(f"{action_desc:<35} | {common_str:<24} | {src_str:<24} | {tgt_str:<24}")
    print("="*115)

# ================= 5. 主程序 =================
if __name__ == "__main__":
    try:
        model, tokenizer = load_model_and_tokenizer()
        # test_prompt = "我老婆让我Design a mechanism with 1 loop，应该怎么回复。"
        # test_prompt = "今天是跨年夜，我适合给老婆买什么礼物？"
        test_prompt = "Design a mechanism with 2 DoFs and 3 loops."
        # test_prompt = "Design a bennett mechanism."
        # test_prompt = "你知道Bennett机构是什么吗？以及这个机构有什么特点？"
        # test_prompt = "先有鸡还是先有蛋。"
        # test_prompt = "请给我设计一个像鸡蛋一样的机构。"
        result = generate_mechanism(model, tokenizer, test_prompt)
        print(f"\n🛠️  Raw Token Stream:\n{result}")
        parse_and_print_mechanism(result)
    except Exception as e:
        print(f"❌ 运行失败: {e}")