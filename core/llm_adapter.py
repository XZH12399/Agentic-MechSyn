import torch
from torch.utils.data import Dataset
from dataset_builder.tokenizer import MechanismTokenizer

class LLMMechanismDataset(Dataset):
    def __init__(self, pt_file_path, llm_tokenizer, max_length=1024):
        """
        :param pt_file_path: 你的 train_dataset.pt 路径
        :param llm_tokenizer: Qwen2-VL 的分词器
        """
        # 1. 加载你生成的原始 .pt 数据
        print(f"📂 Loading PT data from {pt_file_path}...")
        checkpoint = torch.load(pt_file_path)
        self.src_data = checkpoint['src'] # 输入: Specs + Screws
        self.tgt_data = checkpoint['tgt'] # 输出: 机构 Graph
        
        # 2. 恢复你的物理 Tokenizer (为了把 ID 变回字符串)
        self.mech_tokenizer = MechanismTokenizer()
        # 强制同步词表 (非常重要!)
        self.mech_tokenizer.vocab = checkpoint['vocab']
        self.mech_tokenizer.id2token = {i: t for i, t in enumerate(self.mech_tokenizer.vocab)}
        
        self.llm_tokenizer = llm_tokenizer
        self.max_length = max_length
        
        # 预计算 Prompt 模板
        self.system_prompt = "You are an expert in mechanism design. Generate a mechanism topology based on the given motion requirements."

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        # === A. 还原数据 ===
        # 把 ID [10, 25...] 变成字符串 ["<DoF_1>", "<Off_0.5>..."]
        
        # 1. 处理输入 (Source)
        raw_src = self.src_data[idx].tolist()
        src_tokens = [self.mech_tokenizer.id2token.get(i, "") for i in raw_src if i != 0] # 去掉 PAD
        src_str = "".join(src_tokens) # 连成一整条字符串
        
        # 2. 处理输出 (Target)
        raw_tgt = self.tgt_data[idx].tolist()
        tgt_tokens = [self.mech_tokenizer.id2token.get(i, "") for i in raw_tgt if i != 0]
        tgt_str = "".join(tgt_tokens)
        
        # === B. 构建对话格式 ===
        # Qwen2-VL 推荐的对话格式
        conversation = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": f"Design Specs: {src_str}"
            },
            {
                "role": "assistant",
                "content": tgt_str
            }
        ]
        
        # === C. 使用 LLM 的模板工具进行编码 ===
        # apply_chat_template 会自动处理 <|im_start|>user... 等特殊符
        text = self.llm_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 编码为 Tensor
        encoding = self.llm_tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        
        # === D. 构造 Labels (Loss 计算掩码) ===
        # 我们只训练 Assistant 回复的部分，Mask 掉 System 和 User 的部分
        labels = input_ids.clone()
        
        # 简单策略：找到 "assistant" 标签后的内容开始训练
        # Qwen2-VL 的 assistant 引导符通常包含 "\n<|im_start|>assistant\n"
        # 这里为了简化，我们让模型全量预测（Prompt Loss 不会太大影响），或者使用 DataCollatorMask
        # 对于初学者，直接让 labels = input_ids 也是可以跑通的，只是效率略低
        # 为了严谨，我们将 Padding 部分设为 -100
        labels[labels == self.llm_tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }