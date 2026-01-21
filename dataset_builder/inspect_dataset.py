import torch
import os
import random


def inspect_data(pt_file_path):
    if not os.path.exists(pt_file_path):
        print(f"❌ 找不到文件: {pt_file_path}")
        return

    print(f"🚀 正在加载数据集: {pt_file_path} ...")
    # 加载 .pt 文件
    checkpoint = torch.load(pt_file_path)

    src_tensor = checkpoint['src']  # Encoder Input (Task / Motion Screw)
    tgt_tensor = checkpoint['tgt']  # Decoder Target (Mechanism Graph)
    vocab = checkpoint['vocab']  # 词表 list

    # 构建 ID -> Token 映射
    id2token = {i: t for i, t in enumerate(vocab)}

    num_samples = src_tensor.shape[0]
    print(f"📊 数据集包含 {num_samples} 个样本")
    print(f"   Src Shape: {src_tensor.shape} (Task Sequence)")
    print(f"   Tgt Shape: {tgt_tensor.shape} (Mechanism Sequence)")

    # 随机抽取一个样本
    idx = random.randint(0, num_samples - 1)
    print(f"\n🔍 查看样本 ID: {idx}")

    # --- 1. 查看 Input (Task) ---
    raw_src = src_tensor[idx].tolist()
    # 过滤掉 Padding (0)
    readable_src = [id2token.get(i, f"<UNK_{i}>") for i in raw_src if i != 0]

    print(f"\n[Encoder Input - Task/Screw]:")
    print(f"Raw IDs: {raw_src}")
    print(f"Tokens : {readable_src}")

    # --- 2. 查看 Label (Mechanism) ---
    raw_tgt = tgt_tensor[idx].tolist()
    # 过滤掉 Padding (0)
    readable_tgt = [id2token.get(i, f"<UNK_{i}>") for i in raw_tgt if i != 0]

    print(f"\n[Decoder Label - Mechanism Graph]:")
    print(f"Raw IDs: {raw_tgt[:20]} ... (只显示前20个)")

    # 格式化打印 Token 序列，方便阅读
    print(f"Tokens :")
    formatted_output = []
    indent = 0
    for token in readable_tgt:
        # 简单的缩进格式化
        if "Action_New_Node" in token or "Action_Jump_To" in token:
            print(" ".join(formatted_output))
            formatted_output = [token]
        else:
            formatted_output.append(token)
    if formatted_output:
        print(" ".join(formatted_output))


if __name__ == "__main__":
    # 确保路径正确
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pt_path = os.path.join(current_dir, "output", "train_dataset.pt")

    inspect_data(pt_path)