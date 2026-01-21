import os
import json
import torch
from tqdm import tqdm
from tokenizer import MechanismTokenizer


def process_data(data_dir, output_file):
    # 1. 初始化 Tokenizer
    tokenizer = MechanismTokenizer()

    all_encoder_inputs = []
    all_decoder_targets = []

    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"🚀 开始处理 {len(files)} 个文件...")

    max_seq_len = 0
    valid_samples = 0

    for fname in tqdm(files):
        path = os.path.join(data_dir, fname)
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            graph_data = data['graph']
            meta_data = data['meta']  # 获取元数据
            tasks = meta_data.get('tasks', [])

            # 提取机构属性
            dof = meta_data.get('dof', 1)
            num_loops = meta_data.get('num_loops', 1)

            for task in tasks:
                base_ids = task['base_link']
                ee_ids = task['ee_link']
                # ✨ 字段名变更: motion_screw -> motion_screws
                screws = task['motion_screws']

                # --- 1. Encoder Input ---
                spec_tokens = tokenizer.encode_specs(dof, num_loops)

                # encode_task 现在接受列表的列表
                screw_tokens = tokenizer.encode_task(screws)

                # 现在的输入长度应该是: 2 (Specs) + 6*3 (Screws) = 20
                input_tokens = spec_tokens + screw_tokens
                input_ids = [tokenizer.token2id.get(t, 0) for t in input_tokens]

                # --- 2. Decoder Target: Mechanism ---
                mech_tokens = tokenizer.encode_graph(
                    graph_data,
                    base_ids=base_ids,
                    ee_ids=ee_ids
                )
                mech_tokens.append("<EOS>")
                mech_ids = [tokenizer.token2id.get(t, tokenizer.token2id["<PAD>"]) for t in mech_tokens]

                if len(mech_ids) > max_seq_len:
                    max_seq_len = len(mech_ids)

                all_encoder_inputs.append(input_ids)
                all_decoder_targets.append(mech_ids)
                valid_samples += 1

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print(f"📊 样本总数: {valid_samples}")
    print(f"📏 最大序列长度: {max_seq_len}")

    if valid_samples == 0:
        print("❌ 未生成任何样本")
        return

    # --- Padding & Saving ---
    # Encoder Input 现在的长度应该是 2 + 6 = 8
    # 如果未来Specs长度不固定，这里也需要Pad
    max_src_len = max(len(x) for x in all_encoder_inputs)
    src_tensor = torch.full((valid_samples, max_src_len), tokenizer.token2id["<PAD>"], dtype=torch.long)

    for i, seq in enumerate(all_encoder_inputs):
        src_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    tgt_tensor = torch.full((valid_samples, max_seq_len), tokenizer.token2id["<PAD>"], dtype=torch.long)
    for i, seq in enumerate(all_decoder_targets):
        tgt_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    torch.save({
        "src": src_tensor,
        "tgt": tgt_tensor,
        "vocab": tokenizer.vocab,
        "token2id": tokenizer.token2id
    }, output_file)

    print(f"💾 数据集已保存至: {output_file}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "output", "generated_v1")
    output_file = os.path.join(current_dir, "output", "train_dataset.pt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    process_data(input_dir, output_file)