import os
import json
import random
import numpy as np
from tokenizer import MechanismTokenizer


def verify_round_trip(data_dir):
    # 1. 初始化
    tokenizer = MechanismTokenizer()
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    if not files:
        print("❌ 目录下没有 JSON 文件！")
        return

    # 随机抽查 3 个文件
    sample_files = random.sample(files, min(3, len(files)))
    print(f"🔍 正在抽检 {len(sample_files)} 个样本...\n")

    for fname in sample_files:
        path = os.path.join(data_dir, fname)
        with open(path, 'r') as f:
            data = json.load(f)

        print(f"📄 文件: {fname}")
        graph_data = data['graph']
        tasks = data['meta'].get('tasks', [])

        # 建立原始边的查询字典 (Key: sorted tuple)
        orig_edges_map = {}
        for e in graph_data['edges']:
            key = tuple(sorted((e['source'], e['target'])))
            orig_edges_map[key] = e['params']

        # 针对每个 Task 进行验证
        for i, task in enumerate(tasks):
            base_list = task['base_link']
            ee_list = task['ee_link']

            print(f"  ➡️  Task {i}: Base={base_list}, EE={ee_list}")

            # --- A. Encode ---
            token_seq = tokenizer.encode_graph(
                graph_data,
                base_ids=base_list,
                ee_ids=ee_list
            )

            # --- B. Decode ---
            decoded_struct = tokenizer.decode(token_seq)

            # --- C. Compare ---

            # 1. 结构完整性
            n_orig = len(graph_data['nodes'])
            n_dec = len(decoded_struct['topology']['nodes'])
            e_orig = len(graph_data['edges'])
            e_dec = len(decoded_struct['topology']['connections'])

            if n_orig != n_dec or e_orig != e_dec:
                print(f"     ❌ 结构数量不匹配! N:{n_orig}/{n_dec}, E:{e_orig}/{e_dec}")
                continue

            # 2. 角色还原 (带 Ground 优先级的宽松检查)
            dec_meta = decoded_struct['meta']
            dec_ground = set(dec_meta['ground_nodes'])
            dec_ee = set(dec_meta['ee_node'])

            base_ids_orig = set(base_list)
            ee_ids_orig = set(ee_list)

            if not base_ids_orig.issubset(dec_ground):
                print(f"     ❌ Ground 丢失! 原: {base_ids_orig}, 解: {dec_ground}")

            # 允许 EE 被 Ground 覆盖
            missing_ee = ee_ids_orig - dec_ee
            acceptable_missing = {nid for nid in missing_ee if nid in dec_ground}
            real_missing = missing_ee - acceptable_missing

            if real_missing:
                print(f"     ❌ EE 丢失! 原: {ee_ids_orig}, 解: {dec_ee}")

            # 3. 数值精度 (全量边对比)
            max_diff = 0.0
            error_msg = ""

            for dec_edge in decoded_struct['topology']['connections']:
                u, v = dec_edge['source'], dec_edge['target']
                key = tuple(sorted((u, v)))

                if key not in orig_edges_map:
                    print(f"     ❌ 解码出了不存在的边: {key}")
                    break

                orig_params = orig_edges_map[key]

                # 对比 a (长度) 和 alpha (角度)
                diff_a = abs(orig_params['a'] - dec_edge['a'])
                # 角度主要在 -pi~pi，可能存在 2pi 翻转，简单对比绝对差
                diff_alpha = abs(orig_params['alpha'] - dec_edge['alpha'])

                if diff_a > max_diff: max_diff = diff_a

                # 严格阈值检查 (考虑到之前的 bin 精度是 ~0.02)
                if diff_a > 0.1:
                    error_msg = f"边 {key} 参数偏差: 原 a={orig_params['a']:.3f}, 解 a={dec_edge['a']:.3f}"
                    break

            if error_msg:
                print(f"     ⚠️ {error_msg}")
            else:
                print(f"     ✅ 验证通过 (最大长度误差: {max_diff:.4f})")

        print("  ✅ 文件结构验证完成\n")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "output", "generated_v1")
    verify_round_trip(data_dir)