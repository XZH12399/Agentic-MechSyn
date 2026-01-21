import os

# === 1. 修复 OMP 冲突报错 (必须放在最前面) ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import torch
import random
from topology_gen import TopologyGenerator
from geometry_optim import DifferentiableOptimizer
from converter import MechanismConverter


def main():
    # === 配置区域 ===
    OUTPUT_DIR = "../output/generated_v1"
    NUM_SAMPLES = 10  # 想要生成的机构总数
    TASKS_PER_MECH = 5  # 每个机构生成的任务(Base-EE)对数
    TARGET_DOF = 2  # 🎯 目标自由度 (可以设为 1, 2, 3...)

    # 拓扑生成配置
    # 如果想要多自由度，通常需要更复杂的拓扑 (更多的节点/环路)
    # 对于 1-DoF，4-6 节点足够；对于 2-DoF，建议 6-8 节点
    MIN_NODES = 4
    MAX_NODES = 8

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 模块初始化 ===
    # loop_probs: 控制生成 3节点环、4节点环 的概率
    topo_gen = TopologyGenerator(
        min_nodes=MIN_NODES,
        max_nodes=MAX_NODES,
        loop_probs={3: 0.1, 4: 0.9}
    )
    geo_optim = DifferentiableOptimizer(device=DEVICE)
    converter = MechanismConverter(device=DEVICE)

    count = 0
    attempts = 0

    print(f"🚀 开始生成任务: 目标 {NUM_SAMPLES} 个有效机构 (Target DoF={TARGET_DOF})")
    print(f"   📂 输出目录: {OUTPUT_DIR}")

    while count < NUM_SAMPLES:
        attempts += 1
        print(f"\n--- Attempt {attempts} (Collected: {count}/{NUM_SAMPLES}) ---")

        # 1. 生成拓扑
        G, cycles = topo_gen.generate()

        # 2. 几何优化
        # === 修改处：传入 target_dof 参数 ===
        success, P, Z, joint_types, dof, null_motion = geo_optim.optimize_mobility(
            G,
            cycles,
            target_dof=TARGET_DOF
        )

        if success:
            # 3. 数据转换与分析
            mech_id = f"gen_mech_{int(time.time())}_{count:03d}"

            # converter 处理
            json_data = converter.process(
                G, P, Z, joint_types,
                dof, null_motion, mech_id,
                num_task_samples=TASKS_PER_MECH
            )

            # 4. 保存文件
            save_path = os.path.join(OUTPUT_DIR, f"{mech_id}.json")
            with open(save_path, 'w') as f:
                json.dump(json_data, f, indent=4)

            print(f"💾 Saved: {save_path} (DoF: {dof}, Loops: {len(cycles)})")
            count += 1
        else:
            print("❌ Optimization failed, retrying...")


if __name__ == "__main__":
    main()