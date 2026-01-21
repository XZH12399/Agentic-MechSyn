import os
import torch
import numpy as np
import random
import time
import json
import math
from multiprocessing import Pool
from tqdm import tqdm

# 引入你的模块
from dataset_builder.generative.topology_gen import TopologyGenerator
from dataset_builder.generative.geometry_optim import DifferentiableOptimizer
from dataset_builder.generative.converter import MechanismConverter
from dataset_builder.tokenizer import MechanismTokenizer

# ================= 🎛️ 调试开关 (关键修改) =================
# True: 只生成 5 条数据并在终端打印，不保存大文件 (用于测试)
# False: 全速生成 10000 条数据并保存 JSON (用于生产)
TEST_MODE = False 
# ========================================================

# ================= 配置区域 =================
TOTAL_SAMPLES = 5 if TEST_MODE else 10000 
OUTPUT_DIR = "dataset_builder/output/balanced_dataset"
JSON_FILE_PATH = "dataset_builder/output/balanced_dataset/train_dataset_natural.json"

DOF_RATIOS = {1: 0.4, 2: 0.4, 3: 0.2}
NUM_WORKERS = 1 if TEST_MODE else 128  # 测试时单进程，方便看报错

# ================= 1. 螺旋理论分析引擎 =================
def analyze_screw_motion(screw_vectors):
    descriptions = []
    for s_vec in screw_vectors:
        s = np.array(s_vec, dtype=float).flatten()
        if len(s) < 6: continue
        direction, moment = s[:3], s[3:]
        mag_dir, mag_mom = np.linalg.norm(direction), np.linalg.norm(moment)
        EPS = 1e-4

        if mag_dir < EPS and mag_mom > EPS:
            descriptions.append("pure translational motion") # 增加形容词
        elif mag_dir > EPS:
            pitch = np.dot(direction, moment) / (mag_dir**2)
            if abs(pitch) < EPS:
                descriptions.append("pure rotational motion")
            else:
                descriptions.append(f"helical motion (pitch ~{pitch:.1f})")
    
    if not descriptions: return "complex general motion"
    return ", ".join(sorted(list(set(descriptions))))

# ================= 2. 扩充版 Prompt 生成器 =================
class PromptGenerator:
    def __init__(self):
        # A. 基础命令型 (Direct Command)
        self.cmd_templates = [
            "Design a mechanism with {dof} DoF and {loops} loop(s).",
            "Generate a {dof}-DoF, {loops}-loop linkage topology.",
            "Construct a spatial mechanism. Constraints: DoF={dof}, Loops={loops}.",
            "Synthesize a mechanism graph with {dof} degrees of freedom.",
            "Please create a mechanism design with {loops} independent loops.",
        ]
        
        # B. 需求描述型 (User Requirement)
        self.req_templates = [
            "I need a mechanism that has {loops} closed loop(s) and {dof} degree(s) of freedom.",
            "Can you provide a token sequence for a {dof}-DoF spatial mechanism?",
            "Looking for a mechanism solution with exactly {loops} loops.",
            "The target mechanism must possess {dof} DoF. Generate the structure.",
        ]
        
        # C. 工程参数型 (Engineering Specs)
        self.spec_templates = [
            "Input Specs:\n- DoF: {dof}\n- Loops: {loops}\nOutput: Mechanism Topology.",
            "Configuration: [DoF: {dof}, Loops: {loops}]. Generate action sequence.",
            "Mechanism Synthesis Task >> DoF: {dof} | Loops: {loops}.",
        ]

        # D. 运动描述型 (Motion Based - Advanced)
        self.motion_templates = [
            "Design a {dof}-DoF mechanism capable of {motion}.",
            "Create a mechanism with {loops} loop(s) that generates {motion} at the end-effector.",
            "Task: Synthesis of a spatial linkage. Target Motion: {motion}. (DoF={dof})",
            "Generate a mechanism structure where the output is {motion}.",
            "I want a manipulator that performs {motion} with {dof} degrees of freedom.",
        ]

    def get_prompt(self, dof, loops, screw_vectors=None):
        # 策略：混合多种风格，防止模型死记硬背
        
        # 1. 优先尝试运动描述 (40% 概率)
        if screw_vectors is not None and random.random() < 0.4:
            try:
                motion_desc = analyze_screw_motion(screw_vectors)
                template = random.choice(self.motion_templates)
                return template.format(dof=dof, loops=loops, motion=motion_desc)
            except:
                pass # 失败则回退
        
        # 2. 随机选择其他风格
        style = random.choice(['cmd', 'req', 'spec'])
        
        if style == 'cmd':
            return random.choice(self.cmd_templates).format(dof=dof, loops=loops)
        elif style == 'req':
            return random.choice(self.req_templates).format(dof=dof, loops=loops)
        else:
            return random.choice(self.spec_templates).format(dof=dof, loops=loops)

# ===========================================

def generate_one_sample(target_dof):
    """
    工作进程：尝试生成一个符合 target_dof 的样本
    """
    # 1. 强制 PyTorch 和 NumPy 单线程 (防止多进程死锁)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # 防止随机数冲突
    np.random.seed(os.getpid() + int(time.time() * 1000) % 10000)
    
    # 2. 动态设定节点数范围
    # 这里的逻辑是为了提高生成特定 DoF 的成功率
    if target_dof == 1:
        num_nodes = random.randint(4, 6) 
    elif target_dof == 2:
        num_nodes = random.randint(5, 10)
    else:
        num_nodes = random.randint(9, 12)

    try:
        # 3. 初始化工具
        device = 'cpu'
        
        # === 关键修正 ===
        # 在这里初始化，直接锁死 min_nodes 和 max_nodes
        topo_gen = TopologyGenerator(
            min_nodes=num_nodes,
            max_nodes=num_nodes, 
            loop_probs={3: 0.1, 4: 0.9}
        )
        geo_optim = DifferentiableOptimizer(device=device)
        converter = MechanismConverter(device=device)
        
        # A. 生成拓扑
        # 你的类返回的是 G 和 cycles
        G, cycles = topo_gen.generate()
        
        if not cycles: return None 
        
        # B. 几何优化
        # max_steps=1000 足够筛选了，不收敛就放弃
        success, P, Z, final_types, final_dof, null_motion = geo_optim.optimize_mobility(
            G, cycles, target_dof=target_dof, max_steps=1000, verbose=False
        )
        
        # C. 筛选条件：优化成功 且 自由度匹配
        if success and final_dof == target_dof:
            unique_id = f"{int(time.time())}_{os.getpid()}_{random.randint(1000,9999)}"
            data = converter.process(
                G, P, Z, final_types, final_dof, null_motion, 
                mech_id=unique_id, 
                num_task_samples=3 
            )
            return data
            
    except Exception as e:
        # 打印错误方便调试，但不中断进程
        # print(f"Error in worker: {e}") 
        pass
        
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if TEST_MODE:
        print("\n" + "="*60)
        print("🚧 TEST MODE ACTIVE: Only generating 5 samples for inspection 🚧")
        print("="*60 + "\n")
        targets = {1: 2, 2: 2, 3: 1} # 测试时每种都生成一点
    else:
        targets = {dof: int(TOTAL_SAMPLES * ratio) for dof, ratio in DOF_RATIOS.items()}
    
    print(f"🎯 目标设定: {targets}")
    print(f"🚀 启动 {NUM_WORKERS} 个进程...")
    
    tokenizer = MechanismTokenizer()
    prompt_gen = PromptGenerator()
    final_dataset = []
    
    pbars = {dof: tqdm(total=count, desc=f"DoF-{dof}", position=i) for i, (dof, count) in enumerate(targets.items())}
    
    with Pool(NUM_WORKERS) as pool:
        while sum(targets.values()) > 0:
            needed_dofs = [dof for dof, count in targets.items() if count > 0]
            if not needed_dofs: break
            
            # 测试模式下批次小一点
            batch_size = 5 if TEST_MODE else NUM_WORKERS * 4
            task_args = [random.choice(needed_dofs) for _ in range(batch_size)]
            results = pool.map(generate_one_sample, task_args)
            
            for res in results:
                if res is not None:
                    dof = res['meta']['dof']
                    if dof in targets and targets[dof] > 0:
                        targets[dof] -= 1
                        pbars[dof].update(1)
                        
                        graph_data = res['graph']
                        tasks = res['meta']['tasks']
                        num_loops = res['meta']['num_loops']
                        
                        for task in tasks:
                            screws = task['motion_screws']
                            instruction = prompt_gen.get_prompt(dof, num_loops, screws)
                            
                            mech_tokens = tokenizer.encode_graph(
                                graph_data, base_ids=task['base_link'], ee_ids=task['ee_link']
                            )
                            mech_tokens.append("<EOS>")
                            target_string = " ".join(mech_tokens)
                            
                            entry = {"instruction": instruction, "input": "", "output": target_string}
                            final_dataset.append(entry)
                            
                            # === 👁️ 核心测试功能：打印预览 ===
                            if TEST_MODE:
                                print(f"\n[{len(final_dataset)}] Sample Preview:")
                                print(f"📝 Instruction: {instruction}")
                                print(f"🤖 Output (First 50 chars): {target_string[:50]}...")
                                print("-" * 40)
                                if len(final_dataset) >= 5: # 测试够了就强制退出
                                    print("✅ Test limit reached.")
                                    return 
    
    for p in pbars.values(): p.close()
    
    if not TEST_MODE:
        print("\n📦 正在保存 JSON 数据集...")
        with open(JSON_FILE_PATH, 'w') as f:
            json.dump(final_dataset, f, indent=4)
        print(f"✅ 完成！共生成 {len(final_dataset)} 条训练样本")
        print(f"💾 数据集保存至: {JSON_FILE_PATH}")

if __name__ == "__main__":
    main()