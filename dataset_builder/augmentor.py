import json
import copy
import numpy as np
import os
import sys

# 引用项目根目录下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.optimizer_engine import MechanismOptimizer
from utils.tensor_adapter import MechanismTensorAdapter
from tools.evaluator_engine import MechanismEvaluator


class DataAugmentor:
    def __init__(self, optimizer_cfg):
        self.optimizer = MechanismOptimizer(optimizer_cfg)
        self.evaluator = MechanismEvaluator()
        # 假设最大节点数为 8，足够涵盖四杆/六杆机构
        self.adapter = MechanismTensorAdapter(num_nodes=8)

        # === 核心：机构类型约束配置 ===
        # 针对 Bennett 机构，修复时必须开启 bennett_ratio 约束
        self.type_constraints = {
            "Bennett": {
                "tools": ["closure_loop", "bennett_ratio", "mobility_dof"],
                # 给 Bennett 比例约束极高的权重，确保数学性质不被破坏
                "weights": {"closure_loop": 10.0, "bennett_ratio": 100.0}
            },
            "General_Spatial": {
                "tools": ["closure_loop", "mobility_dof"],
                "weights": {"closure_loop": 10.0}
            }
        }

    def augment_from_seed(self, seed_file, num_variants=50, noise_base=0.05):
        """
        读取种子 -> 变异 -> 修复 -> 返回新数据列表
        """
        with open(seed_file, 'r', encoding='utf-8') as f:
            seed_data = json.load(f)

        mech_type = seed_data['meta'].get('mech_type', 'General_Spatial')
        print(f"🔧 Processing Seed: {seed_data['id']} (Type: {mech_type})")

        # 获取该类型的约束配置
        constraint_config = self.type_constraints.get(mech_type, self.type_constraints["General_Spatial"])

        # 1. 提取初始 Tensor (几何)
        seed_tensor = self.adapter.json_to_tensor(seed_data['graph'])

        # 2. 提取初始 q (状态) - 用于热启动优化器
        # 我们需要手动解析 JSON 中的 initial_state 填充到 q_matrix
        N = seed_tensor.shape[1]
        seed_q = np.zeros((N, N))
        for edge in seed_data['graph']['edges']:
            u, v = edge['source'], edge['target']
            q_src = edge.get('initial_state', {}).get('q_source')
            q_tgt = edge.get('initial_state', {}).get('q_target')
            if q_src is not None: seed_q[u, v] = float(q_src)
            if q_tgt is not None: seed_q[v, u] = float(q_tgt)

        generated_dataset = []

        for i in range(num_variants):
            # 动态调整噪声：越往后噪声越大，探索越远
            current_noise = noise_base * (1.0 + (i / num_variants) * 2.0)

            # === Step A: 扰动 (Mutation) ===
            perturbed_tensor = self._apply_noise(seed_tensor, current_noise)

            # === Step B: 修复 (Repair) ===
            # 构造修复任务 (无轨迹要求，只求合法)
            repair_task = {
                "kinematics": {"dof": seed_data['meta']['dof']},
                "targets": {},
                "solver_settings": {"max_iters": 800}
            }

            # 运行优化器
            # 注意：传入 seed_q 作为 q_opt 的初值，避免从零开始乱猜
            repaired_geometry, repaired_q, _ = self.optimizer.run_optimization(
                perturbed_tensor,
                repair_task,
                selected_tool_names=constraint_config['tools'],
                # 注意：这里假设 optimizer 支持传入 q_init，如果没有，它会随机初始化，问题也不大
                # 最好修改 optimizer_engine.py 让其支持 q_init 参数
            )

            # === Step C: 验证 (Validate) ===
            if mech_type == "Bennett":
                # 简单检查是否满足 Bennett 几何条件 (a/sin(alpha) 恒定)
                # 这里复用 evaluator 的检查逻辑
                report = self.evaluator.generate_report(repaired_geometry, ["bennett_geometric"])
                geo_err = report['details'].get('bennett_geometric', {}).get('geometric_error', 1.0)
                if geo_err > 1e-2:  # 容忍度
                    continue  # 修复失败，跳过

            # === Step D: 生成新 Prompt & 封装 ===
            # 简单生成一个 Prompt，实际可以结合轨迹分析
            new_instruction = f"Design a {mech_type} mechanism variant with modified link lengths."

            # 还原为 Graph JSON
            new_graph = self._tensor_to_graph_struct(repaired_geometry, repaired_q, seed_data['graph'])

            record = {
                "id": f"{seed_data['id']}_var_{i:04d}",
                "instruction": new_instruction,
                "meta": seed_data['meta'],  # 继承元数据 (is_spatial, mech_type)
                "graph": new_graph
            }
            generated_dataset.append(record)

            if (i + 1) % 10 == 0:
                print(f"  -> Generated {len(generated_dataset)}/{num_variants} variants")

        return generated_dataset

    def _apply_noise(self, tensor, noise_scale):
        """对 a, alpha, offset 施加噪声"""
        noisy = tensor.copy()
        mask = (tensor[0] > 0.5)  # 只修改存在的边
        # 杆长 a (乘性噪声，保持正值)
        noisy[2][mask] *= np.random.normal(1.0, noise_scale, size=noisy[2][mask].shape)
        # 扭转角 alpha (加性噪声)
        noisy[3][mask] += np.random.normal(0.0, noise_scale * 0.5, size=noisy[3][mask].shape)
        return noisy

    def _tensor_to_graph_struct(self, tensor, q_matrix, template_graph):
        """将 Tensor 数据填回 Graph 结构"""
        new_graph = copy.deepcopy(template_graph)
        N = tensor.shape[1]
        for edge in new_graph['edges']:
            u, v = edge['source'], edge['target']
            if u < N and v < N:
                # 回填几何参数
                edge['params']['a'] = float(tensor[2, u, v])
                edge['params']['alpha'] = float(tensor[3, u, v])
                edge['params']['offset_source'] = float(tensor[4, u, v])
                edge['params']['offset_target'] = float(tensor[4, v, u])
                # 回填状态参数 (关键！)
                if edge.get('initial_state'):
                    if edge['initial_state'].get('q_source') is not None:
                        edge['initial_state']['q_source'] = float(q_matrix[u, v])
                    if edge['initial_state'].get('q_target') is not None:
                        edge['initial_state']['q_target'] = float(q_matrix[v, u])
        return new_graph