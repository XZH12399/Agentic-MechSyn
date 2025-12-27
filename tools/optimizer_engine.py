import torch
import numpy as np
import math
import networkx as nx
from collections import deque


class MechanismOptimizer:
    def __init__(self, physics_config):
        self.cfg = physics_config
        self.lr = physics_config.learning_rate
        self.epochs = physics_config.max_iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 工具注册表
        self.loss_tools = {
            "closure_loop": self._loss_closure,
            "mobility_dof": self._loss_mobility_dof,
            "path_error": self._loss_path_generation,
            "twist_match": self._loss_twist_alignment,
            "bennett_ratio": self._loss_bennett_condition,
            "instantaneous_check": self._loss_instantaneous_check
        }

    def run_optimization(self, initial_tensor, task_template, selected_tool_names, new_tools_definitions=[]):
        print(f"    -> [Optimizer] 启动 PyTorch 协同优化引擎 (Device: {self.device})")

        # 新增: 日志捕获列表
        execution_log = []

        def log_and_print(msg):
            print(msg)
            execution_log.append(msg)

        log_and_print(f"Optimizer Config: Device={self.device}, LR={self.lr}, Epochs={self.epochs}")

        N = initial_tensor.shape[1]

        # 1. Geometry (5, N, N)
        geometry_tensor = torch.tensor(
            initial_tensor, dtype=torch.float32, device=self.device, requires_grad=True
        )

        # 2. Joint Variables (N, N)
        q_opt = torch.empty((N, N), device=self.device).uniform_(-0.5, 0.5).requires_grad_(True)

        # 3. Optimizer & Scheduler
        optimizer = torch.optim.Adam([
            {'params': geometry_tensor, 'lr': self.lr},
            {'params': q_opt, 'lr': self.lr * 2.0}
        ], lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
        )

        # Topology Analysis
        adj_matrix = initial_tensor[0]
        self.cycles = self._find_fundamental_cycles(adj_matrix)
        self.G_graph = self._build_nx_graph(adj_matrix)

        print(f"    -> [Optimizer] 拓扑分析: 发现 {len(self.cycles)} 个基本闭环")

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            loss_components = {}

            # constrained_geometry 是物理合法的张量，且保留了对 geometry_tensor 的梯度
            constrained_geometry = self._get_constrained_geometry(geometry_tensor)

            for tool_name in selected_tool_names:
                if tool_name in self.loss_tools:
                    loss_func = self.loss_tools[tool_name]

                    # 注意：传入的是 constrained_geometry
                    loss_val = loss_func(constrained_geometry, task_template, cycles=self.cycles, q_opt=q_opt)

                    weight = 1.0
                    if tool_name == "closure_loop": weight = 10.0
                    if tool_name == "bennett_ratio": weight = 10.0
                    if tool_name in ["path_error", "twist_match"]: weight = 50.0
                    if tool_name == "mobility_dof": weight = 10.0

                    total_loss += weight * loss_val
                    loss_components[tool_name] = loss_val.item()

            # 如果 total_loss 没有梯度函数 (即它是一个常数)，说明当前所有的 Loss 都失效了
            if total_loss.grad_fn is None:
                # 手动加上一个与参数相关的 0.0，强行建立计算图连接
                # 这样 backward() 就不会报错，只是梯度为 0
                dummy = (geometry_tensor.sum() + q_opt.sum()) * 0.0
                total_loss = total_loss + dummy

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_([geometry_tensor, q_opt], max_norm=1.0)

            optimizer.step()
            scheduler.step(total_loss.item())

            # 防止 float 误差导致连接断开或类型突变
            with torch.no_grad():
                geometry_tensor[0] = torch.tensor(initial_tensor[0], device=self.device)
                geometry_tensor[1] = torch.tensor(initial_tensor[1], device=self.device)

            if epoch % 50 == 0 or epoch == 1:
                current_lr = optimizer.param_groups[0]['lr']
                log_str = f"Epoch {epoch}: Loss={total_loss.item():.4f} (LR={current_lr:.1e})"
                for k, v in loss_components.items():
                    log_str += f" | {k}={v:.4f}"
                # 使用 log_and_print 替代 print
                log_and_print(f"       {log_str}")

        # 返回最终结果时，也要应用一次约束，确保输出的是物理合法值
        final_geometry = self._get_constrained_geometry(geometry_tensor).detach().cpu().numpy()
        q_matrix = q_opt.detach().cpu().numpy()

        return final_geometry, q_matrix, "\n".join(execution_log)

    # =========================================================================
    # ✨ 新增: [IDOF 检测版] 运动可持续性检查 Loss
    # =========================================================================
    def _loss_instantaneous_check(self, tensor, task, cycles=None, q_opt=None):
        """
        检查末端运动是否可持续（判断是否为瞬时机构）。
        原理：计算二阶漂移 (Drift) 是否落在雅可比矩阵的列空间内。
        """
        targets = task.get('targets', {})
        path_to_ee = targets.get('target_path_sequence', [])
        target_twists = targets.get('target_motion_twists', [])
        target_masks = targets.get('target_masks', [])

        # 如果没有定义路径或目标螺旋，无法执行此检查，返回 0 loss
        if not path_to_ee or not target_twists:
            return torch.tensor(0.0, device=self.device)

        dt = 1e-3
        total_loss = torch.tensor(0.0, device=self.device)

        # 处理多个目标 Twist (如果有)
        # 这里简化处理：通常只有一个 path，但可能有多个 twist 目标 (多模式)
        # 我们取第一个 twist 进行检查，或者遍历检查
        num_modes = len(target_twists)

        for k in range(num_modes):
            tgt_twist = torch.tensor(target_twists[k], device=self.device)
            # 如果有 mask 就用，没有就全 1
            if target_masks and k < len(target_masks):
                tgt_mask = torch.tensor(target_masks[k], device=self.device)
            else:
                tgt_mask = torch.ones(6, device=self.device)

            # --- 步骤 1: 获取当前构型的 K 和 x0 ---
            # 这里的 K_curr 实际上就是包含了"虚拟环路约束"（任务约束）的雅可比矩阵
            mapping, K_curr, x0, spectrum = self._solve_anchor_system(
                tensor, q_opt, cycles,
                extended_task_path=path_to_ee,
                target_twist=tgt_twist,
                target_mask=tgt_mask,
                return_full_data=True  # ✨ 请求返回完整数据
            )

            # 基础检查：如果是死锁或当前位置就不闭合，直接跳过 (由其他 Loss 负责)
            if x0 is None or K_curr is None:
                continue

            # 归一化 x0 (单位速度，消除速度大小对 Drift 幅度的影响)
            x_norm = torch.norm(x0)
            if x_norm < 1e-6:
                continue
            x0 = x0 / x_norm

            # --- 步骤 2: 计算二阶漂移 (The "Bill") ---
            # 我们使用有限差分来通过 PyTorch 自动计算 (J_dot * q_dot)

            # 2.1 模拟向前走极小的一步 q_next = q_current + x0 * dt
            q_next = q_opt.clone()

            # 利用 mapping 将 x0 (列向量) 映射回 q (矩阵)
            # mapping: {(u, v): col_idx}
            for (u, v), col_idx in mapping.items():
                if col_idx < len(x0):
                    val = x0[col_idx] * dt
                    q_next[u, v] = q_next[u, v] + val
                    q_next[v, u] = q_next[v, u] - val  # 反向边取反 (如果是 R 关节)

            # 2.2 获取新位置的 K (无需解方程，只要矩阵)
            _, K_next, _, _ = self._solve_anchor_system(
                tensor, q_next, cycles,
                extended_task_path=path_to_ee,
                target_twist=tgt_twist,
                target_mask=tgt_mask,
                return_full_data=True
            )

            if K_next is None:
                total_loss = total_loss + 10.0
                continue

            # 2.3 计算漂移向量 Drift = (K_next - K_curr) * x0 / dt
            # 物理含义：保持关节速度不变时，约束方程产生的破坏速度
            drift_vec = (K_next @ x0 - K_curr @ x0) / dt

            # --- 步骤 3: 投影相容性测试 (The "Payment") ---
            # 检查方程 K_curr * alpha = -drift 是否有解

            # 使用伪逆进行投影: Residual = (I - K * K_pinv) * drift
            # rcond=1e-3 用于忽略极小的奇异值噪声
            try:
                # 为 pinv 添加抖动保护
                if K_curr.requires_grad:
                    jitter = torch.randn_like(K_curr) * 1e-9
                    K_curr_noisy = K_curr + jitter
                else:
                    K_curr_noisy = K_curr

                K_pinv = torch.linalg.pinv(K_curr_noisy, rcond=1e-3)
                alpha_sol = K_pinv @ (-drift_vec)
                compensated_drift = K_curr @ alpha_sol

                residual_vec = (-drift_vec) - compensated_drift
                loss_idof = torch.norm(residual_vec)

                # 标准化：除以 drift 的模长 (Ratio)
                drift_norm = torch.norm(drift_vec)
                if drift_norm > 1e-6:
                    loss_ratio = loss_idof / drift_norm
                else:
                    loss_ratio = torch.tensor(0.0, device=self.device)  # 几乎没有漂移，完美

                total_loss = total_loss + loss_ratio
            except:
                total_loss = total_loss + 1.0  # SVD 失败惩罚

        return total_loss / num_modes

    # =========================================================================
    # Helper: 可微物理约束层 (Differentiable Physics Constraints)
    # =========================================================================
    def _get_constrained_geometry(self, raw_tensor):
        """
        输入: 包含任意实数值的原始张量 (Raw Parameters)
        输出: 符合物理约束的张量 (Physical Parameters)
        特性: 全程可导，梯度可回传
        """
        # 1. 拆分通道 (为了保持梯度，不要使用 detach)
        # tensor shape: (5, N, N)
        exists = raw_tensor[0]
        j_type = raw_tensor[1]
        a = raw_tensor[2]
        alpha = raw_tensor[3]
        offset = raw_tensor[4]

        # 2. 强制对称性 (Symmetry)
        # 连杆属性 a, alpha, exists 必须是对称矩阵
        # 操作: M_sym = (M + M.T) / 2
        # 梯度会平均分配给 M_ij 和 M_ji
        exists_sym = (exists + exists.T) / 2.0
        a_sym = (a + a.T) / 2.0
        alpha_sym = (alpha + alpha.T) / 2.0

        # 3. 关节类型行一致性 (Row Consistency)
        # 节点类型由行决定，取行均值并广播
        j_type_row = j_type.mean(dim=1, keepdim=True).expand_as(j_type)

        # 4. 物理合法性 (Positivity & Periodicity)
        # 杆长 a 必须非负 -> 使用 abs()
        a_phys = torch.abs(a_sym)

        # 扭转角 alpha 周期性 -> 使用 remainder
        # 注意: 这里的梯度是 1 (线性)，不会阻断
        alpha_phys = torch.remainder(alpha_sym, 2 * math.pi)

        # offset 可以为负，不做限制
        offset_phys = offset

        # 5. 拓扑掩码 (Masking)
        # 强制非连接处的参数为 0
        # 使用初始拓扑作为硬 mask (假设 Step 4 不改变拓扑)
        # 这里使用 Sigmoid 近似 step function? 不，直接用 exists > 0.5 的硬 mask 即可
        # 因为我们不想优化不存在的边
        mask = (exists_sym > 0.5).float()

        a_final = a_phys * mask
        alpha_final = alpha_phys * mask
        offset_final = offset_phys * mask
        exists_final = exists_sym * mask  # 实际上这会把 exists 变成 0/1 (如果它是parameter的话)

        # 6. 对角线清零 (No Self-loops)
        N = raw_tensor.shape[1]
        diag_mask = 1.0 - torch.eye(N, device=self.device)

        a_final = a_final * diag_mask
        alpha_final = alpha_final * diag_mask
        offset_final = offset_final * diag_mask
        exists_final = exists_final * diag_mask

        # 7. 重新堆叠
        return torch.stack([exists_final, j_type_row, a_final, alpha_final, offset_final], dim=0)

    # =========================================================================
    # Loss Functions
    # =========================================================================
    def _loss_closure(self, tensor, task, cycles=None, q_opt=None):
        """
        闭环误差计算 (基于节点多重观测一致性)
        """
        # 1. 内部调用核心引擎，获取所有节点的观测状态
        # 假设 base_node 为 0，或者根据 cycles 自动推断
        base_node = 0
        if cycles and len(cycles) > 0:
            base_node = cycles[0][0]  # 尝试使用环路中的第一个节点作为 base

        node_observations = self._compute_multi_path_states(tensor, q_opt, base_node=base_node)

        # for key, value in node_observations.items():
        #     print(key, value)

        total_loss = torch.tensor(0.0, device=self.device)
        count = 0

        # 2. 遍历观测结果，计算方差/偏差
        for node_id, obs_list in node_observations.items():
            # 如果只有 1 个观测值，说明没有闭环冲突，跳过
            if len(obs_list) < 2:
                continue

            # 以第一个观测值为基准 (Anchor)
            ref_P = obs_list[0]['P']
            ref_z = obs_list[0]['z']

            # 强制所有后续观测值与基准一致
            for i in range(1, len(obs_list)):
                curr_P = obs_list[i]['P']
                curr_z = obs_list[i]['z']

                loss_pos = torch.sum((curr_P - ref_P) ** 2)
                loss_align = torch.sum((curr_z - ref_z) ** 2)

                total_loss = total_loss + loss_pos + loss_align
                count += 1

        return total_loss

    def _loss_mobility_dof(self, tensor, task, cycles=None, q_opt=None):
        device = self.device

        # === 核心修复：计算实际活跃节点数，而非 Tensor 维度 ===
        # 原代码: num_nodes = tensor.shape[1]
        if cycles:
            # 统计所有在闭环(cycles)中出现的唯一节点
            active_nodes = set()
            for cycle in cycles:
                active_nodes.update(cycle)
            num_nodes = len(active_nodes)
        else:
            # 如果没有闭环信息（极端情况），回退到 Tensor 维度，或者直接返回
            num_nodes = tensor.shape[1]

        try:
            target_dof = task.get('kinematics', {}).get('dof', 1)
        except:
            target_dof = 1

        _, _, _, spectrum = self._solve_anchor_system(
            tensor, q_opt, cycles,
            extended_task_path=None, target_twist=None, target_mask=None,
            return_spectrum=True
        )

        if spectrum is None: return torch.tensor(10.0, device=device)

        target_zero_count = num_nodes + target_dof

        if len(spectrum) <= target_zero_count:
            # 如果谱的长度比目标还短，说明矩阵太小了，可能还没有形成有效约束
            # 或者这里原本的逻辑是想要惩罚“非零”特征值
            # 这种情况下返回 1.0 可能是不合适的，视具体数学推导而定
            # 但既然之前是因为 8 > 4 导致的误判，现在 num_nodes 修正为 4 后，
            # target_zero_count 变小，这个 if 条件就不容易误触发了。
            return torch.tensor(1.0, device=device)

        zeros_part = spectrum[:target_zero_count]
        loss_zeros = torch.sum(zeros_part ** 2) * 10.0
        return loss_zeros

    def _loss_path_generation(self, tensor, task, cycles=None, q_opt=None):
        targets = task.get('targets', {})
        target_path_seq = targets.get('target_path_sequence', [])
        target_poses = targets.get('target_motion_twists', [])

        if not target_path_seq or not target_poses:
            return torch.tensor(0.0, device=self.device)

        # FK Path: Base -> ... -> EE (去除 Ghost)
        fk_path_nodes = target_path_seq[1:-1]
        ghost_in = target_path_seq[0]

        T_ee = torch.eye(4, device=self.device)
        for i in range(len(fk_path_nodes) - 1):
            curr = fk_path_nodes[i]
            next_n = fk_path_nodes[i + 1]
            prev = ghost_in if i == 0 else fk_path_nodes[i - 1]

            j_type = tensor[1, curr, next_n]
            a = torch.abs(tensor[2, curr, next_n])
            alpha = tensor[3, curr, next_n]

            off_out = tensor[4, curr, next_n]
            off_in = tensor[4, curr, prev]
            d_static = off_out - off_in

            q_out = q_opt[curr, next_n]
            q_in = q_opt[curr, prev]
            q_diff = q_out - q_in

            is_R = (j_type > 0.5).float()
            is_P = 1.0 - is_R
            theta = is_R * (q_diff - PI) + is_P * (d_static - PI)
            d = is_R * d_static + is_P * q_diff

            T_step = self._get_dh_matrix_fast(a, alpha, d, theta)
            T_ee = T_ee @ T_step

        tgt_vec = torch.tensor(target_poses[0], device=self.device)
        T_target = self._vec6_to_matrix(tgt_vec)

        pos_err = torch.norm(T_ee[:3, 3] - T_target[:3, 3])
        rot_err = torch.norm(T_ee[:3, :3] - T_target[:3, :3])

        return pos_err + rot_err

    def _loss_twist_alignment(self, tensor, task, cycles=None, q_opt=None):
        targets = task.get('targets', {})
        target_path_seq = targets.get('target_path_sequence', [])
        target_twists = targets.get('target_motion_twists', [])
        target_masks = targets.get('target_masks', [])

        if not target_path_seq or not target_twists:
            return torch.tensor(0.0, device=self.device)

        tgt_twist = torch.tensor(target_twists[0], device=self.device)
        tgt_mask = torch.tensor(target_masks[0], device=self.device) if target_masks else torch.ones(6,
                                                                                                     device=self.device)

        _, _, _, spectrum = self._solve_anchor_system(
            tensor, q_opt, cycles,
            extended_task_path=target_path_seq,
            target_twist=tgt_twist,
            target_mask=tgt_mask,
            return_spectrum=True
        )

        if spectrum is None: return torch.tensor(10.0, device=self.device)
        num_nodes = tensor.shape[1]
        if len(spectrum) > num_nodes:
            return torch.abs(spectrum[num_nodes]) * 50.0
        return torch.tensor(0.0, device=self.device)

    def _loss_bennett_condition(self, tensor, task, cycles=None, q_opt=None):
        if not cycles: return torch.tensor(0.0, device=self.device)
        total_error = torch.tensor(0.0, device=self.device)
        TWO_PI = 2 * math.pi
        for path in cycles:
            if len(path) != 4: continue
            a_list = []
            alpha_list = []
            offset_loss_accum = torch.tensor(0.0, device=self.device)
            max_a = torch.tensor(1.0, device=self.device)
            L = 4
            for i in range(L):
                curr, next_n, prev = path[i], path[(i + 1) % L], path[(i - 1 + L) % L]
                a = torch.abs(tensor[2, curr, next_n])
                alpha = tensor[3, curr, next_n] % TWO_PI
                max_a = torch.max(max_a, a)
                a_list.append(a)
                alpha_list.append(alpha)
                off_out = tensor[4, curr, next_n]
                off_in = tensor[4, curr, prev]
                d_val = off_out - off_in
                offset_loss_accum += d_val ** 2
            offset_loss_rel = offset_loss_accum / (max_a ** 2 + 1e-6)
            a_vec = torch.stack(a_list)
            alpha_vec = torch.stack(alpha_list)
            sym_loss_a = ((a_vec[0] - a_vec[2]) ** 2) / (a_vec[0] ** 2 + a_vec[2] ** 2 + 1e-6) + \
                         ((a_vec[1] - a_vec[3]) ** 2) / (a_vec[1] ** 2 + a_vec[3] ** 2 + 1e-6)
            # 修改后 (对周期性不敏感)
            # 1 - cos(diff) 在 diff=0 时为 0，在 diff=2pi 时也为 0，完美解决断层
            sym_loss_alpha = (1.0 - torch.cos(alpha_vec[0] - alpha_vec[2])) + \
                             (1.0 - torch.cos(alpha_vec[1] - alpha_vec[3]))
            sin_alpha = torch.sin(alpha_vec)
            term1 = a_vec[0] * sin_alpha[1]
            term2 = a_vec[1] * sin_alpha[0]
            ratio_err1 = (term1 - term2) ** 2 / (term1 ** 2 + term2 ** 2 + 1e-6)
            term3 = a_vec[1] * sin_alpha[2]
            term4 = a_vec[2] * sin_alpha[1]
            ratio_err2 = (term3 - term4) ** 2 / (term3 ** 2 + term4 ** 2 + 1e-6)
            total_error += sym_loss_a + sym_loss_alpha + ratio_err1 + ratio_err2 + offset_loss_rel * 5.0
        # print(f"  > Symmetry A: {sym_loss_a.item():.6f}")
        # print(f"  > Symmetry Alpha: {sym_loss_alpha.item():.6f}")
        # print(f"  > Ratio Error: {ratio_err2.item():.6f}")
        # print(f"  > Offset Error (d=0): {(offset_loss_rel * 5.0).item():.6f}")  # <--- 重点关注这个

        return total_error

    # =========================================================================
    # Helpers
    # =========================================================================
    def _compute_all_joint_screws(self, structure, q_current, base_node, cycles=None, normalize=True):
        """
        计算所有关节螺旋 (Screws)。
        内部调用多路径解算器，提取生成树状态，并进行特征长度归一化。
        """
        # 1. 内部调用核心引擎
        node_observations = self._compute_multi_path_states(structure, q_current, base_node=base_node)

        N = structure.shape[1]
        screws = torch.zeros((N, 6), device=self.device)

        # 2. 计算特征长度 L_char (用于归一化)
        L_char = torch.tensor(1.0, device=self.device)
        if normalize:
            exists_mask = structure[0] > 0.5
            all_a = torch.abs(structure[2][exists_mask])
            valid_a = all_a[all_a > 1e-6]
            if valid_a.numel() > 0:
                L_char = torch.mean(valid_a)

        # 3. 提取螺旋
        for u in range(N):
            # 如果节点未连通，保持 0
            if u not in node_observations:
                continue

            # 总是取列表中的第 0 个观测值作为"权威状态"
            # (Jacobian 计算只需要一套自洽的坐标系)
            state = node_observations[u][0]
            P = state['P']
            z = state['z']

            row_types = structure[1, u, :]
            is_R = not (row_types < -0.5).any()

            if is_R:
                w = z
                v = torch.linalg.cross(P, z)

                # [归一化] 仅缩放力矩部分，使旋转和移动量级匹配
                if normalize:
                    v = v / L_char

                screws[u] = torch.cat([w, v])
            else:
                # P副: w=0, v=z (移动方向)
                w = torch.zeros(3, device=self.device)
                v = z
                screws[u] = torch.cat([w, v])

        return screws

    def _solve_anchor_system(self, structure, q_current, loops, extended_task_path=None, target_twist=None,
                             target_mask=None, return_spectrum=False, return_full_data=False):
        # 1. 确定计算所需的节点和 Screw
        # ------------------------------------------------------------------
        # 收集所有活跃节点以确定 base_node
        active_nodes = set()
        for loop in loops: active_nodes.update(loop)
        if extended_task_path: active_nodes.update([n for n in extended_task_path if n >= 0])

        # 如果没有活跃节点，直接返回
        if not active_nodes: return None, None, None, None

        base_node = extended_task_path[1] if extended_task_path and len(extended_task_path) > 1 else min(active_nodes)
        # 计算所有节点的螺旋轴 (Joint Screws)
        all_screws = self._compute_all_joint_screws(
            structure, q_current, base_node,
            cycles=loops,
            normalize=True  # <--- 确保这里开启
        )
        num_nodes = structure.shape[1]

        # 2. 建立变量映射 (Edge to Column Mapping) - 参考提供的 NumPy 逻辑
        # ------------------------------------------------------------------
        # 收集所有涉及的无向边 (u, v) 其中 u < v
        involved_edges_set = set()

        # A. 从回路中收集边
        for loop in loops:
            L = len(loop)
            for i in range(L):
                u, v = loop[i], loop[(i + 1) % L]
                involved_edges_set.add(tuple(sorted((u, v))))

        # B. 从任务路径中收集边
        if extended_task_path:
            for i in range(len(extended_task_path) - 1):
                u = extended_task_path[i]
                v = extended_task_path[i + 1]
                if u >= 0 and v >= 0:  # 确保节点索引有效
                    involved_edges_set.add(tuple(sorted((u, v))))

        # C. 构建有向边映射 (u->v 和 v->u 对应不同的列)
        edge_to_col = {}
        current_col = 0
        for u, v in involved_edges_set:
            # 添加 (u, v)
            edge_to_col[(u, v)] = current_col
            current_col += 1
            # 添加 (v, u)
            edge_to_col[(v, u)] = current_col
            current_col += 1

        num_vars_reduced = current_col  # 实际的变量数量 (无需再用 mask 过滤)

        if num_vars_reduced == 0: return None, None, None, None

        # 3. 构建 Jacobian 矩阵 K
        # ------------------------------------------------------------------
        num_loops = len(loops)
        has_task = (target_twist is not None)
        total_rows = 6 * (num_loops + (1 if has_task else 0))

        # 直接建立紧凑矩阵，不再建立 num_nodes*num_nodes 的大矩阵
        K_compact = torch.zeros((total_rows, num_vars_reduced), device=self.device)
        b = torch.zeros(total_rows, device=self.device)

        current_row = 0

        # --- 填充回路约束 ---
        for loop in loops:
            L = len(loop)
            for i in range(L):
                curr, next_n, prev = loop[i], loop[(i + 1) % L], loop[(i - 1 + L) % L]
                screw = all_screws[curr]

                # 对应 NumPy 代码: K_local[..., edge_to_col[(curr, next)]] += screw
                if (curr, next_n) in edge_to_col:
                    col_idx = edge_to_col[(curr, next_n)]
                    K_compact[current_row:current_row + 6, col_idx] += screw

                # 对应 NumPy 代码: K_local[..., edge_to_col[(curr, prev)]] -= screw
                if (curr, prev) in edge_to_col:
                    col_idx = edge_to_col[(curr, prev)]
                    K_compact[current_row:current_row + 6, col_idx] -= screw

            current_row += 6

        # --- 填充任务/路径约束 ---
        if has_task and extended_task_path:
            # 同样使用紧凑矩阵的一行
            row_slice = slice(current_row, current_row + 6)

            for i in range(1, len(extended_task_path) - 1):
                curr = extended_task_path[i]
                prev_n = extended_task_path[i - 1]
                next_n = extended_task_path[i + 1]
                screw = all_screws[curr]

                if next_n is not None and next_n >= 0:
                    if (curr, next_n) in edge_to_col:
                        col_idx = edge_to_col[(curr, next_n)]
                        # 注意：这里需要处理 target_mask，稍后统一处理或在此处处理
                        # 为保持逻辑清晰，先填入 K_compact，最后再乘 mask
                        K_compact[row_slice, col_idx] += screw

                if prev_n is not None and prev_n >= 0:
                    if (curr, prev_n) in edge_to_col:
                        col_idx = edge_to_col[(curr, prev_n)]
                        K_compact[row_slice, col_idx] -= screw

            # 应用 Task Mask 和 Target Twist
            if target_mask is not None:
                # K_compact 的对应行乘以 mask (unsqueeze用于广播)
                K_compact[row_slice, :] *= target_mask.unsqueeze(1)
                b[row_slice] = target_twist * target_mask
            else:
                b[row_slice] = target_twist

        # --- 4. SVD 分解与数据返回 (修改部分) ---
        b_unsqueezed = b.unsqueeze(1)
        K_aug = torch.cat([K_compact, -b_unsqueezed], dim=1) if has_task else K_compact

        try:
            # 添加微小抖动以防止 SVD 梯度爆炸
            if getattr(self.cfg, 'enable_jitter', True) and K_aug.requires_grad:
                jitter = torch.randn_like(K_aug) * 1e-9
                K_aug = K_aug + jitter

            # Perform SVD
            U, S, Vh = torch.linalg.svd(K_aug, full_matrices=True)

            # S 补齐逻辑 (保持不变)
            num_vars_aug = K_aug.shape[1]
            full_S = torch.zeros(num_vars_aug, dtype=S.dtype, device=self.device)
            full_S[:S.shape[0]] = S
            spectrum = torch.flip(full_S, dims=[0])

            # ✨ 新增: 如果请求完整数据，计算 x0 并返回
            if return_full_data:
                # x0 是 K_aug 的零空间向量，对应最小奇异值的右奇异向量 (Vh 的最后一行)
                # Vh 是 (min(M, N), N)，SVD 也是按奇异值降序排列的
                # 所以 Vh[-1] 对应最小奇异值
                x0 = Vh[-1, :]
                return edge_to_col, K_aug, x0, spectrum

            return None, None, None, spectrum
        except Exception as e:
            return None, None, None, None

    def _get_dh_matrix_fast(self, a, alpha, d, theta):
        """
        构建标准 DH 变换矩阵 (Standard DH)
        支持广播: 输入可以是标量，也可以是 (Batch, N, N) 等任意维度
        输出: (..., 4, 4)
        """
        # 1. 预计算三角函数
        ct, st = torch.cos(theta), torch.sin(theta)
        ca, sa = torch.cos(alpha), torch.sin(alpha)

        # 2. 准备占位符 (0 和 1)，形状与输入一致，保持 device/dtype 正确
        zero = torch.zeros_like(theta)
        one = torch.ones_like(theta)

        # 3. 逐个元素构建 (行优先)
        # 这里的关键是：不使用 stack 堆叠成列表，而是直接构造最后一维

        # Row 1: [ct, -st*ca, st*sa, a*ct]
        r11, r12, r13, r14 = ct, -st * ca, st * sa, a * ct

        # Row 2: [st, ct*ca, -ct*sa, a*st]
        r21, r22, r23, r24 = st, ct * ca, -ct * sa, a * st

        # Row 3: [0, sa, ca, d]
        r31, r32, r33, r34 = zero, sa, ca, d

        # Row 4: [0, 0, 0, 1]
        r41, r42, r43, r44 = zero, zero, zero, one

        # 4. 堆叠成矩阵 (..., 4, 4)
        # 先堆叠成行 (..., 4)，再堆叠成矩阵
        row1 = torch.stack([r11, r12, r13, r14], dim=-1)
        row2 = torch.stack([r21, r22, r23, r24], dim=-1)
        row3 = torch.stack([r31, r32, r33, r34], dim=-1)
        row4 = torch.stack([r41, r42, r43, r44], dim=-1)

        # 最终组合
        T = torch.stack([row1, row2, row3, row4], dim=-2)

        return T

    def _find_fundamental_cycles(self, adj_matrix):
        rows, cols = np.where(adj_matrix > 0.5)
        G = nx.Graph()
        G.add_edges_from(zip(rows, cols))
        try:
            return [list(cycle) for cycle in nx.cycle_basis(G)]
        except:
            return []

    def _build_nx_graph(self, adj_matrix):
        rows, cols = np.where(adj_matrix > 0.5)
        G = nx.Graph()
        G.add_edges_from(zip(rows, cols))
        return G

    def _vec6_to_matrix(self, vec6):
        x, y, z, rx, ry, rz = vec6
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)
        R = torch.stack([
            torch.stack([cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz]),
            torch.stack([cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz]),
            torch.stack([-sy, cy * sx, cx * cy])
        ])
        T = torch.eye(4, device=self.device)
        T[:3, :3] = R
        T[:3, 3] = torch.stack([x, y, z])
        return T

    # =========================================================================
    # 生成树运动学 (引用邻居作为 Base 参考，基于 ID 查询参数)
    # =========================================================================
    def _compute_multi_path_states(self, structure, q_current, base_node=0):
        """
        功能: 多路径运动学解算器 (修正版: 处理入边反向引起的 180 度相位差)
        """
        N = structure.shape[1]
        node_observations = {}
        expanded_nodes = set()

        # 1. 确定 Base 的参考节点
        base_neighbors = torch.nonzero(structure[0, base_node, :] > 0.5).view(-1).tolist()
        if not base_neighbors: return {}
        ref_node = base_neighbors[0]

        T_base = torch.eye(4, device=self.device)

        # --- Base 初始化 ---
        node_observations[base_node] = []
        node_observations[base_node].append({
            'P': T_base[:3, 3],
            'z': T_base[:3, 2],
            'T': T_base,
            'parent': ref_node
        })

        # 标记 Base 已扩展
        expanded_nodes.add(base_node)

        # Stack: (current, parent, T_current)
        stack = [(base_node, -1, T_base)]

        # 定义 PI 常量
        PI = torch.tensor(math.pi, device=self.device)

        while stack:
            u, p, T_u = stack.pop()

            # --- 准备 u 的入参 ---
            # 这些参数代表 u->p (离开 u) 的绝对位置
            if p == -1:
                off_in = structure[4, u, ref_node]
                q_in = q_current[u, ref_node]
            else:
                off_in = structure[4, u, p]
                q_in = q_current[u, p]

            neighbors = torch.nonzero(structure[0, u, :] > 0.5).view(-1).tolist()

            for v in neighbors:
                # 逻辑父节点判断
                logical_parent = ref_node if p == -1 else p

                # 禁止回头
                if v == logical_parent: continue

                # --- 计算 u -> v ---
                a = structure[2, u, v]
                alpha = structure[3, u, v]
                off_out = structure[4, u, v]
                q_out = q_current[u, v]

                # 原始差分 (Out - Stored_In)
                delta_off = off_out - off_in
                delta_q = q_out - q_in

                row_types = structure[1, u, :]
                is_R = not (row_types < -0.5).any()

                # 🌟 核心修正: 角度 theta 需要 +/- 180 度 (PI)
                # 因为 Stored_In 是 u->p, 但物理 In 是 p->u (方向相反)
                if is_R:
                    # 对于 R 副, theta 由 q 决定
                    # theta = q_out - (q_in + PI) = delta_q - PI
                    # 这里加减 PI 对三角函数结果是一样的 (sin(x+pi) = -sin(x))
                    # 我们统一减去 PI (或者加上 PI)
                    theta = delta_q - PI
                    d = delta_off
                else:
                    # 对于 P 副, theta 通常由 offset 决定 (如果是定义角度的话)
                    # d 由 q 决定 (沿着 z 轴的距离, 方向反转通常不影响 z 轴标量长度, 除非坐标系定义导致 z 轴反向)
                    # 假设 offset 也是绝对角度:
                    theta = delta_off - PI
                    d = delta_q

                # 构建 DH 矩阵
                T_step = self._get_dh_matrix_fast(a, alpha, d, theta)
                T_v = T_u @ T_step

                # --- 记录观测 ---
                if v not in node_observations:
                    node_observations[v] = []

                node_observations[v].append({
                    'P': T_v[:3, 3],
                    'z': T_v[:3, 2],
                    'T': T_v,
                    'parent': u
                })

                # --- 递归控制 ---
                if v not in expanded_nodes:
                    expanded_nodes.add(v)
                    stack.append((v, u, T_v))
                else:
                    pass

        return node_observations