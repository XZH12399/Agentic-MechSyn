import torch
import torch.optim as optim
import numpy as np
import networkx as nx
import os

# 防止 OMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DifferentiableOptimizer:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)

    def _lie_bracket_torch(self, twist1, twist2):
        """PyTorch 版李括号: [T1, T2]"""
        w1, v1 = twist1[:3], twist1[3:]
        w2, v2 = twist2[:3], twist2[3:]
        w_new = torch.linalg.cross(w1, w2)
        v_new = torch.linalg.cross(w1, v2) - torch.linalg.cross(w2, v1)
        return torch.cat([w_new, v_new])

    def _assess_candidates_idof_torch(self, K, candidates, cycles, Screws, edge_to_col):
        """
        批量评估一组候选模式是否为 IDOF。
        返回: drift_ratios (Tensor, shape=[num_candidates])
        """
        ratios = []
        K_pinv = None

        for mode_vec in candidates:
            # 归一化模式向量
            mode_vec = mode_vec / (torch.norm(mode_vec) + 1e-9)

            total_drift_list = []

            for loop in cycles:
                L = len(loop)
                cum_twist = torch.zeros(6, device=self.device)
                loop_drift = torch.zeros(6, device=self.device)

                for i in range(L):
                    curr = loop[i]
                    next_n = loop[(i + 1) % L]
                    prev = loop[(i - 1 + L) % L]
                    screw = Screws[curr]

                    val_out = mode_vec[edge_to_col[(curr, next_n)]] if (curr, next_n) in edge_to_col else 0.0
                    val_in = mode_vec[edge_to_col[(curr, prev)]] if (curr, prev) in edge_to_col else 0.0
                    d_theta = val_out - val_in

                    drift_term = self._lie_bracket_torch(cum_twist, screw) * d_theta
                    loop_drift = loop_drift + drift_term
                    cum_twist = cum_twist + screw * d_theta

                total_drift_list.append(loop_drift)

            D_vec = torch.cat(total_drift_list)

            # 计算投影残差
            if K_pinv is None:
                try:
                    K_pinv = torch.linalg.pinv(K, rcond=1e-3)
                except:
                    ratios.append(torch.tensor(1.0, device=self.device))
                    continue

            projected = K @ (K_pinv @ D_vec)
            residual = D_vec - projected

            res_norm = torch.norm(residual)
            drift_norm = torch.norm(D_vec)

            if drift_norm > 1e-6:
                ratios.append(res_norm / drift_norm)
            else:
                ratios.append(torch.tensor(0.0, device=self.device))

        return torch.stack(ratios)

    def optimize_mobility(self, G, cycles, target_dof=1, max_steps=1500):
        """
        使用 [Detect -> Remove -> Count] 流程进行几何优化
        :param target_dof: 期望的目标自由度数量（默认为1）
        """
        # 统计活跃节点
        active_nodes = set()
        for cycle in cycles: active_nodes.update(cycle)
        num_active_nodes = len(active_nodes)
        num_nodes = len(G.nodes)

        # 1. 随机关节类型
        prob_R = np.random.uniform(0.7, 1.0)
        node_types_prob = torch.rand((num_nodes, 1), device=self.device)
        is_R_mask = (node_types_prob < prob_R).float()
        num_R = int(is_R_mask.sum().item())
        print(
            f"   🎲 Joint Config: R_prob={prob_R:.2f} | R-Joints: {num_R}, P-Joints: {num_nodes - num_R} | Target DoF: {target_dof}")

        # 2. 初始化
        P_data = torch.rand((num_nodes, 3), device=self.device) * 10.0 - 5.0
        P = P_data.clone().detach().requires_grad_(True)
        Z_data = torch.randn((num_nodes, 3), device=self.device)
        Z_raw = Z_data.clone().detach().requires_grad_(True)

        optimizer = optim.Adam([P, Z_raw], lr=0.05)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

        print(f"   🔧 Optimization w/ Integrated IDOF Filtering...")

        # 预处理映射
        involved_edges_set = set()
        for loop in cycles:
            L = len(loop)
            for i in range(L):
                u, v = loop[i], loop[(i + 1) % L]
                involved_edges_set.add(tuple(sorted((u, v))))
        edge_to_col = {}
        idx = 0
        for u, v in involved_edges_set:
            edge_to_col[(u, v)] = idx;
            idx += 1
            edge_to_col[(v, u)] = idx;
            idx += 1
        num_vars = idx

        success = False
        best_loss = float('inf')

        final_P, final_Z = None, None
        final_dof = 0
        final_null_motion = None

        for step in range(max_steps):
            optimizer.zero_grad()

            # --- 几何计算 ---
            Z = Z_raw / (torch.norm(Z_raw, dim=1, keepdim=True) + 1e-8)
            scale = torch.max(torch.norm(P, dim=1)) + 1e-6
            P_norm = P / scale

            W_R, V_R = Z, torch.linalg.cross(P_norm, Z)
            W_P, V_P = torch.zeros_like(Z), Z
            W = is_R_mask * W_R + (1 - is_R_mask) * W_P
            V = is_R_mask * V_R + (1 - is_R_mask) * V_P
            Screws = torch.cat([W, V], dim=1)

            # --- 原始 K 矩阵构建 ---
            K = torch.zeros((6 * len(cycles), num_vars), device=self.device)
            current_row = 0
            for loop in cycles:
                L = len(loop)
                for i in range(L):
                    curr = loop[i]
                    next_n = loop[(i + 1) % L]
                    prev = loop[(i - 1 + L) % L]
                    screw = Screws[curr]
                    if (curr, next_n) in edge_to_col:
                        K[current_row:current_row + 6, edge_to_col[(curr, next_n)]] += screw
                    if (curr, prev) in edge_to_col:
                        K[current_row:current_row + 6, edge_to_col[(curr, prev)]] -= screw
                current_row += 6

            mobility_loss = torch.tensor(10.0, device=self.device)
            current_dof = 0

            try:
                # === 步骤 1: 初次 SVD 寻找候选模态 ===
                U, S, Vh = torch.linalg.svd(K)
                # 选取最后 3 个作为候选 (根据需要调整，若 target_dof 较大，可能需要检测更多)
                # 保证候选数量至少涵盖 target_dof
                num_candidates = max(3, target_dof + 1)
                candidates = Vh[-num_candidates:, :]

                # === 步骤 2: 检测这些候选是否为瞬时机构 ===
                drift_ratios = self._assess_candidates_idof_torch(K, candidates, cycles, Screws, edge_to_col)

                # === 步骤 3: 构造增强矩阵 (Augment K) ===
                rows_to_add = []
                for i in range(len(drift_ratios)):
                    ratio = drift_ratios[i]
                    weight = torch.relu(ratio - 0.05) * 50.0
                    if weight > 0:
                        rows_to_add.append(candidates[i] * weight)

                if rows_to_add:
                    K_aug = torch.cat([K, torch.stack(rows_to_add)], dim=0)
                else:
                    K_aug = K

                # === 步骤 4: 基于增强矩阵计算最终 Loss ===
                U2, S2, Vh2 = torch.linalg.svd(K_aug)

                full_S = torch.zeros(num_vars, device=self.device)
                full_S[:S2.shape[0]] = S2
                spectrum = torch.flip(full_S, dims=[0])

                # [关键修改] 目标：剔除 IDOF 后，至少还要剩下 target_dof 个物理 DoF
                target_zero_idx = num_active_nodes + target_dof

                if len(spectrum) > target_zero_idx:
                    # 惩罚前 target_zero_idx 个奇异值 (让它们都趋向 0)
                    zeros_part = spectrum[:target_zero_idx]
                    mobility_loss = torch.sum(zeros_part ** 2) * 10.0

                # 记录当前的有效模式 (最小的那个)
                current_null_motion = Vh2[-1, :].detach()

                # 计算显示的 DoF
                zero_count = torch.sum(spectrum < 1e-3).item()
                current_dof = max(0, int(zero_count - num_active_nodes))

            except Exception as e:
                # print(f"Error: {e}")
                pass

            # --- 正则化 ---
            edges = list(G.edges())
            dist_loss = torch.tensor(0.0, device=self.device)
            if edges:
                u_idx = [e[0] for e in edges]
                v_idx = [e[1] for e in edges]
                dists = torch.norm(P[u_idx] - P[v_idx], dim=1)
                dist_loss += torch.sum(torch.relu(1.0 - dists) * 20.0)
                dist_loss += torch.sum(torch.relu(dists - 15.0) * 1.0)

            total_loss = mobility_loss + dist_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()

            # --- 判定收敛 ---
            # 要求：DoF 至少达到目标值
            if mobility_loss.item() < 1e-4 and dist_loss.item() < 1e-2 and current_dof >= target_dof:
                final_P = P.detach()
                final_Z = Z.detach()
                final_dof = current_dof
                final_null_motion = current_null_motion
                success = True
                print(
                    f"   ✅ Converged! Clean DoF: {final_dof} (Target: {target_dof}), Loss: {mobility_loss.item():.6f}")
                break

            if step % 200 == 0:
                print(
                    f"   Step {step}: Loss={total_loss.item():.4f} (Mob={mobility_loss.item():.4f}, DoF={current_dof})")

        final_joint_types = ['R' if m > 0.5 else 'P' for m in is_R_mask.cpu().numpy().flatten()]

        return success, final_P, final_Z, final_joint_types, final_dof, final_null_motion