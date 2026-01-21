import torch
import numpy as np
import math
import networkx as nx

class PhysicsKernel:
    """
    统一物理计算内核 (Single Source of Truth)
    集成了物理仿真、运动学求解以及优化所需的 Loss 计算功能。
    """
    def __init__(self, device, cfg=None):
        self.device = device
        self.cfg = cfg
        self.PI = torch.tensor(math.pi, device=device)

    # =========================================================================
    # ✨✨✨ 新增: 高级自由度分析 (模仿 pose_to_kinematics) ✨✨✨
    # =========================================================================
    def compute_effective_dof_statistics(self, tensor, q_opt, cycles, threshold_singular=1e-2, threshold_drift=1e-3):
        """
        计算有效自由度 (Effective DOF)。
        逻辑参考: analyze_mobility_anchor
        1. 计算一阶 SVD，获取所有潜在的零空间运动模式 (Candidate Modes).
        2. 对每个模式进行二阶漂移检查 (Drift Check).
        3. 剔除高漂移的瞬时模式，统计真正的物理自由度。
        
        Returns:
            dict: {
                "raw_dof": int,        # SVD 算出的数学自由度 (包含瞬时)
                "effective_dof": int,  # 剔除瞬时后的物理自由度
                "idof_count": int,     # 瞬时自由度数量
                "details": list        # 每个模态的漂移值
            }
        """
        # 1. 确定规范自由度
        if cycles:
            active_nodes = set()
            for cycle in cycles: active_nodes.update(cycle)
            num_gauge = len(active_nodes)
        else:
            num_gauge = tensor.shape[1]

        # 2. 获取系统雅可比信息
        # 注意：这里不带任何 Task，纯粹分析机构本身
        edge_to_col, K_curr, Vh, spectrum = self.solve_anchor_system(
            tensor, q_opt, cycles, return_spectrum=True, return_full_data=True
        )

        print('spectrum')
        print(spectrum)

        if spectrum is None:
            return {"raw_dof": 0, "effective_dof": 0, "idof_count": 0, "details": []}

        # 3. 分析一阶自由度 (Raw DOF)
        # 奇异值 < 阈值的数量 - 规范自由度
        num_vars = K_curr.shape[1]
        # spectrum 是升序排列 (0...large)
        # 统计接近 0 的奇异值数量
        zero_singular_count = torch.sum(spectrum < threshold_singular).item()
        raw_dof = zero_singular_count - num_gauge

        if raw_dof <= 0:
            return {"raw_dof": 0, "effective_dof": 0, "idof_count": 0, "details": []}

        # 4. 二阶分析 (剔除瞬时模态)
        # Vh 的形状是 (num_vars, num_vars)
        # 最小的奇异值对应的特征向量在 Vh 的最后几行
        # 我们需要提取最后 raw_dof + num_gauge 个向量，其中前 num_gauge 个是规范自由度(无需检查)
        # 真正的物理模态位于: Vh[-(num_gauge + raw_dof) : -num_gauge]
        
        # 修正索引逻辑：
        # spectrum[0] 是最小奇异值 -> Vh[-1]
        # spectrum[num_gauge + k] 是物理模态
        
        effective_dof = 0
        idof_count = 0
        drift_details = []

        # 遍历每一个潜在的物理自由度
        for k in range(int(raw_dof)):
            # 找到对应的特征向量索引
            # 规范自由度占用了最小的 num_gauge 个 (spectrum 0 ~ num_gauge-1)
            # 物理自由度从 spectrum[num_gauge] 开始
            idx_in_spectrum = num_gauge + k
            
            # 如果索引超出了实际奇异值范围 (虽然理论上不会)
            if idx_in_spectrum >= len(spectrum): break
            
            # Vh 是与 spectrum 逆序对应的 (spectrum[0] -> Vh[-1])
            vec_idx = -(idx_in_spectrum + 1)
            
            x_mode = Vh[vec_idx]
            
            # 计算该模态的二阶漂移
            drift_val = self._compute_drift_residual_internal(tensor, q_opt, cycles, x_mode, edge_to_col, K_curr).item()
            
            is_stable = drift_val < threshold_drift
            drift_details.append({"mode": k+1, "drift": drift_val, "is_stable": is_stable})
            
            if is_stable:
                effective_dof += 1
            else:
                idof_count += 1

        return {
            "raw_dof": int(raw_dof),
            "effective_dof": int(effective_dof),
            "idof_count": int(idof_count),
            "details": drift_details
        }

    # =========================================================================
    # 1. 优化器专用 Loss 函数
    # =========================================================================

    def _loss_closure(self, tensor, task, cycles=None, q_opt=None):
        """闭环约束 Loss"""
        base_node = cycles[0][0] if cycles else 0
        obs = self.compute_multi_path_states(tensor, q_opt, base_node=base_node)
        total_loss = torch.tensor(0.0, device=self.device)
        for node_id, obs_list in obs.items():
            if len(obs_list) < 2: continue
            ref_P, ref_z = obs_list[0]['P'], obs_list[0]['z']
            for i in range(1, len(obs_list)):
                curr_P, curr_z = obs_list[i]['P'], obs_list[i]['z']
                total_loss += torch.sum((curr_P - ref_P)**2) + torch.sum((curr_z - ref_z)**2)
        return total_loss

    def _loss_mobility_dof(self, tensor, task, cycles=None, q_opt=None):
        """
        自由度约束 Loss (优化器版)
        这里我们依然使用软约束 (Soft Constraint)，将漂移作为罚函数加进去，
        而不是直接返回整数 DOF。这样梯度才能传导。
        """
        if cycles:
            active_nodes = set()
            for cycle in cycles: active_nodes.update(cycle)
            num_gauge = len(active_nodes)
        else:
            num_gauge = tensor.shape[1]

        target_dof = 1
        if task:
            kinematics = task.get('kinematics')
            if isinstance(kinematics, dict):
                found_dof = None
                for k, v in kinematics.items():
                    if k.lower() == 'dof': found_dof = v; break
                if found_dof is not None:
                    try: target_dof = int(found_dof)
                    except: pass

        _, K_curr, Vh, spectrum = self.solve_anchor_system(
            tensor, q_opt, cycles, return_spectrum=True, return_full_data=True
        )

        if spectrum is None: return torch.tensor(10.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        
        # 优化目标：确保前 target_dof 个模态既有零奇异值，又是低漂移的
        for k in range(target_dof):
            idx = num_gauge + k
            if idx < len(spectrum):
                # 1. 奇异值惩罚 (必须是机构运动)
                total_loss += spectrum[idx] ** 2 * 100.0
                
                # 2. 漂移惩罚 (必须是连续运动)
                vec_idx = -(idx + 1)
                if abs(vec_idx) <= Vh.shape[0]:
                    x = Vh[vec_idx]
                    drift_loss = self._compute_drift_residual_internal(tensor, q_opt, cycles, x, _, K_curr)
                    total_loss += drift_loss * 10.0 # 加大权重
            else:
                total_loss += 10.0
        return total_loss

    def _loss_path_generation(self, tensor, task, cycles=None, q_opt=None):
        targets = task.get('targets', {})
        target_path_seq = targets.get('target_path_sequence', [])
        target_poses = targets.get('target_motion_twists', [])
        if not target_path_seq or not target_poses: return torch.tensor(0.0, device=self.device)
        
        PI = self.PI
        fk_path_nodes = target_path_seq[1:-1]
        ghost_in = target_path_seq[0]
        T_ee = torch.eye(4, device=self.device)
        
        for i in range(len(fk_path_nodes) - 1):
            curr, next_n = fk_path_nodes[i], fk_path_nodes[i+1]
            prev = ghost_in if i==0 else fk_path_nodes[i-1]
            
            j_type_val = tensor[1, curr, next_n]
            a = torch.abs(tensor[2, curr, next_n])
            alpha = tensor[3, curr, next_n]
            off_out, off_in = tensor[4, curr, next_n], tensor[4, curr, prev]
            d_static = off_out - off_in
            q_out, q_in = q_opt[curr, next_n], q_opt[curr, prev]
            q_diff = q_out - q_in
            
            if self._is_revolute(j_type_val):
                theta = q_diff - PI; d = d_static
            else:
                theta = d_static - PI; d = q_diff
            
            T_step = self.get_dh_matrix_fast(a, alpha, d, theta)
            T_ee = T_ee @ T_step

        tgt_vec = torch.tensor(target_poses[0], device=self.device)
        T_target = self._vec6_to_matrix(tgt_vec)
        pos_err = torch.norm(T_ee[:3, 3] - T_target[:3, 3])
        rot_err = torch.norm(T_ee[:3, :3] - T_target[:3, :3])
        return pos_err + rot_err

    def _loss_twist_alignment(self, tensor, task, cycles=None, q_opt=None):
        return self.compute_twist_match_residual(tensor, task, cycles, q_opt)

    def _loss_instantaneous_check(self, tensor, task, cycles=None, q_opt=None):
        """
        瞬时性检查 (Loss版)
        注意：现在这个函数主要用于有任务时。
        如果无任务，优化器主要依靠 _loss_mobility_dof 里的漂移惩罚。
        """
        targets = task.get('targets', {}) if task else {}
        path_to_ee = targets.get('target_path_sequence', [])
        target_twists = targets.get('target_motion_twists', [])
        
        if path_to_ee and target_twists:
            return self.compute_instantaneous_check_loss(tensor, task, cycles, q_opt)
        
        # 无任务时返回0，因为 mobility_dof 已经负责了无任务的瞬时性
        return torch.tensor(0.0, device=self.device)

    def _loss_bennett_condition(self, tensor, task, cycles=None, q_opt=None):
        if not cycles: return torch.tensor(0.0, device=self.device)
        total_error = torch.tensor(0.0, device=self.device)
        for path in cycles:
            if len(path) != 4: continue
            a_list, alpha_list = [], []
            offset_loss = torch.tensor(0.0, device=self.device)
            L = 4
            for i in range(L):
                curr, next_n, prev = path[i], path[(i + 1) % L], path[(i - 1 + L) % L]
                a = torch.abs(tensor[2, curr, next_n])
                alpha = tensor[3, curr, next_n] % (2 * math.pi)
                a_list.append(a); alpha_list.append(alpha)
                off_out, off_in = tensor[4, curr, next_n], tensor[4, curr, prev]
                offset_loss += (off_out - off_in) ** 2
            
            a_vec, alpha_vec = torch.stack(a_list), torch.stack(alpha_list)
            sym_loss_a = ((a_vec[0] - a_vec[2])**2 + (a_vec[1] - a_vec[3])**2)
            sym_loss_alpha = (1.0 - torch.cos(alpha_vec[0] - alpha_vec[2])) + (1.0 - torch.cos(alpha_vec[1] - alpha_vec[3]))
            total_error += sym_loss_a + sym_loss_alpha + offset_loss
        return total_error

    # =========================================================================
    # 2. 核心物理计算
    # =========================================================================

    def compute_twist_match_residual(self, tensor, task, cycles=None, q_opt=None):
        targets = task.get('targets', {})
        target_path_seq = targets.get('target_path_sequence', [])
        target_twists = targets.get('target_motion_twists', [])
        target_masks = targets.get('target_masks', [])

        if not target_path_seq or not target_twists: return torch.tensor(0.0, device=self.device)

        if cycles:
            active_nodes = set()
            for cycle in cycles: active_nodes.update(cycle)
            for n in target_path_seq: 
                if n >= 0: active_nodes.add(n)
            num_gauge = len(active_nodes)
        else:
            num_gauge = tensor.shape[1]

        total_residual = torch.tensor(0.0, device=self.device)
        num_tasks = len(target_twists)

        for k in range(num_tasks):
            tgt_twist = torch.tensor(target_twists[k], device=self.device)
            tgt_mask = torch.tensor(target_masks[k], device=self.device) if target_masks and k < len(target_masks) else torch.ones(6, device=self.device)

            _, _, _, spectrum = self.solve_anchor_system(
                tensor, q_opt, cycles, target_path_seq, tgt_twist, tgt_mask, return_spectrum=True
            )

            if spectrum is None: 
                total_residual += 10.0; continue
            
            if num_gauge < len(spectrum):
                total_residual += spectrum[num_gauge]
            else:
                total_residual += 1.0

        return total_residual / num_tasks

    def compute_instantaneous_check_loss(self, tensor, task, cycles=None, q_opt=None):
        targets = task.get('targets', {})
        path_to_ee = targets.get('target_path_sequence', [])
        target_twists = targets.get('target_motion_twists', [])
        target_masks = targets.get('target_masks', [])

        if not path_to_ee or not target_twists: return torch.tensor(0.0, device=self.device)

        dt = 1e-3
        total_loss = torch.tensor(0.0, device=self.device)
        num_modes = len(target_twists)

        for k in range(num_modes):
            tgt_twist = torch.tensor(target_twists[k], device=self.device)
            tgt_mask = torch.tensor(target_masks[k], device=self.device) if target_masks and k < len(target_masks) else torch.ones(6, device=self.device)

            mapping, K_curr, Vh, _ = self.solve_anchor_system(tensor, q_opt, cycles, path_to_ee, tgt_twist, tgt_mask, return_full_data=True)
            
            if Vh is None: continue
            x0 = Vh[-1, :]
            
            x_norm = torch.norm(x0)
            if x_norm < 1e-6: continue
            x0 = x0 / x_norm

            q_next = q_opt.clone()
            for (u, v), col_idx in mapping.items():
                if col_idx < len(x0):
                    val = x0[col_idx] * dt
                    q_next[u, v] += val; q_next[v, u] -= val
            
            _, K_next, _, _ = self.solve_anchor_system(tensor, q_next, cycles, path_to_ee, tgt_twist, tgt_mask, return_full_data=True)
            if K_next is None:
                total_loss += 10.0; continue
            
            drift_vec = (K_next @ x0 - K_curr @ x0) / dt
            try:
                K_noisy = K_curr + torch.randn_like(K_curr)*1e-9 if K_curr.requires_grad else K_curr
                K_pinv = torch.linalg.pinv(K_noisy, rcond=1e-3)
                residual = (-drift_vec) - K_curr @ (K_pinv @ (-drift_vec))
                loss_ratio = torch.norm(residual) / (torch.norm(drift_vec) + 1e-6)
                total_loss += loss_ratio
            except:
                total_loss += 1.0
        return total_loss / num_modes

    def _compute_drift_residual_internal(self, tensor, q_opt, cycles, x0, mapping, K_curr, dt=1e-3):
        x_norm = torch.norm(x0)
        if x_norm < 1e-6: return torch.tensor(0.0, device=self.device)
        x0 = x0 / x_norm

        q_next = q_opt.clone()
        if mapping is None:
            involved_edges_set = set()
            for loop in cycles:
                L = len(loop)
                for i in range(L): involved_edges_set.add(tuple(sorted((loop[i], loop[(i+1)%L]))))
            edge_to_col = {}
            curr_col = 0
            for u, v in involved_edges_set:
                edge_to_col[(u,v)] = curr_col; curr_col += 1
                edge_to_col[(v,u)] = curr_col; curr_col += 1
            mapping = edge_to_col

        for (u, v), col_idx in mapping.items():
            if col_idx < len(x0):
                val = x0[col_idx] * dt
                q_next[u, v] += val
                q_next[v, u] -= val

        _, K_next, _, _ = self.solve_anchor_system(tensor, q_next, cycles, return_full_data=True)
        if K_next is None: return torch.tensor(1.0, device=self.device)

        drift_vec = (K_next @ x0 - K_curr @ x0) / dt
        try:
            K_pinv = torch.linalg.pinv(K_curr, rcond=1e-3)
            residual = (-drift_vec) - K_curr @ (K_pinv @ (-drift_vec))
            return torch.norm(residual) / (torch.norm(drift_vec) + 1e-6)
        except:
            return torch.tensor(1.0, device=self.device)

    # =========================================================================
    # 3. 基础算法与辅助函数
    # =========================================================================

    def solve_anchor_system(self, structure, q_current, loops, extended_task_path=None, target_twist=None, target_mask=None, return_spectrum=False, return_full_data=False):
        active_nodes = set()
        if loops: [active_nodes.update(l) for l in loops]
        if extended_task_path: active_nodes.update([n for n in extended_task_path if n >= 0])
        if not active_nodes: return None, None, None, None

        base_node = extended_task_path[1] if extended_task_path and len(extended_task_path) > 1 else min(active_nodes)
        all_screws = self.compute_all_joint_screws(structure, q_current, base_node, normalize=True, cycles=loops)

        involved_edges_set = set()
        if loops:
            for loop in loops:
                L = len(loop)
                for i in range(L): involved_edges_set.add(tuple(sorted((loop[i], loop[(i+1)%L]))))
        if extended_task_path:
            for i in range(len(extended_task_path)-1):
                if extended_task_path[i]>=0 and extended_task_path[i+1]>=0:
                    involved_edges_set.add(tuple(sorted((extended_task_path[i], extended_task_path[i+1]))))

        edge_to_col = {}
        current_col = 0
        for u, v in involved_edges_set:
            edge_to_col[(u, v)] = current_col; current_col += 1
            edge_to_col[(v, u)] = current_col; current_col += 1

        num_vars_reduced = current_col
        if num_vars_reduced == 0: return None, None, None, None

        num_loops = len(loops) if loops else 0
        has_task = (target_twist is not None)
        total_rows = 6 * (num_loops + (1 if has_task else 0))
        K_compact = torch.zeros((total_rows, num_vars_reduced), device=self.device)
        b = torch.zeros(total_rows, device=self.device)

        current_row = 0
        if loops:
            for loop in loops:
                L = len(loop)
                for i in range(L):
                    u, v, p = loop[i], loop[(i+1)%L], loop[(i-1+L)%L]
                    s = all_screws[u]
                    if (u,v) in edge_to_col: K_compact[current_row:current_row+6, edge_to_col[(u,v)]] += s
                    if (u,p) in edge_to_col: K_compact[current_row:current_row+6, edge_to_col[(u,p)]] -= s
                current_row += 6

        if has_task and extended_task_path:
            rs = slice(current_row, current_row+6)
            for i in range(1, len(extended_task_path)-1):
                u, p, n = extended_task_path[i], extended_task_path[i-1], extended_task_path[i+1]
                s = all_screws[u]
                if n>=0 and (u,n) in edge_to_col: K_compact[rs, edge_to_col[(u,n)]] += s
                if p>=0 and (u,p) in edge_to_col: K_compact[rs, edge_to_col[(u,p)]] -= s
            if target_mask is not None:
                K_compact[rs, :] *= target_mask.unsqueeze(1)
                b[rs] = target_twist * target_mask
            else:
                b[rs] = target_twist

        b_unsq = b.unsqueeze(1)
        K_aug = torch.cat([K_compact, -b_unsq], dim=1) if has_task else K_compact

        try:
            if getattr(self.cfg, 'enable_jitter', True) and K_aug.requires_grad:
                K_aug = K_aug + torch.randn_like(K_aug)*1e-9
            
            U, S, Vh = torch.linalg.svd(K_aug, full_matrices=True)
            full_S = torch.zeros(K_aug.shape[1], dtype=S.dtype, device=self.device)
            full_S[:S.shape[0]] = S
            spectrum = torch.flip(full_S, dims=[0])

            if return_full_data:
                return edge_to_col, K_aug, Vh, spectrum
            
            return None, None, None, spectrum
        except:
            return None, None, None, None

    def compute_all_joint_screws(self, structure, q_current, base_node, normalize=True, cycles=None):
        obs = self.compute_multi_path_states(structure, q_current, base_node)
        N = structure.shape[1]
        screws = torch.zeros((N, 6), device=self.device)
        L_char = torch.tensor(1.0, device=self.device)
        if normalize:
            exists_mask = structure[0] > 0.5
            all_a = torch.abs(structure[2][exists_mask])
            valid_a = all_a[all_a > 1e-6]
            if valid_a.numel() > 0: L_char = torch.mean(valid_a)
        
        for u in range(N):
            if u not in obs: continue
            state = obs[u][0]
            P, z = state['P'], state['z']
            
            is_R_node = True
            parent = state.get('parent')
            if parent is not None:
                j_type_val = structure[1, parent, u]
                is_R_node = self._is_revolute(j_type_val)
            
            if is_R_node:
                v = torch.linalg.cross(P, z)
                if normalize: v = v / L_char
                screws[u] = torch.cat([z, v])
            else:
                screws[u] = torch.cat([torch.zeros(3).to(self.device), z])
        return screws

    def compute_multi_path_states(self, structure, q_current, base_node=0):
        N = structure.shape[1]
        node_observations, expanded_nodes = {}, set()
        
        base_neighbors = torch.nonzero(structure[0, base_node, :] > 0.5).view(-1).tolist()
        if not base_neighbors: return {}
        ref_node = base_neighbors[0]
        T_base = torch.eye(4, device=self.device)
        node_observations[base_node] = [{'P': T_base[:3, 3], 'z': T_base[:3, 2], 'T': T_base, 'parent': ref_node}]
        expanded_nodes.add(base_node)
        stack = [(base_node, -1, T_base)]
        PI = self.PI

        while stack:
            u, p, T_u = stack.pop()
            if p == -1: off_in, q_in = structure[4, u, ref_node], q_current[u, ref_node]
            else: off_in, q_in = structure[4, u, p], q_current[u, p]
            
            neighbors = torch.nonzero(structure[0, u, :] > 0.5).view(-1).tolist()
            for v in neighbors:
                logical_parent = ref_node if p == -1 else p
                if v == logical_parent: continue
                
                j_type_val = structure[1, u, v]
                a = structure[2, u, v]
                alpha = structure[3, u, v]
                off_out = structure[4, u, v]
                q_out = q_current[u, v]
                
                delta_off, delta_q = off_out - off_in, q_out - q_in
                
                if self._is_revolute(j_type_val):
                    theta = delta_q - PI; d = delta_off
                else:
                    theta = delta_off - PI; d = delta_q
                
                T_step = self.get_dh_matrix_fast(a, alpha, d, theta)
                T_v = T_u @ T_step
                
                if v not in node_observations: node_observations[v] = []
                node_observations[v].append({'P': T_v[:3, 3], 'z': T_v[:3, 2], 'T': T_v, 'parent': u})
                if v not in expanded_nodes: expanded_nodes.add(v); stack.append((v, u, T_v))
        return node_observations

    def get_dh_matrix_fast(self, a, alpha, d, theta):
        ct, st = torch.cos(theta), torch.sin(theta)
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        zero, one = torch.zeros_like(theta), torch.ones_like(theta)
        r1 = torch.stack([ct, -st*ca, st*sa, a*ct], dim=-1)
        r2 = torch.stack([st, ct*ca, -ct*sa, a*st], dim=-1)
        r3 = torch.stack([zero, sa, ca, d], dim=-1)
        r4 = torch.stack([zero, zero, zero, one], dim=-1)
        return torch.stack([r1, r2, r3, r4], dim=-2)

    def find_fundamental_cycles(self, adj_matrix):
        if isinstance(adj_matrix, torch.Tensor): adj_matrix = adj_matrix.detach().cpu().numpy()
        rows, cols = np.where(adj_matrix > 0.5)
        G = nx.Graph()
        G.add_edges_from(zip(rows, cols))
        try: return [list(c) for c in nx.cycle_basis(G)]
        except: return []

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
            torch.stack([cy*cz, cz*sx*sy - cx*sz, cx*cz*sy + sx*sz]),
            torch.stack([cy*sz, cx*cz + sx*sy*sz, -cz*sx + cx*sy*sz]),
            torch.stack([-sy, cy*sx, cx*cy])
        ])
        T = torch.eye(4, device=self.device)
        T[:3, :3] = R
        T[:3, 3] = torch.stack([x, y, z])
        return T

    def _get_constrained_geometry(self, raw_tensor):
        exists, j_type = raw_tensor[0], raw_tensor[1]
        a, alpha, offset = raw_tensor[2], raw_tensor[3], raw_tensor[4]
        exists_sym = (exists + exists.T) / 2.0
        a_sym = (a + a.T) / 2.0
        alpha_sym = (alpha + alpha.T) / 2.0
        j_type_row = j_type.mean(dim=1, keepdim=True).expand_as(j_type)
        a_phys = torch.abs(a_sym)
        alpha_phys = torch.remainder(alpha_sym, 2 * math.pi)
        mask = (exists_sym > 0.5).float()
        diag_mask = 1.0 - torch.eye(raw_tensor.shape[1], device=self.device)
        return torch.stack([exists_sym * mask * diag_mask, j_type_row, a_phys * mask * diag_mask, alpha_phys * mask * diag_mask, offset * mask * diag_mask], dim=0)

    def _is_revolute(self, j_type_val):
        return j_type_val > 0.5

    # 兼容性别名
    _solve_anchor_system = solve_anchor_system
    _compute_multi_path_states = compute_multi_path_states
    _compute_all_joint_screws = compute_all_joint_screws
    _find_fundamental_cycles = find_fundamental_cycles
    _get_dh_matrix_fast = get_dh_matrix_fast