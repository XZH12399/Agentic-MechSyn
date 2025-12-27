import torch
import numpy as np
import networkx as nx
import random
import itertools


class MechanismConverter:
    def __init__(self, device='cpu'):
        self.device = device

    # =========================================================
    # 1. 几何计算与标准化辅助函数
    # =========================================================

    def _get_standard_axis(self, z_vec):
        z = np.array(z_vec)
        norm = np.linalg.norm(z)
        if norm < 1e-9: return z, 1.0
        z = z / norm
        sign = 1.0
        if abs(z[0]) > 1e-6:
            if z[0] < 0: sign = -1.0
        elif abs(z[1]) > 1e-6:
            if z[1] < 0: sign = -1.0
        else:
            if z[2] < 0: sign = -1.0
        return z * sign, sign

    def _compute_geometry_params(self, p1, z1, p2, z2):
        n = np.cross(z1, z2)
        n_norm = np.linalg.norm(n)

        if n_norm < 1e-6:  # 平行
            alpha = 0.0
            n = np.cross(z1, p2 - p1)
            if np.linalg.norm(n) < 1e-6:
                n = np.cross(z1, np.array([1, 0, 0]))
                if np.linalg.norm(n) < 1e-6: n = np.cross(z1, np.array([0, 1, 0]))
            n = n / np.linalg.norm(n)
            a = np.linalg.norm(np.cross(p2 - p1, z1))
            t1 = np.dot(p2 - p1, z1)
            Pa = p1 + t1 * z1
            Pb = p2 + z2 * np.dot(Pa - p2, z2)
        else:  # 异面
            n = n / n_norm
            alpha = np.arctan2(n_norm, np.dot(z1, z2))
            a = np.abs(np.dot(p2 - p1, n))
            dot_12 = np.dot(z1, z2)
            denom = 1 - dot_12 ** 2
            vec = p2 - p1
            t1 = (np.dot(vec, z1) - np.dot(vec, z2) * dot_12) / denom
            t2 = (np.dot(vec, z1) * dot_12 - np.dot(vec, z2)) / denom
            Pa = p1 + z1 * t1
            Pb = p2 + z2 * t2
        return float(a), float(alpha), n, Pa, Pb

    def _normalize_node_params(self, node_id, attachments, axis_vec):
        if not attachments: return {}

        d_values = [item['d_raw'] for item in attachments]
        d_mean = np.mean(d_values)

        base_n = attachments[0]['n_vec']
        local_x = base_n
        local_y = np.cross(axis_vec, local_x)  # 修正: 确保正交系正确

        q_values = []
        for item in attachments:
            n = item['n_vec']
            # 使用 atan2 确保角度范围正确
            # n 在 local_x, local_y 平面的投影
            angle = np.arctan2(np.dot(n, local_y), np.dot(n, local_x))
            q_values.append(angle)
        q_mean = np.mean(q_values)

        normalized_map = {}
        for i, item in enumerate(attachments):
            edge_key = item['edge_key']
            d_final = d_values[i] - d_mean
            q_final = (q_values[i] - q_mean + np.pi) % (2 * np.pi) - np.pi
            normalized_map[edge_key] = {"offset": float(d_final), "q": float(q_final)}
        return normalized_map

    # =========================================================
    # 2. 运动学分析辅助函数
    # =========================================================

    def _build_edge_mapping(self, G):
        cycles = nx.cycle_basis(G)
        involved_edges = set()
        for loop in cycles:
            L = len(loop)
            for i in range(L):
                u, v = loop[i], loop[(i + 1) % L]
                involved_edges.add(tuple(sorted((u, v))))
        edge_to_col = {}
        idx = 0
        for u, v in involved_edges:
            edge_to_col[(u, v)] = idx;
            idx += 1
            edge_to_col[(v, u)] = idx;
            idx += 1
        return edge_to_col

    def _compute_all_screws(self, num_nodes, P_t, Z_t, joint_types):
        Screws = []
        for i in range(num_nodes):
            w = Z_t[i]
            v_vec = torch.linalg.cross(P_t[i], Z_t[i])
            if joint_types[i] == 'P':
                w = torch.zeros(3)
                v_vec = Z_t[i]
            Screws.append(torch.cat([w, v_vec]))
        return torch.stack(Screws)

    def _find_best_path_between_edges(self, G, edge_base, edge_ee):
        u1, v1 = edge_base
        u2, v2 = edge_ee
        candidates = []

        def check_path(start, end, anchor_start, anchor_end):
            try:
                p = nx.shortest_path(G, start, end)
                return p, (anchor_start, start), (end, anchor_end)
            except:
                return None, None, None

        for s, e, anc_s, anc_e in [
            (v1, u2, u1, v2), (v1, v2, u1, u2),
            (u1, u2, v1, v2), (u1, v2, v1, u2)]:
            p, b, e_ = check_path(s, e, anc_s, anc_e)
            candidates.append((p, b, e_))

        valid = [c for c in candidates if c[0]]
        if not valid: return None, None, None
        valid.sort(key=lambda x: len(x[0]))
        return valid[0][1], valid[0][2], valid[0][0]

    def _compute_relative_twist_link_to_link(self, path_nodes, ordered_base, ordered_ee, Screws, edge_to_col,
                                             null_motion):
        twist = torch.zeros(6)
        L = len(path_nodes)
        for i in range(L):
            curr = path_nodes[i]
            prev = ordered_base[0] if i == 0 else path_nodes[i - 1]
            next_n = ordered_ee[1] if i == L - 1 else path_nodes[i + 1]

            val_in, val_out = 0.0, 0.0
            if (prev, curr) in edge_to_col:
                val_in = null_motion[edge_to_col[(prev, curr)]]
            elif (curr, prev) in edge_to_col:
                val_in = -null_motion[edge_to_col[(curr, prev)]]

            if (curr, next_n) in edge_to_col:
                val_out = null_motion[edge_to_col[(curr, next_n)]]
            elif (next_n, curr) in edge_to_col:
                val_out = -null_motion[edge_to_col[(next_n, curr)]]

            twist += Screws[curr] * (val_out - val_in)

        if torch.norm(twist) > 1e-6: twist = twist / torch.norm(twist)
        return twist.tolist()

    # =========================================================
    # 3. 主处理逻辑 (Clean Version)
    # =========================================================

    def process(self, G, P, Z, joint_types, dof_val, null_motion, mech_id, num_task_samples=5):
        num_nodes = len(G.nodes)
        P_np = P.cpu().numpy() if isinstance(P, torch.Tensor) else P
        Z_np = Z.cpu().numpy() if isinstance(Z, torch.Tensor) else Z

        # A. 预备
        if not isinstance(null_motion, torch.Tensor): null_motion = torch.tensor(null_motion)
        null_motion = null_motion.cpu()
        P_t = torch.tensor(P_np, dtype=torch.float32)
        Z_t = torch.tensor(Z_np, dtype=torch.float32)
        edge_to_col = self._build_edge_mapping(G)
        Screws = self._compute_all_screws(num_nodes, P_t, Z_t, joint_types)

        # B. 标准化
        std_Z = []
        for i in range(num_nodes):
            z_new, sign = self._get_standard_axis(Z_np[i])
            std_Z.append(z_new)
        std_Z = np.array(std_Z)

        # C. 几何计算 & 收集
        node_attachments = {i: [] for i in range(num_nodes)}
        raw_edges_data = {}

        for u, v in G.edges():
            a, alpha, n, Pa, Pb = self._compute_geometry_params(
                P_np[u], std_Z[u], P_np[v], std_Z[v]
            )
            d_u_raw = np.dot(Pa - P_np[u], std_Z[u])
            node_attachments[u].append({"edge_key": (u, v), "d_raw": d_u_raw, "n_vec": n})

            d_v_raw = np.dot(Pb - P_np[v], std_Z[v])
            node_attachments[v].append({"edge_key": (v, u), "d_raw": d_v_raw, "n_vec": n})

            raw_edges_data[tuple(sorted((u, v)))] = {"a": a, "alpha": alpha}

        # D. 归一化
        node_normalized_params = {}
        for i in range(num_nodes):
            attachments = node_attachments[i]
            attachments.sort(key=lambda x: x['d_raw'])
            norm_map = self._normalize_node_params(i, attachments, std_Z[i])
            node_normalized_params[i] = norm_map

        # E. 任务采样 (Tasks) - 不再决定 Graph 顶层的 Ground/EE
        edges_list_nx = list(G.edges())
        all_edge_pairs = list(itertools.combinations(edges_list_nx, 2))
        sampled_pairs = random.sample(all_edge_pairs, min(len(all_edge_pairs), num_task_samples))

        tasks_data = []
        for e1, e2 in sampled_pairs:
            ordered_base, ordered_ee, path_nodes = self._find_best_path_between_edges(G, e1, e2)
            if path_nodes:
                twist = self._compute_relative_twist_link_to_link(
                    path_nodes, ordered_base, ordered_ee, Screws, edge_to_col, null_motion
                )
                tasks_data.append({
                    "base_link": list(ordered_base),  # [Virtual, Real]
                    "ee_link": list(ordered_ee),  # [Real, Virtual]
                    "joint_path": path_nodes,
                    "motion_screw": twist
                })

        # F. 构建 JSON - 纯粹的机构描述
        nodes_list = []
        for i in range(num_nodes):
            # 去除 Role 字段，因为 Role 取决于 Task
            nodes_list.append({
                "id": i,
                "type": joint_types[i]
            })

        edges_output = []
        for u, v in G.edges():
            geo = raw_edges_data[tuple(sorted((u, v)))]
            p_src = node_normalized_params[u].get((u, v))
            p_tgt = node_normalized_params[v].get((v, u))
            edges_output.append({
                "source": u, "target": v,
                "params": {
                    "a": geo['a'], "alpha": geo['alpha'],
                    "offset_source": p_src['offset'], "offset_target": p_tgt['offset']
                },
                "initial_state": {
                    "q_source": p_src['q'], "q_target": p_tgt['q']
                }
            })

        final_data = {
            # 移除了顶层的 ground_nodes 和 ee_node
            "graph": {
                "nodes": nodes_list,
                "edges": edges_output
            },
            "meta": {
                "dof": int(dof_val),
                "num_loops": len(nx.cycle_basis(G)),
                "is_spatial": True,
                "tasks": tasks_data  # 所有的应用场景定义都在这里
            }
        }

        return final_data