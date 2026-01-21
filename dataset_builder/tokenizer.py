import numpy as np


class MechanismTokenizer:
    def __init__(self, len_bins=2000, ang_bins=360, off_bins=2000, state_bins=360):
        # 增加 bin 数量以配合更大的数值范围，保持高精度
        self.bins = {
            "len": len_bins,
            "ang": ang_bins,
            "off": off_bins,
            "state": state_bins
        }

        # === 1. 构建词表 ===
        self.vocab = [
            "<PAD>", "<SOS>", "<EOS>",
            # 动作指令
            "<Action_New_Node>", "<Action_Link_To_New>",
            "<Action_Link_To_Old>", "<Action_Jump_To>",
            # 属性指令
            "<Type_R>", "<Type_P>",
            # 角色指令
            "<Role_Ground>", "<Role_EE>", "<Role_Normal>",
            # 状态特殊值
            "<State_Free>"
        ]

        # 动态生成的 ID 和 数值 Token
        self.vocab += [f"<ID_{i}>" for i in range(32)]
        self.vocab += [f"<Len_{i}>" for i in range(self.bins["len"])]
        self.vocab += [f"<Twist_{i}>" for i in range(self.bins["ang"])]
        self.vocab += [f"<Off_{i}>" for i in range(self.bins["off"])]
        self.vocab += [f"<State_{i}>" for i in range(self.bins["state"])]

        # ✨ 新增：DoF 和 Loop 的 Token (支持 0-9)
        self.vocab += [f"<DoF_{i}>" for i in range(10)]
        self.vocab += [f"<Loop_{i}>" for i in range(10)]

        # 全局属性
        self.vocab += [
            "<Space_Planar>", "<Space_Spatial>",
            "<Type_General_Parallel>", "<Type_Delta>",
            "<Type_Bennett>", "<Type_Bricard>", "<Type_Origami>", "<Type_Sarrus>"
        ]

        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for i, t in enumerate(self.vocab)}

    # ==========================================
    # Decode (Tokens -> Graph)
    # ==========================================

    def decode(self, token_sequence):
        nodes = {}
        connections = []

        cursor_id = -1
        iterator = iter(token_sequence)

        try:
            while True:
                token = next(iterator)
                if token == "<EOS>": break
                if token == "<SOS>" or token == "<PAD>": continue

                # 跳过规格 Token (DoF/Loop)，它们仅用于 Encoder 输入
                if token.startswith("<DoF_") or token.startswith("<Loop_"):
                    continue

                # === 模式 A: 建立新节点 ===
                if token == "<Action_New_Node>":
                    node_id = self._parse_id(next(iterator))
                    node_type = self._parse_type(next(iterator))
                    node_role = self._parse_role(next(iterator))

                    self._register_node(nodes, node_id, node_type, node_role)
                    cursor_id = node_id

                # === 模式 B: 连接到新节点 ===
                elif token == "<Action_Link_To_New>":
                    if cursor_id == -1: raise ValueError("Action_Link_To_New without cursor!")

                    new_id = self._parse_id(next(iterator))
                    node_type = self._parse_type(next(iterator))
                    node_role = self._parse_role(next(iterator))

                    self._register_node(nodes, new_id, node_type, node_role)

                    edge_params = self._parse_edge_params(iterator)
                    connections.append({
                        "source": cursor_id, "target": new_id, **edge_params
                    })
                    cursor_id = new_id

                # === 模式 C: 连接到老节点 ===
                elif token == "<Action_Link_To_Old>":
                    if cursor_id == -1: raise ValueError("Action_Link_To_Old without cursor!")

                    target_id = self._parse_id(next(iterator))
                    # Old Node 不需要 Role

                    edge_params = self._parse_edge_params(iterator)
                    connections.append({
                        "source": cursor_id, "target": target_id, **edge_params
                    })

                # === 模式 D: 跳转 ===
                elif token == "<Action_Jump_To>":
                    cursor_id = self._parse_id(next(iterator))

        except StopIteration:
            pass

        # 后处理：提取 Ground 和 EE
        ground_nodes = []
        ee_nodes = []

        for nid, props in nodes.items():
            if props['role'] == 'Ground':
                ground_nodes.append(nid)
            elif props['role'] == 'EE':
                ee_nodes.append(nid)

        return {
            "topology": {
                "nodes": {str(k): v['type'] for k, v in nodes.items()},
                "connections": connections
            },
            "meta": {
                "ground_nodes": ground_nodes,
                "ee_node": ee_nodes,
                "num_nodes": len(nodes),
                "num_edges": len(connections)
            }
        }

    # ==========================================
    # 辅助解析函数
    # ==========================================

    def _register_node(self, nodes, nid, ntype, nrole):
        nodes[nid] = {"type": ntype, "role": nrole}

    def _parse_role(self, token):
        if token == "<Role_Ground>": return "Ground"
        if token == "<Role_EE>": return "EE"
        return "Normal"

    def _parse_edge_params(self, iterator):
        return {
            "a": self._val(next(iterator), "len"),
            "alpha": self._val(next(iterator), "ang", is_angle=True),
            "offset_src": self._val(next(iterator), "off"),
            "offset_tgt": self._val(next(iterator), "off"),
            "state_src": self._val(next(iterator), "state", is_angle=True, allow_free=True),
            "state_tgt": self._val(next(iterator), "state", is_angle=True, allow_free=True)
        }

    def _parse_id(self, token):
        # 移除可能存在的 '>' 符号
        return int(token.split('_')[1].replace('>', ''))

    def _parse_type(self, token):
        return "R" if "Type_R" in token else "P"

    def _val(self, token, bin_type, is_angle=False, allow_free=False):
        """将 Token 转为数值"""
        if allow_free and "Free" in token:
            return None

        try:
            # 格式如 <Len_50> -> 50
            idx = int(token.split('_')[-1].replace(">", ""))
            max_bin = self.bins[bin_type]

            norm_val = idx / (max_bin - 1)

            # 反归一化：数值范围扩大至 [-20, 20]
            if is_angle:
                # [0, 1] -> [-pi, pi]
                val = norm_val * 2 * np.pi - np.pi
            else:
                # [0, 1] -> [-20, 20]
                val = norm_val * 40.0 - 20.0

            return val

        except:
            return 0.0

    # ==========================================
    # Encode (Graph / Specs / Task -> Tokens)
    # ==========================================

    def _get_node_role(self, node_id, base_ids, ee_ids):
        if node_id in base_ids: return "Ground"
        if node_id in ee_ids: return "EE"
        return "Normal"

    def encode_value(self, val, bin_type, is_angle=False):
        """数值转 Token (带 NaN/Inf 保护)"""
        if val is None or np.isnan(val) or np.isinf(val):
            val = 0.0

        max_bin = self.bins[bin_type]

        if is_angle:
            # [-pi, pi] -> [0, 1]
            norm_val = (val + np.pi) / (2 * np.pi)
        else:
            # [-20, 20] -> [0, 1] (扩大范围以容纳异常值)
            norm_val = (val + 20.0) / 40.0

        norm_val = np.clip(norm_val, 0.0, 1.0)

        # 使用 round 四舍五入减少量化误差
        idx = int(round(norm_val * (max_bin - 1)))

        prefix = bin_type.capitalize()
        if bin_type == "ang": prefix = "Twist"
        if bin_type == "state": prefix = "State"
        if bin_type == "off": prefix = "Off"

        return f"<{prefix}_{idx}>"

    def encode_specs(self, dof, num_loops):
        """编码机构规格 (DoF, Loops)"""
        dof = max(0, min(int(dof), 9))
        num_loops = max(0, min(int(num_loops), 9))
        return [f"<DoF_{dof}>", f"<Loop_{num_loops}>"]

    def encode_task(self, motion_screws_list):
        """
        编码运动螺旋列表
        Input: [[w1, v1...], [w2, v2...]] (针对 DoF=2)
        Output: Flattened Tokens [Screw1_Tokens..., Screw2_Tokens..., Pad...]
        """
        # 设定最大支持的任务自由度维度 (例如 3)
        # 如果机构自由度只有 1，剩下的位置补零
        MAX_TASK_DOF = 3

        tokens = []
        for i in range(MAX_TASK_DOF):
            if i < len(motion_screws_list):
                screw = motion_screws_list[i]
                for val in screw:
                    # 乘以 20 利用 [-20, 20] 范围
                    tokens.append(self.encode_value(val * 20.0, "off"))
            else:
                # 填充零螺旋 (表示该维度无运动)
                for _ in range(6):
                    tokens.append(self.encode_value(0.0, "off"))

        return tokens

    def encode_graph(self, graph_data, base_ids=None, ee_ids=None):
        """编码图结构，同时注入任务相关的角色信息"""
        if base_ids is None: base_ids = []
        if ee_ids is None: ee_ids = []
        base_ids = [int(i) for i in base_ids]
        ee_ids = [int(i) for i in ee_ids]

        nodes = {n['id']: n for n in graph_data['nodes']}
        adj = {n: [] for n in nodes}
        all_edges = set()

        for e in graph_data['edges']:
            u, v = e['source'], e['target']
            edge_id = tuple(sorted((u, v)))
            adj[u].append((v, e, edge_id))
            adj[v].append((u, e, edge_id))
            all_edges.add(edge_id)

        visited_nodes = set()
        visited_edges = set()
        token_seq = []

        start_node = base_ids[0] if base_ids else min(nodes.keys())
        curr = start_node
        visited_nodes.add(curr)

        # Action: New Node
        role = self._get_node_role(curr, base_ids, ee_ids)
        token_seq.extend([
            "<Action_New_Node>",
            f"<ID_{curr}>",
            f"<Type_{nodes[curr]['type']}>",
            f"<Role_{role}>"
        ])

        stack = [curr]

        while len(visited_edges) < len(all_edges) or len(visited_nodes) < len(nodes):
            if not stack:
                unvisited = [n for n in nodes if n not in visited_nodes]
                if not unvisited: break

                next_node = unvisited[0]
                role = self._get_node_role(next_node, base_ids, ee_ids)
                token_seq.extend([
                    "<Action_Jump_To>", f"<ID_{next_node}>",
                    "<Action_New_Node>", f"<ID_{next_node}>",
                    f"<Type_{nodes[next_node]['type']}>",
                    f"<Role_{role}>"
                ])
                visited_nodes.add(next_node)
                stack.append(next_node)
                curr = next_node
                continue

            curr = stack[-1]
            neighbors = sorted(adj[curr], key=lambda x: x[0])
            found_move = False

            # 1. Link To New
            for neighbor, edge_data, edge_id in neighbors:
                if edge_id in visited_edges: continue
                if neighbor not in visited_nodes:
                    visited_edges.add(edge_id)
                    visited_nodes.add(neighbor)

                    role = self._get_node_role(neighbor, base_ids, ee_ids)
                    token_seq.extend([
                        "<Action_Link_To_New>",
                        f"<ID_{neighbor}>",
                        f"<Type_{nodes[neighbor]['type']}>",
                        f"<Role_{role}>"
                    ])
                    token_seq.extend(self._encode_edge_params(edge_data, curr, neighbor))
                    stack.append(neighbor)
                    found_move = True
                    break

            if found_move: continue

            # 2. Link To Old
            for neighbor, edge_data, edge_id in neighbors:
                if edge_id in visited_edges: continue
                if neighbor in visited_nodes:
                    visited_edges.add(edge_id)
                    token_seq.extend([
                        "<Action_Link_To_Old>",
                        f"<ID_{neighbor}>"
                    ])
                    token_seq.extend(self._encode_edge_params(edge_data, curr, neighbor))
                    found_move = True
                    break

            if found_move: continue

            stack.pop()
            if stack:
                prev = stack[-1]
                token_seq.extend(["<Action_Jump_To>", f"<ID_{prev}>"])

        return token_seq

    def _encode_edge_params(self, edge, u, v):
        p = edge['params']
        init = edge['initial_state']
        is_forward = (edge['source'] == u)

        off_src = p['offset_source'] if is_forward else p['offset_target']
        off_tgt = p['offset_target'] if is_forward else p['offset_source']
        q_src = init['q_source'] if is_forward else init['q_target']
        q_tgt = init['q_target'] if is_forward else init['q_source']

        return [
            self.encode_value(p['a'], "len"),
            self.encode_value(p['alpha'], "ang", is_angle=True),
            self.encode_value(off_src, "off"),
            self.encode_value(off_tgt, "off"),
            self.encode_value(q_src, "state", is_angle=True),
            self.encode_value(q_tgt, "state", is_angle=True)
        ]