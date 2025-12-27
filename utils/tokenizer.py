import numpy as np


class MechanismTokenizer:
    def __init__(self, len_bins=100, ang_bins=72, off_bins=100):
        self.bins = {
            "len": len_bins, "ang": ang_bins, "off": off_bins, "state": ang_bins
        }

        # === 1. 构建词表 ===
        self.vocab = [
            "<PAD>", "<SOS>", "<EOS>",
            # 动作指令
            "<Action_New_Node>", "<Action_Link_To_New>",
            "<Action_Link_To_Old>", "<Action_Jump_To>",
            # 属性指令
            "<Type_R>", "<Type_P>",
            # ✨ 新增：角色指令 (互斥)
            "<Role_Ground>", "<Role_EE>", "<Role_Normal>",
            # 状态特殊值
            "<State_Free>"
        ]

        # 动态生成的 ID 和 数值 Token 保持不变...
        self.vocab += [f"<ID_{i}>" for i in range(32)]
        self.vocab += [f"<Len_{i}>" for i in range(self.bins["len"])]
        self.vocab += [f"<Twist_{i}>" for i in range(self.bins["ang"])]
        self.vocab += [f"<Off_{i}>" for i in range(self.bins["off"])]
        self.vocab += [f"<State_{i}>" for i in range(self.bins["state"])]

        # === 全局属性 Token (放在序列开头) ===
        self.vocab += [
            "<Space_Planar>", "<Space_Spatial>",  # 空间属性

            # 机构类型 (专家知识)
            "<Type_General_Parallel>",
            "<Type_Delta>",
            "<Type_Bennett>",
            "<Type_Bricard>",
            "<Type_Origami>",
            "<Type_Sarrus>"
        ]

        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for i, t in enumerate(self.vocab)}

    def decode(self, token_sequence):
        """
        解析 Token 序列 -> 结构化拓扑数据
        """
        nodes = {}  # {id: {type, role}}
        connections = []
        ground_nodes = []
        ee_node = None

        cursor_id = -1
        iterator = iter(token_sequence)

        try:
            while True:
                token = next(iterator)
                if token == "<EOS>": break
                if token == "<SOS>" or token == "<PAD>": continue

                # === 模式 A: 建立新节点 ===
                if token == "<Action_New_Node>":
                    node_id = self._parse_id(next(iterator))
                    node_type = self._parse_type(next(iterator))
                    # ✨ 解析角色
                    node_role = self._parse_role(next(iterator))

                    self._register_node(nodes, node_id, node_type, node_role)
                    cursor_id = node_id

                # === 模式 B: 连接到新节点 ===
                elif token == "<Action_Link_To_New>":
                    if cursor_id == -1: raise ValueError("No cursor!")

                    new_id = self._parse_id(next(iterator))
                    node_type = self._parse_type(next(iterator))
                    # ✨ 解析角色
                    node_role = self._parse_role(next(iterator))

                    self._register_node(nodes, new_id, node_type, node_role)

                    edge_params = self._parse_edge_params(iterator)
                    connections.append({
                        "source": cursor_id, "target": new_id, **edge_params
                    })
                    cursor_id = new_id

                # === 模式 C: 连接到老节点 ===
                elif token == "<Action_Link_To_Old>":
                    if cursor_id == -1: raise ValueError("No cursor!")

                    target_id = self._parse_id(next(iterator))
                    # 注意：连回老节点时，不需要重新定义角色，因为它已经存在了

                    edge_params = self._parse_edge_params(iterator)
                    connections.append({
                        "source": cursor_id, "target": target_id, **edge_params
                    })

                # === 模式 D: 跳转 ===
                elif token == "<Action_Jump_To>":
                    cursor_id = self._parse_id(next(iterator))

        except StopIteration:
            pass

        # === 后处理：提取 Ground 和 EE ===
        # 这一步非常关键，把 nodes 里的 role 属性提取到 meta 中
        for nid, props in nodes.items():
            if props['role'] == 'Ground':
                ground_nodes.append(nid)
            elif props['role'] == 'EE':
                # 如果有多个 EE，目前的架构只支持一个，取最后一个定义的
                ee_node = nid

        return {
            "topology": {
                "nodes": {str(k): v['type'] for k, v in nodes.items()},
                "connections": connections
            },
            "meta": {
                "ground_nodes": ground_nodes,
                "ee_node": ee_node if ee_node is not None else (len(nodes) - 1),
                # 初始状态由 connections 里的 state 字段携带，不需要单独在 meta 里
            }
        }

    # === 辅助解析函数 ===
    def _register_node(self, nodes, nid, ntype, nrole):
        nodes[nid] = {"type": ntype, "role": nrole}

    def _parse_role(self, token):
        if token == "<Role_Ground>": return "Ground"
        if token == "<Role_EE>": return "EE"
        return "Normal"

    def _parse_edge_params(self, iterator):
        """解析固定的边属性序列: a, alpha, off_src, off_tgt, q_src, q_tgt"""
        return {
            "a": self._val(next(iterator), "len"),
            "alpha": self._val(next(iterator), "ang", is_angle=True),
            "offset_src": self._val(next(iterator), "off"),
            "offset_tgt": self._val(next(iterator), "off"),
            "state_src": self._val(next(iterator), "state", is_angle=True, allow_free=True),
            "state_tgt": self._val(next(iterator), "state", is_angle=True, allow_free=True)
        }

    def _parse_id(self, token):
        return int(token.split('_')[1])

    def _parse_type(self, token):
        return "R" if "Type_R" in token else "P"

    def _parse_prop(self, token):
        return True if "Prop_Ground" in token else False

    def _val(self, token, bin_type, is_angle=False, allow_free=False):
        """将 Token 转为数值"""
        if allow_free and "Free" in token:
            return None  # 对应 Python None, 优化器处理为随机

        try:
            # 格式如 <Len_50> -> 50
            idx = int(token.split('_')[-1].replace(">", ""))
            max_bin = self.bins[bin_type]

            # 归一化值 [0, 1]
            norm_val = idx / max_bin

            if is_angle:
                return norm_val * 2 * np.pi  # 映射到 0~2pi
            return norm_val  # 长度/偏置保持归一化，由 Adapter 缩放

        except:
            return 0.0  # 容错

    def _format_output(self, nodes, connections):
        """转换为标准输出格式"""
        ground_nodes = [k for k, v in nodes.items() if v['ground']]
        # 简单策略：最后一个被操作的节点作为 EE，或者需要专门的 Token 定义 EE
        # 这里为了简化，暂不特定 EE，留给后续逻辑处理

        return {
            "topology": {
                "nodes": {str(k): v['type'] for k, v in nodes.items()},
                "connections": connections
            },
            "meta": {
                "ground_nodes": ground_nodes,
                # 初始状态提取：将 connection 里的 state 提取出来
                # 注意：这里 state 属于 connection (u->v)，需要在 TensorAdapter 里处理
            }
        }