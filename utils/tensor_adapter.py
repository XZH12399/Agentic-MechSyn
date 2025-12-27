import numpy as np
import json
import traceback


class MechanismTensorAdapter:
    def __init__(self, num_nodes):
        self.N = num_nodes
        # === 维度定义 (5, N, N) ===
        # Ch0: Exist (0/1)
        # Ch1: Type (R=1, P=-1) - 行广播
        # Ch2: a (Length)
        # Ch3: alpha (Twist)
        # Ch4: Offset (Absolute Coordinate at Source)
        self.tensor = np.zeros((5, num_nodes, num_nodes), dtype=np.float32)

    def json_to_tensor(self, data_input):
        """
        将 JSON 数据 (Graph结构) 转换为 物理张量 (Tensor)
        兼容:
        1. Old Schema: nodes={"0":"R"}, connections=[{source, target, a, ...}]
        2. New Schema: nodes=[{id:0, type:"R"}], edges=[{source, target, params:{a...}}]
        """
        # 每次调用前先重置张量，防止叠加
        self.tensor.fill(0)

        try:
            if isinstance(data_input, str):
                data = json.loads(data_input)
            else:
                data = data_input

            # 兼容旧版: 如果有 topology 键，取其内部
            if "topology" in data:
                data = data["topology"]

            # ==========================================
            # 1. 解析节点 (Nodes) -> Ch1: Type
            # ==========================================
            nodes = data.get("nodes", {})

            if isinstance(nodes, list):
                # [Case A] New Schema: List of dicts
                for node in nodes:
                    i = int(node['id'])
                    if i >= self.N: continue

                    type_str = node.get('type', 'R')
                    val = 1.0 if type_str.upper() == "R" else -1.0

                    # 广播到整行：表示从节点 i 出发的关节类型
                    self.tensor[1, i, :] = val

            elif isinstance(nodes, dict):
                # [Case B] Old Schema: Dict {id: type}
                for node_idx, type_str in nodes.items():
                    i = int(node_idx)
                    if i >= self.N: continue

                    val = 1.0 if type_str.upper() == "R" else -1.0
                    self.tensor[1, i, :] = val

            # ==========================================
            # 2. 解析连接 (Edges/Connections) -> Ch0,2,3,4
            # ==========================================
            # 优先找 'edges' (新版)，找不到再找 'connections' (旧版)
            connections = data.get("edges")
            if connections is None:
                connections = data.get("connections", [])

            for conn in connections:
                u = int(conn["source"])
                v = int(conn["target"])

                if u >= self.N or v >= self.N: continue

                # Ch0: Exist (对称)
                self.tensor[0, u, v] = 1.0
                self.tensor[0, v, u] = 1.0

                # 提取参数源：新版在 'params' 中，旧版直接在 conn 中
                params = conn.get("params", conn)

                # Ch2: Length a (对称)
                if "a" in params:
                    val = float(params["a"])
                    self.tensor[2, u, v] = val
                    self.tensor[2, v, u] = val

                # Ch3: Twist alpha (对称)
                if "alpha" in params:
                    val = float(params["alpha"])
                    self.tensor[3, u, v] = val
                    self.tensor[3, v, u] = val

                # Ch4: Offset (非对称)
                # 新版: Explicit source/target offsets
                if "offset_source" in params and "offset_target" in params:
                    self.tensor[4, u, v] = float(params["offset_source"])
                    self.tensor[4, v, u] = float(params["offset_target"])

                # 旧版: Single offset (通常假设是对称或仅指源)
                elif "offset" in params:
                    val = float(params["offset"])
                    self.tensor[4, u, v] = val
                    # self.tensor[4, v, u] = val # 根据需要决定是否对称

            return self.tensor

        except Exception as e:
            print(f"❌ [TensorAdapter] Error parsing JSON: {e}")
            traceback.print_exc()
            return self.tensor