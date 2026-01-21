import torch
import numpy as np
import traceback

class MechanismTensorAdapter:
    def __init__(self, num_nodes=8):
        self.N = num_nodes

    def json_to_tensor(self, topology_data):
        """
        JSON -> Tensor (Final Fix)
        [Fix]: 
        1. 关节类型 (Channel 1) 由 Source Node 决定。
        2. 固定参数 (Channel 4) 由 Source Node 的 offset_source 决定。
        """
        tensor = np.zeros((5, self.N, self.N), dtype=np.float32)

        try:
            nodes = topology_data.get('nodes', {})
            connections = topology_data.get('connections', [])

            # Channel 1: Joint Types (Diagonal)
            for node_id, node_info in nodes.items():
                idx = int(node_id)
                if idx >= self.N: continue
                type_str = node_info.get('type', 'R') if isinstance(node_info, dict) else str(node_info)
                val = 1.0 if type_str.upper() == "R" else -1.0
                tensor[1, idx, idx] = val

            # Channels 0-4: Connections
            for conn in connections:
                if hasattr(conn, 'model_dump'): c = conn.model_dump()
                else: c = conn
                u, v = int(c['source']), int(c['target'])
                if u >= self.N or v >= self.N: continue

                # Channel 0: Adjacency
                tensor[0, u, v] = 1.0
                tensor[0, v, u] = 1.0

                # Channel 1: Joint Type (Source Based)
                # u->v 的类型由 u 决定
                tensor[1, u, v] = tensor[1, u, u]
                # v->u 的类型由 v 决定
                tensor[1, v, u] = tensor[1, v, v]

                # Channel 2: a (Symmetric)
                tensor[2, u, v] = c.get('a', 0.0)
                tensor[2, v, u] = c.get('a', 0.0)

                # Channel 3: alpha (Symmetric)
                tensor[3, u, v] = c.get('alpha', 0.0)
                tensor[3, v, u] = c.get('alpha', 0.0) 

                # ✨✨✨ [Core Fix]: Channel 4 (Offset) 必须与 Source 对应 ✨✨✨
                # u->v 代表从 u 变换到 v，使用 u 处的固定参数
                tensor[4, u, v] = c.get('offset_source', 0.0)
                
                # v->u 代表从 v 变换到 u，使用 v 处的固定参数
                tensor[4, v, u] = c.get('offset_target', 0.0)

            return tensor

        except Exception as e:
            print(f"❌ [TensorAdapter] Error: {e}")
            traceback.print_exc()
            return np.zeros((5, self.N, self.N))
    
    def tensor_to_json(self, tensor):
        """
        逆向转换: Tensor -> JSON
        """
        N = tensor.shape[1]
        nodes = {}
        connections = []

        for i in range(N):
            if tensor[1, i, i] != 0 or np.sum(tensor[0, i, :]) > 0:
                nodes[str(i)] = "R" if tensor[1, i, i] > 0 else "P"

        rows, cols = np.where(np.triu(tensor[0]) > 0)
        for u, v in zip(rows, cols):
            connections.append({
                "source": int(u),
                "target": int(v),
                "a": float(tensor[2, u, v]),
                "alpha": float(tensor[3, u, v]),
                "offset_source": float(tensor[4, u, v]), # 修正后的对应关系
                "offset_target": float(tensor[4, v, u])  # 修正后的对应关系
            })

        return {"nodes": nodes, "connections": connections}