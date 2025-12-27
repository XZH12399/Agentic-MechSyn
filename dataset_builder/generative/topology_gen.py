import networkx as nx
import random
import matplotlib.pyplot as plt


class TopologyGenerator:
    def __init__(self, min_nodes=4, max_nodes=9, loop_probs=None, bridge_prob=0.3):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.bridge_prob = bridge_prob

        # 归一化概率
        if loop_probs is None:
            raw_probs = {3: 0.3, 4: 0.7}
        else:
            raw_probs = loop_probs

        # 确保概率和为1
        total = sum(raw_probs.values())
        self.loop_probs = {k: v / total for k, v in raw_probs.items()}

    def _find_valid_bridge_option(self, G, remaining_quota):
        """
        寻找一个合法的架桥方案，使得形成的新环路符合 loop_probs 的要求。
        返回: (u, v, num_new_nodes) 或 None
        """
        # 1. 获取所有可桥接的点对
        nodes = list(G.nodes())
        candidates = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if not G.has_edge(u, v):
                    candidates.append((u, v))

        if not candidates:
            return None

        # 随机打乱，避免总是选前面的点
        random.shuffle(candidates)

        # 2. 遍历点对，检查能否形成目标大小的环
        # 我们按照 loop_probs 的权重随机尝试目标环大小
        target_sizes = list(self.loop_probs.keys())
        weights = list(self.loop_probs.values())

        # 尝试几次随机的目标大小
        for _ in range(5):
            target_size = random.choices(target_sizes, weights=weights, k=1)[0]

            for u, v in candidates:
                # 计算现有距离
                try:
                    dist = nx.shortest_path_length(G, u, v)
                except nx.NetworkXNoPath:
                    # 如果图还不连通（理论上不会，因为是构造法），忽略
                    continue

                # 计算需要的中间节点数
                # 环大小 = 现有距离 + (新节点数 + 1)
                # => 新节点数 = 环大小 - 现有距离 - 1
                needed_nodes = target_size - dist - 1

                # 检查条件:
                # 1. 节点数非负 (如果 needed_nodes < 0，说明现有距离太远，无法形成这么小的环)
                # 2. 不超过剩余配额
                if needed_nodes >= 0 and needed_nodes <= remaining_quota:
                    return u, v, needed_nodes

        return None

    def _pick_new_nodes_for_expansion(self, remaining_quota):
        """边扩展模式下的节点选择 (逻辑不变)"""
        candidates = []
        weights = []
        for size, weight in self.loop_probs.items():
            needed = size - 2
            if needed <= remaining_quota and needed > 0:
                candidates.append(needed)
                weights.append(weight)

        if not candidates:
            return remaining_quota if remaining_quota > 0 else 0
        return random.choices(candidates, weights=weights, k=1)[0]

    def generate(self):
        while True:
            G = nx.Graph()

            # === Step 1: 初始化基础环 ===
            valid_base_sizes = [s for s in self.loop_probs.keys() if s <= self.max_nodes]
            if not valid_base_sizes: valid_base_sizes = [3, 4]
            base_loop_len = random.choices(valid_base_sizes,
                                           weights=[self.loop_probs.get(s, 0) for s in valid_base_sizes], k=1)[0]

            nx.add_cycle(G, range(base_loop_len))
            current_node_id = base_loop_len

            # === Step 2: 混合扩展 ===
            max_attempts = 100
            attempts = 0
            target_nodes = random.randint(self.min_nodes, self.max_nodes)

            while len(G.nodes) < target_nodes and attempts < max_attempts:
                attempts += 1
                remaining_quota = target_nodes - len(G.nodes)
                if remaining_quota <= 0: break

                # 🎲 决策: 桥接 vs 扩展
                allow_bridge = (len(G.nodes) > base_loop_len) and (random.random() < self.bridge_prob)

                bridge_plan = None
                if allow_bridge:
                    # 尝试寻找符合环路大小要求的桥接方案
                    bridge_plan = self._find_valid_bridge_option(G, remaining_quota)

                if bridge_plan:
                    # --- 执行桥接 ---
                    u, v, num_new_nodes = bridge_plan
                    # print(f"   ⚡ Bridging {u}-{v} (New nodes: {num_new_nodes})")

                    new_nodes = []
                    for _ in range(num_new_nodes):
                        new_nodes.append(current_node_id)
                        current_node_id += 1
                    path_nodes = [u] + new_nodes + [v]
                    nx.add_path(G, path_nodes)

                else:
                    # --- 执行边扩展 ---
                    # 这里的逻辑其实也应该像桥接一样反推，但因为是在"边"上扩展，
                    # 现有距离永远是 1，所以 新节点数 = 目标大小 - 1 - 1 = 目标大小 - 2
                    # 这和 _pick_new_nodes_for_expansion 的逻辑是一致的，不需要改
                    existing_edges = list(G.edges())
                    u, v = random.choice(existing_edges)

                    num_new_nodes = self._pick_new_nodes_for_expansion(remaining_quota)
                    if num_new_nodes == 0: break  # 配额不够做最小的扩展

                    new_nodes = []
                    for _ in range(num_new_nodes):
                        new_nodes.append(current_node_id)
                        current_node_id += 1
                    path_nodes = [u] + new_nodes + [v]
                    nx.add_path(G, path_nodes)

            # === Step 3: 校验 ===
            if len(G.nodes) < self.min_nodes: continue
            if not nx.is_connected(G): continue

            try:
                cycles = nx.cycle_basis(G)
                if len(cycles) >= 1:
                    return G, cycles
            except:
                continue

    def plot_graph(self, G, cycles=None):
        plt.figure(figsize=(8, 6))
        pos = nx.kamada_kawai_layout(G)

        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='white', edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.3)

        if cycles:
            colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231']
            print(f"   🎨 Cycles sizes: {[len(c) for c in cycles]}")

            for i, cycle in enumerate(cycles):
                cycle_edges = []
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    cycle_edges.append(tuple(sorted((u, v))))

                color = colors[i % len(colors)]
                nx.draw_networkx_edges(G, pos, edgelist=cycle_edges,
                                       edge_color=color, width=3, alpha=0.5,
                                       label=f"Loop-{len(cycle)}")

        plt.legend()
        plt.title(f"Checked Topology (Bridge Prob={self.bridge_prob})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 测试：只允许 3 和 4 的环路
    gen = TopologyGenerator(
        min_nodes=6,
        max_nodes=12,
        loop_probs={3: 0.1, 4: 0.9},
        bridge_prob=0.5
    )

    for i in range(3):
        print(f"\n--- Sample {i + 1} ---")
        G, cycles = gen.generate()
        # 验证一下生成的环路大小
        sizes = [len(c) for c in cycles]
        print(f"Cycle Sizes: {sizes}")
        gen.plot_graph(G, cycles)