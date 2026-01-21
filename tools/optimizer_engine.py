import torch
import time
import json
import numpy as np
from tools.physics_kernel import PhysicsKernel
from tools.tool_registry import AVAILABLE_TOOLS_DEF

class MechanismOptimizer:
    def __init__(self, physics_config):
        self.cfg = physics_config
        self.lr = physics_config.learning_rate
        self.epochs = physics_config.max_iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.kernel = PhysicsKernel(self.device, self.cfg)

        # =====================================================
        # 🔧 动态加载优化工具 (自动化绑定)
        # =====================================================
        self.loss_tools = {}
        
        # 遍历注册表中的 optimizer 列表
        for tool_def in AVAILABLE_TOOLS_DEF.get("optimizer", []):
            tool_name = tool_def["name"]
            method_name = tool_def.get("binding_method")
            
            if method_name and hasattr(self.kernel, method_name):
                self.loss_tools[tool_name] = getattr(self.kernel, method_name)
            else:
                print(f"⚠️ [Optimizer] Warning: Method '{method_name}' for tool '{tool_name}' not found in PhysicsKernel.")

    def run_optimization(self, tensor_data, task_dict, selected_tools, new_tools=None, initial_q=None):
        """
        执行几何参数优化
        """
        print(f"    -> [Optimizer] 启动 PyTorch 协同优化引擎 (Device: {self.device})")
        
        # 1. 数据准备
        # 将 numpy 转为可优化的 Tensor
        if isinstance(tensor_data, np.ndarray):
            tensor = torch.tensor(tensor_data, dtype=torch.float32, device=self.device)
        else:
            tensor = tensor_data.to(self.device)
            
        tensor.requires_grad = True

        # 2. 准备 Q (关节状态)
        if initial_q is not None:
            if isinstance(initial_q, np.ndarray):
                q_opt = torch.tensor(initial_q, dtype=torch.float32, device=self.device)
            else:
                q_opt = initial_q.to(self.device)
            print("        ✅ [Init] 使用上游提供的初始位型 (Initial Guess)。")
        else:
            q_opt = torch.zeros((tensor.shape[1], tensor.shape[2]), dtype=torch.float32, device=self.device)
        
        # 检测并打破全0对称性
        if torch.all(q_opt == 0):
            print("        ⚠️ [Init] 检测到初始 Q 全为 0，添加微小扰动以打破对称性。")
            q_opt += torch.randn_like(q_opt) * 0.01

        q_opt.requires_grad = True

        # 3. 拓扑分析 (用于 Loss 计算)
        adj = tensor[0].detach()
        cycles = self.kernel.find_fundamental_cycles(adj)
        print(f"    -> [Optimizer] 拓扑分析: 发现 {len(cycles)} 个基本闭环")

        # 4. 优化器配置
        optimizer = torch.optim.Adam([tensor, q_opt], lr=self.lr)
        
        # 打印目标运动螺旋 (Debug用)
        targets = task_dict.get('targets', {})
        target_twists = targets.get('target_motion_twists', [])
        if target_twists:
            desc = targets.get('description', 'Unknown')
            print(f"    -> [Optimizer] 🎯 目标运动模式 (Target Motion):")
            print(f"        - Description: {desc}")
            for i, tw in enumerate(target_twists):
                # 格式化打印，方便检查是否还是 [0,0,1...]
                fmt_tw = ", ".join([f"{x:.4f}" for x in tw])
                print(f"        - Mode {i+1} Expectation: [{fmt_tw}]")

        print(f"Optimizer Config: Device={self.device}, LR={self.lr}, Epochs={self.epochs}")

        # 5. 优化循环
        history = []
        
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            
            total_loss = torch.tensor(0.0, device=self.device)
            loss_components = {}

            # 计算各项 Loss
            for tool_name in selected_tools:
                if tool_name in self.loss_tools:
                    loss_func = self.loss_tools[tool_name]
                    try:
                        val = loss_func(tensor, task_dict, cycles, q_opt)
                        total_loss += val
                        loss_components[tool_name] = val.item()
                    except Exception as e:
                        print(f"Error in {tool_name}: {e}")

            # 反向传播
            total_loss.backward()
            
            # ✨✨✨ 核心修正: 冻结拓扑梯度 ✨✨✨
            self._freeze_topology_gradients(tensor)

            optimizer.step()

            # 记录与打印
            if epoch == 1 or epoch % 50 == 0:
                log_str = f"        Epoch {epoch}: Loss={total_loss.item():.4f} (LR={self.lr:.1e})"
                for k, v in loss_components.items():
                    log_str += f" | {k}={v:.4f}"
                print(log_str)
                history.append(log_str)

        return tensor.detach(), q_opt.detach(), "\n".join(history)

    def _freeze_topology_gradients(self, tensor):
        """
        🔒 冻结拓扑相关的梯度，防止优化器修改机构结构。
        
        Tensor Channels:
        [0]: Adjacency (连接关系) -> 必须冻结
        [1]: Joint Type (关节类型) -> 必须冻结 (防止 P 变 R)
        [2]: Link Length (a)      -> 可优化
        [3]: Twist Angle (alpha)  -> 可优化
        [4]: Offset (d/theta)     -> 可优化
        """
        if tensor.grad is not None:
            # 将 Channel 0 (Adj) 和 Channel 1 (Type) 的梯度强制设为 0
            tensor.grad[0] = 0.0
            tensor.grad[1] = 0.0