import torch
import numpy as np
import math
# 引入统一物理内核
from tools.physics_kernel import PhysicsKernel
from tools.tool_registry import AVAILABLE_TOOLS_DEF

class MechanismEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ✨ 初始化物理内核
        self.kernel = PhysicsKernel(self.device)
        
        # ==========================================
        # ⚙️ 评估阈值配置
        # ==========================================
        self.THRESH_CLOSURE_POS = 1.0     # mm (位置闭合阈值)
        self.THRESH_CLOSURE_ROT = 0.05    # rad (姿态闭合阈值)
        self.THRESH_BENNETT_LEN = 1.0     # mm
        self.THRESH_SINGULAR_VAL = 1e-2   # SVD 秩判定阈值
        self.THRESH_DRIFT_RESIDUAL = 1e-3 # 瞬时运动漂移阈值
        self.THRESH_PATH_POS = 5.0        # mm (路径位置误差)

        # =====================================================
        # ⚖️ 动态加载评估工具 (自动化绑定)
        # =====================================================
        self.eval_tools = {}
        
        # 遍历注册表中的 evaluator 列表
        for tool_def in AVAILABLE_TOOLS_DEF.get("evaluator", []):
            tool_name = tool_def["name"]
            method_name = tool_def.get("binding_method")
            
            # 这里从 self (Evaluator 实例) 中获取方法
            # 同时也支持兼容别名逻辑 (如果需要)
            if method_name and hasattr(self, method_name):
                self.eval_tools[tool_name] = getattr(self, method_name)
            else:
                print(f"⚠️ [Evaluator] Warning: Method '{method_name}' for tool '{tool_name}' not found.")
                
        # 兼容旧别名 (防止 LLM 幻觉导致报错)
        if "dof_check" in self.eval_tools:
            self.eval_tools["mobility_dof"] = self.eval_tools["dof_check"]
        if "bennett_geometric" in self.eval_tools:
            self.eval_tools["bennett_ratio"] = self.eval_tools["bennett_geometric"]

    def generate_report(self, tensor, selected_metric_names, q_values=None, task=None):
        if isinstance(tensor, np.ndarray): tensor = torch.tensor(tensor, dtype=torch.float32)
        tensor = tensor.to(self.device)
        if q_values is None: q_values = torch.zeros((tensor.shape[1], tensor.shape[1]), device=self.device)
        elif isinstance(q_values, np.ndarray): q_values = torch.tensor(q_values, dtype=torch.float32).to(self.device)

        adj = tensor[0]
        cycles = self.kernel.find_fundamental_cycles(adj)

        report = {"final_score": 0, "details": {}}
        print(f"    -> [Evaluator] 启动检测项: {selected_metric_names}")

        pass_count = 0
        total_count = 0

        for tool_name in selected_metric_names:
            if tool_name == "mobility_dof": tool_name = "dof_check"
            if tool_name == "bennett_ratio": tool_name = "bennett_geometric"

            if tool_name in self.eval_tools:
                total_count += 1
                check_func = self.eval_tools[tool_name]
                try:
                    result = check_func(tensor, q_values, cycles, task=task)
                    report["details"][tool_name] = result
                    if result.get("status") == "pass":
                        pass_count += 1
                except Exception as e:
                    import traceback; traceback.print_exc()
                    report["details"][tool_name] = {"status": "error", "msg": str(e)}

        if total_count > 0:
            report["final_score"] = (pass_count / total_count) * 100
        
        return report

    # =========================================================================
    # 🔍 核心工具实现
    # =========================================================================

    def _check_dof_rigorous(self, tensor, q_values, cycles, task=None):
        """
        [严谨版] 自由度检查 (SVD + Drift Check)
        调用内核的 compute_effective_dof_statistics 融合了二阶分析。
        """
        if not cycles:
            return {"status": "fail", "value": 0, "msg": "No closed loops"}

        # ✨ 调用内核的新功能
        stats = self.kernel.compute_effective_dof_statistics(
            tensor, q_values, cycles,
            threshold_singular=self.THRESH_SINGULAR_VAL,
            threshold_drift=self.THRESH_DRIFT_RESIDUAL
        )

        eff_dof = stats['effective_dof']
        raw_dof = stats['raw_dof']
        idof_count = stats['idof_count']

        is_pass = (eff_dof >= 1)
        
        # 详细诊断信息
        if eff_dof >= 1:
            msg = f"Effective DoF: {eff_dof} (Continuous)"
            if idof_count > 0:
                msg += f" [Warning: {idof_count} instantaneous modes ignored]"
        elif raw_dof > 0 and eff_dof == 0:
            msg = f"Effective DoF: 0 (Found {raw_dof} Instantaneous Modes) -> Shaky/Rigid"
        else:
            msg = "Effective DoF: 0 (Overconstrained)"

        return {
            "status": "pass" if is_pass else "fail",
            "value": int(eff_dof),
            "msg": msg,
            "raw_dof": raw_dof
        }

    def _check_closure_rigorous(self, tensor, q_values, cycles, task=None):
        if not cycles: return {"status": "skip", "msg": "Open chain"}
        base_node = cycles[0][0]
        obs = self.kernel.compute_multi_path_states(tensor, q_values, base_node=base_node)
        max_pos_err, max_rot_err = 0.0, 0.0
        check_count = 0
        for node_id, obs_list in obs.items():
            if len(obs_list) < 2: continue
            ref_P, ref_z = obs_list[0]['P'], obs_list[0]['z']
            for i in range(1, len(obs_list)):
                curr_P, curr_z = obs_list[i]['P'], obs_list[i]['z']
                max_pos_err = max(max_pos_err, torch.norm(curr_P - ref_P).item())
                max_rot_err = max(max_rot_err, torch.acos(torch.clamp(torch.dot(curr_z, ref_z), -1.0, 1.0)).item())
                check_count += 1
        
        is_pass = (max_pos_err < self.THRESH_CLOSURE_POS) and (max_rot_err < self.THRESH_CLOSURE_ROT)
        msg = f"PosErr: {max_pos_err:.3f}mm, AngErr: {max_rot_err:.3f}rad"
        if check_count == 0: msg = "No loop constraints found"
        return {"status": "pass" if is_pass else "fail", "pos_error_mm": round(max_pos_err, 4), "msg": msg}

    def _check_twist_match(self, tensor, q_values, cycles, task=None):
        if not task: return {"status": "skip", "msg": "No task info"}
        loss_val = self.kernel.compute_twist_match_residual(tensor, task, cycles, q_values).item()
        is_pass = loss_val < self.THRESH_SINGULAR_VAL
        return {"status": "pass" if is_pass else "fail", "msg": f"Twist Residual: {loss_val:.2e}"}

    def _check_instantaneous_motion(self, tensor, q_values, cycles, task=None):
        """
        调用内核计算二阶漂移。
        如果是无任务的机构，它也会自动计算固有自由度的漂移。
        """
        # 注意: 即使 task 为 None，内核现在也能处理了 (Task A vs Mode B)
        loss_val = self.kernel._loss_instantaneous_check(tensor, task, cycles, q_values).item()
        
        is_pass = loss_val < self.THRESH_DRIFT_RESIDUAL
        msg = f"Drift Resid: {loss_val:.2e}"
        msg += " -> Continuous" if is_pass else " -> Instantaneous"
        return {"status": "pass" if is_pass else "fail", "msg": msg}

    def _check_path_accuracy(self, tensor, q_values, cycles, task=None):
        if not task: return {"status": "skip", "msg": "No task info"}
        loss_val = self.kernel._loss_path_generation(tensor, task, cycles, q_values).item()
        is_pass = loss_val < self.THRESH_PATH_POS
        return {"status": "pass" if is_pass else "fail", "pos_error_mm": round(loss_val, 3), "msg": f"Total Path Err: {loss_val:.2f}"}

    def _check_bennett_validity(self, tensor, q_values, cycles, task=None):
        target_cycle = None
        for c in cycles:
            if len(c) == 4: target_cycle = c; break
        if not target_cycle: return {"status": "skip", "msg": "No 4-bar loop"}
        params = []
        L = 4
        for i in range(L):
            u, v = target_cycle[i], target_cycle[(i+1)%L]
            idx_u, idx_v = (u, v) if tensor[0, u, v] > 0.5 else (v, u)
            a = tensor[2, idx_u, idx_v].item(); alpha = tensor[3, idx_u, idx_v].item(); off = tensor[4, idx_u, idx_v].item()
            params.append((abs(a), abs(alpha), off))
        diff_a1 = abs(params[0][0] - params[2][0]); diff_a2 = abs(params[1][0] - params[3][0]); max_off = max([abs(p[2]) for p in params])
        is_valid = (diff_a1 < self.THRESH_BENNETT_LEN) and (diff_a2 < self.THRESH_BENNETT_LEN) and (max_off < self.THRESH_BENNETT_LEN)
        return {"status": "pass" if is_valid else "fail", "val_a_diff": round(max(diff_a1, diff_a2), 3), "msg": f"A-Diff: {max(diff_a1, diff_a2):.2f}, Off: {max_off:.2f}"}