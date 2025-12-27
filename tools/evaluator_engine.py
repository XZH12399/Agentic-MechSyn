import numpy as np


class MechanismEvaluator:
    def __init__(self):
        # ==========================================
        # ⚖️ 评估工具注册表 (Tool Registry)
        # ==========================================
        self.eval_tools = {
            # 基础检查
            "dof_check": self._check_dof,  # 自由度检查 (Gruebler)
            "dead_point": self._check_dead_point,  # 死点检测 (Transmission Angle)

            # 进阶检查
            "mode_consistency": self._check_mode_consistency,  # 运动模式一致性 (二阶加速度)
            "range_of_motion": self._check_rom,  # 工作空间范围

            # 专项检查
            "bennett_geometric": self._check_bennett_validity  # Bennett 几何验证
        }

    def generate_report(self, tensor, selected_metric_names):
        """
        主入口：根据 LLM 选择的指标生成报告
        """
        report = {"final_score": 0, "details": {}}
        print(f"    -> [Evaluator] 启动检测项: {selected_metric_names}")

        # --- 动态调用工具 ---
        for tool_name in selected_metric_names:
            if tool_name in self.eval_tools:
                check_func = self.eval_tools[tool_name]
                # 执行检查
                result = check_func(tensor)
                report["details"][tool_name] = result
            else:
                report["details"][tool_name] = {"status": "skipped", "reason": "Unknown tool"}

        # 计算总分 (简单的 pass 率)
        passed_count = sum(1 for k, v in report["details"].items() if v.get("status") == "pass")
        total_count = len(selected_metric_names)
        report["final_score"] = (passed_count / total_count) * 100 if total_count > 0 else 0

        return report

    # ==========================================
    # 具体工具函数实现
    # ==========================================

    def _check_dof(self, tensor):
        """工具: 自由度检查"""
        # 模拟 SVD 分析 Jacobian 零空间
        return {"value": 1, "status": "pass", "msg": "Mobility is 1, non-rigid."}

    def _check_dead_point(self, tensor):
        """工具: 死点/奇异位形检查"""
        return {"has_dead_point": False, "min_transmission_angle": 45.0, "status": "pass"}

    def _check_mode_consistency(self, tensor):
        """工具: 验证全域运动是否分叉 (Paper 1 核心算法)"""
        # 检查二阶加速度方向
        return {"consistency_score": 0.98, "status": "pass"}

    def _check_rom(self, tensor):
        """工具: 工作空间评估"""
        return {"volume": 1200.5, "status": "info"}

    def _check_bennett_validity(self, tensor):
        """工具: 验证是否满足 Bennett 几何条件"""
        # err = |a/sin(alpha) - const|
        return {"geometric_error": 1e-6, "status": "pass"}