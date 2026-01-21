# Agentic-MechSyn/tools/tool_registry.py

AVAILABLE_TOOLS_DEF = {
    "optimizer": [
        {
            "name": "closure_loop",
            "binding_method": "_loss_closure",  # ✨ 指定 PhysicsKernel 中的函数名
            "description": "[基础] 闭环约束。强制机构在当前几何参数下能够通过调整关节变量实现闭合。",
            "input_desc": "Tensor (5, N, N) + Topology Cycles",
            "output_desc": "Loss = Pos_Err_Rel + Rot_Err (Normalized)"
        },
        {
            "name": "mobility_dof",
            "binding_method": "_loss_mobility_dof",
            "description": "[基础] 自由度(DOF)约束 (含二阶瞬时性检查)。强制机构拥有指定数量的全局有效自由度。内置了漂移检查。",
            "input_desc": "Tensor + Target DOF (from Task)",
            "output_desc": "Loss = SVD_Entropy + Drift_Residual"
        },
        {
            "name": "path_error",
            "binding_method": "_loss_path_generation",
            "description": "[任务] 路径生成误差 (基于正运动学 FK)。计算末端执行器在 SE(3) 空间中的位姿与目标路径的偏差。",
            "input_desc": "Tensor + Target Poses",
            "output_desc": "Loss = Position_Error + Rotation_Error"
        },
        {
            "name": "twist_match",
            "binding_method": "_loss_twist_alignment",
            "description": "[任务] 螺旋运动匹配 (基于谱分析)。用于刚体导引 (2T/3R) 或瞬时运动生成。",
            "input_desc": "Tensor + Target Twists + Masks",
            "output_desc": "Loss = Spectrum[N+1] (Rank Deficiency)"
        },
        {
            "name": "bennett_ratio",
            "binding_method": "_loss_bennett_condition",
            "description": "[特殊] Bennett 几何全约束。强制满足 Bennett 机构的所有几何条件 (对边相等, d=0, 扭转角关联)。",
            "input_desc": "Link Parameters",
            "output_desc": "Loss = Symmetry_Err + Ratio_Err"
        }
    ],
    "evaluator": [
        {
            "name": "dof_check",
            "binding_method": "_check_dof_rigorous", # ✨ 指定 Evaluator 中的函数名
            "description": "[基础] 有效自由度检查 (Effective DOF)。基于 SVD 结合二阶漂移分析。已包含瞬时性验证。",
            "input_desc": "Topology Graph + Geometry",
            "output_desc": "Effective DOF (int) + Diagnosis"
        },
        {
            "name": "closure_loop",
            "binding_method": "_check_closure_rigorous",
            "description": "[基础] 闭环精度验证。检查机构在静态或运动状态下是否能精确闭合（误差 < 1mm）。",
            "input_desc": "Topology Graph + Geometry",
            "output_desc": "Pass/Fail + Max Error (mm)"
        },
        {
            "name": "path_error",
            "binding_method": "_check_path_accuracy",
            "description": "[任务] 路径跟踪精度验证。检查末端执行器是否能精确复现目标路径。",
            "input_desc": "FK Solver + Target Path",
            "output_desc": "Pass/Fail + MSE (mm)"
        },
        {
            "name": "twist_match",
            "binding_method": "_check_twist_match",
            "description": "[任务] 螺旋特征验证。检查在目标构型下，机构是否能产生指定方向的瞬时运动。",
            "input_desc": "Jacobian + Target Twist",
            "output_desc": "Pass/Fail + Residual"
        },
        {
            "name": "bennett_geometric",
            "binding_method": "_check_bennett_validity",
            "description": "[特殊] Bennett 几何精度验证。检查对边长度差和偏移量是否接近 0。",
            "input_desc": "Geometric Parameters",
            "output_desc": "Pass/Fail + Diff (mm)"
        }
    ]
}