# Agentic-MechSyn/tools/tool_registry.py

# ==============================================================================
#  工具库注册表 (Tool Library Registry)
#  定义了系统中所有可用的优化器(Optimizer)和评估器(Evaluator)工具
#  这些定义将作为上下文提供给 LLM，帮助其进行工具选择
# ==============================================================================

AVAILABLE_TOOLS_DEF = {
    "optimizer": [
        {
            "name": "closure_loop",
            "description": "[基础] 闭环约束。强制机构在当前几何参数下能够通过调整关节变量实现闭合。这是所有闭链机构必须选择的基础约束。",
            "input_desc": "Tensor (5, N, N) + Topology Cycles",
            "output_desc": "Loss = Pos_Err_Rel + Rot_Err (Normalized)"
        },
        {
            "name": "mobility_dof",
            "description": "[基础] 自由度(DOF)约束 (基于谱分析)。强制机构拥有指定数量的全局自由度 (通常为1)。防止机构退化为刚性桁架 (0-DOF) 或产生多余自由度。",
            "input_desc": "Tensor + Target DOF (from Task)",
            "output_desc": "Loss = Sum(Unexpected_NonZero_Eigenvalues)"
        },
        {
            "name": "path_error",
            "description": "[任务] 路径生成误差 (基于正运动学 FK)。计算末端执行器在 SE(3) 空间中的位姿(位置+姿态)与目标路径点的欧氏距离。",
            "input_desc": "Tensor + Target Poses (Positions & Rotations)",
            "output_desc": "Loss = Position_Error + Rotation_Error"
        },
        {
            "name": "twist_match",
            "description": "[任务] 螺旋运动匹配 (基于谱分析)。用于刚体导引 (2T/3R) 或瞬时运动生成。通过分析锚点系统矩阵的奇异值，判断机构是否具备生成目标瞬时螺旋(Twist)的内禀能力。",
            "input_desc": "Tensor + Target Twists + Masks",
            "output_desc": "Loss = Spectrum[N+1] (Null Space Feasibility)"
        },
        {
            "name": "instantaneous_check",
            "description": "[任务] 运动可持续性检查 (IDOF Detection)。判断当前设计的机构是能够产生连续的运动路径，还是仅仅是一个瞬时机构 (Instantaneous Mechanism)。通过计算二阶漂移(Drift)在雅可比空间中的投影残差来惩罚瞬时运动。",
            "input_desc": "Tensor + Path + Target Twists (Extended K Matrix)",
            "output_desc": "Loss = Projection Residual of 2nd Order Drift"
        },
        {
            "name": "bennett_ratio",
            "description": "[特殊] Bennett 几何全约束。强制满足 Bennett 机构的所有几何条件：对边长度/扭转角对称、a/sin(alpha) 比例恒定、以及平面副约束(d=0)。仅在设计 Bennett 类机构时选择。",
            "input_desc": "Link Parameters (a, alpha, offset)",
            "output_desc": "Loss = Symmetry_Err + Ratio_Err + Offset_Err"
        }
    ],
    "evaluator": [
        {
            "name": "dof_check",
            "description": "[基础] 自由度检查 (Gruebler/Kutzbach)。计算机构的理论自由度。",
            "input_desc": "Topology Graph",
            "output_desc": "DOF Value (int)"
        },
        {
            "name": "dead_point",
            "description": "[基础] 死点与传动角检查。基于雅可比矩阵的条件数分析，评估机构的运动顺畅性。",
            "input_desc": "Jacobian Matrix",
            "output_desc": "Min Transmission Angle (deg)"
        },
        {
            "name": "mode_consistency",
            "description": "[进阶] 运动模式一致性检查。计算二阶加速度，防止机构在运动中发生构型分叉或漂移。",
            "input_desc": "Hessian of Anchor System",
            "output_desc": "Consistency Score (0-1)"
        },
        {
            "name": "bennett_geometric",
            "description": "[特殊] Bennett 几何精度验证。输出几何参数与理论 Bennett 条件的偏差值。",
            "input_desc": "Geometric Parameters",
            "output_desc": "Geometric Error (float)"
        }
    ]
}