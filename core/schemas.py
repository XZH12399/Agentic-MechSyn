from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict


# ==============================================================================
#  Part 1: 任务解析模板 (Step 1) - 移除具体的节点ID定义
# ==============================================================================

class KinematicsSpec(BaseModel):
    dof: int = Field(..., description="目标自由度")
    space_type: Literal["planar", "spatial", "spherical"] = Field(..., description="运动空间类型")
    # [移除] ground_nodes: 具体的 ID 应由 Step 2 决定


class TargetSpec(BaseModel):
    task_type: Literal["path_generation", "motion_generation", "function_generation", "unknown"]
    description: str = Field(..., description="运动目标描述")

    # 目标螺旋 (数值目标) 仍然保留在 Step 1，因为这是用户的需求
    target_motion_twists: Optional[List[List[float]]] = Field(None, description="目标运动螺旋列表")
    target_masks: Optional[List[List[float]]] = Field(None, description="目标掩码")

    # [移除] target_path_sequence: 因为此时拓扑未定，无法给出合法序列


# ... (ConstraintsSpec, SolverSettings, TaskTemplate 保持不变，但需去掉被移除的字段) ...
class ConstraintsSpec(BaseModel):
    num_links_min: Optional[int] = Field(None, description="最少连杆数")
    num_links_max: Optional[int] = Field(None, description="最多连杆数")
    allowed_joints: Optional[List[str]] = Field(None, description="允许的关节类型")
    geometric_condition: Optional[str] = Field(None, description="特殊几何约束")


class SolverSettings(BaseModel):
    max_iters: int = Field(2000, description="优化器最大迭代次数")
    closure_weight: float = Field(1.0, description="闭环约束权重")
    exploration_noise: float = Field(0.01, description="探索噪声等级")


class TaskTemplate(BaseModel):
    user_intent_summary: str = Field(..., description="用户意图总结")
    reasoning_trace: str = Field(..., description="推断逻辑")
    kinematics: KinematicsSpec
    constraints: ConstraintsSpec
    targets: TargetSpec
    solver_settings: SolverSettings


# ==============================================================================
#  Part 2: 拓扑生成模板 (Step 2) - 新增元数据定义
# ==============================================================================

class ConnectionSpec(BaseModel):
    source: int = Field(..., description="起始节点 ID")
    target: int = Field(..., description="终止节点 ID")
    a: float = Field(..., description="连杆长度 a")
    alpha: float = Field(..., description="扭转角 alpha")
    offset_source: float = Field(..., description="源节点处的偏移量")
    offset_target: float = Field(..., description="目标节点处的偏移量")


class TopologySpec(BaseModel):
    nodes: Dict[str, Literal["R", "P"]] = Field(..., description="节点列表")
    connections: List[ConnectionSpec] = Field(..., description="连接关系列表")


# === ✨ 新增: 机构元数据 (绑定具体拓扑) ===
class MechanismMetadata(BaseModel):
    ground_nodes: List[int] = Field(..., description="被选定为机架(Ground)的节点ID列表")
    ee_node: int = Field(..., description="被选定为末端执行器(End-Effector)的节点ID")


class TopologyResponse(BaseModel):
    thought_trace: str = Field(..., description="设计思维链")
    topology: TopologySpec = Field(..., description="核心拓扑数据")

    # ✨ 新增字段
    meta: MechanismMetadata = Field(..., description="基于生成拓扑的元数据定义")


# ... (Part 3: ToolSelectionResponse 保持不变) ...
class ToolDefinition(BaseModel):
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="功能描述")
    input_desc: str = Field(..., description="输入定义")
    output_desc: str = Field(..., description="输出定义")


class ToolSelectionResponse(BaseModel):
    selected_optimization_tools: List[str] = Field(..., description="已选优化工具")
    suggested_new_optimization_tools: List[ToolDefinition] = Field(default_factory=list, description="新优化工具建议")
    selected_evaluation_tools: List[str] = Field(..., description="已选评估工具")
    suggested_new_evaluation_tools: List[ToolDefinition] = Field(default_factory=list, description="新评估工具建议")
    reasoning: str = Field(..., description="选择理由")


# 1. 定义修正动作的枚举
class ReflectionAction(str, Enum):
    KEEP_CURRENT = "keep_current"  # 成功了，无需修改
    RESELECT_ANCHORS = "reselect_anchors"  # 重新选择 Ground/EE (针对 Pain Point 1)
    REINIT_GEOMETRY = "reinit_geometry"  # 重新初始化几何参数 (针对 Pain Point 3)
    REGENERATE_TOPOLOGY = "regenerate_topology"  # 彻底重画拓扑 (针对 Pain Point 2)


# 2. 定义反思结果 Schema
class ReflectionResponse(BaseModel):
    analysis: str = Field(...,
                          description="对失败原因的深度分析 (例如：'Loss NaN 说明初始构型奇异' 或 '路径误差大说明机架位置不合理')")
    action: ReflectionAction = Field(..., description="下一步的修正策略")

    # 如果选择 RESELECT_ANCHORS，必须填写以下字段
    suggested_ground_nodes: Optional[List[int]] = Field(None, description="建议新的机架节点ID列表")
    suggested_ee_node: Optional[int] = Field(None, description="建议新的末端节点ID")

    # 如果选择 REGENERATE_TOPOLOGY，可以给出建议
    topology_suggestion: Optional[str] = Field(None, description="给拓扑生成器的改进建议 (例如：'增加连杆数')")
