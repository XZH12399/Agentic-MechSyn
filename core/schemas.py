from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

# ==============================================================================
#  Part 1: 任务解析模板 (Step 1)
# ==============================================================================

class KinematicsSpec(BaseModel):
    dof: int = Field(..., description="机构的目标自由度 (Degrees of Freedom). e.g. Bennett=1, Spatial=6")
    space_type: Literal["planar", "spatial", "spherical"] = Field(..., description="运动空间类型")

class TargetSpec(BaseModel):
    task_type: Literal["path_generation", "motion_generation", "function_generation", "unknown"]
    description: str = Field(..., description="运动目标描述")
    
    # 保留这些字段，因为后续优化器(Optimizer)仍需要它们作为 Loss 的参考
    target_motion_twists: Optional[List[List[float]]] = Field(None, description="目标运动螺旋列表 [[w,v],...]")
    target_masks: Optional[List[List[float]]] = Field(None, description="目标掩码 (1=关注, 0=忽略)")

class ConstraintsSpec(BaseModel):
    # ✨ Num_Loops 在这里，代表“结构形式”
    num_loops: int = Field(1, description="机构的独立闭环数量 (Fundamental Loops). e.g. Bennett=1, Delta=2+")
    
    num_links_min: Optional[int] = Field(None, description="最少连杆数")
    num_links_max: Optional[int] = Field(None, description="最多连杆数")
    special_constraints: Optional[str] = Field(None, description="特殊机构约束 (e.g. 'bennett_ratio')")

class SolverSettings(BaseModel):
    # 保留优化器配置，DeepSeek 可根据任务难度调整参数
    max_iters: int = Field(2000, description="优化器最大迭代次数")
    closure_weight: float = Field(1.0, description="闭环约束权重")
    exploration_noise: float = Field(0.01, description="探索噪声等级")

class TaskTemplate(BaseModel):
    user_intent_summary: str = Field(..., description="用户意图总结")
    reasoning_trace: str = Field(..., description="推断逻辑: 为什么选择这个DoF和Loop数？")
    vla_instruction: str = Field(..., description="VLA指令 (备用，实际会由 Agent 重组)")
    kinematics: KinematicsSpec
    constraints: ConstraintsSpec
    targets: TargetSpec 
    solver_settings: SolverSettings

# ==============================================================================
#  Part 2: 拓扑响应模板 (Step 2)
# ==============================================================================

class ConnectionSpec(BaseModel):
    source: int = Field(..., description="起始节点 ID")
    target: int = Field(..., description="终止节点 ID")
    a: float = Field(..., description="连杆长度 a")
    alpha: float = Field(..., description="扭转角 alpha")
    offset_source: float = Field(..., description="源节点处的偏移量")
    offset_target: float = Field(..., description="目标节点处的偏移量")
    theta_source: float = Field(0.0, description="源节点处的初始角度/状态")
    theta_target: float = Field(0.0, description="目标节点处的初始角度/状态")

# VLA 的 raw output 转换后的标准结构
class TopologySpec(BaseModel):
    nodes: Dict[str, Dict[str, str]] = Field(..., description="节点列表 {'0': {'type': 'R', 'role': 'ground'}, ...}")
    connections: List[ConnectionSpec] = Field(..., description="连接关系列表")

class MechanismMetadata(BaseModel):
    ground_nodes: List[int] = Field(..., description="机架节点 ID")
    ee_node: int = Field(..., description="末端执行器节点 ID")

class TopologyResponse(BaseModel):
    thought_trace: str = Field(..., description="生成逻辑说明")
    
    # ✨ 核心新增：保留 VLA 的原始 Token 输出，便于 Debug 和反思
    raw_tokens: str = Field(..., description="VLA 模型输出的原始 Token 流 (<Action_...>)")
    
    topology: TopologySpec = Field(..., description="解析后的拓扑结构")
    meta: MechanismMetadata = Field(..., description="元数据")

# ==============================================================================
#  Part 2.5: 语义修正模板 (Step 2.5 - New!)
# ==============================================================================

class ParameterOverride(BaseModel):
    target_type: Literal["node", "connection"] = Field(..., description="修改对象类型")
    target_id: str = Field(..., description="对象的ID (Node ID 或 'src_tgt')")
    param_name: str = Field(..., description="要修改的参数名 (type, role, a, alpha, offset, theta)")
    value: str = Field(..., description="新值 (可以是具体数值，也可以是表达式如 'same_as_link_1')")

class TopologyCorrectionResponse(BaseModel):
    requires_correction: bool = Field(..., description="是否需要修正 VLA 的输出")
    reasoning: str = Field(..., description="修正的理由 (例如: 'Bennett机构需要4个R副且对边相等')")
    corrections: List[ParameterOverride] = Field(default_factory=list, description="具体的修改操作列表")

# ==============================================================================
#  Part 3: 工具选择与反思 (Step 3 & Reflect)
# ==============================================================================

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

class ReflectionAction(str, Enum):
    KEEP_CURRENT = "keep_current"
    RESELECT_ANCHORS = "reselect_anchors"       # 针对路径问题，重选机架/末端
    REINIT_GEOMETRY = "reinit_geometry"         # 针对 VLA 参数初值不理想
    REGENERATE_TOPOLOGY = "regenerate_topology" # 针对 VLA 拓扑生成错误 (如断开、DoF不对)

class ReflectionResponse(BaseModel):
    analysis: str = Field(..., description="失败原因分析")
    action: ReflectionAction = Field(..., description="修正策略")
    
    # 用于指导下一步的修正
    refinement_instruction: Optional[str] = Field(None, description="给 VLA 或 优化器的修正指令")