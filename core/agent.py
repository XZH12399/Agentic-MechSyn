import json
import numpy as np
import traceback
import networkx as nx  # ✨ 新增: 用于拓扑路径计算
from openai import OpenAI

# === 1. 导入 Prompt 模板 ===
from .prompt_templates import (
    TOPOLOGY_GEN_SYSTEM_PROMPT,
    TOOL_SELECTION_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,  # ✨ 新增
    STRUCTURED_OUTPUT_SUFFIX
)

# === 2. 导入 Schema ===
from .schemas import (
    TaskTemplate,
    TopologyResponse,
    ToolSelectionResponse,
    ReflectionResponse,  # ✨ 新增
    ReflectionAction  # ✨ 新增
)

# === 3. 导入核心逻辑模块 ===
from .task_parser import TaskParser
from .memory import ExperienceManager

# === 4. 导入工具模块 ===
from tools.tool_registry import AVAILABLE_TOOLS_DEF
from utils.tensor_adapter import MechanismTensorAdapter
from tools.optimizer_engine import MechanismOptimizer
from tools.evaluator_engine import MechanismEvaluator


class MechanismAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.model = cfg.model_name

        self.parser = TaskParser(self.client, self.model)
        self.memory = ExperienceManager()
        self.optimizer = MechanismOptimizer(cfg.physics)
        self.evaluator = MechanismEvaluator()

        # ✨ 全局轨迹记录器
        self.trace_log = []

    def _record(self, tag, content):
        """辅助函数: 记录轨迹到 trace_log"""
        entry = f"\n[{tag}]\n{content}\n" + "-" * 40
        self.trace_log.append(entry)

    def run_pipeline(self, user_input):
        # 清空上一轮日志
        self.trace_log = []
        print(f"\n🚀 [Start] 启动机构综合流水线 (带自我反思循环)...")
        self._record("User Input", user_input)

        # 1. 任务解析 (只做一次)
        task_template = self._step1_parse_task(user_input)
        self._record("Step 1 Task Template", json.dumps(task_template, indent=2, ensure_ascii=False))

        # 1.5 经验检索
        exp_context = self._step1_5_retrieve_experience(task_template)

        # === 进入迭代循环 ===
        max_retries = self.cfg.agent.get('max_turns', 5)
        current_topology_response = None  # 保存当前的拓扑对象
        tensor_data = None

        # 初始生成 (Step 2)
        current_topology_response, tensor_data = self._step2_generate_topology(task_template, exp_context)

        best_report = {"final_score": -1}
        # best_design = None

        for attempt in range(max_retries):
            print(f"\n🔄 [Iteration {attempt + 1}/{max_retries}] 开始新一轮尝试...")

            # 3. 工具选择 (Step 3)
            # 因为拓扑可能在反思中改变，所以每次都要重新选择
            topology_dict = current_topology_response.topology.model_dump()
            tools_config = self._step3_select_tools(task_template, topology_dict)

            # 4. 优化 (Step 4)
            # 注意: 这里的 task_template 可能已经被 _generate_and_inject_path 修改过 (注入了路径)
            optimized_geometry, opt_log_str = self._step4_optimize(tensor_data, task_template, tools_config)

            # 5. 评估 (Step 5)
            report = self._step5_evaluate(optimized_geometry, tools_config)
            print(f"📉 [Result] 当前得分: {report['final_score']:.2f}")
            self._record(f"Step 5 Evaluation (Attempt {attempt + 1})", json.dumps(report, indent=2))

            # 记录最佳结果
            if report['final_score'] > best_report['final_score']:
                best_report = report
                # best_design = optimized_geometry

            # 成功判定 (例如 >= 90 分)
            if report['final_score'] >= 90.0:
                print("🎉 [Success] 达到目标分数，停止迭代！")
                break

            # === 6. 自我反思 (Step Reflect) ===
            # 如果没达到满分，且还有剩余次数，进行反思
            if attempt < max_retries - 1:
                reflection = self._step_reflect(task_template, current_topology_response, report)

                # === 7. 执行修正策略 ===
                if reflection.action == ReflectionAction.KEEP_CURRENT:
                    print("🤔 [Reflect] Agent 认为当前结果已足够好。")
                    break

                elif reflection.action == ReflectionAction.REINIT_GEOMETRY:
                    print("🎲 [Reflect] 策略: 重新初始化几何参数 (Re-Init Tensor)")
                    # 重新生成张量 (Adapter 会随机初始化参数)
                    # 注意：保留拓扑结构，只重置数值
                    tensor_data = self._convert_and_print_tensor(topology_dict)
                    # 这里的 tensor_data 已经是全新的随机初值了

                elif reflection.action == ReflectionAction.RESELECT_ANCHORS:
                    print(
                        f"⚓ [Reflect] 策略: 重新选择基座/末端 -> Ground: {reflection.suggested_ground_nodes}, EE: {reflection.suggested_ee_node}")
                    # 修改元数据
                    if reflection.suggested_ground_nodes:
                        current_topology_response.meta.ground_nodes = reflection.suggested_ground_nodes
                    if reflection.suggested_ee_node:
                        current_topology_response.meta.ee_node = reflection.suggested_ee_node

                    # ✨ 重要: 修改了 Ground/EE 后，必须重新生成路径！
                    self._generate_and_inject_path(current_topology_response, task_template)
                    # 张量结构不变，但 task_template 里的 Path 变了

                elif reflection.action == ReflectionAction.REGENERATE_TOPOLOGY:
                    print(f"🎨 [Reflect] 策略: 拓扑重绘 (Regenerate)")
                    # 将建议加入到 prompt 中
                    suggestion = reflection.topology_suggestion or "尝试不同的结构"
                    refined_context = f"{exp_context}\n[上一轮失败教训]: {suggestion}"
                    # 重新执行 Step 2 (LLM 生成 -> 路径注入 -> 转张量)
                    current_topology_response, tensor_data = self._step2_generate_topology(task_template,
                                                                                           refined_context)

        # 循环结束，存入经验池
        self._step6_store_experience(user_input, task_template, tensor_data, best_report)
        return best_report

    # =========================================================================
    #                               Step Functions
    # =========================================================================

    def _step1_parse_task(self, user_input):
        print(f"\n--- Step 1: 生成任务模板 ---")
        return self.parser.parse(user_input)

    def _step1_5_retrieve_experience(self, task_template):
        print(f"\n--- Step 1.5: 检索过往经验 ---")
        past_exps = self.memory.retrieve_relevant(task_template)
        exp_context = ""
        if past_exps:
            print(f"📚 发现 {len(past_exps)} 条相关经验...")
            exp_context = f"\n参考过往成功案例:\n{json.dumps(past_exps[0]['report'], indent=2)}"
        return exp_context

    def _step2_generate_topology(self, task_template, exp_context):
        print(f"\n--- Step 2: 生成初始拓扑草图 (DeepSeek Compat) ---")

        # 1. LLM 生成与解析
        response_obj = self._call_topology_llm(task_template, exp_context)

        # 2. ✨ 全自动生成路径并注入到 Task 模板中
        #    注意：这里不再操作 response_obj.meta，而是直接修改 task_template
        self._generate_and_inject_path(response_obj, task_template)

        # 3. 转换为物理张量并打印
        topology_data = response_obj.topology.model_dump()
        tensor_data = self._convert_and_print_tensor(topology_data)

        return response_obj, tensor_data

    # === Step 2 辅助函数 ===

    def _call_topology_llm(self, task_template, exp_context):
        """辅助函数: 负责 Prompt 构建、LLM 调用与基础校验"""
        try:
            # 1. 准备 Schema 和 Prompt
            schema_str = json.dumps(TopologyResponse.model_json_schema(), indent=2, ensure_ascii=False)
            full_system_prompt = TOPOLOGY_GEN_SYSTEM_PROMPT + STRUCTURED_OUTPUT_SUFFIX.format(schema=schema_str)
            user_content = f"任务模板:\n{json.dumps(task_template)}\n{exp_context}\n请生成初始拓扑。"

            self._record("Step 2 LLM Input (Prompt)", f"System:\n{full_system_prompt}\n\nUser:\n{user_content}")

            # 2. 调用 API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"}
            )

            # 3. 解析与校验
            content = response.choices[0].message.content
            print(f"\n🔍 [Debug] LLM 原始输出 (Raw JSON):\n{content}")
            self._record("Step 2 LLM Output (Raw)", content)

            response_obj = TopologyResponse.model_validate_json(content)
            print(f"🧠 [Thought]: {response_obj.thought_trace[:150]}...")

            return response_obj

        except Exception as e:
            print(f"❌ [Topology LLM Error] 生成失败: {e}")
            raise e

    def _generate_and_inject_path(self, response_obj, task_template):
        """
        辅助函数: 根据拓扑图自动计算路径序列，并注入到 task_template 中。
        不再依赖 response_obj.meta.path_sequence。
        """
        try:
            topology_data = response_obj.topology.model_dump()

            # 1. 构建图
            G = nx.Graph()
            for conn in topology_data['connections']:
                G.add_edge(int(conn['source']), int(conn['target']))

            # 2. 获取端点 (LLM 只负责定这两个点)
            if not response_obj.meta.ground_nodes:
                print("⚠️ [Path Gen] 未定义 ground_nodes，无法生成路径。")
                return

            start_node = int(response_obj.meta.ground_nodes[0])
            end_node = int(response_obj.meta.ee_node)

            print(f"🛠️ [Path Gen] 正在计算运动链: {start_node} -> ... -> {end_node}")

            # 3. 计算最短物理路径
            shortest_chain = nx.shortest_path(G, source=start_node, target=end_node)

            # 4. 寻找 Ghost Nodes (辅助节点)
            def get_ghost(current, exclude_list):
                neighbors = list(G.neighbors(current))
                for n in neighbors:
                    if n not in exclude_list: return n
                # 如果没有额外邻居，回退取自身或链上的邻居（仅做防崩处理）
                return neighbors[0] if neighbors else current

            ghost_in = get_ghost(start_node, shortest_chain)
            ghost_out = get_ghost(end_node, shortest_chain)

            # 5. 组装完整路径
            full_path = [ghost_in] + shortest_chain + [ghost_out]

            msg = f"Start: {start_node}, End: {end_node}\nGenerated Path: {full_path}"
            print(f"    - ✅ 自动生成路径: {full_path}")
            self._record("Step 2 Auto-Fix (Path Gen)", msg)

            # 6. ✨ 核心步骤: 注入到 Task Template
            if 'targets' not in task_template:
                task_template['targets'] = {}

            task_template['targets']['target_path_sequence'] = full_path

        except Exception as e:
            print(f"❌ [Path Gen Error] 路径生成失败: {e}")
            traceback.print_exc()
            if 'targets' in task_template:
                task_template['targets'].pop('target_path_sequence', None)

    def _convert_and_print_tensor(self, topology_data):
        """辅助函数: 负责张量转换及详细日志"""
        try:
            # 1. 动态决定 num_nodes
            node_ids = [int(nid) for nid in topology_data['nodes'].keys()]
            if node_ids:
                max_id = max(node_ids)
                calculated_num = max_id + 1
                num_nodes = max(calculated_num, 4)
                print(f"🧩 [Topology] 节点规模: {calculated_num} (Tensor Size: {num_nodes}x{num_nodes})")
            else:
                print("⚠️ [Warning] 未检测到节点，使用默认值 8")
                num_nodes = 8

            # 2. 转化张量
            adapter = MechanismTensorAdapter(num_nodes=num_nodes)
            tensor_data = adapter.json_to_tensor(topology_data)

            print(f"📊 [Tensor] 张量形状: {tensor_data.shape}")

            # 3. 详细打印
            with np.printoptions(threshold=np.inf, linewidth=200, precision=4, suppress=True):
                if np.all(tensor_data[0] == 0):
                    print("⚠️ [Warning] 张量全为 0！")
                else:
                    self._print_connection_details(tensor_data)

            return tensor_data

        except Exception as e:
            print(f"❌ [Tensor Error] 转换失败: {e}")
            traceback.print_exc()
            return np.zeros((5, 8, 8))

    def _print_connection_details(self, tensor_data):
        """辅助函数：打印连接表 (修正版: 显式打印反向节点类型)"""
        rows, cols = np.where(tensor_data[0] > 0)

        print(f"✅ [Success] 解析非零连接数: {len(rows) // 2}")
        print("\n🔍 [Debug] 非零连接详情:")
        print(f"{'Link':<10} | {'Type':<5} | {'a (mm)':<10} | {'alpha':<10} | {'offset':<15}")
        print("-" * 65)

        for r, c in zip(rows, cols):
            if r < c:
                # 1. 正向 r -> c
                type_val_r = tensor_data[1, r, c]
                type_str_r = "R" if type_val_r > 0.5 else ("P" if type_val_r < -0.5 else "?")
                a_val = tensor_data[2, r, c]
                alpha_val = tensor_data[3, r, c]
                off_at_r = tensor_data[4, r, c]

                # 2. 反向 c -> r
                type_val_c = tensor_data[1, c, r]
                type_str_c = "R" if type_val_c > 0.5 else ("P" if type_val_c < -0.5 else "?")
                off_at_c = tensor_data[4, c, r]

                print(
                    f"{r}->{c:<7} | {type_str_r:<5} | {a_val:<10.4f} | {alpha_val:<10.4f} | {off_at_r:<10.4f} (at {r})")
                print(f"{c}->{r:<7} | {type_str_c:<5} | {'^':<10} | {'^':<10} | {off_at_c:<10.4f} (at {c})")
                print("-" * 65)

    def _step3_select_tools(self, task_template, topology_data):
        print(f"\n--- Step 3: 动态选择优化与评估工具 (Structured Tool Library) ---")

        try:
            schema_str = json.dumps(ToolSelectionResponse.model_json_schema(), indent=2, ensure_ascii=False)
            tools_context = json.dumps(AVAILABLE_TOOLS_DEF, indent=2, ensure_ascii=False)
            full_system_prompt = TOOL_SELECTION_SYSTEM_PROMPT + STRUCTURED_OUTPUT_SUFFIX.format(schema=schema_str)

            user_content = f"""
                            任务信息:
                            {json.dumps(task_template, ensure_ascii=False)}
                            当前拓扑:
                            {json.dumps(topology_data, ensure_ascii=False)}
                            === 可用工具库定义 (JSON) ===
                            {tools_context}
                            请根据任务需求选择工具，并按照 Schema 格式输出。
                            """

            self._record("Step 3 Tool Selection Input", user_content)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            response_obj = ToolSelectionResponse.model_validate_json(content)

            print(f"🤔 [Reasoning]: {response_obj.reasoning}")
            print(f"🔧 [Optimizer] Selected: {response_obj.selected_optimization_tools}")
            print(f"⚖️ [Evaluator] Selected: {response_obj.selected_evaluation_tools}")

            if response_obj.suggested_new_optimization_tools:
                print(f"💡 [New Opt Idea] 建议新增优化工具: {len(response_obj.suggested_new_optimization_tools)} 个")

            return {
                "selected_optimization_losses": response_obj.selected_optimization_tools,
                "selected_evaluation_metrics": response_obj.selected_evaluation_tools,
                "full_response": response_obj.model_dump()
            }

        except Exception as e:
            print(f"❌ [Step 3 Error] 工具选择失败: {e}")
            traceback.print_exc()
            return {
                "selected_optimization_losses": ["closure_loop", "regularization"],
                "selected_evaluation_metrics": ["dof_check"]
            }

    def _step4_optimize(self, tensor_data, task_template, tools_config):
        print(f"\n--- Step 4: 执行优化循环 (Epochs={self.cfg.physics.max_iterations}) ---")

        selected_tool_names = tools_config.get('selected_optimization_losses', [])
        full_response = tools_config.get('full_response', {})
        new_tool_definitions = full_response.get('suggested_new_optimization_tools', [])

        if not new_tool_definitions:
            print(f"    -> [Optimizer] 传入新工具建议: [] (Empty)")
        else:
            print(f"    -> [Optimizer] 传入新工具建议: {len(new_tool_definitions)} 个")

        # === 核心修改：接收日志返回 ===
        optimized_geometry, optimized_q, opt_log_str = self.optimizer.run_optimization(
            tensor_data,
            task_template,
            selected_tool_names=selected_tool_names,
            new_tools_definitions=new_tool_definitions
        )

        # 记录优化日志
        self._record("Step 4 Optimizer Log", opt_log_str)

        return optimized_geometry, opt_log_str

    def _step5_evaluate(self, optimized_tensor, tools_config):
        print(f"\n--- Step 5: 生成评估报告 ---")
        return self.evaluator.generate_report(
            optimized_tensor,
            tools_config.get('selected_evaluation_metrics', [])
        )

    def _step6_store_experience(self, user_input, task_template, tensor_data, report):
        print(f"\n--- Step 6: 存入经验池 ---")
        self.memory.store_experience(user_input, task_template, tensor_data, report)

    def _step_reflect(self, task, topology_response, report):
        print(f"\n--- Step Reflect: 失败分析与策略调整 ---")
        try:
            # 1. 准备上下文 (全量历史)
            full_history_text = "\n".join(self.trace_log)

            context = {
                "task_intent": task['user_intent_summary'],
                "current_status": {
                    "topology_nodes": len(topology_response.topology.nodes),
                    "ground": topology_response.meta.ground_nodes,
                    "ee": topology_response.meta.ee_node
                },
                "evaluation_report": report,
                "execution_history_trace": "See full logs provided below"
            }

            schema_str = json.dumps(ReflectionResponse.model_json_schema(), indent=2, ensure_ascii=False)
            system_prompt = REFLECTION_SYSTEM_PROMPT + STRUCTURED_OUTPUT_SUFFIX.format(schema=schema_str)

            user_prompt = f"""
            当前优化结果不理想。以下是完整的执行历史记录（包含 Prompt、Raw Output、Code Logic、Optimization Logs）：

            === EXECUTION HISTORY START ===
            {full_history_text}
            === EXECUTION HISTORY END ===

            请结合上述历史，分析失败原因并给出 JSON 格式的修正策略。
            Context Summary: {json.dumps(context, ensure_ascii=False)}
            """

            # 2. 调用 LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )

            # 3. 解析
            content = response.choices[0].message.content
            self._record("Step Reflect Output", content)

            reflection = ReflectionResponse.model_validate_json(content)

            print(f"🧠 [Analysis]: {reflection.analysis}")
            print(f"👉 [Action]: {reflection.action.value}")

            return reflection

        except Exception as e:
            print(f"❌ [Reflection Error] 反思失败，默认重置初值: {e}")
            traceback.print_exc()
            return ReflectionResponse(
                analysis="Reflection failed",
                action=ReflectionAction.REINIT_GEOMETRY
            )