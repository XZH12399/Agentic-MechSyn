import json
import numpy as np
import traceback
import networkx as nx
import re
import os
from openai import OpenAI

# === 1. 导入 Prompt 模板 ===
from .prompt_templates import (
    TOOL_SELECTION_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    TOPOLOGY_CORRECTION_SYSTEM_PROMPT,
    STRUCTURED_OUTPUT_SUFFIX
)

# === 2. 导入 Schema ===
from .schemas import (
    TaskTemplate,
    TopologyResponse,
    TopologyCorrectionResponse,
    ToolSelectionResponse,
    ReflectionResponse,
    ReflectionAction,
    TopologySpec
)

# === 3. 导入核心逻辑模块 ===
from .task_parser import TaskParser
from .memory import ExperienceManager

# === 4. 导入工具模块 ===
from tools.tool_registry import AVAILABLE_TOOLS_DEF
from utils.tensor_adapter import MechanismTensorAdapter
from tools.optimizer_engine import MechanismOptimizer
from tools.evaluator_engine import MechanismEvaluator

# === 5. 导入 VLA 推理模块 ===
try:
    from vla_module.inference_vla import load_model_and_tokenizer, generate_mechanism
except ImportError:
    from inference_vla import load_model_and_tokenizer, generate_mechanism


class MechanismAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.model = cfg.model_name

        self.parser = TaskParser(self.client, self.model)
        self.memory = ExperienceManager()
        self.optimizer = MechanismOptimizer(cfg.physics)
        self.evaluator = MechanismEvaluator()

        print("🤖 [Init] Loading Fine-tuned VLA Model (Qwen2-VL)...")
        self.vla_model, self.vla_tokenizer = load_model_and_tokenizer()

        self.trace_log = []
        self.conversation_history = []

    def _record(self, tag, content):
        entry = f"\n[{tag}]\n{content}\n" + "-" * 40
        self.trace_log.append(entry)

    def run_pipeline(self, user_input):
        self.trace_log = []
        self.conversation_history = [] 
        
        print(f"\n🚀 [Start] 启动语义-拓扑-几何 Agent 流水线...")
        self._record("User Input", user_input)

        # 1. 任务解析
        task_template_obj = self._step1_parse_task(user_input)
        initial_task_dict = task_template_obj.model_dump()
        self._record("Step 1 Task Template", json.dumps(initial_task_dict, indent=2, ensure_ascii=False))

        # 1.5 经验检索
        exp_context = self._step1_5_retrieve_experience(initial_task_dict)

        # === 进入迭代循环 ===
        max_retries = self.cfg.agent.get('max_turns', 5)
        current_topology_response = None
        tensor_data = None
        initial_q = None # 保存初始 Q
        current_task_dict = initial_task_dict
        
        last_failure_reason = ""
        best_report = {"final_score": -1}
        best_design = None
        best_q = None

        # 初始生成 (Step 2)
        current_topology_response, tensor_data, initial_q, current_task_dict = self._step2_generate_topology(
            task_template_obj, exp_context
        )

        for attempt in range(max_retries):
            print(f"\n🔄 [Iteration {attempt + 1}/{max_retries}] 执行几何优化...")

            # 3. 工具选择
            topology_dict = current_topology_response.topology.model_dump()
            tools_config = self._step3_select_tools(current_task_dict, topology_dict)

            # 4. 优化
            # ✨ 传递 initial_q 到优化器
            optimized_geometry, optimized_q, opt_log_str = self._step4_optimize(
                tensor_data, current_task_dict, tools_config, initial_q=initial_q
            )

            # 5. 评估
            report = self._step5_evaluate(optimized_geometry, tools_config, optimized_q, current_task_dict)
            
            print(f"📉 [Result] 当前得分: {report['final_score']:.2f}")
            self._record(f"Step 5 Evaluation (Attempt {attempt + 1})", json.dumps(report, indent=2))

            if report['final_score'] > best_report['final_score']:
                best_report = report
                best_design = optimized_geometry
                best_q = optimized_q

            if report['final_score'] >= 90.0:
                print("🎉 [Success] 满足设计要求！")
                break

            if attempt < max_retries - 1:
                reflection = self._step_reflect(current_task_dict, current_topology_response, report)
                last_failure_reason = reflection.analysis

                if reflection.action == ReflectionAction.KEEP_CURRENT:
                    break
                elif reflection.action == ReflectionAction.REINIT_GEOMETRY:
                    print(f"🎲 [Reflect] 策略: 请求 DeepSeek 修正几何参数")
                    refined_topology_dict = self._step2_5_refine_topology_with_llm(
                        task_template_obj,
                        current_topology_response.topology.model_dump(),
                        current_topology_response.raw_tokens,
                        retry_context=last_failure_reason
                    )
                    try:
                        current_topology_response.topology = TopologySpec.model_validate(refined_topology_dict)
                    except Exception as e:
                        print(f"⚠️ [Warning] Topology Validation: {e}")
                    
                    # ✨ 重新解析 tensor 和 initial_q
                    tensor_data, initial_q = self._convert_and_print_tensor(refined_topology_dict)
                    
                elif reflection.action == ReflectionAction.RESELECT_ANCHORS:
                    print(f"⚓ [Reflect] 策略: 重选 Ground/EE")
                    if reflection.suggested_ground_nodes:
                        current_topology_response.meta.ground_nodes = reflection.suggested_ground_nodes
                    if reflection.suggested_ee_node:
                        current_topology_response.meta.ee_node = reflection.suggested_ee_node
                    self._generate_and_inject_path(current_topology_response, current_task_dict)
                    
                elif reflection.action == ReflectionAction.REGENERATE_TOPOLOGY:
                    print(f"🎨 [Reflect] 策略: 拓扑重绘")
                    current_topology_response, tensor_data, initial_q, current_task_dict = self._step2_generate_topology(
                        task_template_obj, exp_context, retry_context=last_failure_reason
                    )

        # ✨ 最终结果汇总 ✨
        final_tensor = best_design if best_design is not None else tensor_data
        final_q = best_q if best_q is not None else np.zeros_like(tensor_data[0])

        self._step6_store_experience(user_input, current_task_dict, final_tensor, best_report)
        
        # 导出结果
        self._export_result(final_tensor, final_q, current_topology_response.meta)

        return best_report

    # =========================================================================
    #                               Step Functions
    # =========================================================================

    def _step1_parse_task(self, user_input):
        print(f"\n--- Step 1: 生成任务模板 ---")
        return self.parser.parse_task(user_input)

    def _step1_5_retrieve_experience(self, task_dict):
        print(f"\n--- Step 1.5: 检索过往经验 ---")
        past_exps = self.memory.retrieve_relevant(task_dict)
        if past_exps:
            print(f"📚 发现 {len(past_exps)} 条相关经验...")
            return f"\n参考过往成功案例:\n{json.dumps(past_exps[0]['report'], indent=2)}"
        return ""

    def _step2_generate_topology(self, task_template_obj, exp_context, retry_context=None):
        print(f"\n--- Step 2: VLA 拓扑综合 (Fine-tuned Model) ---")

        if isinstance(task_template_obj, dict):
            task_template_obj = TaskTemplate.model_validate(task_template_obj)

        summary = getattr(task_template_obj, 'user_intent_summary', "a mechanism")
        vla_prompt = getattr(task_template_obj, 'vla_instruction', None)
        
        if not vla_prompt:
            target_dof = task_template_obj.kinematics.dof if hasattr(task_template_obj, 'kinematics') else 1
            target_loops = 1 
            note = ""

            if retry_context:
                retry_lower = retry_context.lower()
                if "0" in retry_lower or "rigid" in retry_lower or "overconstrained" in retry_lower:
                    if "parallel" in summary.lower():
                        target_loops = 2
                        note = " (Increased loop count to avoid overconstraint)"
                        print(f"💡 [Strategy] 检测到过约束: 尝试生成 {target_loops} 环机构。")
                    else:
                        note = " Ensure the mechanism is not rigid."
                elif "instantaneous" in retry_lower or "uncontrollable" in retry_lower:
                    note = " Ensure valid continuous motion, avoid instantaneous positions."

            vla_prompt = f"Design a mechanism with {target_dof} DoF and {target_loops} loop(s) based on: {summary}"
            if retry_context:
                vla_prompt += f" Note: Previous attempt failed ({retry_context}).{note} Try a different structure."

        raw_tokens = generate_mechanism(self.vla_model, self.vla_tokenizer, vla_prompt)
        self._record("VLA Raw Tokens", raw_tokens)
        
        print(f"\n📝 [VLA Adjusted Input]: {vla_prompt}")
        print(f"📝 [VLA Raw Output]:\n{raw_tokens}\n")

        topology_dict, meta_info = self._parse_vla_tokens_to_topology(raw_tokens)
        
        topology_dict = self._step2_5_refine_topology_with_llm(
            task_template_obj, 
            topology_dict, 
            raw_tokens, 
            retry_context=retry_context
        )

        self._print_agentic_mechsyn_report(topology_dict, meta_info)

        response_obj = TopologyResponse(
            thought_trace="Generated via Fine-tuned VLA Module and refined by DeepSeek.",
            raw_tokens=raw_tokens,
            topology=topology_dict,
            meta=meta_info
        )

        updated_task_dict = task_template_obj.model_dump()
        self._generate_and_inject_path(response_obj, updated_task_dict)
        
        # ✨ 解析 tensor 和 initial_q
        tensor_data, initial_q = self._convert_and_print_tensor(response_obj.topology.model_dump())
        self._print_topology_cycles(topology_dict)

        return response_obj, tensor_data, initial_q, updated_task_dict

    def _step2_5_refine_topology_with_llm(self, task_obj, topology_dict, vla_raw_tokens, retry_context=None):
        """
        [全量智能修正]
        ✨ 修复: 使用中文 Prompt，并强制要求 DeepSeek 给出非奇异的初始状态。
        """
        print(f"\n--- Step 2.5: 语义拓扑修正 (DeepSeek Judge) ---")
        
        special_req = task_obj.constraints.special_constraints
        user_intent = task_obj.user_intent_summary
        
        try:
            schema_str = json.dumps(TopologyCorrectionResponse.model_json_schema(), indent=2)
            system_prompt = TOPOLOGY_CORRECTION_SYSTEM_PROMPT + STRUCTURED_OUTPUT_SUFFIX.format(schema=schema_str)
            
            # ✨✨✨ 修改后的中文 Prompt ✨✨✨
            user_content = f"""
            用户设计意图: {user_intent}
            特殊约束: {special_req}
            
            VLA 生成的拓扑结构 (JSON):
            {json.dumps(topology_dict, indent=2)}
            
            重要修正指南:
            1. 对于特殊机构（如 Bennett, Sarrus 等），请确保杆长 (a) 和扭转角 (alpha) 严格满足其几何构成条件。
            2. **关键**: 对于关节初始状态 (theta/state)，**请勿**将其全部设为 0.0。请给出一个合理的、非奇异的初始位型（例如 0.5 rad 或 30 度），以防止机构在初始时刻处于折叠或死锁状态。
            """
            
            messages = [{"role": "system", "content": system_prompt}]
            
            if self.conversation_history:
                messages.extend(self.conversation_history[-4:])
            
            if retry_context:
                user_content += f"\n\n⚠️ 关键上下文: 上一次模拟失败了，报错信息为: '{retry_context}'。\n请严格检查几何参数是否会导致此类错误（如刚性、过约束等）。"
            
            messages.append({"role": "user", "content": user_content})

            print(f"🧠 [Refine] 正请求 DeepSeek 审核拓扑结构 (中文 Prompt)...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            self.conversation_history.append({"role": "user", "content": user_content[:800] + "..."})
            self.conversation_history.append({"role": "assistant", "content": content})

            result = TopologyCorrectionResponse.model_validate_json(content)
            
            if not result.requires_correction:
                print("✅ [Refine] DeepSeek 判定：VLA 输出符合要求，无需修正。")
                return topology_dict

            print(f"🛠️ [Refine] DeepSeek 判定：需要修正。原因: {result.reasoning}")
            
            nodes = topology_dict['nodes']
            connections = topology_dict['connections']
            
            for action in result.corrections:
                print(f"    -> Modifying {action.target_type} {action.target_id}: {action.param_name} = {action.value}")
                
                if action.target_type == "node":
                    if action.target_id in nodes:
                        nodes[action.target_id][action.param_name] = action.value
                
                elif action.target_type == "connection":
                    target_conn = None
                    if "_" in action.target_id:
                        src, tgt = action.target_id.split("_")
                        for conn in connections:
                            c_src = str(conn.get('source')) if isinstance(conn, dict) else str(conn.source)
                            c_tgt = str(conn.get('target')) if isinstance(conn, dict) else str(conn.target)
                            if c_src == src and c_tgt == tgt:
                                target_conn = conn
                                break
                    
                    if target_conn:
                        val = action.value
                        try:
                            val = float(val)
                        except ValueError:
                            print(f"    ⚠️ [Refine] Non-numeric value '{val}' detected, using 0.0")
                            val = 0.0 
                        
                        if isinstance(target_conn, dict):
                            target_conn[action.param_name] = val
                        else:
                            setattr(target_conn, action.param_name, val)

            return {"nodes": nodes, "connections": connections}

        except Exception as e:
            print(f"⚠️ [Refine Error] 修正过程出错: {e}")
            traceback.print_exc()
            return topology_dict

    def _parse_vla_tokens_to_topology(self, token_string):
        import math
        DEG2RAD = math.pi / 180.0
        TWO_PI = 2 * math.pi

        nodes = {}
        connections = []
        all_actions = list(re.finditer(r"(<Action_[a-zA-Z_]+>)", token_string))
        param_pattern = re.compile(r"<([a-zA-Z]+)_([a-zA-Z0-9.\-]+)>")
        current_source = "0"
        
        for i, match in enumerate(all_actions):
            action_tag = match.group(1)
            start_idx = match.end()
            end_idx = all_actions[i+1].start() if i+1 < len(all_actions) else len(token_string)
            content_str = token_string[start_idx:end_idx]
            
            params_list = param_pattern.findall(content_str)
            p_dict = {}
            geo_vals = {'Len': [], 'Twist': [], 'Off': [], 'State': []}
            for key, val in params_list:
                if key in geo_vals: geo_vals[key].append(float(val))
                else: p_dict[key] = val
            
            node_id = p_dict.get('ID', str(len(nodes)))
            raw_role = p_dict.get('Role', 'link')
            normalized_role = raw_role.lower()
            if normalized_role == 'normal': normalized_role = 'link'
            
            if action_tag == "<Action_New_Node>":
                nodes[node_id] = {"type": p_dict.get('Type', 'R'), "role": normalized_role}
                current_source = node_id
            
            elif action_tag in ["<Action_Link_To_New>", "<Action_Link_To_Old>"]:
                if action_tag == "<Action_Link_To_New>" and node_id not in nodes:
                    nodes[node_id] = {"type": p_dict.get('Type', 'R'), "role": normalized_role}
                
                src_type = nodes.get(current_source, {}).get('type', 'R')
                tgt_type = nodes.get(node_id, {}).get('type', 'R')
                is_src_R = (src_type == 'R')
                is_tgt_R = (tgt_type == 'R')

                raw_a = geo_vals['Len'][0] if geo_vals['Len'] else 50.0
                raw_alpha_deg = geo_vals['Twist'][0] if geo_vals['Twist'] else 0.0
                raw_off_src = geo_vals['Off'][0] if len(geo_vals['Off']) > 0 else 0.0
                raw_off_tgt = geo_vals['Off'][1] if len(geo_vals['Off']) > 1 else 0.0
                raw_state_src = geo_vals['State'][0] if len(geo_vals['State']) > 0 else 0.0
                raw_state_tgt = geo_vals['State'][1] if len(geo_vals['State']) > 1 else 0.0

                val_a = raw_a
                val_alpha = (raw_alpha_deg * DEG2RAD) % TWO_PI

                if is_src_R:
                    val_state_src = (raw_state_src * DEG2RAD) % TWO_PI
                    val_offset_src = raw_off_src
                else:
                    val_state_src = raw_state_src
                    val_offset_src = (raw_off_src * DEG2RAD) % TWO_PI

                if is_tgt_R:
                    val_state_tgt = (raw_state_tgt * DEG2RAD) % TWO_PI
                    val_offset_tgt = raw_off_tgt
                else:
                    val_state_tgt = raw_state_tgt
                    val_offset_tgt = (raw_off_tgt * DEG2RAD) % TWO_PI

                conn_entry = {
                    "source": int(current_source),
                    "target": int(node_id),
                    "a": val_a,
                    "alpha": val_alpha,
                    "offset_source": val_offset_src,
                    "offset_target": val_offset_tgt,
                    "theta_source": val_state_src,
                    "theta_target": val_state_tgt
                }
                connections.append(conn_entry)
                current_source = node_id
            
            elif action_tag == "<Action_Jump_To>":
                current_source = node_id

        ground_nodes = [int(nid) for nid, n in nodes.items() if n.get('role') == 'ground']
        if not ground_nodes: ground_nodes = [0]
        ee_nodes = [int(nid) for nid, n in nodes.items() if n.get('role') == 'ee']
        if ee_nodes: ee_node = ee_nodes[0]
        else:
            all_ids = sorted([int(k) for k in nodes.keys()], reverse=True)
            candidates = [nid for nid in all_ids if nid not in ground_nodes]
            ee_node = candidates[0] if candidates else all_ids[0]

        return {"nodes": nodes, "connections": connections}, {"ground_nodes": ground_nodes, "ee_node": ee_node}

    def _convert_and_print_tensor(self, topology_data):
        """
        转换并打印 Tensor。
        ✨ 新增: 同时解析并返回 initial_q 矩阵
        """
        try:
            node_ids = [int(nid) for nid in topology_data['nodes'].keys()]
            num_nodes = max(max(node_ids) + 1, 4) if node_ids else 8
            print(f"🧩 [Topology] 节点规模: {len(node_ids)} (Tensor Size: {num_nodes}x{num_nodes})")
            
            # 1. 转换几何 Tensor
            adapter = MechanismTensorAdapter(num_nodes=num_nodes)
            tensor_data = adapter.json_to_tensor(topology_data)
            print(f"📊 [Tensor] 张量形状: {tensor_data.shape}")
            
            # 2. ✨ 手动解析 Initial Q (State)
            initial_q = np.zeros((num_nodes, num_nodes))
            conns = topology_data.get('connections', [])
            if not isinstance(conns, list): conns = topology_data['connections']
            
            for conn in conns:
                c = conn if isinstance(conn, dict) else conn.model_dump()
                u, v = int(c['source']), int(c['target'])
                
                # 提取状态 (优先 float，失败则 0.0)
                try:
                    ts = float(c.get('theta_source', 0.0))
                except: ts = 0.0
                
                try:
                    tt = float(c.get('theta_target', 0.0))
                except: tt = 0.0
                
                # 填充矩阵
                if u < num_nodes and v < num_nodes:
                    initial_q[u, v] = ts
                    initial_q[v, u] = tt

            if not np.all(tensor_data[0] == 0): 
                self._print_connection_details(tensor_data, topology_data)
            else: 
                print("⚠️ [Warning] 张量全为 0！")
                
            return tensor_data, initial_q
            
        except Exception as e:
            print(f"❌ [Tensor Error] 转换失败: {e}")
            traceback.print_exc()
            # 出错时返回空 Q
            return np.zeros((5, 8, 8)), np.zeros((8, 8))

    def _print_connection_details(self, tensor_data, topology_data=None):
        rows, cols = np.where(tensor_data[0] > 0)
        conn_lookup = {}
        if topology_data:
            for c in topology_data.get('connections', []):
                c_dict = c if isinstance(c, dict) else c.model_dump()
                u, v = int(c_dict['source']), int(c_dict['target'])
                conn_lookup[(u, v)] = c_dict

        print("\n🔍 [Debug] 非零连接详情:")
        print(f"{'Link':<10} | {'Type':<5} | {'a (mm)':<10} | {'alpha (rad)':<12} | {'Offset (Par)':<15} | {'State (Var)':<15}")
        print("-" * 90)
        for r, c in zip(rows, cols):
            r, c = int(r), int(c)
            if r < c:
                t_r = "R" if tensor_data[1, r, c] > 0.5 else "P"
                a = tensor_data[2, r, c]
                al = tensor_data[3, r, c]
                off_r = tensor_data[4, r, c]
                
                state_r_str = "N/A"
                if (r, c) in conn_lookup: 
                    val = conn_lookup[(r, c)].get('theta_source', 0.0)
                    state_r_str = f"{val:.4f}"
                elif (c, r) in conn_lookup: 
                    val = conn_lookup[(c, r)].get('theta_target', 0.0)
                    state_r_str = f"{val:.4f}"
                
                t_c = "R" if tensor_data[1, c, r] > 0.5 else "P"
                off_c = tensor_data[4, c, r]
                state_c_str = "N/A"
                if (c, r) in conn_lookup: 
                    val = conn_lookup[(c, r)].get('theta_source', 0.0)
                    state_c_str = f"{val:.4f}"
                elif (r, c) in conn_lookup: 
                    val = conn_lookup[(r, c)].get('theta_target', 0.0)
                    state_c_str = f"{val:.4f}"
                
                print(f"{r}->{c:<7} | {t_r:<5} | {a:<10.4f} | {al:<12.4f} | {off_r:<15.4f} | {state_r_str:<15}")
                print(f"{c}->{r:<7} | {t_c:<5} | {'^':<10} | {'^':<12} | {off_c:<15.4f} | {state_c_str:<15}")
                print("-" * 90)

    def _print_agentic_mechsyn_report(self, topology_dict, meta_info):
        nodes = topology_dict.get('nodes', {})
        connections = topology_dict.get('connections', [])
        num_nodes = len(nodes)
        num_joints = len(connections)
        dof_spatial = 6 * (num_nodes - 1) - 5 * num_joints
        ground_nodes = meta_info.get('ground_nodes', [])
        ee_node = meta_info.get('ee_node', 'N/A')

        def safe_fmt(val, prec=2):
            try:
                f_val = float(val)
                return f"{f_val:.{prec}f}"
            except:
                return str(val)

        print("\n" + "="*80)
        print("📜 [Agentic-MechSyn Report] 机构全息解码报告")
        print("="*80)
        print(f"🏗️  [Topology Summary]")
        print(f"    • Total Nodes (Links):   {num_nodes}")
        print(f"    • Total Joints (Pairs):  {num_joints}")
        print(f"    • Theoretical DoF (Spatial): {dof_spatial} (Gruebler Est.)")
        print(f"    • Ground Bases: {ground_nodes}")
        print(f"    • End-Effector: {ee_node}")
        
        print(f"\n🧩 [Node Registry]")
        sorted_ids = sorted(nodes.keys(), key=lambda x: int(x))
        for nid in sorted_ids:
            info = nodes[nid]
            role_icon = "🏠" if int(nid) in ground_nodes else ("🎯" if int(nid) == int(ee_node) else "🔗")
            print(f"    - Node {nid}: {role_icon} {info.get('role', 'unknown').title()} ({info.get('type', '?')}-Type)")

        print(f"\n⛓️  [Connectivity Flow] (Dual-Sided Kinematics)")
        print(f"    Format: Source(Type) -> Target(Type) | Common Params")
        print(f"            └─ Source Side: Fixed Offset & Variable State")
        print(f"            └─ Target Side: Fixed Offset & Variable State")
        print("-" * 80)

        for i, conn in enumerate(connections):
            c = conn if isinstance(conn, dict) else conn.model_dump()
            u, v = c['source'], c['target']
            
            src_type = nodes.get(str(u), {}).get('type', 'R')
            tgt_type = nodes.get(str(v), {}).get('type', 'R')
            
            common_str = f"a={safe_fmt(c.get('a',0), 1)}, α={safe_fmt(c.get('alpha',0), 2)}"

            if src_type == 'R':
                src_str = f"Off={safe_fmt(c.get('offset_source',0),1)}mm, State={safe_fmt(c.get('theta_source',0),2)}rad"
            else:
                src_str = f"Off={safe_fmt(c.get('offset_source',0),2)}rad, State={safe_fmt(c.get('theta_source',0),1)}mm"

            if tgt_type == 'R':
                tgt_str = f"Off={safe_fmt(c.get('offset_target',0),1)}mm, State={safe_fmt(c.get('theta_target',0),2)}rad"
            else:
                tgt_str = f"Off={safe_fmt(c.get('offset_target',0),2)}rad, State={safe_fmt(c.get('theta_target',0),1)}mm"

            print(f"    {i+1}. {u}({src_type}) ──> {v}({tgt_type}) | {common_str}")
            print(f"       └─ Src: [{src_str}]")
            print(f"       └─ Tgt: [{tgt_str}]")
            
        print("="*80 + "\n")

    def _print_topology_cycles(self, topology_data):
        try:
            G = nx.Graph()
            conns = topology_data.get('connections', [])
            if not isinstance(conns, list): 
                 conns = topology_data['connections']
                 
            for conn in conns:
                c = conn if isinstance(conn, dict) else conn.model_dump()
                u, v = int(c['source']), int(c['target'])
                G.add_edge(u, v)
            
            cycles = nx.cycle_basis(G)
            
            if cycles:
                print(f"\n🔄 [Topology] 检测到 {len(cycles)} 个基本闭环 (Loops):")
                for i, cycle in enumerate(cycles):
                    path_str = " -> ".join(map(str, cycle + [cycle[0]]))
                    print(f"    - Loop {i+1}: {path_str}")
            else:
                print(f"\n⚠️ [Topology] 未检测到闭环 (这是一个开链机构)")
                
        except Exception as e:
            print(f"\n⚠️ [Topology] 环路检测打印失败: {e}")

    def _construct_smart_path(self, topology_data, base_link_str, ee_link_str):
        G_full = nx.Graph()
        conns = topology_data.get('connections', [])
        if not isinstance(conns, list):
             conns = topology_data['connections']
             
        for conn in conns:
            c = conn if isinstance(conn, dict) else conn.model_dump()
            G_full.add_edge(int(c['source']), int(c['target']))

        def parse_anchor_opts(anchor_str):
            s = str(anchor_str)
            if '_' in s:
                u, v = map(int, s.split('_'))
                return [{'node': u, 'ghost': v}, {'node': v, 'ghost': u}], (u, v)
            else:
                return [{'node': int(s), 'ghost': None}], None

        base_opts, base_edge_to_cut = parse_anchor_opts(base_link_str)
        ee_opts, ee_edge_to_cut = parse_anchor_opts(ee_link_str)

        G_cut = G_full.copy()
        
        if base_edge_to_cut and G_cut.has_edge(*base_edge_to_cut):
            G_cut.remove_edge(*base_edge_to_cut)
        if ee_edge_to_cut and G_cut.has_edge(*ee_edge_to_cut):
            G_cut.remove_edge(*ee_edge_to_cut)

        valid_candidates = []

        for b_opt in base_opts:
            for e_opt in ee_opts:
                try:
                    path = nx.shortest_path(G_cut, source=b_opt['node'], target=e_opt['node'])
                    
                    ghost_in = b_opt['ghost']
                    ghost_out = e_opt['ghost']
                    
                    if ghost_in is None:
                        ghost_in = self._find_fallback_ghost(G_full, path[0], exclude=path)
                    if ghost_out is None:
                        ghost_out = self._find_fallback_ghost(G_full, path[-1], exclude=path)

                    full_path = [ghost_in] + path + [ghost_out]
                    valid_candidates.append((len(path), full_path))
                    
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        
        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[0])
            best_path = valid_candidates[0][1]
            print(f"    🔍 [Smart Path] 发现 {len(valid_candidates)} 条路径，最优长度: {valid_candidates[0][0]}")
            return best_path
        
        return None

    def _find_fallback_ghost(self, G, node, exclude):
        neighbors = list(G.neighbors(node))
        for n in neighbors:
            if n not in exclude: return n
        return neighbors[0] if neighbors else node

    def _generate_and_inject_path(self, response_obj, task_dict):
        try:
            topology_data = response_obj.topology.model_dump()
            meta = response_obj.meta
            
            ground_nodes = sorted(meta.ground_nodes)
            if len(ground_nodes) > 1:
                base_str = "_".join(map(str, ground_nodes))
                print(f"🛠️ [Path Gen] 识别基座为杆件: {base_str} (Link)")
            else:
                base_str = str(ground_nodes[0])
                print(f"🛠️ [Path Gen] 识别基座为节点: {base_str} (Node)")

            ee_str = str(meta.ee_node) 
            full_path = self._construct_smart_path(topology_data, base_str, ee_str)
            
            if full_path:
                print(f"    - ✅ 智能路径生成: {full_path}")
                if 'targets' not in task_dict: task_dict['targets'] = {}
                task_dict['targets']['target_path_sequence'] = full_path
            else:
                print("❌ [Path Gen] 智能路径搜索失败，未找到连通路径！")
                
        except Exception as e:
            print(f"❌ [Path Gen Error] 路径生成失败: {e}")
            traceback.print_exc()

    def _step3_select_tools(self, task_dict, topology_data):
        print(f"\n--- Step 3: 动态选择优化与评估工具 (LLM Autonomous) ---")
        try:
            schema_str = json.dumps(ToolSelectionResponse.model_json_schema(), indent=2, ensure_ascii=False)
            full_system_prompt = TOOL_SELECTION_SYSTEM_PROMPT + STRUCTURED_OUTPUT_SUFFIX.format(schema=schema_str)
            
            # ✨✨✨ Prompt 极简版: 仅提供基本信息，无硬性约束 ✨✨✨
            user_content = f"""
            任务信息: {json.dumps(task_dict, ensure_ascii=False)}
            拓扑结构: {json.dumps(topology_data, ensure_ascii=False)}
            
            请根据任务需求和拓扑结构，从工具注册表中选择最合适的优化工具和评估工具。
            请确保选择的工具能充分验证机构的可行性和性能。
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": full_system_prompt},{"role": "user", "content": user_content}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            response_obj = ToolSelectionResponse.model_validate_json(content)
            
            print(f"🔧 [Optimizer] Final Selection: {response_obj.selected_optimization_tools}")
            print(f"📊 [Evaluator] Final Selection: {response_obj.selected_evaluation_tools}")

            return {
                "selected_optimization_losses": response_obj.selected_optimization_tools,
                "selected_evaluation_metrics": response_obj.selected_evaluation_tools,
                "full_response": response_obj.model_dump()
            }
            
        except Exception as e:
            print(f"⚠️ [Step 3 Error] LLM 选择失败，回退到基础配置: {e}")
            traceback.print_exc()
            return {
                "selected_optimization_losses": ["closure_loop", "mobility_dof"],
                "selected_evaluation_metrics": ["dof_check", "closure_loop"],
                "full_response": {}
            }

    def _step4_optimize(self, tensor_data, task_dict, tools_config, initial_q=None):
        print(f"\n--- Step 4: 执行优化循环 (Epochs={self.cfg.physics.max_iterations}) ---")
        selected = tools_config.get('selected_optimization_losses', [])
        new_tools = tools_config.get('full_response', {}).get('suggested_new_optimization_tools', [])
        # ✨ 传递 initial_q
        return self.optimizer.run_optimization(tensor_data, task_dict, selected, new_tools, initial_q=initial_q)

    def _step5_evaluate(self, optimized_tensor, tools_config, optimized_q=None, task_dict=None):
        print(f"\n--- Step 5: 生成评估报告 ---")
        
        # 1. 调用评估器获取原始报告 (传入 task_dict 以支持 path/twist check)
        report = self.evaluator.generate_report(
            optimized_tensor, 
            tools_config.get('selected_evaluation_metrics', []),
            q_values=optimized_q,
            task=task_dict 
        )
        
        # 2. 打印详细诊断表格
        print(f"\n    {'='*75}")
        print(f"    {'Metric Name':<20} | {'Status':<8} | {'Diagnostic Message'}")
        print(f"    {'-'*75}")
        
        for name, res in report.get("details", {}).items():
            status = res.get("status", "unknown")
            
            # 状态图标化
            if status == "pass":
                icon = "✅"
                status_str = "PASS"
            elif status == "fail":
                icon = "❌"
                status_str = "FAIL"
            elif status == "skip":
                icon = "⏩"
                status_str = "SKIP"
            else:
                icon = "❓"
                status_str = status.upper()

            msg = res.get("msg", "No details provided")
            
            # 打印一行诊断信息
            print(f"    {name:<20} | {icon} {status_str:<4} | {msg}")
            
        print(f"    {'='*75}\n")
        
        return report

    def _step6_store_experience(self, user_input, task_dict, tensor_data, report):
        print(f"\n--- Step 6: 存入经验池 ---")
        self.memory.store_experience(user_input, task_dict, tensor_data, report)

    def _step_reflect(self, task_dict, topology_response, report):
        print(f"\n--- Step Reflect: 失败分析与策略调整 ---")
        try:
            full_history = "\n".join(self.trace_log)
            schema_str = json.dumps(ReflectionResponse.model_json_schema(), indent=2)
            system_prompt = REFLECTION_SYSTEM_PROMPT + STRUCTURED_OUTPUT_SUFFIX.format(schema=schema_str)
            intent = task_dict.get('user_intent_summary', 'unknown')
            user_prompt = f"User Intent: {intent}\n历史记录:\n{full_history}\n分析失败原因并给出策略。"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            reflection = ReflectionResponse.model_validate_json(content)
            print(f"🧠 [Analysis]: {reflection.analysis}")
            print(f"👉 [Action]: {reflection.action.value}")
            return reflection
        except Exception as e:
            print(f"⚠️ [Reflect] Error: {e}")
            return ReflectionResponse(analysis="Error", action=ReflectionAction.KEEP_CURRENT)

    def _export_result(self, tensor, q_matrix, meta_info):
        print(f"\n--- Final Result Export ---")
        
        try:
            if hasattr(tensor, 'detach'): tensor = tensor.detach().cpu().numpy()
            if hasattr(q_matrix, 'detach'): q_matrix = q_matrix.detach().cpu().numpy()
            
            N = tensor.shape[1]
            rows, cols = np.where(tensor[0] > 0.5)
            
            joints = {}
            edges = {}
            
            for r, c in zip(rows, cols):
                if r >= c: continue 
                
                u, v = int(r), int(c)
                edge_key = f"{u}_{v}"
                
                is_R = tensor[1, u, v] > 0.5
                j_type = "R" if is_R else "P"
                
                joints[str(u)] = j_type
                joints[str(v)] = j_type 
                
                a = float(tensor[2, u, v])
                alpha = float(tensor[3, u, v])
                off_source = float(tensor[4, u, v])
                off_target = float(tensor[4, v, u])
                state_source = float(q_matrix[u, v])
                state_target = float(q_matrix[v, u])
                
                edges[edge_key] = {
                    "a": round(a, 10),
                    "alpha": round(alpha, 10),
                    "offset_source": round(off_source, 10),
                    "offset_target": round(off_target, 10),
                    "state_source": round(state_source, 10),
                    "state_target": round(state_target, 10)
                }

            export_data = {
                "data": {
                    "joints": joints,
                    "edges": edges
                },
                "settings": {
                    "root_node": str(meta_info.ground_nodes[0]) if meta_info.ground_nodes else "0",
                    "ee_node": str(meta_info.ee_node) if meta_info.ee_node else str(N-1)
                }
            }
            
            json_str = json.dumps(export_data, indent=2)
            print("📄 [Generated JSON Spec]:")
            print(json_str)
            
            filename = "final_mechanism_design.json"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"\n💾 结果已保存至: {os.path.abspath(filename)}")
            
        except Exception as e:
            print(f"❌ [Export Error] 导出结果失败: {e}")
            traceback.print_exc()