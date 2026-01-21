import json
import traceback
import re
from .schemas import TaskTemplate
from .prompt_templates import (
    TASK_PARSING_SYSTEM_PROMPT,
    STRUCTURED_OUTPUT_SUFFIX
)

class TaskParser:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def parse_task(self, user_input: str) -> TaskTemplate:
        """
        核心方法: 解析用户输入，生成包含 VLA 指令的任务模板
        """
        print(f"🕵️ [TaskParser] Analyzing user intent...")
        
        try:
            # 1. 构造完整的 System Prompt
            schema_str = json.dumps(TaskTemplate.model_json_schema(), indent=2, ensure_ascii=False)
            system_prompt = TASK_PARSING_SYSTEM_PROMPT + STRUCTURED_OUTPUT_SUFFIX.format(schema=schema_str)

            # 2. 调用 LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User Requirement: {user_input}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1 
            )
            
            content = response.choices[0].message.content
            
            # 3. 解析 JSON
            data = json.loads(content)
            
            # =====================================================
            # ✨✨✨ 自动纠错逻辑 (Auto-Correction) ✨✨✨
            # 防止 LLM 混淆 Twist 的 [w, v] 顺序
            # =====================================================
            if 'targets' in data and data['targets'].get('target_motion_twists'):
                twists = data['targets']['target_motion_twists']
                desc = data['targets'].get('description', '').lower()
                intent = data.get('user_intent_summary', '').lower()
                full_context = desc + " " + intent
                
                # 规则：如果文本明确说了 "translation" (平移) 且没有 "rotation" (旋转)
                # 但向量的前三位 (w) 却有值，说明搞反了
                if ("translat" in full_context or "平移" in full_context) and \
                   ("rotat" not in full_context and "旋转" not in full_context and "screw" not in full_context):
                    
                    for i, tw in enumerate(twists):
                        if len(tw) == 6:
                            w_norm = sum([abs(x) for x in tw[:3]])
                            v_norm = sum([abs(x) for x in tw[3:]])
                            
                            # 如果角速度非零，线速度接近零 -> 肯定是填反了
                            if w_norm > 0.1 and v_norm < 0.01:
                                print(f"⚠️ [TaskParser] 检测到螺旋向量格式错误 (Rotation vs Translation). 自动修正中...")
                                print(f"   原向量: {tw}")
                                # 交换 w 和 v: [w, v] -> [0, w] (假设是纯平移)
                                # 实际上是把前三位搬到后三位
                                fixed_tw = [0.0, 0.0, 0.0, tw[0], tw[1], tw[2]]
                                data['targets']['target_motion_twists'][i] = fixed_tw
                                print(f"   修正后: {fixed_tw}")

            # 4. 校验与转换
            task_data = TaskTemplate.model_validate(data)
            
            # Print logs...
            print(f"    - Intent Summary: {task_data.user_intent_summary}")
            print(f"    - Kinematics:     {task_data.kinematics.dof} DoF, {task_data.kinematics.space_type.title()} Space")
            
            special_cons = task_data.constraints.special_constraints
            if special_cons:
                print(f"    - ⚠️ Constraints:  {special_cons}")
            else:
                print(f"    - Constraints:    None")
                
            print(f"    - Structure:      {task_data.constraints.num_loops} Loop(s)")
            
            if hasattr(task_data, 'vla_instruction'):
                print(f"    - VLA Instruction: {task_data.vla_instruction}")
            
            return task_data

        except Exception as e:
            print(f"❌ [TaskParser Error]: {e}")
            traceback.print_exc()
            raise e

    def _clean_markdown(self, text: str) -> str:
        if not text:
            return ""
        pattern = r"```(?:json)?\s*(.*)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return text.strip()