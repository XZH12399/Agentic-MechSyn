import json
import re  # ✨ 新增: 用于正则清洗
from openai import OpenAI

# 导入数据结构 (Schema)
from .schemas import TaskTemplate
# 导入提示词
from .prompt_templates import TASK_PARSING_SYSTEM_PROMPT, STRUCTURED_OUTPUT_SUFFIX


class TaskParser:
    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def parse(self, user_query: str) -> dict:
        """
        将自然语言转换为结构化 TaskTemplate 字典 (兼容 DeepSeek)
        """
        print(f"📋 [TaskParser] 正在解析用户意图: '{user_query}'...", flush=True)  # ✨ flush=True 强制刷新日志

        try:
            # 1. 准备 Schema 字符串
            schema_str = json.dumps(TaskTemplate.model_json_schema(), indent=2, ensure_ascii=False)

            # 2. 组合 System Prompt
            full_system_prompt = TASK_PARSING_SYSTEM_PROMPT + STRUCTURED_OUTPUT_SUFFIX.format(schema=schema_str)

            # 3. 调用 API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_query}
                ],
                response_format={"type": "json_object"}
            )

            # 4. 获取内容
            content = response.choices[0].message.content

            # === ✨ 核心调试：打印原始回复 (防止卡住不知道发生了什么) ===
            print(f"\n🔍 [Debug] LLM 原始回复 (Raw Content):\n{content}\n", flush=True)

            # === ✨ 核心修复：自动去除 Markdown 代码块 ===
            content = self._clean_markdown(content)

            # 5. 校验与返回
            task_template = TaskTemplate.model_validate_json(content)

            print("\n" + "=" * 40)
            print("📝 [生成的任务模板 (Task Template)]")
            print("=" * 40)
            print(json.dumps(task_template.model_dump(), indent=2, ensure_ascii=False))
            print("=" * 40 + "\n", flush=True)

            print(f"✅ [TaskParser] 解析完成。DoF: {task_template.kinematics.dof}", flush=True)

            return task_template.model_dump()

        except Exception as e:
            print(f"❌ [TaskParser] 解析失败: {e}", flush=True)
            import traceback
            traceback.print_exc()  # 打印完整堆栈
            return self._get_fallback_template()

    def _clean_markdown(self, text: str) -> str:
        """
        工具函数：去除 LLM 输出可能包含的 Markdown 代码块标记 (```json ... ```)
        """
        if not text:
            return ""

        # 匹配 ```json ... ``` 或 ``` ... ```
        pattern = r"```(?:json)?\s*(.*)\s*```"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            # 如果匹配到，返回中间的内容
            return match.group(1).strip()
        else:
            # 如果没匹配到，说明可能是纯文本，直接返回
            return text.strip()

    def _get_fallback_template(self):
        return {
            "meta": {"error": "Parsing failed"},
            "kinematics": {"dof": 1, "space_type": "planar"},
            "constraints": {"num_links_max": None}
        }