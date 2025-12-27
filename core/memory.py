import json
import os


class ExperienceManager:
    def __init__(self, memory_file="data/experience_pool.json"):
        self.memory_file = memory_file
        # 确保目录存在
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
        if not os.path.exists(memory_file):
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def retrieve_relevant(self, task_template, top_k=3):
        """
        [简易实现] 根据任务类型检索过往经验。
        实际项目中可以使用向量数据库 (Vector DB) 进行语义检索。
        """
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            # 简单过滤：只看相同自由度的案例
            target_dof = task_template.get("kinematics", {}).get("dof")
            relevant = [exp for exp in history if exp["task"]["kinematics"]["dof"] == target_dof]

            return relevant[-top_k:]  # 返回最近的 k 条
        except Exception:
            return []

    def store_experience(self, user_input, task_template, tensor_data, report):
        """
        存入经验池
        """
        new_entry = {
            "user_input": user_input,
            "task": task_template,
            "tensor_summary": str(tensor_data.shape),  # 实际存可能存张量路径
            "report": report
        }

        try:
            with open(self.memory_file, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data.append(new_entry)
                f.seek(0)
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("💾 [Memory] 经验已存入经验池。")
        except Exception as e:
            print(f"❌ [Memory] 存储失败: {e}")