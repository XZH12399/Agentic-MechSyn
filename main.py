import hydra
from omegaconf import DictConfig, open_dict
from dotenv import load_dotenv
import os
import sys
from core.agent import MechanismAgent

# 加载 .env 文件中的环境变量
load_dotenv()

# ==========================================
# 🔧 API 服务商配置注册表
# 在这里定义不同服务商的 URL 和 对应的环境变量名
# ==========================================
PROVIDER_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat"
    },
    "v36": {  # 你图片中的新服务商
        # 注意：OpenAI Python SDK 通常需要在自定义 URL 后加 /v1
        "base_url": "https://free.v36.cm/v1",
        "api_key_env": "V36_API_KEY",
        "default_model": "gpt-4o-mini"  # 图片中勾选的模型
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o"
    }
}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 使用 open_dict 上下文管理器来允许修改 cfg 结构
    with open_dict(cfg):
        # 1. 获取当前想要使用的 provider (默认为 deepseek)
        # 优先读取 config.yaml 中的 llm.provider，如果没写则默认为 'deepseek'
        current_provider = getattr(cfg.llm, "provider", "deepseek")

        # 2. 获取该 provider 的配置详情
        provider_settings = PROVIDER_CONFIGS.get(current_provider)

        if not provider_settings:
            print(f"❌ 错误: 未知的 provider '{current_provider}'。请在 main.py 中配置。")
            sys.exit(1)

        print(f"🔄 [System] 正在切换 API 服务商: {current_provider}")

        # 3. 注入 Base URL
        # 如果 yaml 里没写 url，就用注册表里的默认值
        if hasattr(cfg.llm, "base_url") and cfg.llm.base_url:
            cfg.base_url = cfg.llm.base_url
        else:
            cfg.base_url = provider_settings["base_url"]

        # 4. 注入 API Key
        # 根据 provider 查找对应的环境变量 (例如 V36_API_KEY)
        env_var_name = provider_settings["api_key_env"]
        api_key = os.getenv(env_var_name)

        if not api_key:
            print(f"❌ 错误: 未找到环境变量 {env_var_name}。请在 .env 文件中设置。")
            sys.exit(1)

        cfg.api_key = api_key

        # 5. 注入 Model Name
        # 如果 yaml 里指定了 model_name，优先使用 yaml 的，否则使用 provider 的默认模型
        if hasattr(cfg.llm, "model_name") and cfg.llm.model_name:
            cfg.model_name = cfg.llm.model_name
        else:
            cfg.model_name = provider_settings["default_model"]

    # 打印最终配置以供检查 (脱敏)
    masked_key = cfg.api_key[:8] + "..." if cfg.api_key else "None"
    print(f"✅ [Config] URL: {cfg.base_url}")
    print(f"✅ [Config] Model: {cfg.model_name}")
    print(f"✅ [Config] Key: {masked_key}")

    # 初始化 Agent
    agent = MechanismAgent(cfg)

    # 模拟用户输入
    # user_input = "设计一个满足'一个沿z轴平移运动'条件的平面单环并联机构。"
    user_input = "设计一个满足'一个转动'条件的平面单环并联机构。"
    # user_input = "设计一个满足'一个自由度'条件的空间单环并联机构。"
    # user_input = "设计一个满足Bennett并联机构。"

    # 运行全流程
    agent.run_pipeline(user_input)


if __name__ == "__main__":
    main()