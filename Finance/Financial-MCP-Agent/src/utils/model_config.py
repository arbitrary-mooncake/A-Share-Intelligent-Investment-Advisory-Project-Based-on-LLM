"""
模型配置工具：将各子Agent映射到其分配的LLM模型。

.env 五模型架构：
  Model 1 (OPENAI_COMPATIBLE):      MiMo-V2.5-Pro  → summary, medium/long scorer
  Model 2 (OPENAI_COMPATIBLE_2):    Qwen3.6-Flash  → 快速查询 + 快筛股票池（不动）
  Model 3 (OPENAI_COMPATIBLE_3):    Qwen3.6-Plus   → technical, news, short scorer
  Model 4 (OPENAI_COMPATIBLE_4):    Kimi K2.6      → fundamental, value
  Model 5 (OPENAI_COMPATIBLE_5):    MiMo-V2.5       → 智能问答主模型
"""
import os
from typing import Dict, Optional

from dotenv import load_dotenv

# 确保 .env 中的环境变量已加载到 os.environ
load_dotenv(override=True)

BASE_PREFIX = "OPENAI_COMPATIBLE"

# 每个 Agent 分配的环境变量后缀
# 空字符串 = Model 1, "_3" = Model 3, "_4" = Model 4
AGENT_MODEL_SUFFIX: Dict[str, str] = {
    # ── Model 1: MiMo-V2.5-Pro ──
    "summary_agent": "",
    "medium_term_scorer": "",
    "long_term_scorer": "",

    # ── Model 3: Qwen3.6-Plus ──
    "technical_agent": "_3",
    "news_agent": "_3",
    "short_term_scorer": "_3",

    # ── Model 4: Kimi K2.6 ──
    "fundamental_agent": "_4",
    "value_agent": "_4",

    # ── Model 5: MiMo-V2.5 (智能问答主模型) ──
    "qa_engine": "_5",
}


def get_model_config_for_agent(
    agent_name: str,
    state_data: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    获取指定 Agent 的模型配置。

    优先级：
    1. state_data 中的覆盖（快筛模式用 Model 2 全量覆盖时生效）
    2. Agent 在 AGENT_MODEL_SUFFIX 中分配的模型

    Returns: {"api_key": ..., "base_url": ..., "model_name": ...}
    """
    state_data = state_data or {}

    # 检查 state 覆盖（快筛模式传入的 model_name / model_api_key / model_base_url）
    override_model = state_data.get("model_name", "")
    override_key = state_data.get("model_api_key", "")
    override_url = state_data.get("model_base_url", "")

    if override_model and override_key and override_url:
        return {
            "api_key": override_key,
            "base_url": override_url,
            "model_name": override_model,
        }

    # 使用 Agent 分配的模型
    # .env 命名: OPENAI_COMPATIBLE_{API_KEY,BASE_URL,MODEL}{suffix}
    #   如 suffix=""  → OPENAI_COMPATIBLE_API_KEY
    #   如 suffix="_3" → OPENAI_COMPATIBLE_API_KEY_3
    suffix = AGENT_MODEL_SUFFIX.get(agent_name, "")

    api_key = os.getenv(f"{BASE_PREFIX}_API_KEY{suffix}", "")
    base_url = os.getenv(f"{BASE_PREFIX}_BASE_URL{suffix}", "")
    model_name = os.getenv(f"{BASE_PREFIX}_MODEL{suffix}", "")

    # 如果分配的模型未配置，回退到 Model 1
    if not all([api_key, base_url, model_name]) and suffix:
        api_key = os.getenv(f"{BASE_PREFIX}_API_KEY", "")
        base_url = os.getenv(f"{BASE_PREFIX}_BASE_URL", "")
        model_name = os.getenv(f"{BASE_PREFIX}_MODEL", "mimo-v2.5-pro")

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
    }
