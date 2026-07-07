"""
模型配置工具：将各子Agent映射到其分配的LLM模型。

双版本架构：
  Full 模式（默认）：六模型架构
    Model 1 (OPENAI_COMPATIBLE):      MiMo-V2.5-Pro  → summary, medium/long scorer, fundamental, value, quality_risk
    Model 2 (OPENAI_COMPATIBLE_2):    Qwen3.6-Flash  → 快速查询 + 快筛股票池（不动）
    Model 3 (OPENAI_COMPATIBLE_3):    Qwen3.7-Plus   → technical, news, short scorer, event, moneyflow
    Model 4 (OPENAI_COMPATIBLE_4):    Kimi K2.6      → (已迁移至 M1，当前无 Agent 分配)
    Model 5 (OPENAI_COMPATIBLE_5):    MiMo-V2.5       → 智能问答主模型
    Model 6 (OPENAI_COMPATIBLE_6):    DeepSeek V4 Pro → 评测系统运筹调度

  Lite 模式（APP_MODE=lite）：单模型架构
    所有 Agent → DeepSeek V4 Pro（DEEPSEEK_API_KEY）
    M2 角色（快速查询）→ DeepSeek Flash

优化调整 (2026-06-02):
  - news_agent: M4(Kimi K2.6) → M3(Qwen3.7-Plus)，新闻摘要无需深度推理，提速 5-10x
  - 各 Agent temperature 统一 1.0（Kimi K2.6 仅接受 temperature=1）
  - 提取 get_thinking_body() 统一 thinking 参数格式（DashScope/Qwen 用 enable_thinking，其余用 thinking.type）
"""
import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# 确保 .env 中的环境变量已加载到 os.environ
load_dotenv(override=True)

BASE_PREFIX = "OPENAI_COMPATIBLE"

# 每个 Agent 分配的环境变量后缀
AGENT_MODEL_SUFFIX: Dict[str, str] = {
    # ── Model 1: MiMo-V2.5-Pro (1M上下文, 深度推理) ──
    "summary_agent": "",
    "medium_term_scorer": "",
    "long_term_scorer": "",
    "fund_scoring_agent": "",         # MiMo-V2.5-Pro — best reasoning for scoring
    "fund_report_agent": "",          # MiMo-V2.5-Pro — best quality for final report
    "fund_perf_risk_agent": "",       # MiMo-V2.5-Pro — 1M context, deep quantitative analysis (500d NAV)
    "fund_holdings_agent": "",        # MiMo-V2.5-Pro — 1M context, 80K+ token input (10 holdings × 3 tools)

    # ── Model 3: Qwen3.7-Plus (快速工具调用，适合ReAct；新闻摘要；基金结构化分析) ──
    "technical_agent": "_3",
    "news_agent": "_3",
    # ── 新增 Agent (2026-06 架构升级) ──
    "event_analyst": "_3",               # Model 3: Qwen3.7-Plus — 事件/公告分析
    "quality_risk_analyst": "",             # Model 1: MiMo-V2.5-Pro — 财务质量/治理风险深度分析
    "moneyflow_analyst": "_3",           # Model 3: Qwen3.7-Plus — 资金面/量价确认
    "short_term_scorer": "_3",
    "fund_manager_agent": "_3",
    "fund_event_agent": "_3",
    "fund_fee_agent": "_3",           # Qwen3.7-Plus — structured fee analysis
    "fund_product_doc_agent": "_3",   # Qwen3.7-Plus — structured data parsing
    "fund_benchmark_agent": "_3",     # Qwen3.7-Plus + thinking=enabled — style drift analysis

    # ── Model 4: Kimi K2.6 (已迁移至 Model 1，当前无 Agent 使用) ──
    "fundamental_agent": "",
    "value_agent": "",

    # ── Model 2: Qwen3.6-Flash (快速查询 / 快筛) ──
    "quick_query": "_2",
    "quick_screen": "_2",

    # ── Model 5: MiMo-V2.5 (智能问答主模型) ──
    "qa_engine": "_5",
    # 复杂问题升级模型 → Model 1 (MiMo-V2.5-Pro)
    "qa_engine_pro": "",

    # ── Model 6: DeepSeek V4 Pro (评测运筹 / 投顾报告) ──
    "advisory_report_writer": "_6",
}


# ── Lite 模式 DeepSeek 配置 ──
def _get_deepseek_config_pro() -> Dict[str, str]:
    """DeepSeek V4 Pro（用于复杂推理任务）
    回退链：DEEPSEEK_API_KEY → M6 Key → M1 Key"""
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY")
                   or os.getenv(f"{BASE_PREFIX}_API_KEY_6")
                   or os.getenv(f"{BASE_PREFIX}_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_BASE_URL")
                    or os.getenv(f"{BASE_PREFIX}_BASE_URL_6")
                    or "https://api.deepseek.com/v1",
        "model_name": os.getenv("DEEPSEEK_MODEL_PRO", "deepseek-v4-pro"),
    }


def _get_deepseek_config_flash() -> Dict[str, str]:
    """DeepSeek V4 Flash（用于快速查询）
    回退链：DEEPSEEK_API_KEY → M6 Key → M1 Key"""
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY")
                   or os.getenv(f"{BASE_PREFIX}_API_KEY_6")
                   or os.getenv(f"{BASE_PREFIX}_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_BASE_URL")
                    or os.getenv(f"{BASE_PREFIX}_BASE_URL_6")
                    or "https://api.deepseek.com/v1",
        "model_name": os.getenv("DEEPSEEK_MODEL_FLASH", "deepseek-v4-flash"),
    }


# Lite 模式下使用 Flash 的 Agent（快速查询角色）
_LITE_FLASH_AGENTS = {"quick_query", "quick_screen"}


def _get_lite_model_config(agent_name: str) -> Dict[str, str]:
    """Lite 模式：所有 Agent 使用 DeepSeek"""
    if agent_name in _LITE_FLASH_AGENTS:
        return _get_deepseek_config_flash()
    return _get_deepseek_config_pro()


def get_thinking_body(base_url: str, enabled: bool = True) -> dict:
    """
    根据 API 提供商返回正确的 thinking 参数格式。

    Qwen/DashScope 使用 enable_thinking，其余（Kimi/MiMo/OpenAI 兼容）使用 thinking.type。
    各 Agent 统一调用此函数，不再各自硬编码。
    """
    if not enabled:
        return {"thinking": {"type": "disabled"}}
    if "dashscope" in base_url.lower():
        return {"enable_thinking": True}
    return {"thinking": {"type": "enabled"}}


# ── 评测系统模型Profiles（总纲 §13.3）──
# 通过 get_eval_model_config() 调用，不走 AGENT_MODEL_SUFFIX
EVAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "eval_analysis": {       # 评测分析Agent（日常高频）→ MiMo-V2.5
        "suffix": "_5",
        "thinking": True,
        "max_tokens": 8000,
        "timeout": 360,
    },
    "eval_orchestrator": {  # 评测运筹调度（按需）→ DeepSeek V4 Pro
        "suffix": "_6",
        "thinking": True,
        "max_tokens": 16000,
        "timeout": 600,
    },
    "eval_llm_free": {      # LLM自由投资线 → DeepSeek V4/V4.1
        "suffix": "_6",
        "thinking": True,
        "max_tokens": 16000,
        "timeout": 600,
    },
}


def get_eval_model_config(profile: str = "eval_analysis") -> Dict[str, str]:
    """
    获取评测系统的模型配置。

    Args:
        profile: 评测模型profile名称
            - "eval_analysis": 日常分析Agent（MiMo-V2.5, 性价比高）
            - "eval_orchestrator": 运筹调度（DeepSeek V4 Pro, 最强分析）
            - "eval_llm_free": LLM自由投资线（DeepSeek V4/V4.1）

    Returns: {"api_key": ..., "base_url": ..., "model_name": ...}
    """
    # Lite 模式：评测系统也使用 DeepSeek
    if os.getenv("APP_MODE", "full").strip().lower() == "lite":
        return _get_deepseek_config_pro()

    profile_cfg = EVAL_PROFILES.get(profile, {"suffix": "_5"})
    suffix = profile_cfg.get("suffix", "_5")

    api_key = os.getenv(f"{BASE_PREFIX}_API_KEY{suffix}", "")
    base_url = os.getenv(f"{BASE_PREFIX}_BASE_URL{suffix}", "")
    model_name = os.getenv(f"{BASE_PREFIX}_MODEL{suffix}", "")

    # 回退: 如果指定模型未配置 → 回退到Model 1
    if not all([api_key, base_url, model_name]):
        api_key = os.getenv(f"{BASE_PREFIX}_API_KEY", "")
        base_url = os.getenv(f"{BASE_PREFIX}_BASE_URL", "")
        model_name = os.getenv(f"{BASE_PREFIX}_MODEL", "mimo-v2.5-pro")

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
    }


def get_model_config_for_agent(
    agent_name: str,
    state_data: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    获取指定 Agent 的模型配置。

    优先级：
    1. state_data 中的覆盖（快筛模式用 Model 2 全量覆盖时生效）
    2. Lite 模式：所有 Agent 使用 DeepSeek
    3. Full 模式：Agent 在 AGENT_MODEL_SUFFIX 中分配的模型

    Returns: {"api_key": ..., "base_url": ..., "model_name": ...}
    """
    state_data = state_data or {}

    # Lite 模式：使用 DeepSeek 单一模型
    if os.getenv("APP_MODE", "full").strip().lower() == "lite":
        return _get_lite_model_config(agent_name)

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
