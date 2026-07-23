"""
打分解释生成器（4.3 定稿：LLM 残余职责之一——只写解释，不碰数字）。

- 分数、维度分、评级由确定性 scorer 算定后作为既成事实传入；
- LLM 仅生成 reasoning 人话解释，输出不接受分数字段；
- 懒加载：池更新只存结构化分数（模板 reasoning），本模块仅在
  SCORE_EXPLANATION_ENABLED=1 时被调用；
- 调用失败则保留模板 reasoning，不产生错误分数。
"""
import os
from typing import Any, Dict

from src.utils.analysis_schema import AnalysisPackage
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

_TERM_NAMES = {"short": "短期(1-5个交易日)", "medium": "中期(1-3个月)", "long": "长期(6个月以上)"}


def explanation_enabled() -> bool:
    return os.getenv("SCORE_EXPLANATION_ENABLED", "0").strip() == "1"


async def generate_score_explanation(
    term: str,
    result: Dict[str, Any],
    pkg: AnalysisPackage,
    company_name: str = "",
) -> str:
    """为确定性分数生成自然语言解释。失败时返回模板 reasoning（兜底）。"""
    fallback = result.get("reasoning", "")
    from langchain_openai import ChatOpenAI
    from src.utils.model_config import get_model_config_for_agent, get_thinking_body

    model_cfg = get_model_config_for_agent("score_explainer")
    if not all([model_cfg.get("api_key"), model_cfg.get("base_url"), model_cfg.get("model_name")]):
        return fallback

    sub = result.get("sub_scores", {})
    sub_lines = "\n".join(f"- {k}: {v}" for k, v in sub.items())
    risk_flags = ", ".join(pkg.global_risk_flags) if pkg.global_risk_flags else "无"
    missing = ", ".join(result.get("missing_core_fields", [])) or "无"

    prompt = (
        f"你是投资分析报告撰写员。以下{_TERM_NAMES.get(term, term)}打分已由确定性模型计算完成，"
        f"分数和评级是既定事实，不得质疑、修改或重新评判。\n\n"
        f"股票: {company_name}\n"
        f"总分: {result.get('score')}（评级: {result.get('rating')}）\n"
        f"维度分:\n{sub_lines}\n"
        f"风险标签: {risk_flags}\n"
        f"缺失数据: {missing}\n\n"
        f"证据摘要:\n{pkg.compact_prompt_context[:3000]}\n\n"
        f"请撰写 150-250 字的打分理由说明，要求：\n"
        f"1. 解释为什么是这个分数——结合维度分和关键证据；\n"
        f"2. 引用具体证据（因子名、数据），不编造数字；\n"
        f"3. 如存在跨维度分歧（如技术弱/基本面强），如实呈现而非调和；\n"
        f"4. 只输出说明文字本身，不要输出分数、JSON 或标题。"
    )

    try:
        llm = ChatOpenAI(
            model=model_cfg["model_name"],
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
            temperature=0.4,
            request_timeout=90,
            max_tokens=2000,
            extra_body=get_thinking_body(model_cfg["base_url"], enabled=False),
        )
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        text = response.content.strip() if hasattr(response, "content") else str(response)
        return text if text else fallback
    except Exception as e:
        logger.warning(f"打分解释生成失败（保留模板 reasoning）: {e}")
        return fallback
