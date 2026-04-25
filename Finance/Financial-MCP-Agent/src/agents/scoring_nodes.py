"""
Scoring Nodes: LangGraph wrapper functions for the 3 scoring agents

3个打分 Agent 的函数签名是 scorer(stock_code, company_name, ...)，
不接受 AgentState。这个模块提供 wrapper 函数，
使它们能作为 LangGraph 节点使用。

工作流数据流：
    4个分析 Agent 并行 → state.data.{fundamental,technical,value,news}_analysis
    ↓
    打分 Agent 节点从 state 中提取中间产物 → 写入 state.data.{period}_term_score
"""
from typing import Dict, Any

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)


async def short_term_scorer_node(state: AgentState) -> Dict[str, Any]:
    """
    短线打分 LangGraph 节点

    依赖：technical_analysis, news_analysis
    输出：state.data.short_term_score
    """
    from src.agents.short_term_scorer import short_term_scorer

    data = state.get("data", {})
    stock_code = data.get("stock_code", "")
    company_name = data.get("company_name", "")

    logger.info(f"{WAIT_ICON} ShortTermScorerNode: 开始对 {company_name}({stock_code}) 进行短线打分")

    try:
        result = await short_term_scorer(
            stock_code=stock_code,
            company_name=company_name,
            technical_analysis=data.get("technical_analysis", ""),
            news_analysis=data.get("news_analysis", ""),
            current_time_info=data.get("current_time_info", ""),
            current_date=data.get("current_date", ""),
            query=data.get("query", ""),
        )

        logger.info(f"{SUCCESS_ICON} ShortTermScorerNode: {company_name} 短线评分={result['score']} ({result['recommendation']})")

        return {"data": {"short_term_score": result}}

    except Exception as e:
        logger.error(f"{ERROR_ICON} ShortTermScorerNode 打分失败: {e}", exc_info=True)
        raise


async def medium_term_scorer_node(state: AgentState) -> Dict[str, Any]:
    """
    中线打分 LangGraph 节点（核心产品）

    依赖：fundamental_analysis, technical_analysis, value_analysis, news_analysis
    输出：state.data.medium_term_score
    """
    from src.agents.medium_term_scorer import medium_term_scorer

    data = state.get("data", {})
    stock_code = data.get("stock_code", "")
    company_name = data.get("company_name", "")

    logger.info(f"{WAIT_ICON} MediumTermScorerNode: 开始对 {company_name}({stock_code}) 进行中线打分")

    try:
        result = await medium_term_scorer(
            stock_code=stock_code,
            company_name=company_name,
            fundamental_analysis=data.get("fundamental_analysis", ""),
            technical_analysis=data.get("technical_analysis", ""),
            value_analysis=data.get("value_analysis", ""),
            news_analysis=data.get("news_analysis", ""),
            current_time_info=data.get("current_time_info", ""),
            current_date=data.get("current_date", ""),
            query=data.get("query", ""),
        )

        logger.info(f"{SUCCESS_ICON} MediumTermScorerNode: {company_name} 中线评分={result['score']} ({result['rating']})")

        return {"data": {"medium_term_score": result}}

    except Exception as e:
        logger.error(f"{ERROR_ICON} MediumTermScorerNode 打分失败: {e}", exc_info=True)
        raise


async def long_term_scorer_node(state: AgentState) -> Dict[str, Any]:
    """
    长线打分 LangGraph 节点

    依赖：fundamental_analysis, technical_analysis, value_analysis, news_analysis
    输出：state.data.long_term_score
    """
    from src.agents.long_term_scorer import long_term_scorer

    data = state.get("data", {})
    stock_code = data.get("stock_code", "")
    company_name = data.get("company_name", "")

    logger.info(f"{WAIT_ICON} LongTermScorerNode: 开始对 {company_name}({stock_code}) 进行长线打分")

    try:
        result = await long_term_scorer(
            stock_code=stock_code,
            company_name=company_name,
            fundamental_analysis=data.get("fundamental_analysis", ""),
            technical_analysis=data.get("technical_analysis", ""),
            value_analysis=data.get("value_analysis", ""),
            news_analysis=data.get("news_analysis", ""),
            current_time_info=data.get("current_time_info", ""),
            current_date=data.get("current_date", ""),
            query=data.get("query", ""),
        )

        logger.info(f"{SUCCESS_ICON} LongTermScorerNode: {company_name} 长线评分={result['score']} ({result['rating']}) 护城河={result.get('moat_type', 'N/A')}")

        return {"data": {"long_term_score": result}}

    except Exception as e:
        logger.error(f"{ERROR_ICON} LongTermScorerNode 打分失败: {e}", exc_info=True)
        raise
