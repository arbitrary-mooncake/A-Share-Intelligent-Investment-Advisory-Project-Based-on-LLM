"""
Scoring Nodes: LangGraph wrapper for 3 scoring agents (v2 — structured evidence)

架构升级:
  - short_term: technical + news + event + moneyflow
  - medium_term + long_term: all 7 analysis agents
  - 每个node先构建AnalysisPackage → 传给scorer → apply risk_gate
"""
import json
from typing import Dict, Any

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)


async def short_term_scorer_node(state: AgentState) -> Dict[str, Any]:
    from src.agents.short_term_scorer import short_term_scorer
    from src.utils.analysis_package_builder import build_analysis_package
    from src.utils.risk_gate import apply_risk_gate
    from src.utils.cache_utils import read_cache, write_cache

    data = state.get("data", {})
    stock_code = data.get("stock_code", "")
    company_name = data.get("company_name", "")
    as_of_date = data.get("current_date", "")
    skip_cache = data.get("skip_cache", False)

    logger.info(f"{WAIT_ICON} ShortTermScorerNode: {company_name}({stock_code}) 短线打分")

    try:
        # 缓存检查: 1天TTL，同一天同股票不重复打分
        if not skip_cache and stock_code and as_of_date:
            cached = read_cache("short_term_scorer", stock_code, as_of_date)
            if cached:
                result = json.loads(cached)
                pkg = build_analysis_package(data, as_of_date)
                gate = apply_risk_gate(pkg, "short", result["score"])
                if gate.score_cap is not None:
                    result["score"] = min(result["score"], gate.score_cap)
                result["risk_gate"] = {
                    "risk_level": gate.risk_level,
                    "risk_flags": gate.risk_flags_found,
                    "score_cap": gate.score_cap,
                    "abstain": gate.abstain,
                    "data_quality_score": gate.data_quality_score,
                }
                logger.info(f"{SUCCESS_ICON} ShortTermScorerNode: {company_name} 短线={result['score']} (缓存命中)")
                return {"data": {"short_term_score": result}}

        pkg = build_analysis_package(data, as_of_date)

        result = await short_term_scorer(
            stock_code=stock_code, company_name=company_name,
            technical_analysis=data.get("technical_analysis", ""),
            news_analysis=data.get("news_analysis", ""),
            event_analysis=data.get("event_analysis", ""),
            moneyflow_analysis=data.get("moneyflow_analysis", ""),
            analysis_package=pkg,
            current_time_info=data.get("current_time_info", ""),
            current_date=as_of_date,
            query=data.get("query", ""),
            model_name=data.get("model_name", ""),
            model_api_key=data.get("model_api_key", ""),
            model_base_url=data.get("model_base_url", ""),
            thinking_enabled=data.get("thinking_enabled", True),
        )

        gate = apply_risk_gate(pkg, "short", result["score"])
        if gate.score_cap is not None:
            result["score"] = min(result["score"], gate.score_cap)
        result["risk_gate"] = {
            "risk_level": gate.risk_level,
            "risk_flags": gate.risk_flags_found,
            "score_cap": gate.score_cap,
            "abstain": gate.abstain,
            "data_quality_score": gate.data_quality_score,
        }

        # 写入缓存 (1天TTL，与子Agent最短TTL一致)
        if not skip_cache and stock_code and as_of_date:
            write_cache("short_term_scorer", stock_code, as_of_date, json.dumps({
                "score": result["score"],
                "reasoning": result.get("reasoning", ""),
                "recommendation": result.get("recommendation", ""),
                "sub_scores": result.get("sub_scores", {}),
                "confidence": result.get("confidence", 0),
                "suggested_action": result.get("suggested_action", ""),
                "risk_warning": result.get("risk_warning", ""),
                "data_quality_score": result.get("data_quality_score", 0),
            }, ensure_ascii=False))

        logger.info(f"{SUCCESS_ICON} ShortTermScorerNode: {company_name} 短线={result['score']} (gate={gate.risk_level})")
        return {"data": {"short_term_score": result}}

    except Exception as e:
        logger.error(f"{ERROR_ICON} ShortTermScorerNode 失败: {e}", exc_info=True)
        raise


async def medium_term_scorer_node(state: AgentState) -> Dict[str, Any]:
    from src.agents.medium_term_scorer import medium_term_scorer
    from src.utils.analysis_package_builder import build_analysis_package
    from src.utils.risk_gate import apply_risk_gate
    from src.utils.cache_utils import read_cache, write_cache

    data = state.get("data", {})
    stock_code = data.get("stock_code", "")
    company_name = data.get("company_name", "")
    as_of_date = data.get("current_date", "")
    skip_cache = data.get("skip_cache", False)

    logger.info(f"{WAIT_ICON} MediumTermScorerNode: {company_name}({stock_code}) 中线打分")

    try:
        if not skip_cache and stock_code and as_of_date:
            cached = read_cache("medium_term_scorer", stock_code, as_of_date)
            if cached:
                result = json.loads(cached)
                pkg = build_analysis_package(data, as_of_date)
                gate = apply_risk_gate(pkg, "medium", result["score"])
                if gate.score_cap is not None:
                    result["score"] = min(result["score"], gate.score_cap)
                result["risk_gate"] = {
                    "risk_level": gate.risk_level,
                    "risk_flags": gate.risk_flags_found,
                    "score_cap": gate.score_cap,
                    "abstain": gate.abstain,
                    "data_quality_score": gate.data_quality_score,
                }
                logger.info(f"{SUCCESS_ICON} MediumTermScorerNode: {company_name} 中线={result['score']} (缓存命中)")
                return {"data": {"medium_term_score": result}}

        pkg = build_analysis_package(data, as_of_date)

        result = await medium_term_scorer(
            stock_code=stock_code, company_name=company_name,
            fundamental_analysis=data.get("fundamental_analysis", ""),
            technical_analysis=data.get("technical_analysis", ""),
            value_analysis=data.get("value_analysis", ""),
            news_analysis=data.get("news_analysis", ""),
            event_analysis=data.get("event_analysis", ""),
            quality_risk_analysis=data.get("quality_risk_analysis", ""),
            moneyflow_analysis=data.get("moneyflow_analysis", ""),
            analysis_package=pkg,
            current_time_info=data.get("current_time_info", ""),
            current_date=as_of_date,
            query=data.get("query", ""),
            model_name=data.get("model_name", ""),
            model_api_key=data.get("model_api_key", ""),
            model_base_url=data.get("model_base_url", ""),
            thinking_enabled=data.get("thinking_enabled", True),
        )

        gate = apply_risk_gate(pkg, "medium", result["score"])
        if gate.score_cap is not None:
            result["score"] = min(result["score"], gate.score_cap)
        result["risk_gate"] = {
            "risk_level": gate.risk_level,
            "risk_flags": gate.risk_flags_found,
            "score_cap": gate.score_cap,
            "abstain": gate.abstain,
            "data_quality_score": gate.data_quality_score,
        }

        if not skip_cache and stock_code and as_of_date:
            write_cache("medium_term_scorer", stock_code, as_of_date, json.dumps({
                "score": result["score"],
                "reasoning": result.get("reasoning", ""),
                "rating": result.get("rating", ""),
                "sub_scores": result.get("sub_scores", {}),
                "confidence": result.get("confidence", 0),
                "time_horizon": result.get("time_horizon", ""),
                "suggested_action": result.get("suggested_action", ""),
                "risk_warning": result.get("risk_warning", ""),
                "data_quality_score": result.get("data_quality_score", 0),
            }, ensure_ascii=False))

        logger.info(f"{SUCCESS_ICON} MediumTermScorerNode: {company_name} 中线={result['score']} (gate={gate.risk_level})")
        return {"data": {"medium_term_score": result}}

    except Exception as e:
        logger.error(f"{ERROR_ICON} MediumTermScorerNode 失败: {e}", exc_info=True)
        raise


async def long_term_scorer_node(state: AgentState) -> Dict[str, Any]:
    from src.agents.long_term_scorer import long_term_scorer
    from src.utils.analysis_package_builder import build_analysis_package
    from src.utils.risk_gate import apply_risk_gate
    from src.utils.cache_utils import read_cache, write_cache

    data = state.get("data", {})
    stock_code = data.get("stock_code", "")
    company_name = data.get("company_name", "")
    as_of_date = data.get("current_date", "")
    skip_cache = data.get("skip_cache", False)

    logger.info(f"{WAIT_ICON} LongTermScorerNode: {company_name}({stock_code}) 长线打分")

    try:
        if not skip_cache and stock_code and as_of_date:
            cached = read_cache("long_term_scorer", stock_code, as_of_date)
            if cached:
                result = json.loads(cached)
                pkg = build_analysis_package(data, as_of_date)
                gate = apply_risk_gate(pkg, "long", result["score"])
                if gate.score_cap is not None:
                    result["score"] = min(result["score"], gate.score_cap)
                result["risk_gate"] = {
                    "risk_level": gate.risk_level,
                    "risk_flags": gate.risk_flags_found,
                    "score_cap": gate.score_cap,
                    "abstain": gate.abstain,
                    "data_quality_score": gate.data_quality_score,
                }
                logger.info(f"{SUCCESS_ICON} LongTermScorerNode: {company_name} 长线={result['score']} (缓存命中)")
                return {"data": {"long_term_score": result}}

        pkg = build_analysis_package(data, as_of_date)

        result = await long_term_scorer(
            stock_code=stock_code, company_name=company_name,
            fundamental_analysis=data.get("fundamental_analysis", ""),
            technical_analysis=data.get("technical_analysis", ""),
            value_analysis=data.get("value_analysis", ""),
            news_analysis=data.get("news_analysis", ""),
            event_analysis=data.get("event_analysis", ""),
            quality_risk_analysis=data.get("quality_risk_analysis", ""),
            moneyflow_analysis=data.get("moneyflow_analysis", ""),
            analysis_package=pkg,
            current_time_info=data.get("current_time_info", ""),
            current_date=as_of_date,
            query=data.get("query", ""),
            model_name=data.get("model_name", ""),
            model_api_key=data.get("model_api_key", ""),
            model_base_url=data.get("model_base_url", ""),
            thinking_enabled=data.get("thinking_enabled", True),
        )

        gate = apply_risk_gate(pkg, "long", result["score"])
        if gate.score_cap is not None:
            result["score"] = min(result["score"], gate.score_cap)
        result["risk_gate"] = {
            "risk_level": gate.risk_level,
            "risk_flags": gate.risk_flags_found,
            "score_cap": gate.score_cap,
            "abstain": gate.abstain,
            "data_quality_score": gate.data_quality_score,
        }

        if not skip_cache and stock_code and as_of_date:
            write_cache("long_term_scorer", stock_code, as_of_date, json.dumps({
                "score": result["score"],
                "reasoning": result.get("reasoning", ""),
                "rating": result.get("rating", ""),
                "sub_scores": result.get("sub_scores", {}),
                "confidence": result.get("confidence", 0),
                "time_horizon": result.get("time_horizon", ""),
                "suggested_action": result.get("suggested_action", ""),
                "risk_warning": result.get("risk_warning", ""),
                "data_quality_score": result.get("data_quality_score", 0),
            }, ensure_ascii=False))

        logger.info(f"{SUCCESS_ICON} LongTermScorerNode: {company_name} 长线={result['score']} (gate={gate.risk_level})")
        return {"data": {"long_term_score": result}}

    except Exception as e:
        logger.error(f"{ERROR_ICON} LongTermScorerNode 失败: {e}", exc_info=True)
        raise
