"""
Scoring Nodes: LangGraph wrapper for 3 scoring agents (v2 — structured evidence)

架构升级:
  - short_term: technical + news + event + moneyflow
  - medium_term + long_term: all 7 analysis agents
  - 每个node先构建AnalysisPackage → 传给scorer → apply risk_gate

  - 4.3 确定性打分: DETERMINISTIC_SCORER_ENABLED=1 时 scorer 由纯函数计算
    （冲突仲裁+解释懒加载），默认 0 保持 LLM scorer 路径不变
"""
import json
import os
import time
from typing import Dict, Any

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)


def _deterministic_scorer_enabled() -> bool:
    """4.3 确定性 scorer 开关（默认关=LLM 路径，离线比对达标后才应开启）。"""
    return os.getenv("DETERMINISTIC_SCORER_ENABLED", "0").strip() == "1"


async def _run_deterministic_scorer(
    term: str, pkg: Any, stock_code: str, company_name: str, is_etf: bool,
    run_id: str = "",
) -> Dict[str, Any]:
    """确定性打分路径：代码公式 + 冲突仲裁（有界折扣）+ 可选 LLM 解释。"""
    started_at = time.perf_counter()
    status = "failed"
    result: Dict[str, Any] = {}
    logger.info(
        f"确定性打分入口 term={term} stock_code={stock_code or 'unknown'} "
        f"run_id={run_id or 'unknown'} scorer_type=deterministic"
    )
    try:
        from src.utils.deterministic_scorer import (
            collect_signals, detect_material_conflicts, compute_score,
        )
        from src.utils.conflict_arbitration import arbitrate_conflicts

        signals = collect_signals(pkg)
        conflicts = detect_material_conflicts(signals)
        arbitration_on = os.getenv("CONFLICT_ARBITRATION_ENABLED", "1").strip() != "0"
        discounts = await arbitrate_conflicts(conflicts, enabled=arbitration_on)
        result = compute_score(term, signals, pkg, is_etf=is_etf, signal_discounts=discounts)

        from src.utils.score_explanation import explanation_enabled, generate_score_explanation
        if explanation_enabled():
            result["reasoning"] = await generate_score_explanation(term, result, pkg, company_name)

        if conflicts:
            logger.info(f"确定性打分[{term}]: {len(conflicts)} 组实质冲突, {len(discounts)} 条信号折扣")
        status = "success"
        return result
    finally:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            f"确定性打分出口 term={term} stock_code={stock_code or 'unknown'} "
            f"run_id={run_id or 'unknown'} scorer_type=deterministic status={status} "
            f"elapsed_ms={elapsed_ms:.1f} score={result.get('score', 'n/a')}"
        )


def _score_cache_contract(pkg: Any, gate: Any, required_agents: int) -> Dict[str, Any]:
    """Attach provenance needed before a persisted score may be reused elsewhere."""
    available = len(getattr(pkg, "available_agents", []) or [])
    coverage = min(1.0, available / required_agents) if required_agents else 0.0
    if getattr(gate, "abstain", False):
        return {
            "validity": "abstain",
            "coverage": coverage,
            "missing_core_fields": [],
            "missing_optional_fields": list(getattr(pkg, "missing_agents", []) or []),
        }
    if available == 0:
        return {
            "validity": "invalid",
            "coverage": 0.0,
            "missing_core_fields": ["analysis_evidence"],
            "missing_optional_fields": [],
        }
    return {
        "validity": "valid",
        "coverage": coverage,
        "missing_core_fields": [],
        "missing_optional_fields": list(getattr(pkg, "missing_agents", []) or []),
    }


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

        if _deterministic_scorer_enabled():
            result = await _run_deterministic_scorer(
                "short", pkg, stock_code, company_name, data.get("is_etf", False),
                run_id=data.get("analysis_timestamp", ""),
            )
        else:
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
                **_score_cache_contract(pkg, gate, 4),
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

        if _deterministic_scorer_enabled():
            result = await _run_deterministic_scorer(
                "medium", pkg, stock_code, company_name, data.get("is_etf", False),
                run_id=data.get("analysis_timestamp", ""),
            )
        else:
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
                **_score_cache_contract(pkg, gate, 7),
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

        if _deterministic_scorer_enabled():
            result = await _run_deterministic_scorer(
                "long", pkg, stock_code, company_name, data.get("is_etf", False),
                run_id=data.get("analysis_timestamp", ""),
            )
        else:
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
                **_score_cache_contract(pkg, gate, 7),
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
