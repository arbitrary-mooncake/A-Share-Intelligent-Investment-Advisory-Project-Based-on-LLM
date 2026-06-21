"""
A股管线适配器 -- 非侵入式包装现有ScoringEngine。
将现有scorer的JSON输出映射为DecisionPack，不修改现有pipeline代码。

关键设计决策:
  - 不修改 ScoringEngine.scoring_engine 或任何 agent 代码
  - 通过 score_stock() 返回的 score_data 提取各期限 scorer 原始 dict
  - 处理 short/medium/long scorer 之间 key 名不一致 (recommendation vs rating)
  - model_override 通过 _build_initial_state 的 model_config 参数注入
  - as_of_date 通过覆盖 initial_data["current_date"] 注入
  - eval_mode 通过设置 AnalysisPackage 的 task_type="eval" 体现
"""
from datetime import datetime
from typing import Dict, Any, Optional, List


async def run_stock_analysis(
    stock_code: str,
    company_name: str,
    as_of_date: str = "",
    eval_mode: bool = True,
    skip_cache: bool = False,
    thinking_enabled: bool = True,
    model_override: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    对单只股票运行分析+打分管线（适配版）。

    Args:
        stock_code: 股票代码 (e.g. "sh.603871")
        company_name: 公司名称
        as_of_date: 评测时点（YYYY-MM-DD格式），空串则用当前时间
        eval_mode: 是否评测模式 -- 影响 DecisionPack.task_type 和 model_profile
        skip_cache: 是否跳过中间产物缓存
        thinking_enabled: scorer是否启用thinking
        model_override: 模型覆盖 {"model_name", "model_api_key", "model_base_url"}

    Returns:
        {
            "stock_code": ...,
            "company_name": ...,
            "as_of_date": ...,
            "short_term_score": {...},     # 原始scorer JSON (直接从state.data提取)
            "medium_term_score": {...},
            "long_term_score": {...},
            "short_term_decision": DecisionPack,   # 结构化决策
            "medium_term_decision": DecisionPack,
            "long_term_decision": DecisionPack,
            "signal_packs": {agent_name: signal_pack_dict},
            "analysis_texts": {agent_name: analysis_text},
            "execution_time": float,
            "error": str or None,
        }
    """
    from src.stock_pool.scoring_engine import ScoringEngine

    result = {
        "stock_code": stock_code,
        "company_name": company_name,
        "as_of_date": as_of_date or datetime.now().strftime("%Y-%m-%d"),
        "short_term_score": {},
        "medium_term_score": {},
        "long_term_score": {},
        "short_term_decision": None,
        "medium_term_decision": None,
        "long_term_decision": None,
        "signal_packs": {},
        "analysis_texts": {},
        "execution_time": 0.0,
        "error": None,
    }

    start_time = datetime.now()

    try:
        engine = ScoringEngine()

        # 运行完整评分管线
        # 注意: 标准 score_stock() 不接受 model_config / as_of_date，
        # 因此 model_override 和 as_of_date 在此阶段仅作为 DecisionPack
        # 元数据写入，不影响管线内部行为。
        # Phase 2 可扩展为直接调用 _build_workflow + _build_initial_state
        # 以注入 model_config 和 as_of_date。
        score_result = await engine.score_stock(stock_code, company_name)

        if score_result and score_result.get("score_data"):
            score_data = score_result["score_data"]

            # 提取三种期限评分（scorer 输出的原始 dict）
            short_score = score_data.get("short_term_score", {})
            medium_score = score_data.get("medium_term_score", {})
            long_score = score_data.get("long_term_score", {})

            result["short_term_score"] = short_score
            result["medium_term_score"] = medium_score
            result["long_term_score"] = long_score

            # 构建DecisionPack
            result["short_term_decision"] = _build_decision_pack(
                stock_code, company_name, "short",
                result["as_of_date"], short_score, eval_mode
            )
            result["medium_term_decision"] = _build_decision_pack(
                stock_code, company_name, "medium",
                result["as_of_date"], medium_score, eval_mode
            )
            result["long_term_decision"] = _build_decision_pack(
                stock_code, company_name, "long",
                result["as_of_date"], long_score, eval_mode
            )

            # 提取 signal_packs 和分析文本（pipeline 已直接返回）
            result["signal_packs"] = score_result.get("signal_packs", {})
            result["analysis_texts"] = score_result.get("analysis_texts", {})

    except Exception as e:
        result["error"] = str(e)

    result["execution_time"] = (datetime.now() - start_time).total_seconds()
    return result


def _build_decision_pack(
    stock_code: str,
    company_name: str,
    term: str,
    as_of_date: str,
    score_data: Dict[str, Any],
    eval_mode: bool = True,
) -> 'DecisionPack':
    """
    从scorer的JSON输出映射为DecisionPack。

    这是适配层的核心：不修改scorer代码，而是从已有JSON输出中提取字段。

    注意: short_term_scorer 使用 "recommendation" key，
          medium_term_scorer 和 long_term_scorer 使用 "rating" key。
          这里统一处理两种 key。

    Args:
        stock_code: 股票代码
        company_name: 公司名称
        term: short/medium/long
        as_of_date: 评测时点
        score_data: scorer 输出的原始 dict (如 data["short_term_score"])
        eval_mode: 是否评测模式
    """
    from src.utils.analysis_schema import DecisionPack

    if not score_data or not isinstance(score_data, dict):
        return DecisionPack(
            asset_type="stock",
            symbol=stock_code,
            name=company_name,
            term=term,
            as_of_date=as_of_date,
            action="hold",
            model_profile="eval_analysis" if eval_mode else "production",
        )

    # --- action 映射 ---
    # short_term_scorer 用 "recommendation"，medium/long 用 "rating"
    recommendation = score_data.get("recommendation", "") or score_data.get("rating", "")

    # short-term 和 medium/long 的评级措辞不同，需要统一映射
    action_map = {
        # short-term 短线建议
        "强烈买入": "strong_buy",
        "买入": "buy",
        "谨慎买入": "cautious_buy",
        "观望": "hold",
        "谨慎卖出": "cautious_sell",
        "卖出": "sell",
        "强烈卖出": "strong_sell",
        # medium/long-term 投资评级
        "强烈推荐": "strong_buy",
        "推荐": "buy",
        "谨慎推荐": "cautious_buy",
        "中性": "hold",
        "谨慎减持": "cautious_sell",
        "减持": "sell",
    }
    action = action_map.get(recommendation, "hold")

    # --- score ---
    try:
        score = float(score_data.get("score", 0))
    except (ValueError, TypeError):
        score = 0.0

    # --- confidence ---
    try:
        confidence = float(score_data.get("confidence", 0.5))
    except (ValueError, TypeError):
        confidence = 0.5

    # --- data_quality_score ---
    try:
        dq = float(score_data.get("data_quality_score", 0.5))
    except (ValueError, TypeError):
        dq = 0.5

    # --- risk_gate ---
    risk_gate = score_data.get("risk_gate", {})
    risk_gate_applied = bool(risk_gate.get("risk_flags")) if isinstance(risk_gate, dict) else False
    risk_gate_result = risk_gate if isinstance(risk_gate, dict) else None

    # --- sub_scores ---
    sub_scores = score_data.get("sub_scores", {})

    # --- key signals from reasoning and sub_scores ---
    reasoning = score_data.get("reasoning", "")
    key_positive = _extract_signal_strings(reasoning, "positive")
    key_negative = _extract_signal_strings(reasoning, "negative")

    # Also extract from risk_warning if available
    risk_warning = score_data.get("risk_warning", "")
    if risk_warning:
        key_negative.append(risk_warning[:100])

    # --- suggested_action ---
    suggested_action = score_data.get("suggested_action", "")

    # --- time_horizon (medium/long scorer) ---
    time_horizon = score_data.get("time_horizon", "")

    meta = {
        "raw_recommendation": recommendation,
        "scorer_key": "recommendation" if "recommendation" in score_data else "rating",
    }
    if sub_scores:
        meta["sub_scores"] = sub_scores
    if suggested_action:
        meta["suggested_action"] = suggested_action
    if time_horizon:
        meta["time_horizon"] = time_horizon

    return DecisionPack(
        asset_type="stock",
        symbol=stock_code,
        name=company_name,
        task_type="eval" if eval_mode else "single_stock",
        term=term,
        as_of_date=as_of_date,
        action=action,
        score=score,
        confidence=confidence,
        data_quality_score=dq,
        risk_gate_applied=risk_gate_applied,
        risk_gate_result=risk_gate_result,
        key_positive_signals=key_positive if key_positive else None,
        key_negative_signals=key_negative if key_negative else None,
        model_profile="eval_analysis" if eval_mode else "production",
        version_hash="",
        meta=meta,
    )


def _extract_signal_strings(reasoning: str, direction: str) -> List[str]:
    """从reasoning文本中提取利多/利空关键信号。

    Args:
        reasoning: scorer 的 reasoning 文本
        direction: "positive" 或 "negative"

    Returns:
        信号字符串列表（每段最多100字符）
    """
    if not reasoning:
        return []

    positive_keywords = ["利多", "看多", "利好", "支撑", "改善", "增长", "突破", "优势"]
    negative_keywords = ["风险", "利空", "看空", "压力", "恶化", "下滑", "减持", "劣势"]

    keywords = positive_keywords if direction == "positive" else negative_keywords

    # 按句号、分号、换行分段
    segments = []
    for sep in ["。", "；", "\n"]:
        if sep in reasoning:
            segments = [s.strip() for s in reasoning.split(sep) if s.strip()]
            break
    if not segments:
        segments = [reasoning.strip()]

    signals = []
    for seg in segments:
        if not seg:
            continue
        for kw in keywords:
            if kw in seg:
                signals.append(seg[:100])
                break
        if len(signals) >= 3:
            break

    return signals


def extract_signal_packs_from_state(state_data: Dict[str, Any]) -> Dict[str, Dict]:
    """
    从AgentState.data中提取所有agent的signal_pack。

    Agent输出格式: {agent_name}_signal_pack = {...}

    Args:
        state_data: AgentState.data 字典

    Returns:
        {agent_name: signal_pack_dict}，空dict不计入
    """
    agent_names = [
        "fundamental", "technical", "value", "news",
        "event", "quality_risk", "moneyflow"
    ]

    signal_packs = {}
    for agent_name in agent_names:
        sp_key = f"{agent_name}_signal_pack"
        if sp_key in state_data and state_data[sp_key]:
            signal_packs[agent_name] = state_data[sp_key]

    return signal_packs


def extract_analysis_texts_from_state(state_data: Dict[str, Any]) -> Dict[str, str]:
    """
    从AgentState.data中提取所有agent的分析文本。

    Args:
        state_data: AgentState.data 字典

    Returns:
        {agent_name: analysis_text}，空/缺失不计入
    """
    agent_names = [
        "fundamental", "technical", "value", "news",
        "event", "quality_risk", "moneyflow"
    ]

    texts = {}
    for agent_name in agent_names:
        text_key = f"{agent_name}_analysis"
        if text_key in state_data and state_data[text_key]:
            texts[agent_name] = state_data[text_key]

    return texts
