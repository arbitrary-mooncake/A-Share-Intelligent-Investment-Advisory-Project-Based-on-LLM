"""
确定性打分器（4.3）：三期限独立纯函数 scorer。

设计原则（终极优化策略.md 4.3 定稿）：
- 分数 100% 由代码计算：维度分 = 50 + Σ(direction × strength_eff × confidence
  × source_weight) × scale，总分 = Σ(维度分 × 维度权重) 归一化；
- 权重表原样继承现有 LLM scorer prompt 中的定义（v1 目标是 1:1 搬迁规则）；
- LLM 不碰数字：冲突仲裁只产出有界折扣系数（见 conflict_arbitration.py），
  解释文字可懒加载（见 score_explanation.py）；
- ETF 期望缺席：fundamental/value Agent 对 ETF early-exit，对应维度按
  期望缺席处理（跳过 + 剩余维度权重归一），不触发缺失惩罚；
- strength 字段质量策略可配（4.9-11 待离线报告定口径，默认 default5）；
- 输出契约与现有 LLM scorer 缓存格式逐字段一致，下游零改动。

可重复性承诺：同样的 signal_pack 输入必然得到同样的分数（纯函数）。
"""
import os
from typing import Any, Dict, List, Optional

from src.utils.analysis_schema import SOURCE_PRIORITY, AnalysisPackage
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# 权重版本：离线比对调权时递增，缓存键携带（4.3 定稿）
WEIGHTS_VERSION = "det_v1"

# ── 期限维度定义（权重原样继承现有 scorer prompt） ──
# 每项: (维度key, 维度中文名, 权重, 归属category列表)
TERM_DIMENSIONS: Dict[str, List[Dict[str, Any]]] = {
    "short": [
        {"key": "technical_trend", "name": "技术状态", "weight": 25, "categories": ["technical_trend"]},
        {"key": "liquidity", "name": "量价/流动性", "weight": 20, "categories": ["liquidity"]},
        {"key": "capital_flow", "name": "资金确认", "weight": 20, "categories": ["capital_flow"]},
        {"key": "catalyst_event", "name": "事件催化", "weight": 20, "categories": ["catalyst_event"]},
        {"key": "sentiment", "name": "新闻叙事/情绪", "weight": 15, "categories": ["sentiment"]},
    ],
    "medium": [
        {"key": "fundamentals", "name": "基本面", "weight": 20, "categories": ["fundamentals_growth", "fundamentals_profit_quality"]},
        {"key": "valuation", "name": "估值", "weight": 15, "categories": ["valuation"]},
        {"key": "quality", "name": "质量风险", "weight": 20, "categories": ["balance_sheet", "cashflow", "governance"]},
        {"key": "catalyst_event", "name": "事件催化", "weight": 15, "categories": ["catalyst_event"]},
        {"key": "technical_trend", "name": "技术面", "weight": 10, "categories": ["technical_trend", "capital_flow", "liquidity"]},
        {"key": "industry_policy", "name": "行业与政策", "weight": 10, "categories": ["industry_policy"]},
        {"key": "sentiment", "name": "情绪面", "weight": 10, "categories": ["sentiment"]},
    ],
    "long": [
        {"key": "returns", "name": "股东回报/成长", "weight": 25, "categories": ["fundamentals_growth", "dividend"]},
        {"key": "quality", "name": "盈利质量", "weight": 20, "categories": ["fundamentals_profit_quality", "balance_sheet", "cashflow"]},
        {"key": "valuation", "name": "估值安全边际", "weight": 15, "categories": ["valuation"]},
        {"key": "moat", "name": "护城河/治理", "weight": 15, "categories": ["governance"]},
        {"key": "capital", "name": "资本结构", "weight": 10, "categories": ["capital_flow", "ownership"]},
        {"key": "policy", "name": "政策环境", "weight": 10, "categories": ["industry_policy"]},
        {"key": "technical_trend", "name": "技术趋势", "weight": 5, "categories": ["technical_trend"]},
    ],
}

# 各期限打分所需的核心 Agent（用于缺失惩罚与 coverage）
TERM_REQUIRED_AGENTS: Dict[str, List[str]] = {
    "short": ["technical", "news", "event", "moneyflow"],
    "medium": ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"],
    "long": ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"],
}

# ETF 期望缺席的维度（fundamental/value Agent 对 ETF early-exit，4.9-3 已确认）：
# 维度无信号且在 droppable 集合内 → 跳过该维度并按剩余维度权重归一，不触发缺失惩罚。
ETF_DROPPABLE_DIMENSIONS: Dict[str, List[str]] = {
    "short": [],
    "medium": ["fundamentals", "valuation"],
    "long": ["returns", "quality", "valuation"],
}

# ETF 期望缺席的 Agent（从核心 Agent 集合中剔除，不计入缺失）
ETF_EXEMPT_AGENTS = {"fundamental", "value"}

# 来源可信度权重
SOURCE_WEIGHTS: Dict[str, float] = {
    "official_like": 1.0,
    "structured": 0.9,
    "news": 0.7,
    "derived": 0.5,
    "proxy": 0.4,
}

# 评级映射表（4.9-4 已确认初值）
RATING_TABLE = [
    (80, "强烈买入"),
    (65, "买入"),
    (50, "中性"),
    (35, "减持"),
    (0, "回避"),
]

SUGGESTED_ACTION_TABLE = {
    "强烈买入": "积极介入",
    "买入": "轻仓参与",
    "中性": "观望",
    "减持": "减仓回避",
    "回避": "坚决回避",
}

# strength=0 处理策略（4.9-11：最终口径待离线报告确认，此处可配）
# - raw: 原样使用（direction≠0 且 strength=0 的信号贡献为 0）
# - default5: 按缺失兜底为 5（默认）
# - confidence_scaled: 按 confidence × 10 折算
STRENGTH_ZERO_POLICY = os.getenv("DETERMINISTIC_SCORER_STRENGTH_POLICY", "default5")

# 维度缩放系数：信号贡献 → 维度分的换算比例
DIMENSION_SCALE = float(os.getenv("DETERMINISTIC_SCORER_DIMENSION_SCALE", "2.0"))


def _normalize_strength(sig: Dict[str, Any]) -> float:
    """把 0-100 量纲的 strength 归一到 0-10，并按策略处理 strength=0 的异常信号。"""
    try:
        raw = int(sig.get("strength", 50))
    except (ValueError, TypeError):
        raw = 50
    direction = sig.get("direction", 0)
    if raw == 0 and direction != 0:
        if STRENGTH_ZERO_POLICY == "raw":
            return 0.0
        if STRENGTH_ZERO_POLICY == "confidence_scaled":
            try:
                return max(1.0, min(10.0, float(sig.get("confidence", 0.5)) * 10))
            except (ValueError, TypeError):
                return 5.0
        return 5.0  # default5
    return max(0.0, min(10.0, raw / 10.0))


def _signal_contribution(sig: Dict[str, Any]) -> float:
    """单条信号对维度分的贡献（未乘维度缩放系数）。"""
    direction = sig.get("direction", 0)
    if direction == 0:
        return 0.0
    s10 = _normalize_strength(sig)
    try:
        confidence = float(sig.get("confidence", 0.5))
    except (ValueError, TypeError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    src_w = SOURCE_WEIGHTS.get(sig.get("source_level", "proxy"), 0.4)
    return direction * s10 * confidence * src_w


def map_score_to_rating(score: float) -> str:
    """分数 → 评级文案（4.9-4 映射表）。"""
    for threshold, label in RATING_TABLE:
        if score >= threshold:
            return label
    return "回避"


def collect_signals(pkg: AnalysisPackage) -> List[Dict[str, Any]]:
    """从 AnalysisPackage 收集全部多头+空头信号（中性信号贡献为 0，不参与）。"""
    signals: List[Dict[str, Any]] = []
    for s in (pkg.bullish_signals or []) + (pkg.bearish_signals or []):
        if isinstance(s, dict):
            signals.append(s)
    return signals


def detect_material_conflicts(
    signals: List[Dict[str, Any]],
    min_strength_eff: float = 5.0,
    min_impact: float = 1.0,
) -> List[Dict[str, Any]]:
    """纯代码实质冲突检测（4.3 定稿：枚举归组 + 数值比较，无词典无正则）。

    触发条件（4.9-6）：同 category、direction 相反、双方 strength_eff ≥ 阈值、
    source_level 不同、单方预估影响 ≥ min_impact。
    归入 other 或无 category 的信号不参与（漏检退回加权平均 = 现状行为）。
    """
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for sig in signals:
        cat = sig.get("category")
        if not cat or cat == "other":
            continue
        by_category.setdefault(cat, []).append(sig)

    conflicts: List[Dict[str, Any]] = []
    for cat, sigs in by_category.items():
        bulls = [s for s in sigs if s.get("direction", 0) > 0]
        bears = [s for s in sigs if s.get("direction", 0) < 0]
        if not bulls or not bears:
            continue
        for b in bulls:
            for s in bears:
                if b.get("source_level") == s.get("source_level"):
                    continue
                b_impact = abs(_signal_contribution(b)) * DIMENSION_SCALE
                s_impact = abs(_signal_contribution(s)) * DIMENSION_SCALE
                if (
                    _normalize_strength(b) >= min_strength_eff
                    and _normalize_strength(s) >= min_strength_eff
                    and b_impact >= min_impact
                    and s_impact >= min_impact
                ):
                    conflicts.append({
                        "category": cat,
                        "bullish": b,
                        "bearish": s,
                    })
    return conflicts


def compute_score(
    term: str,
    signals: List[Dict[str, Any]],
    pkg: AnalysisPackage,
    is_etf: bool = False,
    signal_discounts: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """核心纯函数：计算单期限分数。

    Args:
        term: short/medium/long
        signals: collect_signals() 的输出
        pkg: AnalysisPackage（缺失 Agent / 风险标签来源）
        is_etf: ETF 期望缺席处理
        signal_discounts: 仲裁层输出的折扣 {id(signal): discount}，缺省无折扣

    Returns: 与现有 scorer 缓存契约一致的结果 dict（未过 risk_gate）。
    """
    if term not in TERM_DIMENSIONS:
        raise ValueError(f"未知期限: {term}")

    discounts = signal_discounts or {}
    dimensions = TERM_DIMENSIONS[term]
    droppable = set(ETF_DROPPABLE_DIMENSIONS.get(term, [])) if is_etf else set()

    # 维度归组
    dim_signals: Dict[str, List[Dict[str, Any]]] = {d["key"]: [] for d in dimensions}
    for sig in signals:
        cat = sig.get("category")
        if not cat:
            continue
        for d in dimensions:
            if cat in d["categories"]:
                dim_signals[d["key"]].append(sig)
                break  # 一条信号只进入一个维度

    sub_scores: Dict[str, float] = {}
    kept_dimensions = []
    dropped_dimensions = []
    for d in dimensions:
        sigs = dim_signals[d["key"]]
        if not sigs and d["key"] in droppable:
            dropped_dimensions.append(d["key"])
            continue
        total = 0.0
        for sig in sigs:
            contrib = _signal_contribution(sig)
            discount = discounts.get(id(sig))
            if discount is not None:
                contrib *= max(0.0, min(1.0, discount))
            total += contrib
        sub_scores[d["key"]] = round(max(0.0, min(100.0, 50.0 + total * DIMENSION_SCALE)), 1)
        kept_dimensions.append(d)

    # 总分：kept 维度加权归一
    weight_sum = sum(d["weight"] for d in kept_dimensions)
    if weight_sum <= 0:
        raw_total = 50.0
    else:
        raw_total = sum(sub_scores[d["key"]] * d["weight"] for d in kept_dimensions) / weight_sum

    # 缺失惩罚（核心缺失 ≤40 / 部分缺失 ≤65，沿用现行分级规则）
    required = list(TERM_REQUIRED_AGENTS[term])
    if is_etf:
        required = [a for a in required if a not in ETF_EXEMPT_AGENTS]
    missing_agents = [a for a in (pkg.missing_agents or []) if a in required]
    coverage = (len(required) - len(missing_agents)) / len(required) if required else 1.0

    missing_cap: Optional[int] = None
    if len(missing_agents) >= 2:
        missing_cap = 40
    elif len(missing_agents) == 1:
        missing_cap = 65

    total = raw_total
    if missing_cap is not None:
        total = min(total, float(missing_cap))
    total = round(max(0.0, min(100.0, total)), 1)

    rating = map_score_to_rating(total)

    # confidence：由 coverage 与冲突数推导（4.9-4）
    n_conflicts = len(detect_material_conflicts(signals, min_strength_eff=0, min_impact=0))
    if coverage >= 0.9 and n_conflicts == 0:
        confidence = 0.85
    elif coverage >= 0.7:
        confidence = 0.7
    elif coverage >= 0.5:
        confidence = 0.55
    else:
        confidence = 0.4
    if n_conflicts > 0:
        confidence = max(0.3, round(confidence - 0.05 * n_conflicts, 2))

    # 模板化 reasoning（零 LLM；LLM 解释为可选懒加载，见 score_explanation.py）
    dim_desc = "，".join(
        f"{d['name']}{sub_scores[d['key']]}" for d in kept_dimensions
    )
    reasoning_parts = [f"确定性打分（{WEIGHTS_VERSION}）：{dim_desc}。"]
    if dropped_dimensions:
        reasoning_parts.append(f"ETF期望缺席维度已跳过: {', '.join(dropped_dimensions)}。")
    if missing_agents:
        reasoning_parts.append(f"缺失Agent: {', '.join(missing_agents)}（分数上限{missing_cap}）。")
    if discounts:
        reasoning_parts.append(f"{len(discounts)} 条信号经冲突仲裁折扣。")

    result = {
        "score": total,
        "raw_score_before_missing_cap": round(raw_total, 1),
        "sub_scores": sub_scores,
        "recommendation": rating,
        "rating": rating,
        "suggested_action": SUGGESTED_ACTION_TABLE.get(rating, "观望"),
        "confidence": confidence,
        "reasoning": "".join(reasoning_parts),
        "risk_warning": "；".join(pkg.global_risk_flags) if pkg.global_risk_flags else "",
        "data_quality_score": round(coverage, 3),
        "coverage": round(coverage, 3),
        "missing_core_fields": missing_agents,
        "missing_optional_fields": [a for a in (pkg.missing_agents or []) if a not in missing_agents],
        "dropped_dimensions": dropped_dimensions,
        "validity": "valid" if coverage > 0 else "invalid",
        "scorer_type": "deterministic",
        "weights_version": WEIGHTS_VERSION,
        "strength_zero_policy": STRENGTH_ZERO_POLICY,
    }
    return result
