"""
Risk Gate: 轻量分析后处理，对评分结果做风险门控。

门控规则:
  1. 高严重度风险标签 → score cap + downgrade
  2. 长线仅新闻叙事无事实支撑 → 不能强推荐
  3. 核心agent缺失过多 + data_quality低 → abstain
  4. 短线流动性差 → 不能高分
"""
from typing import List, Optional

from src.utils.analysis_schema import RiskGateResult, SourceLevel, SOURCE_PRIORITY


CRITICAL_RISK_FLAGS = {
    "audit_risk":             {"cap": 60, "downgrade": "谨慎"},
    "regulatory_risk":        {"cap": 55, "downgrade": "谨慎"},
    "high_pledge_risk":       {"cap": 60, "downgrade": "谨慎"},
    "cashflow_mismatch":      {"cap": 65, "downgrade": "观察"},
    "major_event_negative":   {"cap": 55, "downgrade": "谨慎"},
    "delist_risk":            {"cap": 50, "downgrade": "谨慎"},
    "st_risk":                {"cap": 60, "downgrade": "观察"},
    "earnings_quality_concern": {"cap": 65, "downgrade": "观察"},
    "goodwill_risk":          {"cap": 65, "downgrade": "观察"},
    "debt_risk":              {"cap": 65, "downgrade": "观察"},
    "major_shareholder_sell": {"cap": 65, "downgrade": "观察"},
    "liquidity_risk":         {"cap": 65, "downgrade": "观察"},
    "impairment_risk":        {"cap": 65, "downgrade": "观察"},
}


def _count_signals_by_source(package, min_source_level: str) -> int:
    threshold = SOURCE_PRIORITY.get(min_source_level, 0)
    count = 0
    for sigs in [package.bullish_signals, package.bearish_signals]:
        for s in sigs:
            lv = s.get("source_level", SourceLevel.PROXY)
            if SOURCE_PRIORITY.get(lv, 0) >= threshold:
                count += 1
    return count


def apply_risk_gate(package: 'AnalysisPackage', term: str, original_score: int) -> RiskGateResult:
    risk_flags = package.global_risk_flags
    missing_agents = package.missing_agents

    total_possible = 7
    available = len(package.available_agents)
    data_quality = available / total_possible if total_possible > 0 else 1.0

    # 规则1: 关键风险标签
    score_cap = None
    downgrade = None
    found_critical: List[str] = []

    for flag, cfg in CRITICAL_RISK_FLAGS.items():
        if flag in risk_flags:
            found_critical.append(flag)
            if score_cap is None or cfg["cap"] < score_cap:
                score_cap = cfg["cap"]
                downgrade = cfg["downgrade"]

    # 规则2: 仅新闻叙事无事实 (对中长期)
    factual_signals = _count_signals_by_source(package, SourceLevel.STRUCTURED)
    news_only_signals = _count_signals_by_source(package, SourceLevel.NEWS)
    if factual_signals == 0 and news_only_signals > 0 and term in ("medium", "long"):
        if score_cap is None or score_cap > 55:
            score_cap = 55
            downgrade = "观察"
            found_critical.append("news_only_narrative")

    # 规则3: 数据严重不足 → abstain
    if len(missing_agents) >= 2 and data_quality < 0.4:
        return RiskGateResult(
            risk_level="critical" if found_critical else "high",
            risk_flags_found=found_critical,
            score_cap=score_cap,  # 仅当关键风险标签触发了cap时才设值，否则保持None
            action_downgrade=downgrade or "观察",
            abstain=True,
            abstain_reason=f"数据不足(缺失{len(missing_agents)}个agent: {', '.join(missing_agents)})",
            data_quality_score=data_quality,
            warnings=[f"缺失agent: {', '.join(missing_agents)}"],
        )

    # 规则4: 短线流动性
    if term == "short":
        liquidity_sigs = [s for s in package.bearish_signals if "流动" in s.get("factor", "") or "量价" in s.get("factor", "")]
        if liquidity_sigs and any(abs(s.get("strength", 0)) > 60 for s in liquidity_sigs):
            if score_cap is None or score_cap > 50:
                score_cap = 50
                downgrade = "观察"
                found_critical.append("short_liquidity_risk")

    # risk_level
    if found_critical:
        risk_level = "high"
    elif data_quality < 0.6:
        risk_level = "medium"
    else:
        risk_level = "low"

    # 计算有效分数
    effective_score = min(original_score, score_cap) if score_cap else original_score

    warnings = []
    if found_critical:
        warnings.append(f"检测到关键风险: {', '.join(found_critical)}")
    if data_quality < 0.5:
        warnings.append(f"数据质量低({data_quality:.0%})")

    return RiskGateResult(
        risk_level=risk_level,
        risk_flags_found=found_critical,
        score_cap=score_cap,
        action_downgrade=downgrade,
        abstain=False,
        abstain_reason="",
        data_quality_score=data_quality,
        warnings=warnings,
    )
