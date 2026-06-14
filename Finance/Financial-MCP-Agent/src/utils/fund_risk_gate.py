"""
Fund Risk Gate: Post-processing risk gate for fund/ETF scoring.
Adapted from A-stock risk_gate pattern for fund-specific risk factors.
"""
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class FundRiskGateResult:
    risk_level: str          # "low"|"medium"|"high"
    risk_flags_found: List[str]
    score_cap: Optional[int]
    action_downgrade: Optional[str]
    abstain: bool
    abstain_reason: str
    warnings: List[str]


# Fund-specific critical risk flags
FUND_CRITICAL_RISK_FLAGS = {
    "manager_just_changed":    {"cap": 55, "downgrade": "观察"},  # 基金经理近期更换
    "high_fee_structure":      {"cap": 60, "downgrade": "谨慎"},  # 费率显著高于同类
    "tiny_fund_size":          {"cap": 55, "downgrade": "谨慎"},  # 规模过小（<5000万）
    "high_tracking_error":     {"cap": 60, "downgrade": "观察"},  # ETF跟踪误差过大
    "large_discount_premium":  {"cap": 60, "downgrade": "观察"},  # 折溢价异常
    "high_volatility":         {"cap": 65, "downgrade": "观察"},  # 波动率过高
    "poor_long_term_perf":     {"cap": 65, "downgrade": "观察"},  # 长期业绩不佳
    "frequent_manager_change": {"cap": 50, "downgrade": "谨慎"},  # 基金经理频繁变更
    "liquidity_risk":          {"cap": 60, "downgrade": "谨慎"},  # 流动性风险
    "concentration_risk":      {"cap": 65, "downgrade": "观察"},  # 持仓过度集中
}


def apply_fund_risk_gate(
    signal_packs: dict,
    fund_analysis_package: dict,
    original_score: int,
) -> FundRiskGateResult:
    """Apply fund-specific risk gating rules."""

    # Collect risk flags from all signal_packs
    all_risk_flags = []
    data_quality_total = 0.0
    data_quality_count = 0

    for key, sp in signal_packs.items():
        if isinstance(sp, dict):
            all_risk_flags.extend(sp.get("risk_flags", []))
            dqs = sp.get("data_quality_score", 0)
            if dqs:
                data_quality_total += float(dqs) if not isinstance(dqs, (int, float)) else dqs
                data_quality_count += 1

    unique_risk_flags = list(set(all_risk_flags))
    data_quality = data_quality_total / max(data_quality_count, 1)

    # Rule 1: Check critical risk flags
    score_cap = None
    downgrade = None
    found_critical = []

    for flag, cfg in FUND_CRITICAL_RISK_FLAGS.items():
        if flag in unique_risk_flags:
            found_critical.append(flag)
            if score_cap is None or cfg["cap"] < score_cap:
                score_cap = cfg["cap"]
                downgrade = cfg["downgrade"]

    # Rule 2: Too few signal_packs available
    available_count = sum(1 for sp in signal_packs.values() if isinstance(sp, dict) and sp.get("confidence", 0) > 0.3)
    if available_count < 3:
        return FundRiskGateResult(
            risk_level="high",
            risk_flags_found=found_critical,
            score_cap=min(score_cap or 50, 50),
            action_downgrade=downgrade or "观察",
            abstain=True,
            abstain_reason=f"可用分析agent不足({available_count}个，需至少3个)",
            warnings=[f"缺失过多agent数据"],
        )

    # Risk level
    if found_critical:
        risk_level = "high"
    elif data_quality < 0.5:
        risk_level = "medium"
    else:
        risk_level = "low"

    warnings = []
    if found_critical:
        warnings.append(f"检测到关键风险: {', '.join(found_critical)}")

    return FundRiskGateResult(
        risk_level=risk_level,
        risk_flags_found=found_critical,
        score_cap=score_cap,
        action_downgrade=downgrade,
        abstain=False,
        abstain_reason="",
        warnings=warnings,
    )
