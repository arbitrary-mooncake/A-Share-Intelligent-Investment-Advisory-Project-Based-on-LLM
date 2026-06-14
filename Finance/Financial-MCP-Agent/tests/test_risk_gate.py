"""Test risk_gate: score cap, abstain, downgrade rules"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.analysis_package_builder import build_analysis_package


def test_audit_risk_triggers_cap():
    """审计风险应触发score_cap=60"""
    from src.utils.risk_gate import apply_risk_gate
    data = {
        "quality_risk_signal_pack": {
            "agent_name": "quality_risk", "bias": "neutral",
            "confidence": 0.7, "data_quality_score": 0.8,
            "key_points": [], "signals": [],
            "risk_flags": ["audit_risk"], "missing_data": [],
            "source_summary": "", "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.score_cap == 60
    assert result.action_downgrade == "谨慎"


def test_regulatory_risk_triggers_cap():
    """监管风险应触发score_cap"""
    from src.utils.risk_gate import apply_risk_gate
    data = {
        "quality_risk_signal_pack": {
            "agent_name": "quality_risk", "bias": "neutral",
            "confidence": 0.7, "data_quality_score": 0.8,
            "key_points": [], "signals": [],
            "risk_flags": ["regulatory_risk"], "missing_data": [],
            "source_summary": "", "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.score_cap == 55


def test_no_critical_risk_no_cap():
    """无关键风险时不应cap"""
    from src.utils.risk_gate import apply_risk_gate
    pkg = build_analysis_package({}, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.score_cap is None


def test_news_only_narrative_caps_long():
    """长线仅靠新闻叙事应被cap"""
    from src.utils.risk_gate import apply_risk_gate
    data = {
        "news_signal_pack": {
            "agent_name": "news", "bias": "bullish",
            "confidence": 0.9, "data_quality_score": 0.7,
            "key_points": ["媒体一致看多"], "signals": [
                {"factor": "媒体情绪", "direction": 1, "strength": 70,
                 "source_level": "news", "note": ""}
            ],
            "risk_flags": [], "missing_data": [],
            "source_summary": "新闻接口", "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "long", 90)
    assert result.score_cap is not None
    assert result.score_cap <= 55


def test_abstain_when_many_missing():
    """缺失agent过多且data_quality低时abstain"""
    from src.utils.risk_gate import apply_risk_gate
    pkg = build_analysis_package({}, "2026-06-14")
    result = apply_risk_gate(pkg, "long", 80)
    assert result.abstain is True


def test_low_risk_on_full_data():
    """数据齐全无风险标签时应为low risk"""
    from src.utils.risk_gate import apply_risk_gate
    data = {}
    for agent in ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"]:
        data[f"{agent}_signal_pack"] = {
            "agent_name": agent, "bias": "neutral",
            "confidence": 0.8, "data_quality_score": 0.9,
            "key_points": [], "signals": [],
            "risk_flags": [], "missing_data": [],
            "source_summary": "", "as_of_date": "2026-06-14",
        }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.risk_level == "low"
    assert result.abstain is False


def test_delist_risk_most_conservative():
    """退市风险应触发最低cap"""
    from src.utils.risk_gate import apply_risk_gate
    data = {
        "quality_risk_signal_pack": {
            "agent_name": "quality_risk", "bias": "neutral",
            "confidence": 0.9, "data_quality_score": 0.9,
            "key_points": [], "signals": [],
            "risk_flags": ["delist_risk"], "missing_data": [],
            "source_summary": "", "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.score_cap == 50


def test_multiple_critical_uses_min_cap():
    """多个关键风险标签时取最低cap"""
    from src.utils.risk_gate import apply_risk_gate
    data = {
        "quality_risk_signal_pack": {
            "agent_name": "quality_risk", "bias": "neutral",
            "confidence": 0.9, "data_quality_score": 0.9,
            "key_points": [], "signals": [],
            "risk_flags": ["audit_risk", "regulatory_risk", "high_pledge_risk"],
            "missing_data": [],
            "source_summary": "", "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.score_cap == 55  # 取最严格的
