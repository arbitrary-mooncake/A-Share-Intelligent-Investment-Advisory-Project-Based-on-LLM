"""Tests for analysis_schema.py structured data definitions."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_signal_dataclass_creation():
    """Signal和SignalPack应可正常创建"""
    from src.utils.analysis_schema import Signal, SignalPack, SourceLevel

    s = Signal(
        factor="ROE持续性",
        direction=1,
        strength=80,
        confidence=0.85,
        time_horizon=["medium", "long"],
        source_level=SourceLevel.STRUCTURED,
        freshness="quarterly",
        risk_flags=[],
        note="ROE连续3年>15%"
    )
    assert s.factor == "ROE持续性"
    assert s.direction == 1
    assert s.source_level == "structured"

    sp = SignalPack(
        agent_name="fundamental",
        analysis_text="基本面良好",
        bias="bullish",
        confidence=0.8,
        data_quality_score=0.9,
        key_points=["ROE高", "现金流好"],
        signals=[s],
        risk_flags=[],
        missing_data=[],
        source_summary="Tushare财报数据",
        as_of_date="2026-06-14",
    )
    assert sp.agent_name == "fundamental"
    assert len(sp.signals) == 1


def test_analysis_package_creation():
    """AnalysisPackage应可正常创建"""
    from src.utils.analysis_schema import AnalysisPackage

    pkg = AnalysisPackage(
        as_of_date="2026-06-14",
        executed_agents=["fundamental", "technical"],
        available_agents=["fundamental", "technical"],
        missing_agents=["event", "quality_risk", "moneyflow", "news", "value"],
        global_risk_flags=[],
        global_missing_data=["event数据缺失"],
        bullish_signals=[],
        bearish_signals=[],
        conflicting_signals=[],
        source_priority_summary={"counts": {"structured": 2}},
        compact_prompt_context="## 测试",
    )
    assert len(pkg.missing_agents) == 5


def test_risk_gate_result_creation():
    """RiskGateResult应可正常创建"""
    from src.utils.analysis_schema import RiskGateResult

    r = RiskGateResult(
        risk_level="low",
        risk_flags_found=[],
        score_cap=None,
        action_downgrade=None,
        abstain=False,
        abstain_reason="",
        data_quality_score=0.85,
        warnings=[],
    )
    assert r.risk_level == "low"


def test_source_priority_order():
    """Source priority应正确排序"""
    from src.utils.analysis_schema import SOURCE_PRIORITY, SourceLevel
    assert SOURCE_PRIORITY[SourceLevel.OFFICIAL] > SOURCE_PRIORITY[SourceLevel.NEWS]
    assert SOURCE_PRIORITY[SourceLevel.NEWS] > SOURCE_PRIORITY[SourceLevel.PROXY]


def test_fallback_signal_pack_constant():
    """FALLBACK_SIGNAL_PACK常量应包含所有必要字段"""
    from src.utils.analysis_schema import FALLBACK_SIGNAL_PACK
    required = ["agent_name", "bias", "confidence", "data_quality_score", "key_points", "signals", "risk_flags"]
    for key in required:
        assert key in FALLBACK_SIGNAL_PACK
    assert FALLBACK_SIGNAL_PACK["bias"] == "neutral"
    assert FALLBACK_SIGNAL_PACK["confidence"] == 0.3
