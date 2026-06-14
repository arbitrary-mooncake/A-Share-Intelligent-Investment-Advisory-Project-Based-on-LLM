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


# ═══════════════════════════════════════════════════════
# Tests for analysis_package_builder.py
# ═══════════════════════════════════════════════════════

def test_text_to_signal_pack_bullish():
    from src.utils.analysis_package_builder import text_to_signal_pack
    sp = text_to_signal_pack("ROE持续提升，毛利率改善显著，现金流充裕，强烈看多买入", "fundamental", "2026-06-14")
    assert sp["agent_name"] == "fundamental"
    assert sp["bias"] == "bullish"
    assert sp["confidence"] < 0.5  # derived from text => low confidence
    assert len(sp["key_points"]) >= 0
    assert "structured_output_missing" not in sp.get("risk_flags", [])


def test_text_to_signal_pack_bearish():
    from src.utils.analysis_package_builder import text_to_signal_pack
    sp = text_to_signal_pack("质押风险极高，现金流不匹配，建议卖出减持看空", "quality_risk", "2026-06-14")
    assert sp["bias"] == "bearish"


def test_text_to_signal_pack_neutral():
    from src.utils.analysis_package_builder import text_to_signal_pack
    sp = text_to_signal_pack("数据平平，无明显方向", "unknown", "2026-06-14")
    assert sp["bias"] == "neutral"


def test_text_to_signal_pack_empty():
    from src.utils.analysis_package_builder import text_to_signal_pack
    sp = text_to_signal_pack("", "test", "2026-01-01")
    assert isinstance(sp, dict)
    assert sp["bias"] == "neutral"


def test_build_analysis_package_all_missing():
    """全部agent未执行时仍应返回有效pkg，不抛异常"""
    from src.utils.analysis_package_builder import build_analysis_package
    pkg = build_analysis_package({}, "2026-06-14")
    assert len(pkg.missing_agents) == 7
    assert pkg.global_risk_flags is not None
    assert len(pkg.compact_prompt_context) > 0
    assert isinstance(pkg.compact_prompt_context, str)


def test_build_analysis_package_with_text_only():
    """旧agent只产出了文本，应fallback生成signal_pack"""
    from src.utils.analysis_package_builder import build_analysis_package
    data = {
        "fundamental_analysis": "ROE持续提升，毛利率改善，基本面看多",
        "technical_analysis": "均线多头排列，MACD金叉，技术面看多",
    }
    pkg = build_analysis_package(data, "2026-06-14")
    assert "fundamental" in pkg.available_agents
    assert "technical" in pkg.available_agents


def test_build_analysis_package_with_signal_packs():
    """已有signal_pack时应直接使用"""
    from src.utils.analysis_package_builder import build_analysis_package
    data = {
        "fundamental_signal_pack": {
            "agent_name": "fundamental", "bias": "bullish", "confidence": 0.8,
            "data_quality_score": 0.9, "key_points": ["ROE=18%", "现金流健康"],
            "signals": [
                {"factor": "ROE持续性", "direction": 1, "strength": 80,
                 "source_level": "structured", "time_horizon": ["medium"],
                 "note": "ROE持续>15%"}
            ],
            "risk_flags": [], "missing_data": [], "source_summary": "Tushare",
            "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    assert "fundamental" in pkg.available_agents
    assert len(pkg.bullish_signals) >= 1
    assert pkg.bullish_signals[0]["factor"] == "ROE持续性"


def test_build_package_conflict_detection():
    """方向冲突的因子应被检测"""
    from src.utils.analysis_package_builder import build_analysis_package
    data = {
        "fundamental_signal_pack": {
            "agent_name": "fundamental", "bias": "bullish", "confidence": 0.7,
            "key_points": ["好"], "data_quality_score": 0.8,
            "signals": [
                {"factor": "ROE质量", "direction": 1, "strength": 80,
                 "source_level": "structured", "time_horizon": ["medium"],
                 "note": "ROE高"}
            ],
            "risk_flags": [], "missing_data": [], "source_summary": "",
            "as_of_date": "2026-06-14",
        },
        "quality_risk_signal_pack": {
            "agent_name": "quality_risk", "bias": "bearish", "confidence": 0.8,
            "key_points": ["差"], "data_quality_score": 0.8,
            "signals": [
                {"factor": "ROE质量", "direction": -1, "strength": 75,
                 "source_level": "structured", "time_horizon": ["medium"],
                 "note": "ROE质量差"}
            ],
            "risk_flags": [], "missing_data": [], "source_summary": "",
            "as_of_date": "2026-06-14",
        },
    }
    pkg = build_analysis_package(data, "2026-06-14")
    assert len(pkg.conflicting_signals) > 0


def test_compact_context_contains_sections():
    """compact_prompt_context应包含各个必需段落"""
    from src.utils.analysis_package_builder import build_analysis_package
    data = {
        "fundamental_signal_pack": {
            "agent_name": "fundamental", "bias": "bullish", "confidence": 0.7,
            "key_points": ["ROE>15%"], "signals": [
                {"factor": "ROE", "direction": 1, "strength": 80,
                 "source_level": "structured", "time_horizon": ["medium"],
                 "note": "ROE持续改善"}
            ],
            "risk_flags": ["cashflow_mismatch"], "missing_data": ["未获取质押数据"],
            "source_summary": "Tushare", "as_of_date": "2026-06-14",
            "data_quality_score": 0.7,
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    ctx = pkg.compact_prompt_context
    assert "分析执行概况" in ctx
    assert "关键看多信号" in ctx or "看多" in ctx
