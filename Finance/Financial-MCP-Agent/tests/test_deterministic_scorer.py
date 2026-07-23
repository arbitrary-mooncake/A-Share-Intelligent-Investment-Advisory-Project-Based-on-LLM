"""
确定性打分器测试（4.3）：公式、ETF 归一、缺失惩罚、评级映射、冲突检测、可重复性。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.analysis_schema import AnalysisPackage
from src.utils import deterministic_scorer as ds


def make_pkg(missing=None, risk_flags=None):
    return AnalysisPackage(
        as_of_date="2026-07-20",
        executed_agents=[],
        available_agents=[],
        missing_agents=missing or [],
        global_risk_flags=risk_flags or [],
        global_missing_data=[],
        bullish_signals=[],
        bearish_signals=[],
        conflicting_signals=[],
        source_priority_summary={},
        compact_prompt_context="",
    )


def sig(category, direction, strength=80, confidence=0.9,
         source="structured", factor="测试因子"):
    return {
        "factor": factor,
        "direction": direction,
        "strength": strength,
        "confidence": confidence,
        "source_level": source,
        "category": category,
    }


# ── 基本公式 ────────────────────────────────────────────

def test_bullish_signal_raises_score_above_50():
    signals = [sig("technical_trend", 1)]
    result = ds.compute_score("short", signals, make_pkg())
    assert result["score"] > 50.0
    assert result["sub_scores"]["technical_trend"] > 50.0


def test_bearish_signal_lowers_score_below_50():
    signals = [sig("technical_trend", -1)]
    result = ds.compute_score("short", signals, make_pkg())
    assert result["score"] < 50.0


def test_uncategorized_signal_is_ignored():
    signals = [{"factor": "无类目", "direction": 1, "strength": 100,
                "confidence": 1.0, "source_level": "official_like"}]
    result = ds.compute_score("short", signals, make_pkg())
    assert result["score"] == 50.0


def test_neutral_direction_contributes_zero():
    signals = [sig("technical_trend", 0, strength=100)]
    result = ds.compute_score("short", signals, make_pkg())
    assert result["sub_scores"]["technical_trend"] == 50.0


def test_determinism_same_input_same_output():
    signals = [sig("technical_trend", 1), sig("sentiment", -1, strength=40)]
    r1 = ds.compute_score("short", signals, make_pkg())
    r2 = ds.compute_score("short", signals, make_pkg())
    assert r1["score"] == r2["score"]
    assert r1["sub_scores"] == r2["sub_scores"]


# ── strength=0 策略 ─────────────────────────────────────

def test_strength_zero_default5_counts_signal(monkeypatch):
    monkeypatch.setattr(ds, "STRENGTH_ZERO_POLICY", "default5")
    signals = [sig("technical_trend", 1, strength=0, confidence=0.95)]
    result = ds.compute_score("short", signals, make_pkg())
    assert result["score"] > 50.0  # default5 兜底，信号有贡献


def test_strength_zero_raw_discards_signal(monkeypatch):
    monkeypatch.setattr(ds, "STRENGTH_ZERO_POLICY", "raw")
    signals = [sig("technical_trend", 1, strength=0, confidence=0.95)]
    result = ds.compute_score("short", signals, make_pkg())
    assert result["sub_scores"]["technical_trend"] == 50.0  # raw：贡献为 0


# ── ETF 期望缺席（4.9-3） ───────────────────────────────

def test_etf_drops_fundamental_dimensions_and_renormalizes():
    signals = [sig("technical_trend", 1), sig("catalyst_event", 1)]
    result = ds.compute_score("medium", signals, make_pkg(), is_etf=True)
    assert "fundamentals" in result["dropped_dimensions"]
    assert "valuation" in result["dropped_dimensions"]
    assert "fundamentals" not in result["sub_scores"]
    # 不被缺失惩罚压制（ETF 缺席是期望行为）
    assert result["missing_core_fields"] == []


def test_non_etf_keeps_fundamental_dimension():
    signals = [sig("fundamentals_growth", 1)]
    result = ds.compute_score("medium", signals, make_pkg(), is_etf=False)
    assert "fundamentals" in result["sub_scores"]
    assert result["dropped_dimensions"] == []


# ── 缺失惩罚 ────────────────────────────────────────────

def test_two_missing_agents_cap_at_40():
    strong_bullish = [sig("technical_trend", 1, strength=100, confidence=1.0,
                          source="official_like")]
    pkg = make_pkg(missing=["fundamental", "value"])
    result = ds.compute_score("medium", strong_bullish, pkg)
    assert result["score"] <= 40.0
    assert result["coverage"] < 1.0


def test_one_missing_agent_cap_at_65():
    strong_bullish = [sig("technical_trend", 1, strength=100, confidence=1.0,
                          source="official_like")]
    pkg = make_pkg(missing=["fundamental"])
    result = ds.compute_score("medium", strong_bullish, pkg)
    assert result["score"] <= 65.0


# ── 评级映射（4.9-4） ───────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (85, "强烈买入"), (80, "强烈买入"),
    (79, "买入"), (65, "买入"),
    (64, "中性"), (50, "中性"),
    (49, "减持"), (35, "减持"),
    (34, "回避"), (0, "回避"),
])
def test_rating_mapping(score, expected):
    assert ds.map_score_to_rating(score) == expected


# ── 冲突检测 ────────────────────────────────────────────

def test_material_conflict_detected():
    signals = [
        sig("catalyst_event", 1, strength=80, source="official_like", factor="回购公告"),
        sig("catalyst_event", -1, strength=70, source="news", factor="资金链传闻"),
    ]
    conflicts = ds.detect_material_conflicts(signals)
    assert len(conflicts) == 1
    assert conflicts[0]["category"] == "catalyst_event"


def test_same_source_level_not_material():
    signals = [
        sig("catalyst_event", 1, strength=80, source="news"),
        sig("catalyst_event", -1, strength=70, source="news"),
    ]
    assert ds.detect_material_conflicts(signals) == []


def test_weak_signals_not_material():
    signals = [
        sig("catalyst_event", 1, strength=20, source="official_like"),
        sig("catalyst_event", -1, strength=20, source="news"),
    ]
    assert ds.detect_material_conflicts(signals) == []


def test_other_category_not_checked():
    signals = [
        sig("other", 1, strength=90, source="official_like"),
        sig("other", -1, strength=90, source="news"),
    ]
    assert ds.detect_material_conflicts(signals) == []


# ── 折扣应用 ────────────────────────────────────────────

def test_discount_reduces_signal_contribution():
    s = sig("technical_trend", 1, strength=100, confidence=1.0, source="official_like")
    base = ds.compute_score("short", [s], make_pkg())
    discounted = ds.compute_score("short", [s], make_pkg(),
                                  signal_discounts={id(s): 0.0})
    assert discounted["sub_scores"]["technical_trend"] < base["sub_scores"]["technical_trend"]


# ── 输出契约 ────────────────────────────────────────────

def test_output_contract_fields():
    result = ds.compute_score("short", [sig("technical_trend", 1)], make_pkg())
    for field in ("score", "sub_scores", "recommendation", "rating",
                  "suggested_action", "confidence", "reasoning",
                  "risk_warning", "data_quality_score", "coverage",
                  "missing_core_fields", "validity", "scorer_type",
                  "weights_version"):
        assert field in result, f"缺少契约字段: {field}"
    assert result["scorer_type"] == "deterministic"
    assert 0 <= result["score"] <= 100
