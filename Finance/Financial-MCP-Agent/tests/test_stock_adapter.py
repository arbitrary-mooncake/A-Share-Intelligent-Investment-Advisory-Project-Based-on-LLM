"""Tests for stock pipeline adapter"""
import pytest

from src.eval.adapters.stock_pipeline_adapter import (
    _build_decision_pack,
    _extract_signal_strings,
    extract_signal_packs_from_state,
    extract_analysis_texts_from_state,
)
from src.eval.score_assessment import ScoreAssessmentSchemaError


def _valid_score(**fields):
    """Build a score payload that satisfies the explicit P0 validity contract."""
    return {
        "validity": "valid",
        "coverage": 1.0,
        "missing_core_fields": [],
        "missing_optional_fields": [],
        **fields,
    }


class TestBuildDecisionPack:
    """Test _build_decision_pack with various scorer output formats."""

    def test_build_from_short_term_scorer_output(self):
        """short_term_scorer uses 'recommendation' key."""
        score_data = _valid_score(
            score=75,
            recommendation="买入",
            confidence=0.8,
            data_quality_score=0.7,
            risk_gate={"risk_level": "low", "risk_flags": []},
            reasoning="技术面利多，但存在估值风险",
            sub_scores={"technical_state": 20, "volume_liquidity": 15},
            risk_warning="注意流动性风险",
            suggested_action="短线持有",
        )
        dp = _build_decision_pack("sh.603871", "嘉友国际", "short", "2026-06-19", score_data)

        assert dp.asset_type == "stock"
        assert dp.symbol == "sh.603871"
        assert dp.name == "嘉友国际"
        assert dp.term == "short"
        assert dp.score == 75.0
        assert dp.action == "buy"
        assert dp.confidence == 0.8
        assert dp.data_quality_score == 0.7
        assert dp.risk_gate_applied is False  # empty risk_flags
        assert dp.risk_gate_result is not None
        assert dp.risk_gate_result["risk_level"] == "low"
        assert dp.task_type == "eval"  # eval_mode=True by default
        assert dp.model_profile == "eval_analysis"
        assert dp.meta is not None
        assert dp.meta["raw_recommendation"] == "买入"
        assert dp.meta["suggested_action"] == "短线持有"

    def test_build_from_medium_term_scorer_output(self):
        """medium_term_scorer uses 'rating' key."""
        score_data = _valid_score(
            score=68,
            rating="推荐",
            confidence=0.75,
            data_quality_score=0.65,
            risk_gate={"risk_level": "medium", "risk_flags": ["data_missing"]},
            reasoning="基本面稳健，技术面中性，估值偏高",
            sub_scores={"fundamental_score": 20, "valuation_score": 12},
            time_horizon="3-6个月",
        )
        dp = _build_decision_pack("sz.000001", "测试银行", "medium", "2026-06-19", score_data)

        assert dp.action == "buy"
        assert dp.risk_gate_applied is True  # has risk_flags
        assert dp.meta["raw_recommendation"] == "推荐"
        assert dp.meta["scorer_key"] == "rating"
        assert dp.meta["time_horizon"] == "3-6个月"

    def test_build_from_long_term_scorer_output(self):
        """long_term_scorer uses 'rating' key with 强烈推荐."""
        score_data = _valid_score(
            score=88,
            rating="强烈推荐",
            confidence=0.92,
        )
        dp = _build_decision_pack("sh.600519", "贵州茅台", "long", "2026-06-19", score_data)
        assert dp.action == "strong_buy"
        assert dp.score == 88.0

    def test_build_strong_sell(self):
        score_data = _valid_score(
            score=15,
            recommendation="强烈卖出",
            confidence=0.9,
            risk_gate={"risk_level": "critical", "risk_flags": ["delist_risk"]},
        )
        dp = _build_decision_pack("sz.000001", "测试", "medium", "2026-06-19", score_data)
        assert dp.action == "strong_sell"
        assert dp.score == 15.0
        assert dp.risk_gate_applied is True

    def test_unknown_recommendation_returns_none(self):
        score_data = _valid_score(score=50, recommendation="未知建议", confidence=0.5)
        dp = _build_decision_pack("sh.000001", "测试", "short", "2026-06-19", score_data)
        assert dp is None

    def test_medium_term_中性(self):
        score_data = _valid_score(score=45, rating="中性")
        dp = _build_decision_pack("sh.000001", "测试", "medium", "2026-06-19", score_data)
        assert dp.action == "hold"

    def test_medium_term_减持_maps_to_sell(self):
        """减持 maps to sell; only 谨慎减持 maps to cautious_sell."""
        score_data = _valid_score(score=30, rating="减持")
        dp = _build_decision_pack("sh.000001", "测试", "medium", "2026-06-19", score_data)
        assert dp.action == "sell"

    def test_medium_term_谨慎减持_maps_to_cautious_sell(self):
        """谨慎减持 maps to cautious_sell."""
        score_data = _valid_score(score=35, rating="谨慎减持")
        dp = _build_decision_pack("sh.000001", "测试", "medium", "2026-06-19", score_data)
        assert dp.action == "cautious_sell"

    def test_production_mode(self):
        score_data = _valid_score(score=60, recommendation="观望")
        dp = _build_decision_pack(
            "sh.000001", "测试", "short", "2026-06-19", score_data, eval_mode=False
        )
        assert dp.task_type == "single_stock"
        assert dp.model_profile == "production"

    def test_empty_score_data_returns_none(self):
        dp = _build_decision_pack("sh.000001", "测试", "short", "2026-06-19", {})
        assert dp is None

    def test_none_score_data_returns_none(self):
        dp = _build_decision_pack("sh.000001", "测试", "short", "2026-06-19", None)
        assert dp is None

    def test_string_score_is_schema_error(self):
        score_data = _valid_score(score="80", recommendation="买入", confidence="0.85")
        with pytest.raises(ScoreAssessmentSchemaError):
            _build_decision_pack("sh.000001", "测试", "short", "2026-06-19", score_data)

    def test_key_signals_extracted_from_reasoning(self):
        score_data = _valid_score(
            score=72,
            recommendation="买入",
            reasoning="技术面利多信号明确，MACD金叉突破。但存在估值风险和流动性压力。",
            risk_warning="短期波动可能加大",
        )
        dp = _build_decision_pack("sh.000001", "测试", "short", "2026-06-19", score_data)
        assert dp.key_positive_signals is not None
        assert len(dp.key_positive_signals) > 0
        assert any("利多" in s for s in dp.key_positive_signals)
        assert dp.key_negative_signals is not None
        assert any("风险" in s for s in dp.key_negative_signals) or any(
            "短期波动" in s for s in dp.key_negative_signals
        )


class TestExtractSignalStrings:
    """Test the helper that extracts signal strings from reasoning text."""

    def test_positive_signals(self):
        reasoning = "技术面利多信号明显。基本面支撑较强。但存在估值风险。"
        signals = _extract_signal_strings(reasoning, "positive")
        assert len(signals) >= 1
        assert any("利多" in s for s in signals)

    def test_negative_signals(self):
        reasoning = "技术面利多信号明显。但存在估值风险。流动性压力较大。"
        signals = _extract_signal_strings(reasoning, "negative")
        assert len(signals) >= 1
        assert any("风险" in s for s in signals)

    def test_empty_reasoning(self):
        assert _extract_signal_strings("", "positive") == []
        assert _extract_signal_strings(None, "positive") == []

    def test_no_matching_keywords(self):
        reasoning = "无明显方向性信号。继续观察。"
        assert _extract_signal_strings(reasoning, "positive") == []
        assert _extract_signal_strings(reasoning, "negative") == []


class TestExtractSignalPacks:
    """Test extract_signal_packs_from_state."""

    def test_extract_signal_packs(self):
        state_data = {
            "fundamental_signal_pack": {"bias": "bullish", "confidence": 0.8},
            "technical_signal_pack": {"bias": "bullish", "confidence": 0.7},
            "value_signal_pack": {},  # empty dict -- not included
            "news_analysis": "some text",
            "fundamental_analysis": "some text",
        }
        packs = extract_signal_packs_from_state(state_data)
        assert "fundamental" in packs
        assert "technical" in packs
        assert "value" not in packs  # empty dict
        assert "news" not in packs  # not present at all
        assert "event" not in packs

    def test_all_empty(self):
        assert extract_signal_packs_from_state({}) == {}


class TestExtractAnalysisTexts:
    """Test extract_analysis_texts_from_state."""

    def test_extract_analysis_texts(self):
        state_data = {
            "fundamental_analysis": "基本面分析文本",
            "technical_analysis": "技术面分析文本",
        }
        texts = extract_analysis_texts_from_state(state_data)
        assert "fundamental" in texts
        assert "technical" in texts
        assert "news" not in texts

    def test_empty_texts_not_included(self):
        state_data = {
            "fundamental_analysis": "",
            "technical_analysis": None,
            "news_analysis": "有内容",
        }
        texts = extract_analysis_texts_from_state(state_data)
        assert "fundamental" not in texts  # empty string
        assert "technical" not in texts  # None
        assert "news" in texts

    def test_all_missing(self):
        assert extract_analysis_texts_from_state({}) == {}
