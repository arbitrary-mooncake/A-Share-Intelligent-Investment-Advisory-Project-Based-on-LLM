"""Tests for DecisionPack"""
from src.utils.analysis_schema import DecisionPack

def test_decision_pack_normalize_full():
    data = {
        "asset_type": "stock",
        "symbol": "sh.603871",
        "name": "嘉友国际",
        "task_type": "single_stock",
        "term": "short",
        "as_of_date": "2026-06-19",
        "action": "buy",
        "score": 75,
        "confidence": 0.8,
        "data_quality_score": 0.7,
        "risk_gate_applied": True,
        "risk_gate_result": {"risk_level": "low"},
        "supporting_agents": ["fundamental", "technical"],
        "missing_agents": ["news"],
        "key_positive_signals": ["技术突破"],
        "key_negative_signals": ["数据缺失"],
        "conflicts": ["短中矛盾"],
        "model_profile": "eval_analysis",
        "version_hash": "abc123",
        "meta": {"extra": "info"}
    }
    dp = DecisionPack.normalize(data)
    assert dp.asset_type == "stock"
    assert dp.symbol == "sh.603871"
    assert dp.score == 75.0
    assert dp.confidence == 0.8
    assert dp.risk_gate_applied == True
    assert dp.supporting_agents == ["fundamental", "technical"]

def test_decision_pack_normalize_string_numbers():
    data = {"asset_type": "stock", "symbol": "sz.300308", "score": "80", "confidence": "0.9"}
    dp = DecisionPack.normalize(data)
    assert dp.score == 80.0
    assert dp.confidence == 0.9

def test_decision_pack_normalize_empty():
    dp = DecisionPack.normalize({})
    assert dp.asset_type == "unknown"
    assert dp.symbol == "unknown"
    assert dp.score == 0.0

def test_decision_pack_normalize_none():
    dp = DecisionPack.normalize(None)
    assert dp.asset_type == "unknown"

def test_decision_pack_to_json():
    dp = DecisionPack(asset_type="stock", symbol="sh.603871", score=75.0)
    j = dp.to_json()
    assert j["asset_type"] == "stock"
    assert j["score"] == 75.0

def test_decision_pack_defaults():
    dp = DecisionPack(asset_type="stock", symbol="sh.603871")
    assert dp.name == ""
    assert dp.action == ""
    assert dp.confidence == 0.0
    assert dp.risk_gate_applied == False
