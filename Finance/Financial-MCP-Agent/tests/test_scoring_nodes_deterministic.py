"""打分节点确定性路径端到端测试（4.3d）：flag 开启时走纯代码打分，flag 关闭路径不变。"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.state_definition import AgentState


def _pack(agent, bias, signals):
    return {
        "agent_name": agent,
        "bias": bias,
        "confidence": 0.8,
        "data_quality_score": 0.8,
        "key_points": ["测试要点"],
        "signals": signals,
        "risk_flags": [],
        "missing_data": [],
        "source_summary": "test",
        "as_of_date": "2026-07-20",
    }


def _sig(factor, direction, category, strength=80, source="structured"):
    return {"factor": factor, "direction": direction, "strength": strength,
            "confidence": 0.9, "source_level": source, "category": category,
            "note": "测试"}


def _make_state():
    return AgentState(
        messages=[],
        data={
            "stock_code": "sh.999999",
            "company_name": "测试股份",
            "current_date": "2026-07-20",
            "current_time_info": "2026年07月20日 星期一",
            "skip_cache": True,  # 不读写真实缓存
            "is_etf": False,
            "technical_signal_pack": _pack("technical", "bullish",
                                           [_sig("均线多头", 1, "technical_trend")]),
            "news_signal_pack": _pack("news", "neutral",
                                      [_sig("舆情平稳", 0, "sentiment", strength=30)]),
            "event_signal_pack": _pack("event", "bullish",
                                       [_sig("回购公告", 1, "catalyst_event",
                                             source="official_like")]),
            "moneyflow_signal_pack": _pack("moneyflow", "bullish",
                                           [_sig("主力流入", 1, "capital_flow")]),
        },
        metadata={},
    )


def test_short_node_deterministic_path(monkeypatch):
    monkeypatch.setenv("DETERMINISTIC_SCORER_ENABLED", "1")
    monkeypatch.setenv("CONFLICT_ARBITRATION_ENABLED", "0")  # 纯代码默认折扣
    monkeypatch.setenv("SCORE_EXPLANATION_ENABLED", "0")
    from src.agents.scoring_nodes import short_term_scorer_node
    result = asyncio.run(short_term_scorer_node(_make_state()))
    score = result["data"]["short_term_score"]
    assert score["scorer_type"] == "deterministic"
    assert 0 < score["score"] <= 100
    assert score["score"] > 50.0  # 三条多头信号
    assert "risk_gate" in score
    assert score["recommendation"]
    assert score["sub_scores"]["technical_trend"] > 50.0


def test_medium_node_deterministic_path(monkeypatch):
    monkeypatch.setenv("DETERMINISTIC_SCORER_ENABLED", "1")
    monkeypatch.setenv("CONFLICT_ARBITRATION_ENABLED", "0")
    monkeypatch.setenv("SCORE_EXPLANATION_ENABLED", "0")
    from src.agents.scoring_nodes import medium_term_scorer_node
    result = asyncio.run(medium_term_scorer_node(_make_state()))
    score = result["data"]["medium_term_score"]
    assert score["scorer_type"] == "deterministic"
    # medium 缺 fundamental/value/quality_risk 三个核心 Agent → 分数上限 40
    assert score["score"] <= 40.0
    assert set(score["missing_core_fields"]) == {"fundamental", "value", "quality_risk"}


def test_llm_path_unchanged_when_flag_off(monkeypatch):
    monkeypatch.setenv("DETERMINISTIC_SCORER_ENABLED", "0")
    from src.agents.scoring_nodes import _deterministic_scorer_enabled
    assert _deterministic_scorer_enabled() is False
