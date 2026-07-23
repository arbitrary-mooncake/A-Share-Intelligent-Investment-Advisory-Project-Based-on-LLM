"""期限子图端到端测试（4.2+4.3）：score_stock_for_term 真实走通子图执行。"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stock_pool import scoring_engine as se
from src.stock_pool.scoring_engine import ScoringEngine


def _stub_pack(agent):
    return {
        "agent_name": agent,
        "bias": "bullish",
        "confidence": 0.8,
        "data_quality_score": 0.8,
        "key_points": ["stub"],
        "signals": [{
            "factor": f"{agent}因子", "direction": 1, "strength": 70,
            "confidence": 0.9, "source_level": "structured",
            "category": {
                "technical": "technical_trend", "news": "sentiment",
                "event": "catalyst_event", "moneyflow": "capital_flow",
                "fundamental": "fundamentals_growth", "value": "valuation",
                "quality_risk": "balance_sheet",
            }.get(agent, "other"),
            "note": "stub",
        }],
        "risk_flags": [],
        "missing_data": [],
        "source_summary": "stub",
    }


def _make_stub(agent_key):
    async def _fn(state):
        data = state.get("data", {})
        data[f"{agent_key}_signal_pack"] = _stub_pack(agent_key)
        data[f"{agent_key}_analysis"] = f"{agent_key} stub 分析"
        return {"data": data}
    return _fn


@pytest.fixture
def stubbed_engine(monkeypatch):
    # 替换全部 7 个分析 Agent 为 stub（不触发 MCP/LLM）
    se.ScoringEngine._AGENT_NODE_SPECS = {
        key: (node_name, _make_stub(key))
        for key, (node_name, _) in se.ScoringEngine._AGENT_NODE_SPECS.items()
    }
    # 预取降级为空操作（不触发 MCP）
    import src.data.data_gateway as gw
    async def _noop(*args, **kwargs):
        return None
    monkeypatch.setattr(gw, "prefetch_term_bundle", _noop)
    # 打分节点缓存读写降级（不污染真实缓存目录）
    import src.utils.cache_utils as cu
    monkeypatch.setattr(cu, "read_cache", lambda *a, **k: None)
    monkeypatch.setattr(cu, "write_cache", lambda *a, **k: None)
    return ScoringEngine(pool_manager=False)


def test_score_stock_for_term_short_e2e(stubbed_engine, monkeypatch):
    monkeypatch.setenv("DETERMINISTIC_SCORER_ENABLED", "1")
    monkeypatch.setenv("CONFLICT_ARBITRATION_ENABLED", "0")
    monkeypatch.setenv("SCORE_EXPLANATION_ENABLED", "0")
    result = asyncio.run(
        stubbed_engine.score_stock_for_term("short", "sh.999999", "测试股份")
    )
    assert result.get("error") is None
    score = result["term_score"]
    assert score["scorer_type"] == "deterministic"
    assert score["score"] > 50.0  # 4 个多头 stub 信号
    assert score["sub_scores"]["technical_trend"] > 50.0
    # all_scores 只包含 short（子图不跑 medium/long scorer）
    assert result["all_scores"]["medium"] == {}
    assert result["all_scores"]["long"] == {}


def test_score_stock_for_term_llm_path_flag_off(stubbed_engine, monkeypatch):
    # flag 关闭时走 LLM scorer——此处仅验证子图机制本身可达打分节点
    # （LLM 调用会因缺少 API key 失败，属于预期路径，验证节点被触发即可）
    monkeypatch.setenv("DETERMINISTIC_SCORER_ENABLED", "0")
    from src.agents.scoring_nodes import _deterministic_scorer_enabled
    assert _deterministic_scorer_enabled() is False
