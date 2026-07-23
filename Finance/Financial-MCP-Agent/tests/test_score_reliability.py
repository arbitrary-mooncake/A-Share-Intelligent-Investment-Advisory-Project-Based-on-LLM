"""Regression tests for production score persistence failure boundaries."""

import copy

import pytest

from src.stock_pool.scoring_engine import ScoringEngine
from src.stock_pool.stock_pool_manager import StockPoolManager


CODE = "sh.600000"


def _term_score(score: float):
    return {
        "score": score,
        "rating": "推荐",
        "recommendation": "推荐",
    }


def _valid_score_data():
    return {
        "score": 72,
        "recommendation": "推荐",
        "short_term_score": _term_score(68),
        "medium_term_score": _term_score(72),
        "long_term_score": _term_score(76),
        "company_name": "浦发银行",
        "status": "scored",
    }


def _seed_pool(tmp_path):
    manager = StockPoolManager(str(tmp_path / "stock_pool.json"))
    for term in ("short", "medium", "long"):
        manager.add_stock_to_term(term, CODE, "浦发银行")
    manager.update_stock_score(CODE, _valid_score_data())
    return manager


class _FakeWorkflow:
    def __init__(self, data):
        self.data = data

    async def ainvoke(self, state):
        return {"data": self.data}


@pytest.mark.asyncio
async def test_score_stock_failure_preserves_previous_scores(tmp_path, monkeypatch):
    manager = _seed_pool(tmp_path)
    before = copy.deepcopy(manager.pools)
    engine = ScoringEngine(pool_manager=manager)

    def fail_workflow():
        raise RuntimeError("upstream unavailable")

    async def no_prefetch(*args, **kwargs):
        return None

    monkeypatch.setattr(engine, "_build_workflow", fail_workflow)
    monkeypatch.setattr(engine, "_prefetch", no_prefetch)

    result = await engine.score_stock(CODE, "浦发银行")

    assert result["score_data"] is None
    assert "upstream unavailable" in result["error"]
    for term in ("short", "medium", "long"):
        current = manager.pools[term]["stocks"][CODE]
        assert current["score"] == before[term]["stocks"][CODE]["score"]
        assert current["term_score"] == before[term]["stocks"][CODE]["term_score"]
        assert current["score_history"] == before[term]["stocks"][CODE]["score_history"]
        assert current["status"] == "failed"


def test_failed_payload_does_not_add_or_clear_stock(tmp_path):
    manager = _seed_pool(tmp_path)
    before = copy.deepcopy(manager.pools)

    manager.update_stock_score(
        "sz.000001",
        {
            "score": None,
            "company_name": "平安银行",
            "status": "failed",
            "short_term_score": {},
            "medium_term_score": {},
            "long_term_score": {},
        },
    )
    manager.update_stock_score(
        CODE,
        {
            "score": None,
            "company_name": "浦发银行",
            "status": "failed",
            "short_term_score": {},
            "medium_term_score": {},
            "long_term_score": {},
        },
    )

    assert all(
        "sz.000001" not in manager.pools[term]["stocks"]
        for term in ("short", "medium", "long")
    )
    for term in ("short", "medium", "long"):
        current = manager.pools[term]["stocks"][CODE]
        assert current["score"] == before[term]["stocks"][CODE]["score"]
        assert current["term_score"] == before[term]["stocks"][CODE]["term_score"]
        assert current["status"] == "failed"


@pytest.mark.asyncio
async def test_valid_score_stock_still_persists_all_terms(tmp_path, monkeypatch):
    manager = _seed_pool(tmp_path)
    engine = ScoringEngine(pool_manager=manager)
    data = {
        "short_term_score": _term_score(61),
        "medium_term_score": _term_score(64),
        "long_term_score": _term_score(67),
    }

    async def no_prefetch(*args, **kwargs):
        return None

    monkeypatch.setattr(engine, "_build_workflow", lambda: _FakeWorkflow(data))
    monkeypatch.setattr(engine, "_prefetch", no_prefetch)

    result = await engine.score_stock(CODE, "浦发银行")

    assert result["score_data"]["score"] == 64
    assert manager.pools["short"]["stocks"][CODE]["score"] == 61
    assert manager.pools["medium"]["stocks"][CODE]["score"] == 64
    assert manager.pools["long"]["stocks"][CODE]["score"] == 67
    assert all(
        manager.pools[term]["stocks"][CODE]["status"] == "scored"
        for term in ("short", "medium", "long")
    )


@pytest.mark.asyncio
async def test_term_score_none_is_failure_and_preserves_previous_value(tmp_path, monkeypatch):
    manager = _seed_pool(tmp_path)
    before = manager.pools["medium"]["stocks"][CODE]["score"]
    engine = ScoringEngine(pool_manager=manager)

    async def no_prefetch(*args, **kwargs):
        return None

    monkeypatch.setattr(
        engine,
        "_build_term_workflow",
        lambda term: _FakeWorkflow({"medium_term_score": {"score": None}}),
    )
    monkeypatch.setattr(engine, "_prefetch", no_prefetch)

    result = await engine.score_stock_for_term("medium", CODE, "浦发银行")

    assert "error" in result
    assert manager.pools["medium"]["stocks"][CODE]["score"] == before
    assert manager.pools["medium"]["stocks"][CODE]["status"] == "failed"


def test_invalid_term_payload_does_not_overwrite_score(tmp_path):
    manager = _seed_pool(tmp_path)
    before = manager.pools["long"]["stocks"][CODE]["score"]

    manager.update_term_score(
        "long", CODE, {"score": float("nan"), "status": "failed"}
    )

    assert manager.pools["long"]["stocks"][CODE]["score"] == before
    assert manager.pools["long"]["stocks"][CODE]["status"] == "failed"
