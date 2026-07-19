import shutil
import uuid
import json
from datetime import datetime
from pathlib import Path

import pytest

from src.eval import pool_manager as pool_manager_module
from src.eval import pool_screening
from src.eval.pool_manager import PoolManager


@pytest.fixture
def local_tmp_dir():
    """Use a relative workspace temp dir on Windows paths with non-ASCII users."""
    path = Path(".pool_contract_tests") / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _stock(code: str, score: float = 80.0):
    return {
        "code": code,
        "name": code,
        "final_score": score,
        "validity": "valid",
        "coverage": 1.0,
        "missing_core_fields": [],
        "scored_at": datetime.now().isoformat(),
    }


def _term_scores(score: float):
    return {
        "score": score,
        "recommendation": "推荐",
        "short_term_score": {"score": score, "recommendation": "买入"},
        "medium_term_score": {"score": score, "rating": "推荐"},
        "long_term_score": {"score": score, "rating": "推荐"},
    }


def _signal_packs():
    return {
        name: {"bias": "neutral", "signals": []}
        for name in (
            "fundamental",
            "technical",
            "value",
            "news",
            "event",
            "quality_risk",
            "moneyflow",
        )
    }


@pytest.mark.asyncio
async def test_layer1_invalid_result_is_excluded_not_neutral(monkeypatch):
    async def fake_fetch(stocks, **kwargs):
        return [
            {**stock, "status": "fetched", "data": {"pe": 10}}
            for stock in stocks
        ]

    async def fake_score(stocks, **kwargs):
        stocks[0]["score"] = {
            "level": "中性",
            "confidence": "中",
            "reason": "明确的业务评级",
            "validity": "valid",
            "coverage": 1.0,
            "missing_core_fields": [],
        }
        stocks[1]["score"] = {
            "level": None,
            "reason": "LLM未返回该股票",
            "validity": "invalid",
            "coverage": 0.0,
            "missing_core_fields": ["level"],
            "error_code": "missing_llm_item",
        }
        return stocks

    monkeypatch.setattr("src.api.batch_scorer.fetch_batch", fake_fetch)
    monkeypatch.setattr("src.api.batch_scorer.score_batch", fake_score)
    result = await pool_screening.batch_score_layer1(
        [
            {"ts_code": "600000.SH", "name": "A"},
            {"ts_code": "000001.SZ", "name": "B"},
        ],
        "medium",
    )

    assert result["whitelist"] == []
    assert result["initial_pool"] == []
    assert result["blacklist"] == []
    assert len(result["invalid"]) == 1
    assert result["invalid"][0]["code"] == "sz.000001"
    assert result["invalid"][0]["error_type"] == "missing_llm_item"


@pytest.mark.asyncio
async def test_formal_score_excludes_failure_instead_of_assigning_40_or_50(
    monkeypatch,
):
    class FakeScoringEngine:
        def __init__(self, **kwargs):
            pass

        async def score_stock(self, code, name):
            if code == "sh.600001":
                return {"error": "upstream unavailable", "score_data": None}
            return {
                "score_data": _term_scores(81),
                "signal_packs": _signal_packs(),
            }

    async def no_threshold_failure(scores, target_size):
        return 0.0

    monkeypatch.setattr(
        "src.stock_pool.scoring_engine.ScoringEngine", FakeScoringEngine
    )
    legacy_cache = json.dumps({
        "score_data": {"medium_term_score": {"score": 99, "rating": "推荐"}}
    })
    monkeypatch.setattr(
        "src.utils.cache_utils.read_cache", lambda *args: legacy_cache
    )
    monkeypatch.setattr("src.utils.cache_utils.write_cache", lambda *args: None)
    monkeypatch.setattr(pool_screening, "_dynamic_threshold", no_threshold_failure)

    result = await pool_screening.formal_score_layer3(
        [
            {"code": "sh.600000", "name": "good"},
            {"code": "sh.600001", "name": "failed"},
        ],
        [],
        term="medium",
        target_size=4,
        ratio=1.0,
    )

    assert [item["code"] for item in result["pool"]] == ["sh.600000"]
    assert result["pool"][0]["validity"] == "valid"
    assert result["stats"]["scored"] == 1
    assert result["stats"]["invalid"] == 1
    assert result["invalid"][0]["code"] == "sh.600001"
    assert "final_score" not in result["invalid"][0]


def test_checkpoint_is_invisible_and_publish_failure_keeps_current(
    local_tmp_dir, monkeypatch
):
    path = local_tmp_dir / "refined_pools.json"
    monkeypatch.setattr(pool_manager_module, "POOL_FILE", str(path))
    pm = PoolManager()
    pool_screening._publish_pool_generation(
        pm, "short", [_stock("sh.600000", 70)], reserve=[]
    )
    generation = pm._generation

    token = pool_screening._stage_pool_checkpoint(
        pm, "short", [_stock("sh.600001", 99)]
    )
    assert pm.repository.read()["short"]["stocks"][0]["code"] == "sh.600000"
    assert pm.repository.generation(pm.repository.read()) == generation
    assert pm.repository.status()["status"] == "updating"
    pool_screening._discard_staged_checkpoint(pm, token)

    def fail_publish(*args, **kwargs):
        raise OSError("fault injected before pointer switch")

    monkeypatch.setattr(pm.repository, "publish_staged", fail_publish)
    with pytest.raises(OSError, match="fault injected"):
        pool_screening._publish_pool_generation(
            pm, "short", [_stock("sh.600002", 88)], reserve=[]
        )

    persisted = pool_manager_module.RefinedPoolRepository(str(path)).read()
    assert persisted["short"]["stocks"][0]["code"] == "sh.600000"
    assert pool_manager_module.RefinedPoolRepository.generation(persisted) == generation
    assert pm.pools["short"]["stocks"][0]["code"] == "sh.600000"
    assert pm.repository.status()["staging"] is False


@pytest.mark.asyncio
async def test_partial_update_insufficient_valid_scores_keeps_previous_generation(
    local_tmp_dir, monkeypatch
):
    path = local_tmp_dir / "refined_pools.json"
    monkeypatch.setattr(pool_manager_module, "POOL_FILE", str(path))
    pm = PoolManager()
    current = [_stock(f"sh.{600000 + index:06d}", 60 + index / 10) for index in range(100)]
    reserve = [_stock(f"sz.{100000 + index:06d}", 70) for index in range(40)]
    pool_screening._publish_pool_generation(
        pm, "short", current, reserve=reserve
    )
    before = pm.repository.read()
    before_generation = pm.repository.generation(before)

    class FailingScoringEngine:
        def __init__(self, **kwargs):
            pass

        async def score_stock(self, code, name):
            return {"error": "model timeout", "score_data": None}

    monkeypatch.setattr(
        "src.stock_pool.scoring_engine.ScoringEngine", FailingScoringEngine
    )
    result = await pool_screening.run_pool_update_partial("short")

    assert result["error"] == "insufficient_valid_candidates"
    assert result["stats"]["valid_rescored"] == 0
    assert result["stats"]["invalid_rescored"] == 40
    after = pm.repository.read()
    assert pm.repository.generation(after) == before_generation
    assert after["short"]["stocks"] == before["short"]["stocks"]
    assert after["short"]["reserve"] == before["short"]["reserve"]


def test_pool_capacity_requires_explicitly_valid_safe_minimum():
    assert pool_screening._pool_capacity_failure("short", 89)
    assert pool_screening._pool_capacity_failure("short", 90) is None
