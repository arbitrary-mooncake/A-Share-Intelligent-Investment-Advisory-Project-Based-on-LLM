"""Regression tests for V3 Layer 2 failure isolation and dispatch safety."""

import asyncio

import pytest

from src.eval import pool_screening


def _layer1_batch(offset=0):
    batch = []
    for index in range(100):
        level = "强烈推荐" if index < 75 else "推荐"
        batch.append({
            "code": f"sh.{600000 + offset + index:06d}",
            "name": f"测试{index}",
            "layer1_level": level,
            "_score_validity": "valid",
            "_score_coverage": 1.0,
            "_raw_data": {"last_price": 10 + index},
        })
    return batch


class _FakePoolManager:
    def __init__(self):
        self.pools = {
            "blacklist": {"short": [], "medium": [], "long": []},
        }

    def snapshot_term(self, term):
        return []

    def validate_pool_replacement(self, term, pool, previous_snapshot=None):
        return None


async def _fake_layer3_consumer(
    queue, results, whitelist_final, blacklist_final, invalid_results,
    stage_failures, lock, term, progress=None, heartbeat_interval=30.0,
    per_task_timeout=900.0,
):
    while True:
        stock = await queue.get()
        try:
            if stock is None:
                return
            score = float(stock.get("layer2_score", 85.0))
            entry = {
                **stock,
                "final_score": score,
                "recommendation": "推荐",
                "validity": "valid",
                "coverage": 1.0,
                "missing_core_fields": [],
                "scored_at": "2026-07-23T00:00:00",
            }
            results.append(entry)
        finally:
            queue.task_done()


def _install_v3_fakes(monkeypatch, l2_function, batches=1):
    published = {}

    async def fake_hard_screen(term=None):
        return [
            {"ts_code": f"{600000 + index:06d}.SH", "name": "测试"}
            for index in range(batches * 100)
        ]

    async def fake_layer1_stream(*args, **kwargs):
        for batch_index in range(batches):
            yield _layer1_batch(batch_index * 100)

    def fake_publish(pm, term, pool, **kwargs):
        published["pool"] = pool

    monkeypatch.setattr(pool_screening, "PoolManager", _FakePoolManager)
    monkeypatch.setattr(pool_screening, "hard_screen", fake_hard_screen)
    monkeypatch.setattr(
        pool_screening, "batch_score_layer1_stream", fake_layer1_stream
    )
    monkeypatch.setattr(pool_screening, "score_layer2_batch", l2_function)
    monkeypatch.setattr(
        pool_screening, "_layer3_consumer", _fake_layer3_consumer
    )
    monkeypatch.setattr(pool_screening, "_publish_pool_generation", fake_publish)
    return published


def test_layer2_retry_classifier_distinguishes_data_gap():
    assert pool_screening._layer2_results_need_retry([
        {"validity": "invalid", "error_type": "layer2_timeout"}
    ])
    assert pool_screening._layer2_results_need_retry([
        {"validity": "abstain", "error_type": "insufficient_data"}
    ]) is False
    assert pool_screening._normalize_layer2_results(
        [{"code": "sh.600000"}], []
    )[0]["error_type"] == "missing_layer2_item"


def test_heap_deduplicates_and_releases_reserved_candidates():
    heap = pool_screening.StreamingTopAlphaHeap(
        alpha_fn=lambda: 1.0, stable_batches=1
    )
    stock = {"code": "sh.600000", "name": "测试"}
    heap.feed(80.0, stock)
    heap.feed(90.0, {"code": "sh.600000", "name": "重复"})

    assert heap.total == 1
    candidates = heap.finalize()
    assert [item["code"] for item in candidates] == ["sh.600000"]
    assert heap.finalize() == []

    heap.release_dispatch("sh.600000")
    assert [item["code"] for item in heap.finalize()] == ["sh.600000"]
    heap.mark_dispatched("sh.600000")
    assert heap.finalize() == []


@pytest.mark.asyncio
async def test_v3_retries_transient_layer2_chunk_and_publishes(monkeypatch):
    calls = 0

    async def flaky_l2(chunk, term, pre_fetched_data=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                {
                    "code": stock["code"],
                    "validity": "invalid",
                    "coverage": 0.0,
                    "missing_core_fields": ["score"],
                    "error_type": "layer2_timeout",
                    "error_message": "temporary provider timeout",
                }
                for stock in chunk
            ]
        return [
            {
                "code": stock["code"],
                "score": 70.0,
                "validity": "valid",
                "coverage": 1.0,
                "missing_core_fields": [],
            }
            for stock in chunk
        ]

    published = _install_v3_fakes(monkeypatch, flaky_l2)
    result = await pool_screening.run_pool_update_v3("short")

    assert "error" not in result
    assert calls == 2
    assert len(result["pool"]) == 100
    assert len(published["pool"]) == 100
    assert result["stats"]["layer2_invalid"] == 0


@pytest.mark.asyncio
async def test_v3_keeps_previous_pool_when_layer2_failure_leaves_underfill(
    monkeypatch,
):
    calls = 0

    async def unavailable_l2(chunk, term, pre_fetched_data=None):
        nonlocal calls
        calls += 1
        return [
            {
                "code": stock["code"],
                "validity": "invalid",
                "coverage": 0.0,
                "missing_core_fields": ["score"],
                "error_type": "layer2_timeout",
                "error_message": "provider unavailable",
            }
            for stock in chunk
        ]

    published = _install_v3_fakes(monkeypatch, unavailable_l2)
    result = await pool_screening.run_pool_update_v3("short")

    assert result["error"] == "pool_result_guard"
    assert calls == 2
    assert "pool" not in published
    assert result["stats"]["pool_size"] == 75
    assert result["stats"]["layer2_invalid"] == 25


@pytest.mark.asyncio
async def test_v3_continues_when_one_layer2_chunk_fails(monkeypatch):
    calls = 0
    failed_code = "sh.600075"  # first Layer 1 batch's first recommended chunk

    async def one_chunk_broken_l2(chunk, term, pre_fetched_data=None):
        nonlocal calls
        calls += 1
        if chunk and chunk[0]["code"] == failed_code:
            raise ConnectionError("temporary provider outage")
        return [
            {
                "code": stock["code"],
                "score": 70.0,
                "validity": "valid",
                "coverage": 1.0,
                "missing_core_fields": [],
            }
            for stock in chunk
        ]

    published = _install_v3_fakes(monkeypatch, one_chunk_broken_l2, batches=2)
    result = await pool_screening.run_pool_update_v3("short")

    assert "error" not in result
    assert calls > 2
    assert result["stats"]["layer2_invalid"] == 25
    assert len(published["pool"]) == 100


@pytest.mark.asyncio
async def test_v3_keeps_fatal_layer2_exception_fail_closed(monkeypatch):
    async def broken_l2(chunk, term, pre_fetched_data=None):
        raise RuntimeError("schema invariant broken")

    published = _install_v3_fakes(monkeypatch, broken_l2)
    result = await pool_screening.run_pool_update_v3("short")

    assert result["error"] == "layer2_stage_failed"
    assert "schema invariant broken" in result["message"]
    assert "pool" not in published


@pytest.mark.asyncio
async def test_v3_limits_global_layer2_batch_concurrency(monkeypatch):
    active = 0
    max_active = 0

    async def observed_l2(chunk, term, pre_fetched_data=None):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.01)
            return [
                {
                    "code": stock["code"],
                    "score": 70.0,
                    "validity": "valid",
                    "coverage": 1.0,
                    "missing_core_fields": [],
                }
                for stock in chunk
            ]
        finally:
            active -= 1

    published = _install_v3_fakes(monkeypatch, observed_l2, batches=2)
    result = await pool_screening.run_pool_update_v3("short")

    assert "error" not in result
    assert max_active == 1
    assert len(published["pool"]) == 100
