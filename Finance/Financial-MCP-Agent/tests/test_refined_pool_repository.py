import json
import multiprocessing
import os
import threading
from datetime import datetime, timedelta, timezone

import pytest

from src.advisory.recommendation import RecommendationEngine
from src.eval.refined_pool_repository import (
    RefinedPoolConflictError,
    RefinedPoolRepository,
    RefinedPoolValidationError,
)


def _pool(code="sh.600000", score=80):
    now = datetime.now().isoformat()
    return {
        "short": {"stocks": [{
            "code": code,
            "name": "浦发银行",
            "final_score": score,
            "validity": "valid",
            "coverage": 1.0,
            "missing_core_fields": [],
            "scored_at": now,
        }], "updated_at": now, "version": 1},
        "medium": {"stocks": [], "updated_at": now, "version": 1},
        "long": {"stocks": [], "updated_at": now, "version": 1},
    }


def _publish_worker(path, token, queue):
    """Top-level target so Windows' spawn context can import it."""
    try:
        RefinedPoolRepository(path).publish_staged(token, expected_generation=1)
        queue.put("published")
    except RefinedPoolConflictError:
        queue.put("conflict")
    except Exception as exc:  # pragma: no cover - reported to parent assertion
        queue.put(f"error:{type(exc).__name__}:{exc}")


def test_legacy_top_level_document_is_readable(tmp_path):
    path = tmp_path / "refined_pools.json"
    legacy = {
        "short": {
            "stocks": [{"code": "sh.600000", "final_score": 80}],
            "updated_at": "",
            "version": 0,
        }
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    repo = RefinedPoolRepository(str(path))
    assert repo.generation(repo.read()) == 0
    assert repo.status()["status"] == "current"


def test_new_publish_rejects_legacy_or_partial_shape(tmp_path):
    repo = RefinedPoolRepository(str(tmp_path / "refined_pools.json"))
    with pytest.raises(RefinedPoolValidationError):
        repo.publish({"short": {"stocks": [], "updated_at": "", "version": 0}})

    legacy_items = _pool()
    legacy_items["short"]["stocks"][0].pop("scored_at")
    with pytest.raises(RefinedPoolValidationError, match="scored_at"):
        repo.publish(legacy_items)


def test_staging_is_not_visible_until_atomic_publish(tmp_path):
    repo = RefinedPoolRepository(str(tmp_path / "refined_pools.json"))
    repo.publish(_pool(score=70), expected_generation=0)
    token = repo.stage(_pool(score=99), expected_generation=1)
    assert repo.read()["short"]["stocks"][0]["final_score"] == 70
    assert repo.status()["status"] == "updating"
    published = repo.publish_staged(token, expected_generation=1)
    assert published["short"]["stocks"][0]["final_score"] == 99
    assert repo.generation(published) == 2


def test_generation_cas_rejects_stale_writer(tmp_path):
    repo = RefinedPoolRepository(str(tmp_path / "refined_pools.json"))
    repo.publish(_pool(), expected_generation=0)
    stale = repo.stage(_pool(score=71), expected_generation=1)
    repo.publish(_pool(score=72), expected_generation=1)
    with pytest.raises(RefinedPoolConflictError):
        repo.publish_staged(stale, expected_generation=1)


def test_concurrent_writers_are_locked_and_only_one_cas_wins(tmp_path):
    path = str(tmp_path / "refined_pools.json")
    RefinedPoolRepository(path).publish(_pool(), expected_generation=0)
    barrier = threading.Barrier(2)
    outcomes = []

    def writer(score):
        repo = RefinedPoolRepository(path)
        token = repo.stage(_pool(score=score), expected_generation=1)
        barrier.wait()
        try:
            repo.publish_staged(token, expected_generation=1)
            outcomes.append("published")
        except RefinedPoolConflictError:
            outcomes.append("conflict")

    threads = [threading.Thread(target=writer, args=(score,)) for score in (81, 82)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["conflict", "published"]


def test_independent_process_writers_only_one_cas_wins(tmp_path):
    path = str(tmp_path / "refined_pools.json")
    repo = RefinedPoolRepository(path)
    repo.publish(_pool(), expected_generation=0)
    tokens = [
        repo.stage(_pool(score=score), expected_generation=1)
        for score in (83, 84)
    ]
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_publish_worker, args=(path, token, queue))
        for token in tokens
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    outcomes = [queue.get(timeout=5) for _ in processes]
    assert sorted(outcomes) == ["conflict", "published"]
    assert repo.generation(repo.read()) == 2


def test_replace_fault_keeps_previous_current_generation(tmp_path, monkeypatch):
    path = str(tmp_path / "refined_pools.json")
    repo = RefinedPoolRepository(path)
    repo.publish(_pool(score=70), expected_generation=0)
    token = repo.stage(_pool(score=99), expected_generation=1)
    original_replace = os.replace

    def fail_current_replace(source, target):
        if os.path.abspath(target) == os.path.abspath(path):
            raise OSError("fault injected before current switch")
        return original_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_current_replace)
    with pytest.raises(OSError, match="fault injected"):
        repo.publish_staged(token, expected_generation=1)

    current = repo.read()
    assert repo.generation(current) == 1
    assert current["short"]["stocks"][0]["final_score"] == 70


def test_directory_fsync_fault_is_best_effort_after_atomic_replace(
    tmp_path, monkeypatch
):
    repo = RefinedPoolRepository(str(tmp_path / "refined_pools.json"))

    def fail_directory_fsync(_directory):
        raise OSError("directory fsync unavailable")

    monkeypatch.setattr(RefinedPoolRepository, "_fsync_directory", fail_directory_fsync)
    published = repo.publish(_pool(score=86), expected_generation=0)
    assert repo.generation(published) == 1
    assert repo.read()["short"]["stocks"][0]["final_score"] == 86


def test_abandoned_staging_is_reported_stale_not_updating(tmp_path):
    repo = RefinedPoolRepository(str(tmp_path / "refined_pools.json"))
    repo.publish(_pool(score=70), expected_generation=0)
    token = repo.stage(_pool(score=71), expected_generation=1)
    stage_path = os.path.join(repo.staging_dir, token + ".json")
    staged = json.loads(open(stage_path, encoding="utf-8").read())
    staged["created_at"] = (
        datetime.now(timezone.utc) - timedelta(days=2)
    ).isoformat()
    with open(stage_path, "w", encoding="utf-8") as handle:
        json.dump(staged, handle)

    status = repo.status()
    assert status["status"] == "current"
    assert status["staging"] is False
    assert status["stale_staging"] is True


def test_advisory_preserves_same_stock_in_multiple_terms(tmp_path):
    now = datetime.now().isoformat()
    document = _pool(score=88)
    document["medium"]["stocks"] = [{
        "code": "sh.600000",
        "name": "浦发银行",
        "final_score": 76,
        "validity": "valid",
        "coverage": 1.0,
        "missing_core_fields": [],
        "scored_at": now,
    }]
    document["medium"]["updated_at"] = now
    pool_path = tmp_path / "refined_pools.json"
    RefinedPoolRepository(str(pool_path)).publish(document, expected_generation=0)
    engine = RecommendationEngine(str(pool_path), str(tmp_path / "cache"))
    stocks = engine.load_pool_stocks()
    assert len(stocks) == 1
    assert stocks[0]["terms"] == ["short", "medium"]
    assert stocks[0]["term_scores"]["short"]["score"] == 88
    assert stocks[0]["term_scores"]["medium"]["score"] == 76


def test_term_specific_cache_does_not_cross_terms(tmp_path):
    pool_path = tmp_path / "refined_pools.json"
    RefinedPoolRepository(str(pool_path)).publish(_pool(), expected_generation=0)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "short_term_scorer_sh_600000_2099-01-02.json").write_text(
        json.dumps({"content": json.dumps({
            "score": 91,
            "validity": "valid",
            "coverage": 1.0,
            "missing_core_fields": [],
        })}), encoding="utf-8"
    )
    (cache_dir / "medium_term_scorer_sh_600000_2099-01-01.json").write_text(
        json.dumps({
            "score": 61,
            "validity": "valid",
            "coverage": 1.0,
            "missing_core_fields": [],
        }), encoding="utf-8"
    )
    engine = RecommendationEngine(str(pool_path), str(cache_dir))
    assert engine.get_score_cache_info("sh.600000", "short")["score"] == 91
    assert engine.get_score_cache_info("sh.600000", "medium")["score"] == 61


def test_legacy_cache_is_readable_but_not_actionable(tmp_path):
    pool_path = tmp_path / "refined_pools.json"
    RefinedPoolRepository(str(pool_path)).publish(_pool(), expected_generation=0)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "short_term_scorer_sh_600000_2099-01-02.json").write_text(
        json.dumps({"content": json.dumps({"score": 91})}), encoding="utf-8"
    )
    engine = RecommendationEngine(str(pool_path), str(cache_dir))
    info = engine.get_score_cache_info("sh.600000", "short")
    assert info["score"] is None
    assert info["validity"] == "legacy_non_actionable"
