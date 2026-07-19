"""Layered contract tests for the formal replay PIT boundary."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from src.eval.pit_boundary import (
    ContentAddressedSnapshotStore,
    PITDataGateway,
    PITSnapshotCorruptError,
)
from src.eval.replay_backtest_engine import BacktestConfig, ReplayBacktestEngine
from src.eval.schemas import PredictionSnapshot


ANCHOR = "2024-06-14"


def _requester(*, revised: bool = False, include_future: bool = False):
    anchor = date.fromisoformat(ANCHOR)

    def request(api_name, params, fields):
        ts_code = params["ts_code"]
        if api_name == "daily":
            rows = []
            for offset in range(90):
                day = anchor - timedelta(days=offset)
                close = 100.0 - offset * 0.2
                rows.append({
                    "ts_code": ts_code,
                    "trade_date": day.strftime("%Y%m%d"),
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "pre_close": close - 0.1,
                    "vol": 1000,
                    "amount": 100000,
                })
            if include_future:
                rows.append({
                    "ts_code": ts_code,
                    "trade_date": "20240615",
                    "open": 999,
                    "high": 999,
                    "low": 999,
                    "close": 999,
                    "pre_close": 100,
                    "vol": 1,
                    "amount": 1,
                })
            return {"items": rows}
        if api_name == "daily_basic":
            return {"items": [{
                "ts_code": ts_code,
                "trade_date": "20240614",
                "pe": 18.0,
                "pb": 2.0,
                "turnover_rate": 1.2,
                "total_mv": 1000000,
            }]}
        if api_name == "fina_indicator":
            rows = [{
                "ts_code": ts_code,
                "ann_date": "20240420",
                "end_date": "20240331",
                "update_flag": "1" if revised else "0",
                "roe": 14.0,
                "roa": 7.0,
                "grossprofit_margin": 35.0,
                "netprofit_margin": 12.0,
                "or_yoy": 15.0,
                "profit_yoy": 18.0,
                "debt_to_assets": 42.0,
                "current_ratio": 1.8,
                "ocf_to_or": 0.15,
            }]
            if include_future:
                rows.append({
                    **rows[0],
                    "ann_date": "20240701",
                    "end_date": "20240630",
                    "roe": 99.0,
                })
            return {"items": rows}
        if api_name == "moneyflow":
            return {"items": [{
                "ts_code": ts_code,
                "trade_date": (anchor - timedelta(days=offset)).strftime("%Y%m%d"),
                "net_mf_amount": 1000.0,
            } for offset in range(20)]}
        raise AssertionError(f"unexpected API: {api_name}")

    return request


def _prices(code: str, *, multiplier: float = 1.0):
    start = date.fromisoformat(ANCHOR)
    return {
        code: {
            (start + timedelta(days=offset)).isoformat():
                (100.0 + 2.0 * offset) * multiplier
            for offset in range(5)
        }
    }


def _engine(tmp_path, *, requester=None, snapshot_ids=None, minimum=1):
    store = ContentAddressedSnapshotStore(tmp_path / "snapshots")
    gateway = PITDataGateway(requester or _requester(), snapshot_store=store)
    config = BacktestConfig(
        term="medium",
        holding_days=2,
        minimum_common_samples=minimum,
    )
    return ReplayBacktestEngine(
        config,
        pit_gateway=gateway,
        snapshot_ids=snapshot_ids,
    ), store


@pytest.mark.asyncio
async def test_future_prices_are_labels_and_cannot_change_score(tmp_path):
    engine, _ = _engine(tmp_path, requester=_requester(include_future=True))
    score_before = await engine._score_stock_pit("sh.600000", "x", ANCHOR, "full")
    result_a = await engine.run_single_anchor(
        ANCHOR, ["sh.600000"], _prices("sh.600000")
    )
    result_b = await engine.run_single_anchor(
        ANCHOR, ["sh.600000"], _prices("sh.600000", multiplier=50.0)
    )
    score_after = await engine._score_stock_pit("sh.600000", "x", ANCHOR, "full")

    assert score_before == score_after
    assert result_a["ablation_results"]["full"]["scores"] == [score_before]
    assert result_b["ablation_results"]["full"]["scores"] == [score_before]
    sample = result_a["ablation_results"]["full"]["samples"][0]
    assert sample["sample_id"] == f"{ANCHOR}|sh.600000"
    snapshot = engine._gateway().snapshot_store.get(sample["data_snapshot_id"])
    assert all(row["trade_date"] <= "20240614" for row in snapshot["normalized_inputs"]["daily"])
    assert all(row["ann_date"] <= "20240614" for row in snapshot["normalized_inputs"]["fina_indicator"])


@pytest.mark.asyncio
async def test_common_valid_join_excludes_missing_outcome_without_zero_return(tmp_path):
    engine, _ = _engine(tmp_path, minimum=1)
    pool = ["sh.600000", "sz.000001"]
    result = await engine.run_single_anchor(ANCHOR, pool, _prices("sh.600000"))

    expected_id = f"{ANCHOR}|sh.600000"
    assert result["common_valid_sample_ids"] == [expected_id]
    assert result["coverage"]["common_valid_samples"] == 1
    assert result["coverage"]["common_valid_ratio"] == pytest.approx(0.5)
    for variant in result["ablation_results"].values():
        assert [row["sample_id"] for row in variant["samples"]] == [expected_id]
        assert variant["returns"] != [0.0]
    summary = engine._summarize_contributions([result])
    assert summary["agent_contribution_eligible"] is False
    assert summary["experiment_type"] == "factor_domain_ablation"


@pytest.mark.asyncio
async def test_best_effort_financial_vintage_abstains_from_formal_replay(tmp_path):
    engine, _ = _engine(tmp_path, requester=_requester(revised=True))
    result = await engine.run_single_anchor(
        ANCHOR, ["sh.600000"], _prices("sh.600000")
    )
    assert result["validity"] == "abstain"
    assert result["ablation_results"]["full"]["scores"] == []
    assert any(item["error_type"] == "PITDataUnavailableError" for item in result["abstentions"])


@pytest.mark.asyncio
async def test_snapshot_offline_replay_and_tamper_detection(tmp_path):
    live, store = _engine(tmp_path)
    first = live._score_factor_record("sh.600000", ANCHOR, "full")
    assert first["validity"] == "valid"

    def offline(*_args, **_kwargs):
        raise AssertionError("offline replay must not fetch")

    replay_gateway = PITDataGateway(offline, snapshot_store=store)
    replay = ReplayBacktestEngine(
        live.config,
        pit_gateway=replay_gateway,
        snapshot_ids={(ANCHOR, "sh.600000"): first["data_snapshot_id"]},
    )
    second = replay._score_factor_record("sh.600000", ANCHOR, "full")
    assert second["score"] == first["score"]
    assert second["context_fingerprint"] == first["context_fingerprint"]

    digest = first["data_snapshot_id"].split(":", 1)[1]
    path = store.root / f"{digest}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["normalized_inputs"]["daily"][0]["close"] = 123456
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PITSnapshotCorruptError):
        store.get(first["data_snapshot_id"])


@pytest.mark.asyncio
async def test_unsupported_formal_entrances_are_explicit(tmp_path):
    engine, _ = _engine(tmp_path)
    regime = await engine.run_regime_analysis([], {}, benchmark_data=None)
    calibration = await engine.run_quarterly_calibration()
    assert regime["status"] == "unsupported"
    assert regime["agent_contribution_eligible"] is False
    assert calibration["status"] == "unsupported"
    assert calibration["error_type"] == "pit_dynamic_universe_unavailable"


def test_legacy_prediction_snapshot_is_not_exact_by_default():
    snapshot = PredictionSnapshot.from_dict({
        "snapshot_id": "old", "score": 50, "pit_mode": "exact"
    })
    assert snapshot.pit_mode == "legacy_non_pit"
    assert snapshot.score_validity == "invalid"
    assert snapshot.agent_contribution_eligible is False


def test_legacy_replay_rows_cannot_enter_formal_factor_trend(tmp_path):
    engine, _ = _engine(tmp_path)
    legacy = {
        "anchor_date": ANCHOR,
        "pool_size": 1,
        "pit_mode": "legacy_non_pit",
        "ablation_results": {
            "full": {"samples": [{"sample_id": "old", "score": 80, "return": 0.2}]},
            "-fundamental": {"samples": [{"sample_id": "old", "score": 60, "return": 0.2}]},
        },
    }
    summary = engine._summarize_contributions([legacy])
    assert summary["validity"] == "abstain"
    assert summary["domains"] == {}
    assert summary["coverage"]["common_valid_samples"] == 0
