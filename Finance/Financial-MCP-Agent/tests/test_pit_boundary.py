"""No-network tests for strict point-in-time data boundaries."""
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.eval.pit_boundary import (
    AsOfContext,
    ContentAddressedSnapshotStore,
    EvidenceChannel,
    FeatureLabelBoundary,
    PITDataGateway,
    PITDataUnavailableError,
    PITMode,
    PITSnapshotCorruptError,
    PITTemporalViolation,
    TimedEvidence,
    select_fina_indicator_rows,
)


def _context(**overrides):
    values = {
        "trade_date": "2024-03-31",
        "knowledge_cutoff": "2024-03-31",
        "model_profile": "eval_m5",
        "prompt_version": "pit-score-v1",
        "cache_namespace": "pit_v1",
    }
    values.update(overrides)
    return AsOfContext(**values)


def test_as_of_context_is_immutable_and_fingerprint_covers_all_axes():
    context = _context()

    with pytest.raises(FrozenInstanceError):
        context.model_profile = "production"  # type: ignore[misc]

    assert context.fingerprint != _context(model_profile="other").fingerprint
    assert context.fingerprint != _context(prompt_version="v2").fingerprint
    assert context.fingerprint != _context(cache_namespace="other").fingerprint
    assert context.fingerprint != context.with_snapshot("sha256:" + "a" * 64).fingerprint


def test_feature_after_cutoff_is_rejected_but_future_label_is_isolated():
    context = _context()
    future_feature = TimedEvidence(
        domain="daily",
        source="fixture",
        available_at="2024-04-01",
        fetched_at="2024-04-02",
        payload={"close": 11},
    )
    future_label = TimedEvidence(
        domain="realized_return",
        source="fixture",
        available_at="2024-04-10",
        fetched_at="2024-04-10",
        payload={"return": 0.1},
        channel=EvidenceChannel.OUTCOME_LABEL,
    )

    with pytest.raises(PITTemporalViolation):
        FeatureLabelBoundary(context=context, features=(future_feature,))

    boundary = FeatureLabelBoundary(context=context, outcome_labels=(future_label,))
    assert boundary.scorer_payload() == ()


def test_financial_selection_uses_same_row_vintage_and_rejects_future_disclosure():
    rows = [
        {
            "ts_code": "000001.SZ", "ann_date": "20240330",
            "end_date": "20231231", "update_flag": "0", "roe": 12.0,
        },
        {
            "ts_code": "000001.SZ", "ann_date": "20240430",
            "end_date": "20231231", "update_flag": "0", "roe": 99.0,
        },
        # A row without its own ann_date cannot borrow another row's disclosure date.
        {
            "ts_code": "000001.SZ", "ann_date": "",
            "end_date": "20230930", "update_flag": "0", "roe": 88.0,
        },
    ]

    selected, mode = select_fina_indicator_rows(rows, _context())

    assert mode is PITMode.EXACT
    assert len(selected) == 1
    assert selected[0]["roe"] == 12.0
    assert selected[0]["ann_date"] == "20240330"


def test_revised_financial_values_are_best_effort_never_exact():
    rows = [{
        "ann_date": "20240330", "end_date": "20231231",
        "update_flag": "1", "roe": 20,
    }]

    with pytest.raises(PITDataUnavailableError, match="best_effort"):
        select_fina_indicator_rows(rows, _context(), require_exact=True)

    selected, mode = select_fina_indicator_rows(rows, _context(), require_exact=False)
    assert selected[0]["roe"] == 20
    assert mode is PITMode.BEST_EFFORT


class _FixtureRequester:
    def __init__(self):
        self.calls = []

    def __call__(self, api_name, params, fields):
        self.calls.append((api_name, dict(params), fields))
        rows = {
            "daily": [
                {
                    "ts_code": "000001.SZ", "trade_date": "20240329",
                    "open": 10, "high": 11, "low": 9, "close": 10.5,
                    "pre_close": 10, "vol": 100, "amount": 1000,
                },
                {
                    "ts_code": "000001.SZ", "trade_date": "20240401",
                    "open": 100, "high": 100, "low": 100, "close": 100,
                    "pre_close": 10.5, "vol": 100, "amount": 1000,
                },
            ],
            "daily_basic": [{
                "ts_code": "000001.SZ", "trade_date": "20240329",
                "pe": 8, "pb": 1, "turnover_rate": 1.2, "total_mv": 10000,
            }],
            "fina_indicator": [
                {
                    "ts_code": "000001.SZ", "ann_date": "20240330",
                    "end_date": "20231231", "update_flag": "0", "roe": 13,
                },
                {
                    "ts_code": "000001.SZ", "ann_date": "20240430",
                    "end_date": "20231231", "update_flag": "0", "roe": 99,
                },
            ],
            "moneyflow": [{
                "ts_code": "000001.SZ", "trade_date": "20240329",
                "net_mf_amount": 123,
            }],
        }[api_name]
        names = fields.split(",")
        return {
            "fields": names,
            "items": [[row.get(name) for name in names] for row in rows],
        }


def test_gateway_builds_content_addressed_replayable_immutable_snapshot(tmp_path):
    requester = _FixtureRequester()
    store = ContentAddressedSnapshotStore(tmp_path / "snapshots")
    fixed_now = datetime(2026, 7, 19, tzinfo=timezone.utc)
    gateway = PITDataGateway(requester, snapshot_store=store, clock=lambda: fixed_now)

    bundle = gateway.build_stock_bundle("000001.SZ", _context())
    replayed = gateway.replay_stock_bundle(bundle.snapshot_id)

    assert bundle.snapshot_id.startswith("sha256:")
    assert bundle.context.data_snapshot_id == bundle.snapshot_id
    assert bundle.exact
    assert len(bundle.daily) == 1  # the future 2024-04-01 row was removed
    assert bundle.fina_indicator[0]["roe"] == 13
    assert replayed.snapshot_id == bundle.snapshot_id
    assert replayed.fina_indicator[0]["roe"] == 13
    with pytest.raises(TypeError):
        bundle.daily[0]["close"] = 999  # type: ignore[index]

    for api_name, params, _ in requester.calls:
        if "end_date" in params:
            assert params["end_date"] <= "20240331"
        if api_name == "daily_basic":
            assert params["trade_date"] <= "20240331"


def test_snapshot_tampering_is_detected(tmp_path):
    store = ContentAddressedSnapshotStore(tmp_path / "snapshots")
    snapshot_id = store.put({"schema_version": "pit_v1", "value": 1})
    digest = snapshot_id.split(":", 1)[1]
    path = Path(store.root) / f"{digest}.json"
    path.write_text('{"schema_version":"pit_v1","value":2}', encoding="utf-8")

    with pytest.raises(PITSnapshotCorruptError):
        store.get(snapshot_id)

