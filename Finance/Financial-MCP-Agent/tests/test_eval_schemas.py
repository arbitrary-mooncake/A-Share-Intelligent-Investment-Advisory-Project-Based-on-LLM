"""Tests for eval schemas"""
from src.eval.schemas import (
    EvalBatch, PredictionSnapshot, RealizedLabel,
    ExperimentRun, ModuleLoss, AgentContribution, OptimizationTicket
)


def test_eval_batch_to_from_dict():
    b = EvalBatch(batch_id="test_001", status="running", trigger_source="ui")
    d = b.to_dict()
    b2 = EvalBatch.from_dict(d)
    assert b2.batch_id == "test_001"
    assert b2.status == "running"


def test_prediction_snapshot_roundtrip():
    s = PredictionSnapshot(
        snapshot_id="snap_001", batch_id="batch_001", line_id="S-L0",
        symbol="sh.603871", term="short", as_of_date="2026-06-19", score=75.0
    )
    d = s.to_dict()
    s2 = PredictionSnapshot.from_dict(d)
    assert s2.snapshot_id == "snap_001"
    assert s2.score == 75.0


def test_realized_label_defaults():
    r = RealizedLabel(snapshot_id="snap_001")
    assert r.horizon_days == 1
    assert r.is_valid == True


def test_optimization_ticket_types():
    t = OptimizationTicket(ticket_id="t_001", ticket_type="PARAM_TUNE", route="auto")
    assert t.ticket_type == "PARAM_TUNE"
    assert t.route == "auto"
    d = t.to_dict()
    assert d["ticket_type"] == "PARAM_TUNE"
