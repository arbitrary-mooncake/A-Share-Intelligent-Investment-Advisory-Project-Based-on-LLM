"""Tests for eval database and repositories"""
import os
import tempfile
from src.eval.database import init_db, get_connection, generate_id
from src.eval.repositories import (
    create_batch, get_batch, get_latest_batch, update_batch_status,
    create_snapshot, get_snapshots_by_batch,
    create_module_loss, get_module_losses,
    create_agent_contribution, get_agent_contributions,
    create_ticket, get_tickets_by_batch, update_ticket_status,
)
from src.eval.schemas import (
    EvalBatch, PredictionSnapshot, ModuleLoss, AgentContribution, OptimizationTicket
)


def setup_module():
    """初始化数据库并创建测试用的基准批次"""
    init_db()
    # 清理可能存在的旧测试数据，避免UNIQUE冲突
    conn = get_connection()
    try:
        conn.execute("DELETE FROM optimization_ticket WHERE batch_id LIKE 'test_%'")
        conn.execute("DELETE FROM agent_contribution WHERE batch_id LIKE 'test_%'")
        conn.execute("DELETE FROM module_loss WHERE batch_id LIKE 'test_%'")
        conn.execute("DELETE FROM experiment_run WHERE batch_id LIKE 'test_%'")
        conn.execute("DELETE FROM prediction_snapshot WHERE batch_id LIKE 'test_%'")
        conn.execute("DELETE FROM realized_label WHERE snapshot_id LIKE 'snap_test_%'")
        conn.execute("DELETE FROM eval_batch WHERE batch_id LIKE 'test_%'")
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()
    # 创建一个固定batch_id的基准批次，供后续依赖外键的测试使用
    from src.eval.repositories import create_batch
    batch = EvalBatch(batch_id="test_batch_001", status="completed", trigger_source="test")
    create_batch(batch)


def test_create_and_get_batch():
    batch = EvalBatch(status="running", trigger_source="cli")
    bid = create_batch(batch)
    assert bid

    result = get_batch(bid)
    assert result is not None
    assert result["status"] == "running"
    assert result["trigger_source"] == "cli"


def test_update_batch_status():
    batch = EvalBatch(status="queued")
    bid = create_batch(batch)
    update_batch_status(bid, "completed", finished_at="2026-06-19T15:00:00", optimize_ready=1)
    result = get_batch(bid)
    assert result["status"] == "completed"
    assert result["optimize_ready"] == 1


def test_get_latest_batch():
    batch1 = EvalBatch(status="completed")
    batch2 = EvalBatch(status="running")
    create_batch(batch1)
    create_batch(batch2)
    latest = get_latest_batch()
    assert latest is not None


def test_create_and_get_snapshot():
    snap = PredictionSnapshot(
        batch_id="test_batch_001",
        line_id="S-L0", symbol="sh.603871", term="short",
        as_of_date="2026-06-19", score=75.0
    )
    sid = create_snapshot(snap)
    assert sid

    snapshots = get_snapshots_by_batch("test_batch_001")
    assert len(snapshots) >= 1
    found = [s for s in snapshots if s["snapshot_id"] == sid]
    assert len(found) == 1
    assert found[0]["score"] == 75.0


def test_create_and_get_module_loss():
    ml = ModuleLoss(
        batch_id="test_batch_001", module_name="stock_short_term",
        line_id="S-L0", L_total=0.25, score_total=75.0, sample_size=100
    )
    lid = create_module_loss(ml)
    assert lid

    losses = get_module_losses("test_batch_001", "stock_short_term")
    assert len(losses) >= 1


def test_create_and_get_agent_contribution():
    ac = AgentContribution(
        batch_id="test_batch_001", term="short", agent_name="fundamental",
        delta_L_total=0.042, significance="significant_positive", stars="★★★",
        sample_size=30, eval_mode="real"
    )
    cid = create_agent_contribution(ac)
    assert cid

    contribs = get_agent_contributions("test_batch_001", "short")
    assert len(contribs) >= 1


def test_create_and_get_ticket():
    ticket = OptimizationTicket(
        batch_id="test_batch_001", ticket_type="PARAM_TUNE",
        severity="medium", title="测试ticket", route="auto", status="pending"
    )
    tid = create_ticket(ticket)
    assert tid

    tickets = get_tickets_by_batch("test_batch_001")
    assert len(tickets) >= 1


def test_update_ticket():
    ticket = OptimizationTicket(batch_id="test_batch_001", ticket_type="PARAM_TUNE", route="auto")
    tid = create_ticket(ticket)
    update_ticket_status(tid, "accepted", after_loss=0.22)
    tickets = get_tickets_by_batch("test_batch_001")
    updated = [t for t in tickets if t["ticket_id"] == tid]
    if updated:
        assert updated[0]["status"] == "accepted"


def test_generate_id():
    id1 = generate_id()
    id2 = generate_id()
    assert len(id1) == 16
    assert id1 != id2
