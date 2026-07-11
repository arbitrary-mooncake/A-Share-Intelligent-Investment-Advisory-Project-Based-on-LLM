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
    # 按外键依赖顺序删除（先删子表 realized_label → prediction_snapshot →
    # 其余子表 → eval_batch）。database.py 开启了 PRAGMA foreign_keys=ON,
    # 若顺序颠倒, 删 prediction_snapshot 时会因 realized_label 的外键引用而失败,
    # 此前用 except: pass 吞掉异常导致 eval_batch 未清理 → test_batch_001 残留 →
    # create_batch 触发 UNIQUE 冲突（本地 eval.db 有残留时复现, CI 干净 db 不触发）。
    conn = get_connection()
    try:
        # realized_label 引用 prediction_snapshot, 必须先删（含引用 test_ 批次快照的行）
        conn.execute(
            "DELETE FROM realized_label WHERE snapshot_id LIKE 'snap_test_%' "
            "OR snapshot_id IN (SELECT snapshot_id FROM prediction_snapshot "
            "WHERE batch_id LIKE 'test_%')"
        )
        conn.execute("DELETE FROM prediction_snapshot WHERE batch_id LIKE 'test_%'")
        for tbl in ("optimization_ticket", "agent_contribution",
                    "module_loss", "experiment_run"):
            conn.execute(f"DELETE FROM {tbl} WHERE batch_id LIKE 'test_%'")
        conn.execute("DELETE FROM eval_batch WHERE batch_id LIKE 'test_%'")
        conn.commit()
    except Exception as e:
        import warnings
        warnings.warn(f"setup_module cleanup failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()
    # 创建一个固定batch_id的基准批次，供后续依赖外键的测试使用
    from src.eval.repositories import create_batch
    batch = EvalBatch(batch_id="test_batch_001", status="completed", trigger_source="test")
    try:
        create_batch(batch)
    except Exception:
        # 清理未完全时 test_batch_001 可能已存在; UPDATE 确保状态一致, 避免阻塞后续测试
        conn2 = get_connection()
        try:
            conn2.execute(
                "UPDATE eval_batch SET status='completed', trigger_source='test' "
                "WHERE batch_id='test_batch_001'"
            )
            conn2.commit()
        finally:
            conn2.close()


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
