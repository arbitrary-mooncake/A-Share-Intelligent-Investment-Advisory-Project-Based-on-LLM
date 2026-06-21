"""
数据访问层 — 所有SQL操作集中在此，禁止在业务模块中直接写SQL。
"""
import json
from typing import Optional, List, Dict, Any
from src.eval.database import get_connection, generate_id
from src.eval.schemas import (
    EvalBatch, PredictionSnapshot, RealizedLabel,
    ExperimentRun, ModuleLoss, AgentContribution, OptimizationTicket
)


# ═══════════════════════════════════════════════════════
# EvalBatch
# ═══════════════════════════════════════════════════════

def create_batch(batch: EvalBatch) -> str:
    """创建新批次，返回batch_id"""
    if not batch.batch_id:
        batch.batch_id = f"eval_{generate_id()}"
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO eval_batch (batch_id, status, trigger_source, started_at,
                market_session, data_cutoff_time, stable_version, candidate_version, run_profile)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (batch.batch_id, batch.status, batch.trigger_source, batch.started_at,
              batch.market_session, batch.data_cutoff_time, batch.stable_version,
              batch.candidate_version, batch.run_profile))
        conn.commit()
        return batch.batch_id
    finally:
        conn.close()


def update_batch_status(batch_id: str, status: str, **kwargs):
    """更新批次状态和其他字段"""
    conn = get_connection()
    try:
        set_clauses = ["status = ?"]
        params = [status]
        for key, val in kwargs.items():
            set_clauses.append(f"{key} = ?")
            params.append(val)
        params.append(batch_id)
        conn.execute(f"UPDATE eval_batch SET {', '.join(set_clauses)} WHERE batch_id = ?", params)
        conn.commit()
    finally:
        conn.close()


def get_batch(batch_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM eval_batch WHERE batch_id = ?", (batch_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_latest_batch(status: Optional[str] = None) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    if status:
        row = conn.execute(
            "SELECT * FROM eval_batch WHERE status = ? ORDER BY created_at DESC LIMIT 1",
            (status,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM eval_batch ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_recent_batches(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM eval_batch ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════
# PredictionSnapshot
# ═══════════════════════════════════════════════════════

def create_snapshot(snap: PredictionSnapshot) -> str:
    if not snap.snapshot_id:
        snap.snapshot_id = f"snap_{generate_id()}"
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO prediction_snapshot
            (snapshot_id, batch_id, line_id, asset_type, symbol, name, term,
             as_of_date, pit_mode, eval_mode, score, action,
             signal_pack_bundle_json, analysis_package_json, decision_pack_json,
             model_profile, version_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snap.snapshot_id, snap.batch_id, snap.line_id, snap.asset_type,
            snap.symbol, snap.name, snap.term, snap.as_of_date,
            snap.pit_mode, snap.eval_mode, snap.score, snap.action,
            snap.signal_pack_bundle_json, snap.analysis_package_json,
            snap.decision_pack_json, snap.model_profile, snap.version_hash
        ))
        conn.commit()
        return snap.snapshot_id
    finally:
        conn.close()


def get_snapshots_by_batch(batch_id: str, line_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    if line_id:
        rows = conn.execute(
            "SELECT * FROM prediction_snapshot WHERE batch_id = ? AND line_id = ? ORDER BY symbol",
            (batch_id, line_id)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM prediction_snapshot WHERE batch_id = ? ORDER BY line_id, symbol",
            (batch_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_snapshots_by_date(as_of_date: str, line_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    if line_id:
        rows = conn.execute(
            "SELECT * FROM prediction_snapshot WHERE as_of_date = ? AND line_id = ?",
            (as_of_date, line_id)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM prediction_snapshot WHERE as_of_date = ?",
            (as_of_date,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════
# RealizedLabel
# ═══════════════════════════════════════════════════════

def create_label(label: RealizedLabel) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO realized_label
            (snapshot_id, line_id, term, horizon_days, outcome_date,
             entry_price, exit_price, asset_return_pct, benchmark_return_pct,
             excess_return_pct, max_drawdown_pct, volatility_pct,
             is_valid, settlement_notes, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            label.snapshot_id, label.line_id, label.term, label.horizon_days,
            label.outcome_date, label.entry_price, label.exit_price,
            label.asset_return_pct, label.benchmark_return_pct,
            label.excess_return_pct, label.max_drawdown_pct, label.volatility_pct,
            int(label.is_valid), label.settlement_notes, label.meta_json
        ))
        conn.commit()
        label_id = cursor.lastrowid
        return label_id
    finally:
        conn.close()


def get_labels_by_snapshot(snapshot_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM realized_label WHERE snapshot_id = ?", (snapshot_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_unsettled_snapshots(term: str, horizon_days: int,
                             current_date: str) -> List[Dict[str, Any]]:
    """获取已到期但未结算的snapshot（as_of_date + horizon_days <= current_date）"""
    from datetime import datetime, timedelta
    try:
        current = datetime.strptime(current_date, "%Y-%m-%d")
        cutoff = (current - timedelta(days=horizon_days)).strftime("%Y-%m-%d")
    except ValueError:
        cutoff = current_date  # 格式异常时回退

    conn = get_connection()
    rows = conn.execute("""
        SELECT ps.* FROM prediction_snapshot ps
        WHERE ps.term = ?
        AND ps.as_of_date <= ?
        AND ps.snapshot_id NOT IN (
            SELECT rl.snapshot_id FROM realized_label rl
            WHERE rl.horizon_days = ?
        )
        ORDER BY ps.as_of_date
    """, (term, cutoff, horizon_days)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════
# ModuleLoss
# ═══════════════════════════════════════════════════════

def create_module_loss(ml: ModuleLoss) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO module_loss
            (batch_id, module_name, line_id, L_return, L_risk, L_structure,
             L_total, score_total, sub_breakdown_json, sample_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ml.batch_id, ml.module_name, ml.line_id, ml.L_return, ml.L_risk,
              ml.L_structure, ml.L_total, ml.score_total, ml.sub_breakdown_json, ml.sample_size))
        conn.commit()
        lid = cursor.lastrowid
        return lid
    finally:
        conn.close()


def get_module_losses(batch_id: str, module_name: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    if module_name:
        rows = conn.execute(
            "SELECT * FROM module_loss WHERE batch_id = ? AND module_name = ?",
            (batch_id, module_name)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM module_loss WHERE batch_id = ?", (batch_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════
# AgentContribution
# ═══════════════════════════════════════════════════════

def create_agent_contribution(ac: AgentContribution) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO agent_contribution
            (batch_id, term, agent_name, delta_L_total, delta_L_return, delta_L_risk,
             delta_L_structure, ci_95_lower, ci_95_upper, significance, stars,
             sample_size, eval_mode, market_regime_breakdown_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ac.batch_id, ac.term, ac.agent_name, ac.delta_L_total, ac.delta_L_return,
              ac.delta_L_risk, ac.delta_L_structure, ac.ci_95_lower, ac.ci_95_upper,
              ac.significance, ac.stars, ac.sample_size, ac.eval_mode,
              ac.market_regime_breakdown_json))
        conn.commit()
        cid = cursor.lastrowid
        return cid
    finally:
        conn.close()


def get_agent_contributions(batch_id: str, term: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    if term:
        rows = conn.execute(
            "SELECT * FROM agent_contribution WHERE batch_id = ? AND term = ? ORDER BY delta_L_total DESC",
            (batch_id, term)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM agent_contribution WHERE batch_id = ? ORDER BY term, delta_L_total DESC",
            (batch_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_agent_trend(agent_name: str, term: str, limit: int = 30) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM agent_contribution
        WHERE agent_name = ? AND term = ?
        ORDER BY created_at DESC LIMIT ?
    """, (agent_name, term, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════
# ExperimentRun
# ═══════════════════════════════════════════════════════

def create_experiment_run(exp: ExperimentRun) -> str:
    if not exp.experiment_id:
        exp.experiment_id = f"exp_{generate_id()}"
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO experiment_run
            (experiment_id, batch_id, experiment_type, variant_key,
             asset_type, symbol, term, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (exp.experiment_id, exp.batch_id, exp.experiment_type, exp.variant_key,
              exp.asset_type, exp.symbol, exp.term, exp.metrics_json))
        conn.commit()
        return exp.experiment_id
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════
# OptimizationTicket
# ═══════════════════════════════════════════════════════

def create_ticket(ticket: OptimizationTicket) -> str:
    if not ticket.ticket_id:
        ticket.ticket_id = f"tkt_{generate_id()}"
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO optimization_ticket
            (ticket_id, batch_id, ticket_type, severity, title, summary,
             evidence_json, route, status, patch_path, manual_package_path,
             before_loss, after_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticket.ticket_id, ticket.batch_id, ticket.ticket_type, ticket.severity,
              ticket.title, ticket.summary, ticket.evidence_json, ticket.route,
              ticket.status, ticket.patch_path, ticket.manual_package_path,
              ticket.before_loss, ticket.after_loss))
        conn.commit()
        return ticket.ticket_id
    finally:
        conn.close()


def update_ticket_status(ticket_id: str, status: str, **kwargs):
    conn = get_connection()
    try:
        set_clauses = ["status = ?"]
        params = [status]
        for key, val in kwargs.items():
            set_clauses.append(f"{key} = ?")
            params.append(val)
        params.append(ticket_id)
        conn.execute(f"UPDATE optimization_ticket SET {', '.join(set_clauses)} WHERE ticket_id = ?", params)
        conn.commit()
    finally:
        conn.close()


def get_tickets_by_batch(batch_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM optimization_ticket WHERE batch_id = ? ORDER BY severity, ticket_type",
        (batch_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_pending_tickets() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM optimization_ticket WHERE status = 'pending' ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
