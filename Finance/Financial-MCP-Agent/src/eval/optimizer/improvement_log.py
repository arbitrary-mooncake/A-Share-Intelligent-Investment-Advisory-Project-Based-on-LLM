"""
改动追踪日志 — 记录所有优化变动及效果。

总纲 §13: 评测优化系统需维护完整的改动审计轨迹:
  - 每次优化变动记录 before/after 状态
  - 冷却期检查：同一模块在min_days内不允许重复修改
  - 累积失败检测：连续失败超过max_consecutive次触发告警
  - 改进趋势：过去N天的模块级效果趋势

设计原则:
  - 默认使用SQLite存储（db_path），支持内存模式（db_path=None）
  - 所有操作是幂等的（重复log同一条change会更新而非新增）
  - 线程安全（sqlite3默认serialized模式）
"""
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────── SQL Schema ────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    module TEXT NOT NULL,
    change_type TEXT NOT NULL,
    before_state_json TEXT,
    after_state_json TEXT,
    before_loss REAL,
    after_loss REAL,
    status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    notes TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_changes_module ON changes(module);
CREATE INDEX IF NOT EXISTS idx_changes_ticket ON changes(ticket_id);
CREATE INDEX IF NOT EXISTS idx_changes_created ON changes(created_at);
CREATE INDEX IF NOT EXISTS idx_changes_module_created ON changes(module, created_at);

CREATE TABLE IF NOT EXISTS improvement_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    module TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    recorded_at TEXT NOT NULL,
    UNIQUE(module, metric_name, recorded_at)
);

CREATE INDEX IF NOT EXISTS idx_metrics_module ON improvement_metrics(module);
CREATE INDEX IF NOT EXISTS idx_metrics_recorded ON improvement_metrics(recorded_at);
"""


class ImprovementLog:
    """改动追踪日志 — 记录所有优化变动及效果。

    Usage:
        log = ImprovementLog("data/eval/improvement_log.db")
        log.log_change(
            ticket_id="TKT-001",
            module="short_term_scorer",
            change_type="PARAM_TUNE",
            before_state={"weight_tech": 0.25},
            after_state={"weight_tech": 0.30},
        )
        can_modify, reason = log.check_cooldown("short_term_scorer", min_days=7)
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or ":memory:"
        self._is_memory = (self.db_path == ":memory:")
        self._persistent_conn = None
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with self._lock:
            conn = self._get_conn_raw()
            try:
                conn.executescript(SCHEMA_SQL)
                conn.commit()
            finally:
                if not self._is_memory:
                    self._close_conn(conn)
                # For :memory:, keep the persistent connection alive

    def _get_conn_raw(self) -> sqlite3.Connection:
        """Get a raw connection. For :memory:, reuse persistent to avoid schema loss."""
        if self._is_memory:
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            return self._persistent_conn
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-safe connection with row_factory set."""
        conn = self._get_conn_raw()
        conn.row_factory = sqlite3.Row
        return conn

    def _close_conn(self, conn: sqlite3.Connection):
        """Close connection only for file-based DBs; keep :memory: persistent conn alive."""
        if not self._is_memory:
            self._close_conn(conn)

    # ──────────────────────────── Log Change ────────────────────────────

    def log_change(
        self,
        ticket_id: str,
        module: str,
        change_type: str,
        before_state: Dict[str, Any] = None,
        after_state: Dict[str, Any] = None,
        before_loss: float = None,
        after_loss: float = None,
        status: str = "pending",
        notes: str = "",
    ) -> int:
        """Record a change with before/after state.

        Args:
            ticket_id: 关联的ticket ID
            module: 被修改的模块名
            change_type: PARAM_TUNE / PROMPT_PATCH / LOGIC_FIX / ARCH_CHANGE / RESEARCH
            before_state: 修改前的状态快照
            after_state: 修改后的状态快照
            before_loss: 修改前的loss值
            after_loss: 修改后的loss值（实施后回填）
            status: pending / accepted / rejected / implemented / rolled_back
            notes: 备注

        Returns:
            The row ID of the inserted/updated record.
        """
        now = datetime.now().isoformat()
        before_json = json.dumps(before_state or {}, ensure_ascii=False)
        after_json = json.dumps(after_state or {}, ensure_ascii=False)

        with self._lock:
            conn = self._get_conn()
            try:
                # Check if this ticket already exists (idempotent)
                existing = conn.execute(
                    "SELECT id FROM changes WHERE ticket_id = ?", (ticket_id,)
                ).fetchone()

                if existing:
                    conn.execute(
                        """UPDATE changes SET
                            module=?, change_type=?, before_state_json=?,
                            after_state_json=?, before_loss=?, after_loss=?,
                            status=?, updated_at=?, notes=?
                        WHERE ticket_id=?""",
                        (module, change_type, before_json, after_json,
                         before_loss, after_loss, status, now, notes, ticket_id),
                    )
                    row_id = existing["id"]
                else:
                    cursor = conn.execute(
                        """INSERT INTO changes
                        (ticket_id, module, change_type, before_state_json,
                         after_state_json, before_loss, after_loss, status,
                         created_at, updated_at, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (ticket_id, module, change_type, before_json, after_json,
                         before_loss, after_loss, status, now, now, notes),
                    )
                    row_id = cursor.lastrowid

                conn.commit()
                logger.info(f"Logged change {ticket_id} for module {module} ({change_type})")
                return row_id
            finally:
                self._close_conn(conn)

    # ──────────────────────────── Cooldown Check ────────────────────────────

    def check_cooldown(self, module: str, min_days: int = 7) -> Tuple[bool, str]:
        """Check if module is in cooldown period.

        Args:
            module: 模块名
            min_days: 最小冷却天数，默认7天

        Returns:
            (can_modify: bool, reason: str)
            can_modify=True 表示可以修改（不在冷却期），False表示需要等待。
        """
        conn = self._get_conn()
        try:
            cutoff = (datetime.now() - timedelta(days=min_days)).isoformat()

            row = conn.execute(
                """SELECT ticket_id, created_at, change_type
                FROM changes
                WHERE module = ? AND created_at > ?
                ORDER BY created_at DESC
                LIMIT 1""",
                (module, cutoff),
            ).fetchone()

            if row is None:
                return True, f"模块 {module} 在最近{min_days}天内无修改记录，可以修改"

            return False, (
                f"模块 {module} 在 {row['created_at'][:10]} 有过修改 "
                f"(ticket={row['ticket_id']}, type={row['change_type']})，"
                f"冷却期至 {(datetime.fromisoformat(row['created_at']) + timedelta(days=min_days)).strftime('%Y-%m-%d')}，"
                f"请等待冷却期结束后再修改"
            )
        finally:
            self._close_conn(conn)

    # ─────────────────────── Cumulative Failures ────────────────────────

    def check_cumulative_failures(
        self, module: str, max_consecutive: int = 3
    ) -> Tuple[bool, str, int]:
        """Check if module has too many consecutive failures.

        A "failure" is defined as a change where after_loss > before_loss
        (i.e., the change made things worse) or status is 'rolled_back'.

        Args:
            module: 模块名
            max_consecutive: 最大连续失败次数，默认3

        Returns:
            (can_proceed: bool, reason: str, consecutive_failures: int)
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT ticket_id, before_loss, after_loss, status, created_at
                FROM changes
                WHERE module = ? AND before_loss IS NOT NULL
                ORDER BY created_at DESC
                LIMIT ?""",
                (module, max_consecutive + 1),
            ).fetchall()

            consecutive = 0
            last_failure_ticket = ""
            for row in rows:
                is_failure = False
                if row["status"] == "rolled_back":
                    is_failure = True
                elif (row["before_loss"] is not None and
                      row["after_loss"] is not None and
                      row["after_loss"] > row["before_loss"]):
                    is_failure = True

                if is_failure:
                    consecutive += 1
                    last_failure_ticket = row["ticket_id"]
                else:
                    break  # Streak broken

            if consecutive >= max_consecutive:
                return False, (
                    f"模块 {module} 已连续失败{consecutive}次 "
                    f"(最近: ticket={last_failure_ticket})，"
                    f"超过阈值{max_consecutive}次。建议人工审核后再进行修改。"
                ), consecutive

            return True, (
                f"模块 {module} 连续失败{consecutive}次，"
                f"未超过阈值{max_consecutive}次"
            ), consecutive
        finally:
            self._close_conn(conn)

    # ─────────────────────── Improvement Trend ──────────────────────────

    def get_improvement_trend(
        self, module: str, days: int = 90
    ) -> Dict[str, Any]:
        """Get improvement trend for a module over the past N days.

        Args:
            module: 模块名
            days: 统计天数，默认90天

        Returns:
            {
                "module": str,
                "period_days": int,
                "total_changes": int,
                "improvements": int,       # after_loss < before_loss
                "regressions": int,        # after_loss > before_loss
                "neutral": int,            # no loss change
                "net_delta_loss": float,   # sum(before - after), positive = improvement
                "avg_improvement_per_change": float,
                "trend": "improving" / "stable" / "degrading",
                "recent_changes": List[dict],
            }
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT ticket_id, change_type, before_loss, after_loss,
                   status, created_at, notes
                FROM changes
                WHERE module = ? AND created_at >= ? AND before_loss IS NOT NULL
                ORDER BY created_at DESC""",
                (module, cutoff),
            ).fetchall()

            if not rows:
                return {
                    "module": module,
                    "period_days": days,
                    "total_changes": 0,
                    "improvements": 0,
                    "regressions": 0,
                    "neutral": 0,
                    "net_delta_loss": 0.0,
                    "avg_improvement_per_change": 0.0,
                    "trend": "insufficient_data",
                    "recent_changes": [],
                }

            improvements = 0
            regressions = 0
            neutral = 0
            net_delta = 0.0
            recent = []

            for row in rows:
                before = row["before_loss"]
                after = row["after_loss"]
                if before is not None and after is not None:
                    delta = before - after  # positive = improvement
                    net_delta += delta

                    if delta > 0.001:
                        improvements += 1
                    elif delta < -0.001:
                        regressions += 1
                    else:
                        neutral += 1

                recent.append({
                    "ticket_id": row["ticket_id"],
                    "change_type": row["change_type"],
                    "before_loss": before,
                    "after_loss": after,
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "notes": row["notes"],
                })

            total = improvements + regressions + neutral
            avg_improvement = net_delta / max(total, 1)

            # Determine trend
            if net_delta > 0.02:
                trend = "improving"
            elif net_delta < -0.02:
                trend = "degrading"
            else:
                trend = "stable"

            return {
                "module": module,
                "period_days": days,
                "total_changes": total,
                "improvements": improvements,
                "regressions": regressions,
                "neutral": neutral,
                "net_delta_loss": round(net_delta, 4),
                "avg_improvement_per_change": round(avg_improvement, 4),
                "trend": trend,
                "recent_changes": recent[:20],
            }
        finally:
            self._close_conn(conn)

    # ──────────────────────────── Get Change ────────────────────────────

    def get_change(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific change by ticket_id."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM changes WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()

            if row is None:
                return None

            return {
                "id": row["id"],
                "ticket_id": row["ticket_id"],
                "module": row["module"],
                "change_type": row["change_type"],
                "before_state": json.loads(row["before_state_json"] or "{}"),
                "after_state": json.loads(row["after_state_json"] or "{}"),
                "before_loss": row["before_loss"],
                "after_loss": row["after_loss"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "notes": row["notes"],
            }
        finally:
            self._close_conn(conn)

    def list_changes(
        self, module: str = None, status: str = None,
        change_type: str = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List changes with optional filtering."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM changes WHERE 1=1"
            params = []

            if module:
                query += " AND module = ?"
                params.append(module)
            if status:
                query += " AND status = ?"
                params.append(status)
            if change_type:
                query += " AND change_type = ?"
                params.append(change_type)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()

            return [{
                "id": r["id"],
                "ticket_id": r["ticket_id"],
                "module": r["module"],
                "change_type": r["change_type"],
                "before_loss": r["before_loss"],
                "after_loss": r["after_loss"],
                "status": r["status"],
                "created_at": r["created_at"],
                "notes": r["notes"],
            } for r in rows]
        finally:
            self._close_conn(conn)

    # ───────────────────────── Metrics Tracking ─────────────────────────

    def log_metric(self, module: str, metric_name: str,
                   metric_value: float) -> None:
        """Record a metric snapshot for trend analysis."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO improvement_metrics
                    (module, metric_name, metric_value, recorded_at)
                    VALUES (?, ?, ?, ?)""",
                    (module, metric_name, metric_value, now),
                )
                conn.commit()
            finally:
                self._close_conn(conn)

    def get_metric_trend(self, module: str, metric_name: str,
                         days: int = 30) -> List[Dict[str, Any]]:
        """Get metric time series for trend visualization."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT metric_value, recorded_at
                FROM improvement_metrics
                WHERE module = ? AND metric_name = ? AND recorded_at >= ?
                ORDER BY recorded_at ASC""",
                (module, metric_name, cutoff),
            ).fetchall()
            return [{
                "value": r["metric_value"],
                "recorded_at": r["recorded_at"],
            } for r in rows]
        finally:
            self._close_conn(conn)
