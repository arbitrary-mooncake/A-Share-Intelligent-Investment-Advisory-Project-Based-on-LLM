"""
SQLite数据库管理 — 评测系统持久化层。
单文件数据库，零部署依赖，Windows/WSL兼容。
"""
import os
import sqlite3
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime


def _get_db_path() -> str:
    """获取数据库文件路径"""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_dir = os.path.join(base, "data", "eval")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "eval.db")


def _dict_factory(cursor, row):
    """Row factory: 返回dict而非tuple"""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def get_connection() -> sqlite3.Connection:
    """获取数据库连接"""
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = _dict_factory
    conn.execute("PRAGMA journal_mode=WAL")  # 更好的并发
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """初始化数据库表（幂等操作）"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS eval_batch (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            trigger_source TEXT DEFAULT 'ui',
            started_at TEXT DEFAULT '',
            finished_at TEXT DEFAULT '',
            market_session TEXT DEFAULT 'post_close',
            data_cutoff_time TEXT DEFAULT '',
            stable_version TEXT DEFAULT '',
            candidate_version TEXT DEFAULT '',
            run_profile TEXT DEFAULT '',
            summary_metrics_json TEXT DEFAULT '',
            report_md_path TEXT DEFAULT '',
            report_pdf_path TEXT DEFAULT '',
            optimize_ready INTEGER DEFAULT 0,
            error_message TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS prediction_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT UNIQUE NOT NULL,
            batch_id TEXT NOT NULL,
            line_id TEXT NOT NULL,
            asset_type TEXT NOT NULL DEFAULT 'stock',
            symbol TEXT NOT NULL,
            name TEXT DEFAULT '',
            term TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            pit_mode TEXT DEFAULT 'exact',
            eval_mode TEXT DEFAULT 'real',
            score REAL DEFAULT 0.0,
            action TEXT DEFAULT '',
            signal_pack_bundle_json TEXT DEFAULT '',
            analysis_package_json TEXT DEFAULT '',
            decision_pack_json TEXT DEFAULT '',
            model_profile TEXT DEFAULT '',
            version_hash TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (batch_id) REFERENCES eval_batch(batch_id)
        );

        CREATE TABLE IF NOT EXISTS realized_label (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL,
            line_id TEXT NOT NULL,
            term TEXT NOT NULL,
            horizon_days INTEGER NOT NULL DEFAULT 1,
            outcome_date TEXT NOT NULL,
            entry_price REAL DEFAULT 0.0,
            exit_price REAL DEFAULT 0.0,
            asset_return_pct REAL DEFAULT 0.0,
            benchmark_return_pct REAL DEFAULT 0.0,
            excess_return_pct REAL DEFAULT 0.0,
            max_drawdown_pct REAL DEFAULT 0.0,
            volatility_pct REAL DEFAULT 0.0,
            is_valid INTEGER DEFAULT 1,
            settlement_notes TEXT DEFAULT '',
            meta_json TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (snapshot_id) REFERENCES prediction_snapshot(snapshot_id)
        );

        CREATE TABLE IF NOT EXISTS module_loss (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            module_name TEXT NOT NULL,
            line_id TEXT NOT NULL,
            L_return REAL DEFAULT 0.0,
            L_risk REAL DEFAULT 0.0,
            L_structure REAL DEFAULT 0.0,
            L_total REAL DEFAULT 0.0,
            score_total REAL DEFAULT 0.0,
            sub_breakdown_json TEXT DEFAULT '',
            sample_size INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (batch_id) REFERENCES eval_batch(batch_id)
        );

        CREATE TABLE IF NOT EXISTS agent_contribution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            term TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            delta_L_total REAL DEFAULT 0.0,
            delta_L_return REAL DEFAULT 0.0,
            delta_L_risk REAL DEFAULT 0.0,
            delta_L_structure REAL DEFAULT 0.0,
            ci_95_lower REAL DEFAULT 0.0,
            ci_95_upper REAL DEFAULT 0.0,
            significance TEXT DEFAULT '',
            stars TEXT DEFAULT '',
            sample_size INTEGER DEFAULT 0,
            eval_mode TEXT DEFAULT '',
            market_regime_breakdown_json TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (batch_id) REFERENCES eval_batch(batch_id)
        );

        CREATE TABLE IF NOT EXISTS experiment_run (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT UNIQUE NOT NULL,
            batch_id TEXT NOT NULL,
            experiment_type TEXT NOT NULL,
            variant_key TEXT DEFAULT '',
            asset_type TEXT DEFAULT 'stock',
            symbol TEXT DEFAULT '',
            term TEXT DEFAULT '',
            metrics_json TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (batch_id) REFERENCES eval_batch(batch_id)
        );

        CREATE TABLE IF NOT EXISTS optimization_ticket (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT UNIQUE NOT NULL,
            batch_id TEXT NOT NULL,
            ticket_type TEXT NOT NULL,
            severity TEXT DEFAULT 'medium',
            title TEXT DEFAULT '',
            summary TEXT DEFAULT '',
            evidence_json TEXT DEFAULT '',
            route TEXT DEFAULT '',
            status TEXT DEFAULT 'pending',
            patch_path TEXT DEFAULT '',
            manual_package_path TEXT DEFAULT '',
            before_loss REAL DEFAULT 0.0,
            after_loss REAL DEFAULT 0.0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (batch_id) REFERENCES eval_batch(batch_id)
        );
    """)

    # 索引
    cursor.executescript("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_batch
            ON prediction_snapshot(batch_id);
        CREATE INDEX IF NOT EXISTS idx_snapshots_line
            ON prediction_snapshot(line_id);
        CREATE INDEX IF NOT EXISTS idx_snapshots_date
            ON prediction_snapshot(as_of_date);
        CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_date
            ON prediction_snapshot(symbol, as_of_date);
        CREATE INDEX IF NOT EXISTS idx_labels_snapshot
            ON realized_label(snapshot_id);
        CREATE INDEX IF NOT EXISTS idx_loss_module
            ON module_loss(module_name, batch_id);
        CREATE INDEX IF NOT EXISTS idx_contribution_agent
            ON agent_contribution(agent_name, term);
        CREATE INDEX IF NOT EXISTS idx_contribution_batch
            ON agent_contribution(batch_id);
        CREATE INDEX IF NOT EXISTS idx_tickets_batch
            ON optimization_ticket(batch_id);
        CREATE INDEX IF NOT EXISTS idx_tickets_status
            ON optimization_ticket(status);
    """)

    conn.commit()
    conn.close()


def generate_id() -> str:
    """生成唯一ID"""
    return uuid.uuid4().hex[:16]
