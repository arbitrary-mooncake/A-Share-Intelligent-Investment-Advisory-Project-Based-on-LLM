"""精筛池更新 Job 管理器。

UI 侧: JobManager.start_job / poll / find_running / stop / cleanup
Worker 侧: AtomicJobWriter.update (2Hz 节流由 worker 自己控制)
共享: Job dataclass, JobStatus enum, new_job_id, job_path, ensure_job_dir
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # Finance/Financial-MCP-Agent
JOB_DIR = PROJECT_ROOT / "data" / "pool_update_jobs"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ORPHANED = "orphaned"


@dataclass
class Job:
    job_id: str
    term: str
    status: JobStatus = JobStatus.PENDING
    pid: Optional[int] = None
    parent_pid: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    completed_stocks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "term": self.term,
            "status": self.status.value,
            "pid": self.pid,
            "parent_pid": self.parent_pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "progress": self.progress,
            "completed_stocks": self.completed_stocks,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Job":
        return cls(
            job_id=d["job_id"],
            term=d["term"],
            status=JobStatus(d.get("status", "pending")),
            pid=d.get("pid"),
            parent_pid=d.get("parent_pid"),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            error=d.get("error"),
            progress=d.get("progress") or {},
            completed_stocks=d.get("completed_stocks") or [],
        )


class PoolUpdateJobError(RuntimeError):
    """Job 管理器业务异常基类。"""


def new_job_id(term: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = secrets.token_hex(2)  # 4 hex chars
    return f"pool_update_{term}_{ts}_{rand}"


def job_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def ensure_job_dir() -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
