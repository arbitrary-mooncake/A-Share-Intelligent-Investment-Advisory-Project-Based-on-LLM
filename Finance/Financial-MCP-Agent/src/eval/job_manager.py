"""精筛池更新 Job 管理器。

UI 侧: JobManager.start_job / poll / find_running / stop / cleanup
Worker 侧: AtomicJobWriter.update (2Hz 节流由 worker 自己控制)
共享: Job dataclass, JobStatus enum, new_job_id, job_path, ensure_job_dir
"""
from __future__ import annotations

import json
import os
import secrets
import time
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


class AtomicJobWriter:
    """原子写入 job 文件: write -> .tmp -> os.replace。

    防止 UI 读到半写 JSON。所有 update 操作都基于文件当前内容做 merge。
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.path = job_path(job_id)
        self.tmp_path = self.path.with_suffix(".json.tmp")
        ensure_job_dir()
        if not self.path.exists():
            # 首次写入: 用 Job 默认值落盘
            base = Job(job_id=job_id, term=_extract_term(job_id))
            self._atomic_dump(base.to_dict())

    def _atomic_dump(self, data: Dict[str, Any]) -> None:
        # Windows 下 OneDrive/杀毒 可能短暂锁定 .tmp, 重试 3 次
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        for attempt in range(3):
            try:
                self.tmp_path.write_text(payload, encoding="utf-8")
                os.replace(self.tmp_path, self.path)
                return
            except (PermissionError, OSError):
                if attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def read(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def update(self, **fields: Any) -> None:
        data = self.read()
        data.update(fields)
        self._atomic_dump(data)

    def merge_progress(self, progress: Dict[str, Any]) -> None:
        data = self.read()
        merged = dict(data.get("progress") or {})
        merged.update(progress)
        data["progress"] = merged
        self._atomic_dump(data)

    def append_completed_stock(self, stock: Dict[str, Any]) -> None:
        data = self.read()
        stocks = list(data.get("completed_stocks") or [])
        stocks.append(stock)
        data["completed_stocks"] = stocks
        self._atomic_dump(data)


def _extract_term(job_id: str) -> str:
    # pool_update_{term}_{ts}_{rand}
    parts = job_id.split("_")
    if len(parts) >= 3 and parts[0] == "pool" and parts[1] == "update":
        return parts[2]
    raise PoolUpdateJobError(f"invalid job_id_format: {job_id}")


# ---------------------------------------------------------------------------
# Task 3: cross-platform detached spawn + orphan detection
# ---------------------------------------------------------------------------
import subprocess
import sys


def is_pid_alive(pid: Optional[int]) -> bool:
    """Check whether *pid* refers to a currently-running process."""
    if pid is None:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle:
                exit_code = ctypes.c_ulong()
                ctypes.windll.kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)
                )
                ctypes.windll.kernel32.CloseHandle(handle)
                return exit_code.value == 259  # STILL_ACTIVE
            return False
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


def detect_orphan(job_dict: Dict[str, Any]) -> bool:
    """Return True when job claims *running* but its PID is dead."""
    if job_dict.get("status") != "running":
        return False
    return not is_pid_alive(job_dict.get("pid"))


def _spawn_worker(
    job_id: str,
    term: str,
    log_path: str,
    worker_module_override: Optional[str] = None,
) -> subprocess.Popen:
    """Launch the pool-update worker as a detached child process.

    *worker_module_override* (test-only): path to a replacement Python script
    instead of ``-m src.eval.pool_update_worker``.
    """
    ensure_job_dir()
    if worker_module_override:
        cmd = [sys.executable, worker_module_override]
    else:
        cmd = [
            sys.executable, "-m", "src.eval.pool_update_worker",
            "--job-id", job_id, "--term", term,
        ]

    kwargs: Dict[str, Any] = dict(
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        env={**os.environ, "POOL_UPDATE_JOB_ID": job_id, "POOL_UPDATE_TERM": term},
    )

    if sys.platform == "win32":
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008
        kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    return subprocess.Popen(cmd, **kwargs)
