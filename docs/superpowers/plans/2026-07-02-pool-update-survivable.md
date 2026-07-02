# 精筛池更新跨重启存活 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Streamlit 页面触发的精筛池全量更新（V3 四层管线）能在 Streamlit 重启/浏览器关闭后**继续跑**，新 session 自动 attach 进度，子进程崩溃时**过程数据不丢**。

**Architecture:** Streamlit 通过 `subprocess.Popen` 启动 detached worker 子进程（Windows `CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS`，POSIX `start_new_session=True`），worker 跑 `EvalOrchestrator.run_pool_update`，通过 2Hz 原子写入 `data/pool_update_jobs/{job_id}.json` 把进度回传到磁盘，Streamlit UI 每秒 poll 该文件渲染进度条。

**Tech Stack:** Python 3.13、asyncio、subprocess、Streamlit、pytest、既有 `src.eval.orchestrator.EvalOrchestrator` 和 `PipelineProgress`。

## Global Constraints

- **不修改**：`src/eval/orchestrator.py`、`src/eval/pool_screening.py`、`src/stock_pool/scoring_engine.py`、`src/agents/*`、`src/tools/mcp_client.py`、`src/utils/cache_utils.py`、`src/utils/tushare_client.py`
- **Job 文件写入**：2Hz 节流（500ms），`write → .tmp → os.replace` 原子重命名
- **同 term 幂等**：同一 term 同时只允许 1 个 `running` job；重复触发自动 attach 现有
- **保留策略**：按 mtime 保留 1 年内所有 job 文件，超期全删
- **Worker stdout**：重定向到 `data/pool_update_jobs/{job_id}.log`
- **Orphan 检测**：`status=running` 但 `pid` 不存在 → 自动标 `orphaned`
- **测试**：每个 task 必须先写失败测试 → 实现 → 跑通 → commit
- **Platform**：Windows 11 + Python 3.13 为主；POSIX 路径同步实现（不强制本地跑）
- **Job ID 格式**：`pool_update_{term}_{YYYYMMDD_HHMMSS}_{rand4}`

## File Structure

```
Finance/Financial-MCP-Agent/
├── src/eval/
│   ├── job_manager.py              (新增, ~280 行, JobManager + AtomicJobWriter + helpers)
│   ├── pool_update_worker.py       (新增, ~85 行, worker __main__ 入口)
│   └── web_adapter.py              (修改: 删除 run_pool_update_streaming 方法, -40 行)
├── src/app/
│   ├── components/eval/
│   │   └── ops_panel.py            (修改: 调 JobManager 替换 adapter, ~40 行改动)
│   └── pages/
│       └── 06_模拟分析与迭代.py     (修改: 健康卡片"全量更新"走同一路径, ~15 行改动)
├── data/
│   └── pool_update_jobs/
│       └── .gitignore              (新增, 忽略 *.json/*.log, 保留 .gitkeep)
└── tests/
    ├── test_job_manager.py         (新增, 8 个用例)
    ├── test_pool_update_worker.py  (新增, 4 个用例)
    └── test_job_ui_integration.py  (新增, 3 个用例)
```

职责切分：
- `job_manager.py`：唯一对外入口，UI 调 `JobManager`，Worker 用 `AtomicJobWriter`。集中处理 schema、原子 I/O、detach spawn、orphan 检测、cleanup。
- `pool_update_worker.py`：独立可执行模块，**不被任何代码 import**。只被 `subprocess.Popen` 以 `-m` 方式启动。
- `web_adapter.py`：删除旧 `run_pool_update_streaming`（UI 全部改走 JobManager，不再需要）。

---

### Task 1: 基础层 — Job schema + JobDir 配置 + 异常

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/eval/job_manager.py`（本 task 只写 schema 部分，约 80 行；后续 task 追加）
- Test: `Finance/Financial-MCP-Agent/tests/test_job_manager.py`（本 task 写 2 个用例）

**Interfaces:**
- Produces: `JobStatus` enum、`Job` dataclass、`JOB_DIR` 路径常量、`PoolUpdateJobError` 异常类、`new_job_id(term)`、`job_path(job_id)`、`ensure_job_dir()`

- [ ] **Step 1: 写 schema 测试**

```python
# tests/test_job_manager.py
import os
from datetime import datetime

def test_job_status_enum_values():
    from src.eval.job_manager import JobStatus
    assert set(s.value for s in JobStatus) == {
        "pending", "running", "completed", "failed", "orphaned"
    }


def test_job_dataclass_defaults():
    from src.eval.job_manager import Job, JobStatus
    j = Job(job_id="pool_update_medium_20260702_143052_7a3f", term="medium")
    assert j.status == JobStatus.PENDING
    assert j.pid is None
    assert j.error is None
    assert j.progress == {}
    assert j.completed_stocks == []


def test_new_job_id_format():
    from src.eval.job_manager import new_job_id
    jid = new_job_id("medium")
    assert jid.startswith("pool_update_medium_")
    parts = jid.split("_")
    # pool_update_medium_YYYYMMDD_HHMMSS_rand
    assert len(parts) == 5
    assert len(parts[3]) == 6   # HHMMSS
    assert len(parts[4]) == 4   # rand


def test_job_path_returns_under_job_dir():
    from src.eval.job_manager import job_path, JOB_DIR
    p = job_path("pool_update_short_20260702_143052_abcd")
    assert str(p).endswith(".json")
    assert str(p).startswith(str(JOB_DIR))


def test_ensure_job_dir_creates_directory(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    target = tmp_path / "pool_update_jobs"
    monkeypatch.setattr(jm, "JOB_DIR", target)
    jm.ensure_job_dir()
    assert target.exists() and target.is_dir()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_job_manager.py -v`
Expected: FAIL（`ModuleNotFoundError: src.eval.job_manager`）

- [ ] **Step 3: 实现 schema**

```python
# src/eval/job_manager.py (第 1 段, 后续 task 追加)
"""精筛池更新 Job 管理器。

UI 侧: JobManager.start_job / poll / find_running / stop / cleanup
Worker 侧: AtomicJobWriter.update (2Hz 节流由 worker 自己控制)
共享: Job dataclass, JobStatus enum, new_job_id, job_path, ensure_job_dir
"""
from __future__ import annotations

import os
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_job_manager.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/job_manager.py tests/test_job_manager.py
git commit -m "feat(job-manager): schema + Job dataclass + new_job_id"
```

---

### Task 2: 原子写入器 AtomicJobWriter

**Files:**
- Modify: `src/eval/job_manager.py`（追加 `AtomicJobWriter` 类，约 70 行）
- Test: `tests/test_job_manager.py`（追加 3 个用例）

**Interfaces:**
- Consumes: `Job`、`job_path`、`ensure_job_dir`（Task 1）
- Produces: `AtomicJobWriter(job_id)` 类，方法 `update(**fields)`、`merge_progress(progress_dict)`、`append_completed_stock(stock_dict)`、`read()`

- [ ] **Step 1: 写 AtomicJobWriter 测试**

```python
# tests/test_job_manager.py 追加
import json
import time


def test_atomic_writer_creates_file(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()
    w = jm.AtomicJobWriter("pool_update_short_20260702_143052_abcd")
    w.update(status="running", pid=1234)
    data = json.loads((tmp_path / "pool_update_short_20260702_143052_abcd.json").read_text())
    assert data["status"] == "running"
    assert data["pid"] == 1234
    assert data["job_id"] == "pool_update_short_20260702_143052_abcd"


def test_atomic_writer_no_tmp_left_after_write(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()
    w = jm.AtomicJobWriter("pool_update_short_20260702_143052_abcd")
    w.update(status="running")
    files = list(tmp_path.iterdir())
    assert all(not f.name.endswith(".tmp") for f in files)


def test_atomic_writer_merge_progress(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()
    w = jm.AtomicJobWriter("pool_update_short_20260702_143052_abcd")
    w.update(status="running")
    w.merge_progress({"overall_pct": 42.3, "eta_str": "30min"})
    w.merge_progress({"overall_pct": 45.0, "queue_depth": 3})
    data = w.read()
    assert data["progress"]["overall_pct"] == 45.0
    assert data["progress"]["eta_str"] == "30min"
    assert data["progress"]["queue_depth"] == 3


def test_atomic_writer_append_completed_stock(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()
    w = jm.AtomicJobWriter("pool_update_short_20260702_143052_abcd")
    w.update(status="running")
    w.append_completed_stock({"code": "sz.000333", "name": "美的集团", "final_score": 78})
    w.append_completed_stock({"code": "sh.600519", "name": "贵州茅台", "final_score": 82})
    data = w.read()
    assert len(data["completed_stocks"]) == 2
    assert data["completed_stocks"][0]["code"] == "sz.000333"


def test_atomic_writer_retries_on_oserror(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()
    w = jm.AtomicJobWriter("pool_update_short_20260702_143052_abcd")

    # 模拟 os.replace 第一次失败, 第二次成功
    calls = {"n": 0}
    real_replace = os.replace

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("simulated lock")
        return real_replace(src, dst)

    monkeypatch.setattr(jm.os, "replace", flaky_replace)
    w.update(status="running", pid=99)
    data = w.read()
    assert data["status"] == "running"
    assert data["pid"] == 99
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_job_manager.py -v -k atomic_writer`
Expected: 4 FAIL（`AttributeError: AtomicJobWriter not found`）

- [ ] **Step 3: 实现 AtomicJobWriter**

在 `src/eval/job_manager.py` 末尾追加：

```python
# src/eval/job_manager.py (第 2 段)
class AtomicJobWriter:
    """原子写入 job 文件: write → .tmp → os.replace。

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
            except (PermissionError, OSError) as e:
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
    raise PoolUpdateJobError(f"invalid job_id format: {job_id}")


# 顶部 import 追加:
import json
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_job_manager.py -v`
Expected: 10 PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/job_manager.py tests/test_job_manager.py
git commit -m "feat(job-manager): AtomicJobWriter (tmp+rename, merge progress, append stock)"
```

---

### Task 3: 跨平台 Detached Spawn + Orphan 检测

**Files:**
- Modify: `src/eval/job_manager.py`（追加 `_spawn_worker`、`is_pid_alive`、`detect_orphan`，约 70 行）
- Test: `tests/test_job_manager.py`（追加 3 个用例）

**Interfaces:**
- Consumes: `JOB_DIR`、`ensure_job_dir`、`AtomicJobWriter`、`Job`、`JobStatus`（Task 1-2）
- Produces: `_spawn_worker(job_id, term, log_path)` 返回 `subprocess.Popen`；`is_pid_alive(pid)` 返回 bool；`detect_orphan(job_dict)` 返回 bool

- [ ] **Step 1: 写 spawn + orphan 测试**

```python
# tests/test_job_manager.py 追加
import subprocess
import sys
import time


def test_is_pid_alive_current_process():
    import src.eval.job_manager as jm
    assert jm.is_pid_alive(os.getpid()) is True


def test_is_pid_dead_for_invalid_pid():
    import src.eval.job_manager as jm
    # 一个极不可能存在的高 PID
    assert jm.is_pid_alive(999_999_999) is False


def test_spawn_worker_creates_detached_process(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()
    job_id = "pool_update_short_20260702_143052_abcd"
    log_path = tmp_path / f"{job_id}.log"

    # 用一个极简 worker 替代真实 worker: 写一行就退出
    fake_worker = tmp_path / "fake_worker.py"
    fake_worker.write_text(
        "import os, sys, time\n"
        f"open(r'{tmp_path}' + '/ran.flag', 'w').write(str(os.getpid()))\n"
        "time.sleep(0.5)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys, "executable", sys.executable,
    )
    proc = jm._spawn_worker(
        job_id=job_id,
        term="short",
        log_path=str(log_path),
        worker_module_override=str(fake_worker),
    )
    # 父进程不 wait; 子进程应独立跑
    time.sleep(1.0)
    assert (tmp_path / "ran.flag").exists()
    # 子进程已退出也无所谓 — 关键是它跑了


def test_detect_orphan_flags_dead_pid(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    job_dict = {
        "job_id": "x",
        "term": "short",
        "status": "running",
        "pid": 999_999_999,
        "started_at": "2026-07-02T14:30:52",
    }
    assert jm.detect_orphan(job_dict) is True


def test_detect_orphan_false_for_completed(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    job_dict = {
        "job_id": "x",
        "term": "short",
        "status": "completed",
        "pid": 999_999_999,
    }
    # 非 running 不应被误判
    assert jm.detect_orphan(job_dict) is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_job_manager.py -v -k "spawn or orphan or pid"`
Expected: 5 FAIL（`AttributeError`）

- [ ] **Step 3: 实现 spawn + orphan**

在 `src/eval/job_manager.py` 末尾追加：

```python
# src/eval/job_manager.py (第 3 段)
import subprocess
import sys


def is_pid_alive(pid: Optional[int]) -> bool:
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
                ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
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
    if job_dict.get("status") != "running":
        return False
    return not is_pid_alive(job_dict.get("pid"))


def _spawn_worker(
    job_id: str,
    term: str,
    log_path: str,
    worker_module_override: Optional[str] = None,
) -> subprocess.Popen:
    """以 detached 方式启动 worker 子进程。

    worker_module_override: 仅测试用, 指向一个替代的 Python 脚本路径。
    """
    ensure_job_dir()
    if worker_module_override:
        cmd = [sys.executable, worker_module_override]
    else:
        cmd = [sys.executable, "-m", "src.eval.pool_update_worker",
               "--job-id", job_id, "--term", term]

    kwargs = dict(
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_job_manager.py -v`
Expected: 14 PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/job_manager.py tests/test_job_manager.py
git commit -m "feat(job-manager): cross-platform detached spawn + orphan detection"
```

---

### Task 4: 读取 / 清理 / JobManager 外观

**Files:**
- Modify: `src/eval/job_manager.py`（追加 `JobManager` 类 + `parse_worker_args` + `record_failure`，约 120 行）
- Test: `tests/test_job_manager.py`（追加 4 个用例）

**Interfaces:**
- Consumes: `Job`、`AtomicJobWriter`、`_spawn_worker`、`detect_orphan`、`new_job_id`、`job_path`（Task 1-3）
- Produces: `JobManager` 类，方法 `start_job` / `poll` / `find_running` / `find_latest` / `stop` / `cleanup` / `read_log`；顶层 `parse_worker_args()`、`record_failure(job_id, exc)`

- [ ] **Step 1: 写 JobManager 测试**

```python
# tests/test_job_manager.py 追加
import datetime


def test_jobmanager_start_creates_running_job(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()

    # 用一个不会真跑 pool_update 的假 worker
    fake = tmp_path / "fake.py"
    fake.write_text("import time; time.sleep(2)\n", encoding="utf-8")
    monkeypatch.setattr(
        jm, "_spawn_worker",
        lambda job_id, term, log_path, worker_module_override=None:
            jm._spawn_worker(job_id, term, log_path, worker_module_override=str(fake))
    )

    mgr = jm.JobManager()
    jid = mgr.start_job("medium")
    assert jid.startswith("pool_update_medium_")
    job = mgr.poll(jid)
    assert job["status"] in ("pending", "running")


def test_jobmanager_find_running_returns_same(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()

    # 手写一个 running job 文件
    jid = "pool_update_short_20260702_143052_abcd"
    (tmp_path / f"{jid}.json").write_text(json.dumps({
        "job_id": jid, "term": "short", "status": "running",
        "pid": os.getpid(), "parent_pid": os.getpid(),
        "started_at": "2026-07-02T14:30:52",
        "progress": {}, "completed_stocks": [],
    }), encoding="utf-8")

    mgr = jm.JobManager()
    found = mgr.find_running("short")
    assert found and found["job_id"] == jid


def test_jobmanager_poll_marks_orphan(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()

    jid = "pool_update_short_20260702_143052_abcd"
    (tmp_path / f"{jid}.json").write_text(json.dumps({
        "job_id": jid, "term": "short", "status": "running",
        "pid": 999_999_999, "parent_pid": os.getpid(),
        "started_at": "2026-07-02T14:30:52",
        "progress": {}, "completed_stocks": [],
    }), encoding="utf-8")

    mgr = jm.JobManager()
    job = mgr.poll(jid)
    assert job["status"] == "orphaned"


def test_jobmanager_cleanup_removes_old_files(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()

    # 造 2 个文件: 一个新一个旧
    new_f = tmp_path / "pool_update_short_20260702_143052_new.json"
    old_f = tmp_path / "pool_update_short_20200101_000000_old.json"
    new_f.write_text('{"job_id":"x","term":"short","status":"completed"}', encoding="utf-8")
    old_f.write_text('{"job_id":"y","term":"short","status":"completed"}', encoding="utf-8")
    # 把 old_f mtime 设到 2 年前
    old_ts = time.time() - 2 * 365 * 24 * 3600
    os.utime(old_f, (old_ts, old_ts))

    mgr = jm.JobManager()
    mgr.cleanup()
    assert new_f.exists()
    assert not old_f.exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_job_manager.py -v -k jobmanager`
Expected: 4 FAIL（`AttributeError: JobManager`）

- [ ] **Step 3: 实现 JobManager + helpers**

在 `src/eval/job_manager.py` 末尾追加：

```python
# src/eval/job_manager.py (第 4 段)
import argparse
import logging

logger = logging.getLogger(__name__)

_JOB_RETENTION_SECONDS = 365 * 24 * 3600  # 1 年


class JobManager:
    """精筛池更新任务管理器 (UI 侧使用)。"""

    def start_job(self, term: str) -> str:
        existing = self.find_running(term)
        if existing:
            return existing["job_id"]

        job_id = new_job_id(term)
        ensure_job_dir()
        log_path = str(job_path(job_id).with_suffix(".log"))

        # 先写 pending 文件, 再 spawn worker (worker 自己改成 running)
        writer = AtomicJobWriter(job_id)
        writer.update(
            status=JobStatus.PENDING.value,
            parent_pid=os.getpid(),
            started_at=datetime.now().isoformat(),
        )

        _spawn_worker(job_id=job_id, term=term, log_path=log_path)
        self.cleanup()
        return job_id

    def poll(self, job_id: str) -> Optional[Dict[str, Any]]:
        p = job_path(job_id)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        if detect_orphan(data):
            data["status"] = JobStatus.ORPHANED.value
            data["finished_at"] = datetime.now().isoformat()
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    def find_running(self, term: str) -> Optional[Dict[str, Any]]:
        ensure_job_dir()
        for f in JOB_DIR.glob(f"pool_update_{term}_*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("status") == "running":
                if detect_orphan(data):
                    data["status"] = JobStatus.ORPHANED.value
                    f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    continue
                return data
        return None

    def find_latest(self, term: str) -> Optional[Dict[str, Any]]:
        ensure_job_dir()
        candidates = sorted(
            JOB_DIR.glob(f"pool_update_{term}_*.json"),
            key=lambda f: f.stat().st_mtime, reverse=True,
        )
        for f in candidates:
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
        return None

    def stop(self, job_id: str) -> bool:
        data = self.poll(job_id)
        if not data or data.get("status") != "running":
            return False
        pid = data.get("pid")
        if not pid:
            return False
        try:
            if sys.platform == "win32":
                os.kill(pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                os.kill(pid, signal.SIGTERM)
            return True
        except Exception:
            return False

    def cleanup(self) -> int:
        ensure_job_dir()
        cutoff = time.time() - _JOB_RETENTION_SECONDS
        removed = 0
        for f in JOB_DIR.glob("pool_update_*.json"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    log = f.with_suffix(".log")
                    if log.exists():
                        log.unlink()
                    removed += 1
            except Exception:
                continue
        return removed

    def read_log(self, job_id: str, tail: int = 200) -> str:
        log = job_path(job_id).with_suffix(".log")
        if not log.exists():
            return ""
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-tail:])
        except Exception:
            return ""


def parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pool_update_worker")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--term", required=True, choices=("short", "medium", "long"))
    return parser.parse_args()


def record_failure(job_id: str, exc: BaseException) -> None:
    import traceback
    try:
        writer = AtomicJobWriter(job_id)
        writer.update(
            status=JobStatus.FAILED.value,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            finished_at=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error("record_failure could not write job file: %s", e)


# 顶部 import 追加:
import signal
import time
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_job_manager.py -v`
Expected: 18 PASS

- [ ] **Step 5: 跑 pyflakes 确保无遗漏 import**

Run: `python -m pyflakes src/eval/job_manager.py`
Expected: 无输出

- [ ] **Step 6: Commit**

```bash
git add src/eval/job_manager.py tests/test_job_manager.py
git commit -m "feat(job-manager): JobManager facade (start/poll/find/stop/cleanup)"
```

---

### Task 5: Worker 入口 `pool_update_worker.py`

**Files:**
- Create: `src/eval/pool_update_worker.py`（~85 行）
- Test: `tests/test_pool_update_worker.py`（4 个用例）

**Interfaces:**
- Consumes: `AtomicJobWriter`、`parse_worker_args`、`record_failure`（Task 2-4）；`EvalOrchestrator.run_pool_update`（既有接口，**不修改**）
- Produces: `main()` 函数 + `__main__` 入口；可被 `python -m src.eval.pool_update_worker --job-id X --term Y` 执行

- [ ] **Step 1: 写 worker 测试**

```python
# tests/test_pool_update_worker.py
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path


WORKER_MODULE = "src.eval.pool_update_worker"


def _run_worker_in_subprocess(job_id: str, term: str, tmp_path: Path, extra_env=None):
    env = {
        **os.environ,
        "POOL_UPDATE_JOB_ID": job_id,
        "POOL_UPDATE_TERM": term,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "Finance" / "Financial-MCP-Agent"),
        "_FAKE_ORCH": "1",  # worker 看到此变量就用假 orchestrator
    }
    if extra_env:
        env.update(extra_env)
    log_path = tmp_path / f"{job_id}.log"
    proc = subprocess.Popen(
        [sys.executable, "-m", WORKER_MODULE, "--job-id", job_id, "--term", term],
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc, log_path


def test_worker_writes_running_then_completed(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()

    job_id = "pool_update_short_20260702_143052_abcd"
    proc, log_path = _run_worker_in_subprocess(job_id, "short", tmp_path)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise

    data = json.loads((tmp_path / f"{job_id}.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["pid"] == proc.pid
    assert "progress" in data


def test_worker_records_failure_on_orchestrator_error(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()

    job_id = "pool_update_short_20260702_143052_fail"
    proc, _ = _run_worker_in_subprocess(
        job_id, "short", tmp_path, extra_env={"_FAKE_ORCH_RAISE": "1"}
    )
    proc.wait(timeout=30)
    data = json.loads((tmp_path / f"{job_id}.json").read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert "RuntimeError" in data.get("error", "")


def test_worker_args_parser_rejects_invalid_term(capsys):
    import src.eval.job_manager as jm
    import argparse
    import sys as _sys
    old = _sys.argv
    _sys.argv = ["pool_update_worker", "--job-id", "x", "--term", "bogus"]
    try:
        try:
            jm.parse_worker_args()
        except SystemExit:
            pass
    finally:
        _sys.argv = old


def test_worker_log_file_captures_stdout(tmp_path, monkeypatch):
    import src.eval.job_manager as jm
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    jm.ensure_job_dir()

    job_id = "pool_update_short_20260702_143052_log"
    proc, log_path = _run_worker_in_subprocess(job_id, "short", tmp_path)
    proc.wait(timeout=30)
    log_text = log_path.read_text(encoding="utf-8")
    assert "pool_update_worker" in log_text or "running" in log_text.lower()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_pool_update_worker.py -v`
Expected: 4 FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 worker**

```python
# src/eval/pool_update_worker.py
"""精筛池更新 worker — detached subprocess 入口。

由 JobManager 以 subprocess.Popen 启动, 与 Streamlit 进程生命周期解耦。
通过 AtomicJobWriter 把进度 2Hz 写入 job 文件, 供 UI 轮询读取。

运行方式:
    python -m src.eval.pool_update_worker --job-id <id> --term short|medium|long
"""
import asyncio
import datetime
import logging
import os
import sys
import time
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pool_update_worker")


def _make_fake_orchestrator():
    """测试用假 orchestrator: 模拟 run_pool_update 触发 3 次进度回调然后返回。"""
    class _Fake:
        async def run_pool_update(self, term, on_progress=None, on_stage=None):
            for pct in (10.0, 50.0, 100.0):
                if on_progress:
                    on_progress({
                        "overall_pct": pct, "elapsed_s": 1, "eta_s": 1,
                        "eta_str": "1s", "queue_depth": 0, "stall_s": 0,
                        "stages": {},
                    })
                await asyncio.sleep(0.05)
            return {"term": term, "pool": [], "final_pool_size": 0, "stats": {}}
    return _Fake()


def main() -> None:
    from src.eval.job_manager import (
        AtomicJobWriter, parse_worker_args, JobStatus,
    )

    args = parse_worker_args()
    job_id = args.job_id
    term = args.term

    logger.info("pool_update_worker starting: job_id=%s term=%s", job_id, term)
    writer = AtomicJobWriter(job_id)
    writer.update(
        status=JobStatus.RUNNING.value,
        pid=os.getpid(),
        parent_pid=os.getppid(),
    )

    if os.environ.get("_FAKE_ORCH") == "1":
        orch = _make_fake_orchestrator()
        if os.environ.get("_FAKE_ORCH_RAISE") == "1":
            async def _raise(**kw):
                raise RuntimeError("simulated orchestrator failure")
            orch.run_pool_update = _raise
    else:
        from src.eval.orchestrator import EvalOrchestrator
        orch = EvalOrchestrator()

    last_write = 0.0
    throttle = 0.5  # 2Hz

    def on_progress(data):
        nonlocal last_write
        now = time.time()
        if now - last_write < throttle:
            return
        last_write = now
        writer.merge_progress(data)
        # 如果 data 里带了 completed_stocks 增量, 也追加 (V3 管线目前不带, 留扩展)
        for s in data.get("new_completed_stocks", []) or []:
            writer.append_completed_stock(s)

    def on_stage(stage_name, message):
        # 阶段切换也触发一次进度刷新 (即使未到 500ms)
        writer.merge_progress({"current_stage": stage_name, "current_stage_msg": message})

    result = asyncio.run(
        orch.run_pool_update(term, on_progress=on_progress, on_stage=on_stage)
    )
    writer.update(
        status=JobStatus.COMPLETED.value,
        finished_at=datetime.datetime.now().isoformat(),
        progress={"result": _json_safe(result)},
    )
    logger.info("pool_update_worker completed: job_id=%s", job_id)


def _json_safe(obj):
    """确保 result 可 JSON 序列化: 非序列化对象转 str。"""
    import json
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return {"repr": repr(obj)[:500]}


if __name__ == "__main__":
    from src.eval.job_manager import record_failure
    try:
        main()
    except Exception as e:
        job_id = os.environ.get("POOL_UPDATE_JOB_ID", "unknown")
        logger.error("worker failed: %s", e)
        record_failure(job_id, e)
        sys.exit(1)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_pool_update_worker.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/pool_update_worker.py tests/test_pool_update_worker.py
git commit -m "feat(worker): detached pool_update_worker entry point (2Hz progress write)"
```

---

### Task 6: UI 改造 — `ops_panel.py`

**Files:**
- Modify: `src/app/components/eval/ops_panel.py`（~40 行替换）

**Interfaces:**
- Consumes: `JobManager`、`JobStatus`（Task 4）；`st.session_state` 存 `pool_job_{term}`
- Produces: "🎯 更新精筛池" 按钮行为从同步调用 adapter 改为 spawn worker + 轮询 job 文件

- [ ] **Step 1: 阅读现状（手动）**

Run: `grep -n "run_pool_update\|run_pool_update_streaming" src/app/components/eval/ops_panel.py`
Expected: 看到 1 处 `orch.run_pool_update(pool_term, on_stage=on_stage)` 在 `_run_async` 包裹里

- [ ] **Step 2: 替换调用方（约 30 行改动）**

把"🎯 更新精筛池"按钮分支内的 try/except 块替换为：

```python
# src/app/components/eval/ops_panel.py 替换位置 (原 try 块整段)
if st.button("🎯 更新精筛池", use_container_width=True, type="primary"):
    if eval_ready:
        from src.eval.job_manager import JobManager, JobStatus
        jm = JobManager()

        status_box = st.status("启动 worker 中...", expanded=True)
        progress_bar = st.progress(0, text="初始化...")

        try:
            existing = jm.find_running(pool_term)
            if existing:
                job_id = existing["job_id"]
                st.info(
                    f"已有正在跑的更新 ({existing.get('started_at', '')}), 自动 attach"
                )
            else:
                job_id = jm.start_job(pool_term)

            st.session_state[f"pool_job_{pool_term}"] = job_id

            # 轮询进度
            while True:
                job = jm.poll(job_id)
                if not job:
                    st.error("任务信息丢失")
                    break

                status = job.get("status")
                prog = job.get("progress") or {}
                pct = prog.get("overall_pct", 0) / 100.0
                eta = prog.get("eta_str", "计算中...")
                stall = prog.get("stall_s", 0)

                progress_bar.progress(
                    min(pct, 1.0),
                    f"总进度 {pct*100:.0f}% | ETA: {eta}"
                    + (f" | ⚠️ 卡顿{stall:.0f}s" if stall > 60 else "")
                )
                status_box.update(
                    label=f"running: {prog.get('current_stage_msg', '')[:60]}",
                    state="running",
                )

                if status == JobStatus.COMPLETED.value:
                    progress_bar.progress(1.0, text="完成!")
                    status_box.update(label="精筛池更新完成", state="complete")
                    st.success(
                        f"精筛池[{pool_term}]更新完成！"
                        f"共{job.get('progress', {}).get('result', {}).get('final_pool_size', 0)}只"
                    )
                    break
                if status in (JobStatus.FAILED.value, JobStatus.ORPHANED.value):
                    status_box.update(label="更新失败", state="error")
                    st.error(job.get("error") or f"任务状态: {status}")
                    with st.expander("📋 Worker 日志", expanded=False):
                        st.code(jm.read_log(job_id, tail=200), language="log")
                    break

                time.sleep(1.0)

            st.rerun()
        except Exception as e:
            progress_bar.progress(100, text="失败")
            status_box.update(label="更新失败", state="error")
            st.error(f"精筛池更新启动失败: {e}")
```

- [ ] **Step 3: 手动验证（需要 Streamlit 运行环境）**

Run: 启动 Streamlit → 打开"模拟分析与迭代"页面 → 点"🎯 更新精筛池"
Expected:
- 进度条每秒刷新
- Worker 进程独立存在（`tasklist | findstr python` 能看到 2 个 PID）
- 跑到一半关 Streamlit → worker 仍跑 → 重开 Streamlit → 按钮上显示 "自动 attach"
- （本步骤在 Task 9 做全流程验证，本 task 只做语法级校验）

- [ ] **Step 4: py_compile 校验**

Run: `python -c "import ast; ast.parse(open('src/app/components/eval/ops_panel.py', encoding='utf-8').read()); print('OK')"`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/app/components/eval/ops_panel.py
git commit -m "feat(ui): ops_panel 改用 JobManager 启动 worker + 轮询 job 文件"
```

---

### Task 7: UI 改造 — `06_模拟分析与迭代.py` 健康卡片 + Worker 日志面板

**Files:**
- Modify: `src/app/pages/06_模拟分析与迭代.py`（健康卡片"🔄 全量更新"分支，~15 行）

**Interfaces:**
- Consumes: `JobManager`、`JobStatus`
- Produces: 健康卡片按钮走与 Task 6 相同路径；新增 worker 日志 expander

- [ ] **Step 1: 阅读现状**

Run: `grep -n "全量更新\|adapter.run_pool_update_streaming\|run_pool_update_streaming" src/app/pages/06_模拟分析与迭代.py`
Expected: 看到一处 `adapter.run_pool_update_streaming(term, ...)`

- [ ] **Step 2: 替换健康卡片逻辑**

把"🔄 全量更新{label}池"按钮分支整段替换为（结构与 Task 6 一致，仅文案不同）：

```python
# src/app/pages/06_模拟分析与迭代.py 替换位置 (原 holder = adapter.run_pool_update_streaming(...) 整段)
if st.button(
    f"🔄 全量更新{label}池", key=f"health_full_{term}",
    use_container_width=True,
):
    from src.eval.job_manager import JobManager, JobStatus
    jm = JobManager()
    progress_bar = st.progress(0, "Layer 0: 硬筛中...")
    status_col1, status_col2, status_col3 = st.columns(3)
    eta_display = st.empty()
    stage_lines = st.empty()

    try:
        existing = jm.find_running(term)
        if existing:
            job_id = existing["job_id"]
            st.info(f"已有正在跑的更新 ({existing.get('started_at', '')}), 自动 attach")
        else:
            job_id = jm.start_job(term)

        while True:
            job = jm.poll(job_id)
            if not job:
                st.error("任务信息丢失")
                break
            status = job.get("status")
            prog = job.get("progress") or {}
            pct = prog.get("overall_pct", 0) / 100.0
            eta = prog.get("eta_str", "计算中...")
            stall = prog.get("stall_s", 0)

            progress_bar.progress(
                min(pct, 1.0),
                f"总进度 {pct*100:.0f}% | ETA: {eta}"
                + (f" | ⚠️ 卡顿{stall:.0f}s" if stall > 60 else "")
            )
            stages = prog.get("stages") or {}
            stage_text = ""
            for k, v in stages.items():
                if v.get("done", 0) > 0:
                    stage_text += (
                        f"**{v['label']}**: {v['pct']:.0f}% "
                        f"({v.get('done', 0)}/{v.get('total', 0)})  "
                    )
            stage_lines.markdown(stage_text or "准备中...")
            eta_display.caption(
                f"已运行 {(prog.get('elapsed_s', 0) // 60)}min"
                f" | ETA: {eta}"
                f" | 队列: {prog.get('queue_depth', 0)}只"
            )

            if status == JobStatus.COMPLETED.value:
                progress_bar.progress(1.0, "完成!")
                result = prog.get("result") or {}
                st.success(
                    f"更新完成！共 {result.get('final_pool_size', 0)} 只"
                    f" | 耗时 {result.get('stats', {}).get('elapsed_s', 0) // 60}min"
                )
                break
            if status in (JobStatus.FAILED.value, JobStatus.ORPHANED.value):
                st.error(job.get("error") or f"任务状态: {status}")
                with st.expander("📋 Worker 日志", expanded=False):
                    st.code(jm.read_log(job_id, tail=200), language="log")
                break
            time.sleep(1.0)

        st.rerun()
    except Exception as e:
        st.error(f"更新启动失败: {e}")
```

- [ ] **Step 3: py_compile 校验**

Run: `python -c "import ast; ast.parse(open('src/app/pages/06_模拟分析与迭代.py', encoding='utf-8').read()); print('OK')"`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add src/app/pages/06_模拟分析与迭代.py
git commit -m "feat(ui): 06_模拟分析与迭代 健康卡片改用 JobManager + worker 日志面板"
```

---

### Task 8: 删除旧接口 `WebAdapter.run_pool_update_streaming`

**Files:**
- Modify: `src/eval/web_adapter.py`（删除 1 个方法，约 40 行）

**Interfaces:**
- 无新增；仅移除旧方法

- [ ] **Step 1: 确认无其他调用方**

Run: `grep -rn "run_pool_update_streaming" src/`
Expected: 0 hits（Task 6、7 已替换）

- [ ] **Step 2: 删除方法**

编辑 `src/eval/web_adapter.py`，删除整个 `run_pool_update_streaming` 方法（包括内部 `_run` 嵌套函数和 `_thread_target`）。保留 `run_pool_update`（同步版仍有用）。

- [ ] **Step 3: 跑现有测试确认不破**

Run: `python -m pytest tests/ -v --ignore=tests/test_job_manager.py --ignore=tests/test_pool_update_worker.py -x`
Expected: 全部 PASS，无 `AttributeError: run_pool_update_streaming`

- [ ] **Step 4: Commit**

```bash
git add src/eval/web_adapter.py
git commit -m "refactor(web-adapter): 删除 run_pool_update_streaming (UI 已切到 JobManager)"
```

---

### Task 9: 跨平台 Detach 验证（手动）

**Files:** 无代码改动，纯手动验收

- [ ] **Step 1: Windows detach 验证**

1. 启 Streamlit：`.\run.ps1 start`
2. 打开浏览器 `http://localhost:8501` → "模拟分析与迭代" 页面
3. 点"🎯 更新精筛池" → 选"中线"
4. 记下 worker PID（UI 上应该显示；或 `tasklist | findstr python`）
5. `.\run.ps1 stop` 杀 Streamlit
6. `tasklist | findstr <PID>` 确认 worker 仍存活
7. `.\run.ps1 start` 重启 Streamlit → 浏览器打开 → UI 应显示 "自动 attach" 并继续显示进度

- [ ] **Step 2: Orphan 检测验证**

1. 启 worker（同 Step 1 前 4 步）
2. 用 `taskkill /PID <worker_pid> /F` 强杀 worker
3. 等 2 秒 → UI 应显示 `status=orphaned` + 错误信息
4. Job 文件 `status` 字段应为 `"orphaned"`

- [ ] **Step 3: Crash 后恢复验证**

1. 完成 Step 2 后，点"🎯 更新精筛池"重新触发
2. 新 worker 启动 → 跑 V3 管线
3. 之前已完成的股票因 per-agent 缓存命中，应直接跳过 LLM 调用
4. 总耗时明显短于首次冷启动

- [ ] **Step 4: 多 term 并行验证**

1. 同时点"短线"/"中线"/"长线"三个"🎯 更新精筛池"
2. 应有 3 个 worker 进程独立运行
3. 各自 job 文件互不干扰
4. UI 三套进度条独立推进

- [ ] **Step 5: 幂等验证**

1. 点"🎯 更新精筛池"
2. 进度 10% 时再点一次同 term 按钮
3. 第二次应显示"已有正在跑的更新，自动 attach"，不启新 worker

---

### Task 10: 5 轮深度代码检查

**Files:** 无代码改动（必要时修 bug）

- [ ] **Round 1: 语法 + 静态分析**

```bash
python -c "import ast; ast.parse(open('src/eval/job_manager.py', encoding='utf-8').read()); print('job_manager OK')"
python -c "import ast; ast.parse(open('src/eval/pool_update_worker.py', encoding='utf-8').read()); print('worker OK')"
python -c "import ast; ast.parse(open('src/app/components/eval/ops_panel.py', encoding='utf-8').read()); print('ops_panel OK')"
python -c "import ast; ast.parse(open('src/app/pages/06_模拟分析与迭代.py', encoding='utf-8').read()); print('06 OK')"
python -m pyflakes src/eval/job_manager.py src/eval/pool_update_worker.py src/app/components/eval/ops_panel.py src/app/pages/06_模拟分析与迭代.py
```

Expected: 全部 OK；pyflakes 无新增 warning（仅允许既有的预存 warning）

- [ ] **Round 2: Bug 检查（人审）**

逐项确认：
- [ ] `AtomicJobWriter.update` 是否每次都读盘 merge？避免内存缓存导致父子进程不一致
- [ ] `_spawn_worker` 的 `stdout=open(...)` 文件句柄是否在 `Popen` 返回后仍保持（是的，由子进程持有）
- [ ] `JobManager.poll` 检测到 orphan 时**写入文件**而不是只返回 dict（防止下次 poll 仍误判）
- [ ] `cleanup` 不会删正在跑的 running job（mtime 远小于 cutoff）
- [ ] `record_failure` 在 `job_id` 不存在时不 crash（先建 writer，writer 会自动建文件）
- [ ] `on_progress` 节流 500ms 不会漏写最后一次（worker 完成前有一次显式 `writer.update`）

- [ ] **Round 3: 隐藏功能问题**

- [ ] Job 文件在 Windows 下若被 OneDrive 同步锁住，`os.replace` 会失败。`AtomicJobWriter._atomic_dump` 应加 3 次重试 + 100ms sleep（**若未实现则在此 task 补**）
- [ ] UI `while True` 循环里没有"用户中途停止"按钮 → 加一个 `st.button("❌ 取消更新")`，点击调 `JobManager.stop(job_id)`
- [ ] `find_running` 只扫本 term 的 `pool_update_{term}_*.json`，不会误 attach 其他 term
- [ ] `completed_stocks` 目前 V3 管线 `on_progress` 不带 `new_completed_stocks`，因此始终为空 → 这是预期行为（per-agent 缓存已经做了续跑），UI 只显示 L3 stage 进度即可

- [ ] **Round 4: 副作用检查**

- [ ] 确认 `orchestrator.py`、`pool_screening.py`、`scoring_engine.py`、`src/agents/*`、`mcp_client.py` **零改动**（`git diff --stat` 检查）
- [ ] 跑全量 `pytest tests/` 应全绿
- [ ] 启动 Streamlit 走一遍非 pool update 的路径（单股票分析、基金分析、QA）确认不受影响

- [ ] **Round 5: 回归语法 + 测试**

```bash
python -m pytest tests/ -v
python -m pyflakes src/eval/job_manager.py src/eval/pool_update_worker.py
```

Expected: 全绿 + 无新增 warning

- [ ] **Commit（如有修 bug）**

```bash
git add -A
git commit -m "fix(job-manager): round 3/4 发现的问题修复"
```

---

### Task 11: 收尾 — gitignore + 文档

**Files:**
- Create: `Finance/Financial-MCP-Agent/data/pool_update_jobs/.gitignore`
- Create: `Finance/Financial-MCP-Agent/data/pool_update_jobs/.gitkeep`
- Modify: `CLAUDE.md`（追加"精筛池更新跨重启存活"小节）

- [ ] **Step 1: 写 .gitignore**

```gitignore
# data/pool_update_jobs/.gitignore
# Job 文件 + worker 日志不进 git
*.json
*.log
*.tmp
```

- [ ] **Step 2: 写 .gitkeep 占位**

```bash
touch Finance/Financial-MCP-Agent/data/pool_update_jobs/.gitkeep
```

- [ ] **Step 3: CLAUDE.md 追加小节**

在 `## Commands` 章节前插入：

```markdown
## 精筛池更新跨重启存活

Streamlit 页面触发的"🎯 更新精筛池"操作以 detached subprocess 方式运行, 与 Streamlit 进程生命周期解耦。

**核心文件**:
- `src/eval/job_manager.py`: `JobManager` (UI 侧) + `AtomicJobWriter` (worker 侧)
- `src/eval/pool_update_worker.py`: worker 入口, `python -m src.eval.pool_update_worker`
- `data/pool_update_jobs/`: job 文件 + worker 日志 (1 年保留, 自动清理)

**关键行为**:
- 同 term 同时只允许 1 个 running job, 重复触发自动 attach
- 2Hz 原子写入 (write → .tmp → os.replace), 防止读半写文件
- Orphan 检测: `status=running` 但 pid 不存在 → 自动标 `orphaned`
- Windows: `CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS`; POSIX: `start_new_session=True`

**调试**: UI 提供"📋 Worker 日志"折叠面板; 直接看 `data/pool_update_jobs/{job_id}.log`
```

- [ ] **Step 4: 最终 commit**

```bash
git add -A
git commit -m "chore: 跨重启存活功能完整交付 (gitignore + CLAUDE.md 文档)"
```

---

## 回滚预案

所有改动分布在 7 个独立 commit，可分级回滚：

| 出问题范围 | 回滚命令 | 后果 |
|---|---|---|
| 整套跨重启功能 | `git revert 13a5dcd4..HEAD`（spec + 7 task commits） | 完全回到本计划前 |
| 仅 UI 异常 | `git revert <Task 6 commit> <Task 7 commit>` | UI 失去 worker 入口，但旧 `run_pool_update_streaming` 也已删，**需要同时 revert Task 8 才能恢复** |
| 仅 worker 异常 | `git revert <Task 5 commit>` | UI 启动 worker 会失败，但 JobManager 还在 |

**推荐回滚路径**：`git revert HEAD~7..HEAD`（一次性回退 7 个 task commits），保留 spec 文件供复盘。
