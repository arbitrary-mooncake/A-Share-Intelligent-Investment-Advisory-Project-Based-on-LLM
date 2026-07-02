# tests/test_job_manager.py
import json
import os


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
    # pool_update_medium_YYYYMMDD_HHMMSS_rand → 6 parts
    assert len(parts) == 6
    assert len(parts[3]) == 8   # YYYYMMDD
    assert len(parts[4]) == 6   # HHMMSS
    assert len(parts[5]) == 4   # rand


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


# ---- Task 3: spawn + orphan tests ----
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
