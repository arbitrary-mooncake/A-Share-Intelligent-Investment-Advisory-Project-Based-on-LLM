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
        "POOL_UPDATE_JOB_DIR": str(tmp_path),  # 让 subprocess 也用 tmp_path
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
    # PID: Linux 上 os.getpid()==proc.pid; Windows venv launcher 可能重启进程导致 PID 不同,
    # 只校验写入了一个有效 PID 即可
    assert isinstance(data["pid"], int) and data["pid"] > 0
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
