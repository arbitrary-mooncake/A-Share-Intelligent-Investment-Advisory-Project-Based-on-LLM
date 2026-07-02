# tests/test_job_manager.py


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
