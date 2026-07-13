"""Tests for line manager and pool manager"""
import shutil
import tempfile
from pathlib import Path

import pytest

import src.eval.pool_manager as pool_manager_module
from src.eval.line_manager import LineManager, LineState, LINE_DEFINITIONS
from src.eval.pool_manager import PoolManager


@pytest.fixture(autouse=True)
def isolated_pool_storage(monkeypatch):
    """所有池管理测试使用临时文件，禁止改写真实精筛池。"""
    tmpdir = tempfile.mkdtemp(
        prefix=".pool_test_",
        dir=Path(__file__).resolve().parent,
    )
    monkeypatch.setattr(pool_manager_module, "POOL_DIR", tmpdir)
    monkeypatch.setattr(pool_manager_module, "POOL_FILE", f"{tmpdir}/refined_pools.json")
    # 这些测试验证内存中的池管理逻辑，不需要落盘；禁止任何文件写入。
    monkeypatch.setattr(pool_manager_module.PoolManager, "_save", lambda self: None)
    yield
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_line_definitions():
    assert len(LINE_DEFINITIONS) == 14
    assert "S-L0" in LINE_DEFINITIONS
    assert "M-L0" in LINE_DEFINITIONS
    assert "L-L0" in LINE_DEFINITIONS
    assert LINE_DEFINITIONS["S-L0"]["term"] == "short"
    assert LINE_DEFINITIONS["S-L0"]["type"] == "ablation_base"


def test_line_state_init():
    line = LineState("S-L0", 1000000)
    assert line.line_id == "S-L0"
    assert line.cash == 1000000
    assert len(line.holdings) == 0
    assert line.status == "active"


def test_line_state_update_value():
    line = LineState("S-L0", 1000000)
    line.holdings = {"sh.601888": 10000}
    line.cash = 500000
    line.update_value({"sh.601888": 80.0})
    assert line.total_value == 500000 + 800000  # cash + 10000*80


def test_line_manager_init():
    lm = LineManager(1000000)
    assert len(lm.lines) == 14
    assert lm.get_line("S-L0") is not None


def test_line_manager_get_ablation():
    lm = LineManager()
    lines = lm.get_ablation_lines("short")
    assert len(lines) == 8  # S-L0 through S-L7


def test_sync_ablation():
    lm = LineManager()
    base = lm.get_ablation_base("short")
    base.holdings = {"sh.601888": 5000}
    base.cash = 500000
    lm.sync_ablation_holdings("short")
    for line in lm.get_ablation_lines("short"):
        if line.line_id == base.line_id:
            continue
        assert "sh.601888" in line.holdings
        assert line.holdings["sh.601888"] == 5000


def test_pool_manager_init():
    pm = PoolManager()
    pool = pm.get_pool("short")
    assert isinstance(pool, list)


def test_pool_manager_update():
    pm = PoolManager()
    stocks = [{"code": "sh.601888", "name": "中国中免", "score": 82}]
    pm.update_pool("short", stocks)
    pool = pm.get_pool_with_scores("short")
    assert len(pool) == 1
    assert pool[0]["score"] == 82


def test_pool_result_guard_preserves_existing_pool():
    pm = PoolManager()
    stocks = [
        {"code": f"sh.{600000 + i}", "name": f"测试{i}", "final_score": 80}
        for i in range(100)
    ]
    pm.update_pool("short", stocks)
    snapshot = pm.snapshot_term("short")

    reason = pm.validate_pool_replacement("short", [], snapshot)
    assert reason is not None

    pm.restore_term_snapshot("short", snapshot)
    assert len(pm.get_pool("short")) == 100


def test_pool_manager_blacklist():
    pm = PoolManager()
    pm.add_to_blacklist("sh.999999", reason="测试")
    assert pm.is_blacklisted("sh.999999")
    assert not pm.is_blacklisted("sh.601888")


def test_pool_summary():
    pm = PoolManager()
    summary = pm.get_pool_summary("medium")
    assert summary["term"] == "medium"
    assert "target_size" in summary


def test_needs_update():
    pm = PoolManager()
    result = pm.needs_update("short")
    assert "needs_update" in result
    assert "days_since_update" in result
