"""Tests for CLI entry point"""
import sys
import subprocess
import os
import pytest

_api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY", "").lower()
_tushare = os.getenv("TUSHARE_TOKEN", "").lower()
_PLACEHOLDERS = ("test_key", "test_token", "your_", "xxx", "placeholder", "dummy", "fake", "sk-xxxx")
_HAS_REAL_API = (
    len(_api_key) > 10
    and not any(p in _api_key for p in _PLACEHOLDERS)
    and len(_tushare) > 5
    and not any(p in _tushare for p in _PLACEHOLDERS)
)
requires_api = pytest.mark.skipif(not _HAS_REAL_API, reason="Requires real API keys")


def test_cli_help():
    """Test that CLI can be invoked and shows help"""
    result = subprocess.run(
        [sys.executable, "-m", "src.eval.cli", "--help"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        timeout=10
    )
    assert result.returncode == 0
    assert "check" in result.stdout or "check" in result.stderr


def test_cli_status():
    """Test status command"""
    result = subprocess.run(
        [sys.executable, "-m", "src.eval.cli", "status"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        timeout=10
    )
    # status should not crash
    assert result.returncode == 0


@requires_api
def test_cli_check():
    """Test check command — requires real Tushare data.

    check 运行完整 eval 流程（补回+结算+调仓评分+收盘结算）。当精筛池非空时,
    评分需数分钟, 远超测试超时; 仅在池为空时执行 smoke 测试, 非空时跳过
    （check 是重操作, 不适合在测试套件中跑全量评分）。
    """
    # 精筛池非空时 check 需跑评分, 跳过避免超时
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from src.eval.pool_manager import PoolManager
        pm = PoolManager()
        pool_size = (len(pm.get_pool("short"))
                     + len(pm.get_pool("medium"))
                     + len(pm.get_pool("long")))
    except Exception:
        pool_size = 0
    if pool_size > 0:
        pytest.skip(
            f"check command slow with non-empty pool ({pool_size} stocks); "
            "skip to avoid timeout"
        )

    result = subprocess.run(
        [sys.executable, "-m", "src.eval.cli", "check"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        timeout=120
    )
    assert result.returncode == 0
