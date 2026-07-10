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
    """Test check command — requires real Tushare data and pool"""
    result = subprocess.run(
        [sys.executable, "-m", "src.eval.cli", "check"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        timeout=60
    )
    assert result.returncode == 0
