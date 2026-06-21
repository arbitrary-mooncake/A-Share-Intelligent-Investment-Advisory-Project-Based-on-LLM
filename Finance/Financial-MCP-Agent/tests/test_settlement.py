"""Tests for settlement engine and label builder"""
from src.eval.label_builder import (
    compute_holding_return, compute_drawdown, compute_volatility, build_realized_label
)


def test_compute_holding_return_profit():
    result = compute_holding_return(10.0, 12.0, 20)
    assert result["valid"]
    assert result["asset_return_pct"] == 20.0
    assert result["excess_return_pct"] == 20.0  # no benchmark


def test_compute_holding_return_loss():
    result = compute_holding_return(10.0, 8.0, 10)
    assert result["asset_return_pct"] == -20.0


def test_compute_holding_return_with_benchmark():
    result = compute_holding_return(10.0, 12.0, 20, 100.0, 105.0)
    assert result["asset_return_pct"] == 20.0
    assert result["benchmark_return_pct"] == 5.0
    assert result["excess_return_pct"] == 15.0


def test_compute_holding_return_invalid():
    result = compute_holding_return(0, 12.0, 20)
    assert not result["valid"]


def test_compute_drawdown():
    prices = [100, 110, 95, 105, 90, 100]
    mdd, cur = compute_drawdown(prices)
    assert mdd > 0.15  # from 110 to 90


def test_compute_drawdown_no_loss():
    prices = [100, 110, 120, 130]
    mdd, cur = compute_drawdown(prices)
    assert mdd == 0.0


def test_compute_volatility():
    returns = [0.01, -0.005, 0.02, -0.01, 0.005]
    vol = compute_volatility(returns)
    assert vol > 0


def test_build_realized_label():
    snap = {"snapshot_id": "test_001", "line_id": "S-L0", "term": "short"}
    price_data = {"entry_price": 10.0, "exit_price": 12.0, "prices": [10, 10.5, 11, 11.5, 12],
                  "start_date": "2024-01-01", "end_date": "2024-01-05"}
    benchmark = {"entry_price": 3000, "exit_price": 3100}
    label = build_realized_label(snap, price_data, benchmark)
    assert label["snapshot_id"] == "test_001"
    assert label["asset_return_pct"] == 20.0
    assert label["benchmark_return_pct"] > 0
