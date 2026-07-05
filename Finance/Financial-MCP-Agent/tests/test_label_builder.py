"""
Tests for label_builder: holding return, drawdown, volatility, and realized labels.

Complements tests/test_settlement.py which covers basic compute_holding_return,
compute_drawdown, compute_volatility, and build_realized_label.

This file adds:
  - Annualized return edge cases (zero holding days, fractional years)
  - Drawdown edge cases (single price, empty list, flat prices)
  - Volatility edge cases (single return, constant returns)
  - RealizedLabel edge cases (zero returns, missing benchmark, zero-length prices)
  - Label accuracy with known outcomes
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.eval.label_builder import (
    compute_holding_return,
    compute_drawdown,
    compute_volatility,
    build_realized_label,
)


# ── Holding Return Edge Cases ──────────────────────────────────────────────


def test_annualized_return_zero_days():
    """持有天数为0时年化收益率等于绝对收益率"""
    result = compute_holding_return(10.0, 11.0, 0)
    assert result["asset_return_pct"] == 10.0
    assert result["annualized_return_pct"] == 10.0


def test_annualized_return_one_year():
    """持有252天时年化近似绝对收益（复利效应小）"""
    result = compute_holding_return(10.0, 11.0, 252)
    assert result["asset_return_pct"] == 10.0
    # Annualized should be very close to 10% when holding exactly 1 year
    assert abs(result["annualized_return_pct"] - 10.0) < 0.1


def test_holding_return_exit_price_zero():
    """卖出价为0时标记为invalid"""
    result = compute_holding_return(10.0, 0.0, 10)
    assert not result["valid"]


def test_holding_return_entry_price_negative():
    """买入价为负时标记为invalid"""
    result = compute_holding_return(-5.0, 10.0, 10)
    assert not result["valid"]


def test_holding_return_as_flat():
    """价格未变时收益率为0"""
    result = compute_holding_return(50.0, 50.0, 100)
    assert result["asset_return_pct"] == 0.0
    assert result["excess_return_pct"] == 0.0


def test_holding_return_excess_underperformance():
    """跑输基准时超额收益为负"""
    result = compute_holding_return(10.0, 11.0, 20, 100.0, 130.0)
    assert result["asset_return_pct"] == 10.0
    assert result["benchmark_return_pct"] == 30.0
    assert result["excess_return_pct"] == -20.0


# ── Drawdown Edge Cases ────────────────────────────────────────────────────


def test_drawdown_single_price():
    """只有一个价格时回撤为0"""
    mdd, cur = compute_drawdown([100.0])
    assert mdd == 0.0
    assert cur == 0.0


def test_drawdown_empty_list():
    """空列表回撤为0"""
    mdd, cur = compute_drawdown([])
    assert mdd == 0.0
    assert cur == 0.0


def test_drawdown_flat_prices():
    """价格不变时回撤为0"""
    mdd, cur = compute_drawdown([100, 100, 100, 100])
    assert mdd == 0.0
    assert cur == 0.0


def test_drawdown_deep_v():
    """深V走势：回撤大但当前回撤小"""
    prices = [100, 80, 50, 60, 90, 110, 120]
    mdd, cur = compute_drawdown(prices)
    assert mdd > 0.4  # from 100 to 50
    assert cur == 0.0  # recovered above peak


def test_drawdown_current_at_bottom():
    """当前处于最低点时，当前回撤=最大回撤"""
    prices = [100, 110, 90, 80, 70]
    mdd, cur = compute_drawdown(prices)
    assert mdd > 0.0
    assert abs(mdd - cur) < 0.001


def test_drawdown_two_spikes():
    """两次下跌，第二次更深"""
    prices = [100, 90, 100, 70, 100]
    mdd, cur = compute_drawdown(prices)
    assert mdd == 0.3  # (100-70)/100


# ── Volatility Edge Cases ──────────────────────────────────────────────────


def test_volatility_single_return():
    """只有一个日收益时波动率为0"""
    vol = compute_volatility([0.01])
    assert vol == 0.0


def test_volatility_empty_returns():
    """空列表波动率为0"""
    vol = compute_volatility([])
    assert vol == 0.0


def test_volatility_constant_returns():
    """恒定日收益时波动率为0"""
    vol = compute_volatility([0.01, 0.01, 0.01, 0.01])
    assert abs(vol) < 0.0001


def test_volatility_high_variance():
    """高波动时波动率显著大于0"""
    returns = [0.05, -0.04, 0.03, -0.05, 0.06, -0.02]
    vol = compute_volatility(returns)
    assert vol > 0.05  # daily vol ~5%


def test_volatility_not_annualized():
    """非年化时波动率明显小于年化"""
    returns = [0.01, -0.005, 0.02, -0.01, 0.005]
    vol_daily = compute_volatility(returns, annualize=False)
    vol_annual = compute_volatility(returns, annualize=True)
    assert vol_daily < vol_annual
    # Annualized ≈ daily * sqrt(252)
    expected_ratio = vol_annual / max(vol_daily, 1e-9)
    assert 10 < expected_ratio < 20  # sqrt(252) ≈ 15.87


# ── RealizedLabel Edge Cases ────────────────────────────────────────────────


def test_build_realized_label_zero_return():
    """零收益时标签正确"""
    snap = {"snapshot_id": "snap-ZR", "line_id": "S-L0", "term": "short"}
    price_data = {
        "entry_price": 100.0,
        "exit_price": 100.0,
        "prices": [100.0, 100.0, 100.0],
        "start_date": "2024-03-01",
        "end_date": "2024-03-03",
    }
    label = build_realized_label(snap, price_data)
    assert label["asset_return_pct"] == 0.0
    assert label["is_valid"] is True


def test_build_realized_label_missing_benchmark():
    """无基准数据时基准收益为0"""
    snap = {"snapshot_id": "snap-MB", "line_id": "M-L0", "term": "medium"}
    price_data = {
        "entry_price": 10.0,
        "exit_price": 12.0,
        "prices": [10, 10.5, 11, 11.5, 12],
        "start_date": "2024-01-01",
        "end_date": "2024-01-05",
    }
    label = build_realized_label(snap, price_data)  # no benchmark
    assert label["benchmark_return_pct"] == 0.0
    assert label["excess_return_pct"] == label["asset_return_pct"]


def test_build_realized_label_single_price():
    """只有一个价格点（无法计算波动率）"""
    snap = {"snapshot_id": "snap-SP", "line_id": "S-L0", "term": "short"}
    price_data = {
        "entry_price": 100.0,
        "exit_price": 100.0,
        "prices": [100.0],
        "start_date": "2024-06-01",
        "end_date": "2024-06-01",
    }
    label = build_realized_label(snap, price_data)
    assert label["volatility_pct"] == 0.0
    assert label["max_drawdown_pct"] == 0.0


def test_build_realized_label_with_snapshot_term():
    """标签继承snapshot的term信息"""
    for term in ["short", "medium", "long"]:
        snap = {"snapshot_id": f"snap-{term}", "line_id": "T-L0", "term": term}
        price_data = {
            "entry_price": 10.0, "exit_price": 12.0,
            "prices": [10, 11, 12],
            "start_date": "2024-01-01", "end_date": "2024-01-03",
        }
        label = build_realized_label(snap, price_data)
        assert label["term"] == term


def test_build_realized_label_large_drawdown():
    """大幅回撤标签记录mdd"""
    snap = {"snapshot_id": "snap-LD", "line_id": "L-L0", "term": "long"}
    price_data = {
        "entry_price": 100.0,
        "exit_price": 90.0,
        "prices": [100, 110, 80, 95, 90],
        "start_date": "2024-01-01",
        "end_date": "2024-01-05",
    }
    label = build_realized_label(snap, price_data)
    assert label["max_drawdown_pct"] > 20.0  # from 110 to 80 = 27.27%


def test_build_realized_label_meta_json_default():
    """meta_json默认为空字符串"""
    snap = {"snapshot_id": "snap-META", "line_id": "S-L0", "term": "short"}
    price_data = {
        "entry_price": 10.0, "exit_price": 12.0,
        "prices": [10, 11, 12],
        "start_date": "2024-01-01", "end_date": "2024-01-03",
    }
    label = build_realized_label(snap, price_data)
    assert label["meta_json"] == ""
    assert label["settlement_notes"] == ""


# ── Accuracy Tests ──────────────────────────────────────────────────────────


def test_holding_return_exact_calculation():
    """精确计算验证"""
    # Buy at 100, sell at 105, hold 5 days
    result = compute_holding_return(100.0, 105.0, 5)
    assert result["asset_return_pct"] == 5.0

    # Excess with no benchmark = asset return
    assert result["excess_return_pct"] == 5.0

    # Annualized: (1.05)^(252/5) - 1
    expected_annualized = ((1.05) ** (252 / 5) - 1) * 100
    assert abs(result["annualized_return_pct"] - expected_annualized) < 0.01


def test_holding_return_benchmark_accuracy():
    """跑赢基准时的超额收益精确计算"""
    # Stock: 100 -> 110 (+10%), Benchmark: 1000 -> 1050 (+5%)
    result = compute_holding_return(100.0, 110.0, 10, 1000.0, 1050.0)
    assert result["asset_return_pct"] == 10.0
    assert result["benchmark_return_pct"] == 5.0
    assert result["excess_return_pct"] == 5.0
