"""Tests for market simulator"""
from src.eval.market_simulator import (
    MarketSimulator, MarketData, Order, OrderStatus, RejectReason
)


def make_market(**kwargs):
    defaults = {
        "stock_code": "sh.603871", "open": 12.0, "high": 12.5,
        "low": 11.8, "close": 12.39, "pre_close": 11.76,
        "volume": 141000, "amount": 174699000.0, "turnover_rate": 0.03,
    }
    defaults.update(kwargs)
    return MarketData(**defaults)


def test_buy_order_normal():
    sim = MarketSimulator()
    order = Order(stock_code="sh.603871", direction="buy", target_value=100000)
    md = make_market()
    result = sim.execute_order(order, md)
    assert result.status == OrderStatus.FILLED.value
    assert result.actual_price > md.close  # slippage
    assert result.commission > 0
    assert result.net_cost > 0  # 买入支出


def test_sell_order_normal():
    sim = MarketSimulator()
    order = Order(stock_code="sh.603871", direction="sell", target_value=100000)
    md = make_market()
    result = sim.execute_order(order, md)
    assert result.status == OrderStatus.FILLED.value
    assert result.stamp_tax > 0  # 卖出有印花税
    assert result.actual_price < md.close  # slippage


def test_limit_up_rejected():
    sim = MarketSimulator()
    order = Order(stock_code="sh.603871", direction="buy", target_value=100000)
    md = make_market(is_limit_up=True)
    result = sim.execute_order(order, md)
    assert result.status == OrderStatus.REJECTED.value
    assert result.reject_reason == RejectReason.LIMIT_UP.value


def test_limit_down_rejected():
    sim = MarketSimulator()
    order = Order(stock_code="sh.603871", direction="sell", target_value=100000)
    md = make_market(is_limit_down=True)
    result = sim.execute_order(order, md)
    assert result.status == OrderStatus.REJECTED.value
    assert result.reject_reason == RejectReason.LIMIT_DOWN.value


def test_suspended_rejected():
    sim = MarketSimulator()
    order = Order(stock_code="sh.603871", direction="buy", target_value=100000)
    md = make_market(is_suspended=True)
    result = sim.execute_order(order, md)
    assert result.status == OrderStatus.REJECTED.value


def test_commission_minimum():
    sim = MarketSimulator()
    # target_value=5000: enough for ~400 shares at ~12.4/share,
    # but commission (~1.24 yuan) is below the 5 yuan floor
    order = Order(stock_code="sh.603871", direction="buy", target_value=5000)
    md = make_market()
    result = sim.execute_order(order, md)
    assert result.status == OrderStatus.FILLED.value
    assert result.commission == 5.0  # 最低佣金


def test_batch_execute():
    sim = MarketSimulator()
    orders = [
        Order(stock_code="sh.603871", direction="buy", target_value=100000),
        Order(stock_code="sz.300308", direction="sell", target_value=50000),
    ]
    md_map = {
        "sh.603871": make_market(),
        "sz.300308": make_market(stock_code="sz.300308", close=886.0, pre_close=870.0),
    }
    results = sim.execute_orders(orders, md_map)
    assert len(results) == 2
    assert results[0].status == OrderStatus.FILLED.value


def test_reset_daily_state():
    sim = MarketSimulator()
    sim._daily_bought["sh.603871"] = 50000
    sim._t1_locked["sh.603871"] = 1
    sim.reset_daily_state()
    assert "sh.603871" not in sim._t1_locked  # T+1 expired after reset
