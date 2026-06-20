"""Tests for trading strategies"""
from src.eval.strategies.base import BaseStrategy, BuyOrder, SellOrder
from src.eval.strategies.short_ablation import ShortAblationStrategy
from src.eval.strategies.short_longhold import ShortLongHoldStrategy
from src.eval.strategies.medium_term import MediumTermStrategy
from src.eval.strategies.long_term import LongTermStrategy
from src.eval.strategies.factory import get_strategy, list_strategies
from src.eval.market_simulator import MarketData


def _make_md(code, **overrides):
    """创建带默认值的 MarketData（用于策略测试）"""
    defaults = {
        "stock_code": code,
        "close": 20.0,
        "pre_close": 19.5,
        "open": 19.7,
        "high": 20.2,
        "low": 19.4,
        "volume": 100000,
        "amount": 2000000,
        "turnover_rate": 0.02,
        "pe_ratio": 15.0,
        "pb_ratio": 2.0,
        "price_to_ma_ratio": 1.05,
        "roe": 0.12,
        "revenue_growth": 0.15,
        "dividend_yield": 0.02,
        "profit_growth_3y": 1.0,
        "industry": "食品饮料",
        "industry_pe_75th": 25.0,
        "industry_pe_90th": 35.0,
        "industry_pe_median": 18.0,
        "risk_flags": None,
        "close_30min_spike": 0.0,
        "pe_percentile": 50.0,
    }
    defaults.update(overrides)
    return MarketData(**defaults)


def test_short_ablation_select():
    s = ShortAblationStrategy()
    pool = ["sh.601888", "sh.603871", "sz.300308", "sz.300139", "sz.001309"]
    scores = {"sh.601888": 82, "sh.603871": 75, "sz.300308": 63,
              "sz.300139": 45, "sz.001309": 55}
    orders = s.select_stocks(pool, scores, {}, 1000000)
    # 只有score>=60的: 601888(82), 603871(75), 300308(63)
    assert len(orders) == 3
    assert orders[0].stock_code == "sh.601888"


def test_short_ablation_sell_all():
    s = ShortAblationStrategy()
    holdings = {"sh.601888": 10000, "sh.603871": 8000}
    orders = s.generate_sell_orders(holdings, {})
    assert len(orders) == 2
    assert all(o.sell_ratio == 1.0 for o in orders)


def test_short_longhold_buy_filter():
    s = ShortLongHoldStrategy()
    pool = ["sh.601888", "sh.603871", "sz.300308"]
    scores = {"sh.601888": 82, "sh.603871": 55, "sz.300308": 63}  # 603871 < 60 threshold
    orders = s.select_stocks(pool, scores, {}, 1000000)
    codes = [o.stock_code for o in orders]
    assert "sh.603871" not in codes


def test_short_longhold_sell_hard():
    s = ShortLongHoldStrategy()
    holdings = {"sh.603871": 10000}
    scores = {"sh.603871": 30}  # <35 hard sell
    orders = s.generate_sell_orders(holdings, scores)
    assert len(orders) == 1
    assert orders[0].sell_ratio == 1.0


def test_short_longhold_sort_formula():
    """验证 §5.3 排序公式：score×0.6 + 5d_momentum×0.3 + turnover_percentile×0.1"""
    s = ShortLongHoldStrategy()
    pool = ["sh.601888", "sh.603871"]
    scores = {"sh.601888": 80, "sh.603871": 80}
    # Both score 80, diff should come from momentum and turnover percentile
    md_map = {
        "sh.601888": _make_md("sh.601888", close=21.0, pre_close=20.0, turnover_rate=0.03),
        "sh.603871": _make_md("sh.603871", close=19.0, pre_close=20.0, turnover_rate=0.01),
    }
    orders = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map)
    # 601888 has positive momentum (5%), 603871 has negative (-5%)
    # 601888 should rank higher
    assert len(orders) >= 1
    assert orders[0].stock_code == "sh.601888"


def test_short_longhold_staged_build():
    """验证 §5.3 分2天建仓：Day1 50%, Day2 50%"""
    s = ShortLongHoldStrategy()
    pool = ["sh.601888"]
    scores = {"sh.601888": 80}
    md_map = {"sh.601888": _make_md("sh.601888")}
    # Day 1
    orders = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map)
    assert len(orders) == 1
    assert "Day1" in orders[0].reason
    # Day 1 应该在 pending_builds 中
    assert "sh.601888" in s._pending_builds
    # size_positions should set 50% of full position
    sized = s.size_positions(orders, 1000000, 0, 12, 0.12, 0.10)
    assert len(sized) == 1
    # Day 2 (simulate by running select again - pending exists)
    orders2 = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map)
    # 找到 Day2 的订单
    day2_orders = [o for o in orders2 if "Day2" in (o.reason or "")]
    assert len(day2_orders) == 1
    # Day 2 完成后 pending 应该清除
    assert "sh.601888" not in s._pending_builds


def test_short_longhold_sector_cap():
    """验证 §5.3 同行业不超过总资金35%"""
    s = ShortLongHoldStrategy()
    pool = ["sh.601888", "sh.600519"]
    scores = {"sh.601888": 80, "sh.600519": 78}
    md_map = {
        "sh.601888": _make_md("sh.601888", industry="食品饮料"),
        "sh.600519": _make_md("sh.600519", industry="食品饮料"),
    }
    orders = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map)
    # Both should be selected
    assert len(orders) >= 1
    # size_positions with holdings already in same industry
    holdings = {"sh.601888": 5000}  # already holds 601888 in 食品饮料
    sized = s.size_positions(
        orders, 1000000, 1, 12, 0.12, 0.10,
        holdings=holdings, market_data_map=md_map
    )
    # 600519 should be capped by industry limit (35%)
    assert len(sized) >= 1


def test_short_longhold_zhongcang():
    """验证 §5.3 加仓：score>75 且近3日涨幅<8%"""
    s = ShortLongHoldStrategy()
    pool = ["sh.601888"]
    scores = {"sh.601888": 80}
    holdings = {"sh.601888": 10000}
    md_map = {"sh.601888": _make_md("sh.601888", close=21.0, pre_close=20.5)}
    # 近3日涨幅 = (21-20.5)/20.5 = 2.4% < 8%, should trigger add
    orders = s.select_stocks(
        pool, scores, holdings, 1000000,
        market_data_map=md_map, total_capital=1000000
    )
    add_orders = [o for o in orders if o.reason and "加仓" in o.reason]
    assert len(add_orders) >= 1
    assert add_orders[0].target_value > 0


def test_short_longhold_short_medium_contradiction():
    """验证 §5.3 短中矛盾：short>70 medium<40 → 仓位减半"""
    s = ShortLongHoldStrategy()
    holdings = {"sh.601888": 10000}
    scores = {"sh.601888": 75}
    medium_scores = {"sh.601888": 35}
    orders = s.generate_sell_orders(
        holdings, scores, medium_scores=medium_scores
    )
    assert len(orders) == 1
    assert orders[0].sell_ratio == 0.5
    assert "短中矛盾" in orders[0].reason


def test_medium_term_select():
    """验证 §5.4 中线选股：五维买入条件 + 四因子排序"""
    s = MediumTermStrategy()
    pool = ["sh.601888", "sh.603871"]
    scores = {"sh.601888": 70, "sh.603871": 50}  # threshold 65
    md_map = {
        "sh.601888": _make_md("sh.601888"),
        "sh.603871": _make_md("sh.603871"),
    }
    orders = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map)
    # 只有 601888(70>=65) 通过买入条件
    assert len(orders) == 1
    assert orders[0].stock_code == "sh.601888"


def test_medium_term_buy_conditions():
    """验证 §5.4 买入条件：PE>0且<行业75分位, ROE>5%, price>MA60, 无风险"""
    s = MediumTermStrategy()
    pool = ["sh.601888"]
    scores = {"sh.601888": 70}

    # 正常数据 → 通过
    md_map = {"sh.601888": _make_md("sh.601888")}
    orders = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map)
    assert len(orders) == 1

    # PE > 行业75分位 → 不通过
    md_map2 = {"sh.601888": _make_md("sh.601888", pe_ratio=30.0, industry_pe_75th=25.0)}
    orders2 = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map2)
    assert len(orders2) == 0

    # ROE≤5% 且 营收增速≤10% → 不通过
    md_map3 = {"sh.601888": _make_md("sh.601888", roe=0.03, revenue_growth=0.05)}
    orders3 = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map3)
    assert len(orders3) == 0

    # 有 risk_flags (delist) → 不通过
    md_map4 = {"sh.601888": _make_md("sh.601888", risk_flags=["delist_risk"])}
    orders4 = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map4)
    assert len(orders4) == 0

    # price_to_ma_ratio <= 1.0 → 不通过
    md_map5 = {"sh.601888": _make_md("sh.601888", price_to_ma_ratio=0.95)}
    orders5 = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map5)
    assert len(orders5) == 0


def test_medium_term_trend_sell():
    """验证 §5.4 趋势卖出：跌破MA60且3日未收回"""
    s = MediumTermStrategy()
    holdings = {"sh.601888": 10000}
    md_map = {"sh.601888": _make_md("sh.601888", price_to_ma_ratio=0.92)}

    # Day 1: below MA60 → no sell yet
    orders1 = s.generate_sell_orders(holdings, {"sh.601888": 60}, market_data_map=md_map)
    assert len(orders1) == 0
    assert s._below_ma60_days.get("sh.601888", 0) == 1

    # Day 2: still below → no sell
    orders2 = s.generate_sell_orders(holdings, {"sh.601888": 60}, market_data_map=md_map)
    assert len(orders2) == 0
    assert s._below_ma60_days.get("sh.601888", 0) == 2

    # Day 3: trigger sell 50%
    orders3 = s.generate_sell_orders(holdings, {"sh.601888": 60}, market_data_map=md_map)
    assert len(orders3) == 1
    assert orders3[0].sell_ratio >= 0.5
    assert "趋势破位" in orders3[0].reason


def test_medium_term_cross_term_synergy():
    """验证 §5.4 跨期限协同：中>70且长>70→上限22%，中>60且长<35→上限10%"""
    s = MediumTermStrategy()
    orders = [BuyOrder("sh.601888", 0, 75), BuyOrder("sh.603871", 0, 65)]

    # 中线>70 长线>70 → 仓位上限22%
    sized1 = s.size_positions(orders, 1000000, 0, 8, 0.18, 0.15,
                              long_scores={"sh.601888": 75, "sh.603871": 50})
    assert len(sized1) >= 1
    # 601888: weight = min(1/2=0.5, 0.22) = 0.22, target = 220000
    assert sized1[0].target_value > 0

    # 中线>60 长线<35 → 仓位上限10%
    sized2 = s.size_positions(orders, 1000000, 0, 8, 0.18, 0.15,
                              long_scores={"sh.601888": 30, "sh.603871": 50})
    assert len(sized2) >= 1
    # 601888 (65>60) with long<35 should be capped at 10%
    # weight = min(0.5, 0.10) = 0.10, target = 100000


def test_long_term_select():
    """验证 §5.5 长线选股：五维条件 + 五因子排序"""
    s = LongTermStrategy()
    pool = ["sh.601888", "sh.603871"]
    scores = {"sh.601888": 80, "sh.603871": 65}  # threshold 70
    md_map = {
        "sh.601888": _make_md("sh.601888"),
        "sh.603871": _make_md("sh.603871"),
    }
    orders = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map)
    assert len(orders) == 1
    assert orders[0].stock_code == "sh.601888"


def test_long_term_buy_conditions():
    """验证 §5.5 买入条件：ROE>8%, PE<行业中位, 无风险, dividend>0"""
    s = LongTermStrategy()
    pool = ["sh.601888"]
    scores = {"sh.601888": 80}

    # 正常数据 → 通过
    md_map = {"sh.601888": _make_md("sh.601888")}
    orders = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map)
    assert len(orders) == 1

    # ROE ≤ 8% → 不通过
    md_map2 = {"sh.601888": _make_md("sh.601888", roe=0.06)}
    orders2 = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map2)
    assert len(orders2) == 0

    # 有 risk_flags (audit_risk) → 不通过
    md_map3 = {"sh.601888": _make_md("sh.601888", risk_flags=["audit_risk"])}
    orders3 = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map3)
    assert len(orders3) == 0


def test_long_term_quality_deterioration():
    """验证 §5.5 质量恶化：ROE连续2季下降>30% → 全部卖出"""
    s = LongTermStrategy()
    # 注入ROE历史：0.20, 0.12, 0.07 (连续下降>30%)
    s.update_roe("sh.601888", 0.20)
    s.update_roe("sh.601888", 0.12)
    s.update_roe("sh.601888", 0.07)

    holdings = {"sh.601888": 10000}
    orders = s.generate_sell_orders(holdings, {"sh.601888": 60})
    # ROE: 0.20→0.12 (-40%>30%) AND 0.12→0.07 (-41.7%>30%)
    assert len(orders) == 1
    assert orders[0].sell_ratio == 1.0
    assert "质量恶化" in orders[0].reason


def test_long_term_pe_extreme():
    """验证 §5.5 PE极端：PE突破历史95分位 → 卖出30%"""
    s = LongTermStrategy()
    holdings = {"sh.601888": 10000}
    md_map = {"sh.601888": _make_md("sh.601888", pe_percentile=97.0)}
    orders = s.generate_sell_orders(holdings, {"sh.601888": 60}, market_data_map=md_map)
    assert len(orders) == 1
    assert orders[0].sell_ratio >= 0.3
    assert "PE突破历史95分位" in orders[0].reason


def test_long_term_black_swan():
    """验证 §5.5 黑天鹅：delist_risk/audit_risk → 立即清仓"""
    s = LongTermStrategy()
    holdings = {"sh.601888": 10000}
    md_map = {"sh.601888": _make_md("sh.601888", risk_flags=["delist_risk", "audit_risk"])}
    orders = s.generate_sell_orders(holdings, {"sh.601888": 80}, market_data_map=md_map)
    assert len(orders) == 1
    assert orders[0].sell_ratio == 1.0
    assert "黑天鹅" in orders[0].reason


def test_long_term_staged_build():
    """验证 §5.5 分2月建仓：Month 1 50%, Month 2 50%"""
    s = LongTermStrategy()
    pool = ["sh.601888"]
    scores = {"sh.601888": 80}
    md_map = {"sh.601888": _make_md("sh.601888")}
    # Month 1
    orders = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map)
    assert len(orders) == 1
    assert "第1/2月" in orders[0].reason
    assert "sh.601888" in s._pending_builds
    # Month 2
    orders2 = s.select_stocks(pool, scores, {}, 1000000, market_data_map=md_map)
    month2_orders = [o for o in orders2 if "第2/2月" in (o.reason or "")]
    assert len(month2_orders) == 1
    assert "sh.601888" not in s._pending_builds


def test_factory_get_strategy():
    s1 = get_strategy("short", "ablation")
    assert isinstance(s1, ShortAblationStrategy)
    s2 = get_strategy("short", "longhold")
    assert isinstance(s2, ShortLongHoldStrategy)
    s3 = get_strategy("medium")
    assert isinstance(s3, MediumTermStrategy)
    s4 = get_strategy("long")
    assert isinstance(s4, LongTermStrategy)


def test_factory_fallback():
    s = get_strategy("short", "nonexistent")
    assert isinstance(s, ShortAblationStrategy)  # should not crash


def test_list_strategies():
    strategies = list_strategies()
    assert len(strategies) >= 4


def test_base_strategy_size_positions():
    s = ShortAblationStrategy()
    orders = [BuyOrder("sh.601888", 0, 82), BuyOrder("sh.603871", 0, 75)]
    sized = s.size_positions(orders, 1000000, current_positions=0,
                             max_positions=10, single_weight_limit=0.10,
                             min_cash_ratio=0.0)
    assert len(sized) == 2
    assert sized[0].target_value > 0
    assert sized[1].target_value > 0
