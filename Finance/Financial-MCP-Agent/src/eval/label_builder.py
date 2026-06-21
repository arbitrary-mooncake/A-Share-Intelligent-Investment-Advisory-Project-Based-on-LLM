"""
标签生成器 — 计算预测的实际市场收益。
纯代码实现，不调用LLM。通过Tushare日线数据计算持有期收益。
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple


def compute_holding_return(
    entry_price: float,
    exit_price: float,
    holding_days: int,
    benchmark_entry: float = 0.0,
    benchmark_exit: float = 0.0,
) -> Dict[str, Any]:
    """
    计算持有期收益。

    Returns:
        {
            "asset_return_pct": 0.05,      # 5% absolute return
            "benchmark_return_pct": 0.03,   # 3% benchmark return
            "excess_return_pct": 0.02,      # 2% excess return
            "annualized_return_pct": 0.25,  # annualized
        }
    """
    if entry_price <= 0 or exit_price <= 0:
        return {
            "asset_return_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "excess_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "valid": False,
        }

    asset_return = (exit_price - entry_price) / entry_price
    excess = asset_return

    if benchmark_entry > 0 and benchmark_exit > 0:
        benchmark_return = (benchmark_exit - benchmark_entry) / benchmark_entry
        excess = asset_return - benchmark_return
    else:
        benchmark_return = 0.0

    # Annualized (252 trading days)
    if holding_days > 0:
        annualized = (1 + asset_return) ** (252 / holding_days) - 1
    else:
        annualized = asset_return

    return {
        "asset_return_pct": round(asset_return * 100, 4),
        "benchmark_return_pct": round(benchmark_return * 100, 4) if benchmark_entry > 0 else 0.0,
        "excess_return_pct": round(excess * 100, 4),
        "annualized_return_pct": round(annualized * 100, 4),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "holding_days": holding_days,
        "valid": True,
    }


def compute_drawdown(prices: List[float]) -> Tuple[float, float]:
    """计算最大回撤和当前回撤"""
    if not prices or len(prices) < 2:
        return 0.0, 0.0

    peak = prices[0]
    mdd = 0.0
    for p in prices:
        peak = max(peak, p)
        dd = (peak - p) / peak if peak > 0 else 0
        mdd = max(mdd, dd)

    current_dd = (peak - prices[-1]) / peak if peak > 0 else 0
    return mdd, current_dd


def compute_volatility(daily_returns: List[float], annualize: bool = True) -> float:
    """计算波动率"""
    n = len(daily_returns)
    if n < 2:
        return 0.0
    mean = sum(daily_returns) / n
    variance = sum((r - mean) ** 2 for r in daily_returns) / (n - 1)
    if annualize:
        return (variance * 252) ** 0.5
    return variance ** 0.5


def build_realized_label(
    snapshot: Dict[str, Any],
    price_data: Dict[str, Any],
    benchmark_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    从snapshot和市场价格数据构建RealizedLabel。

    Args:
        snapshot: PredictionSnapshot dict (含as_of_date, symbol, term)
        price_data: {entry_price, exit_price, prices[] (持有期所有日收盘价),
                      max_price, min_price, start_date, end_date}
        benchmark_data: 同理，基准指数数据（CSI 300等）

    Returns:
        RealizedLabel dict ready for storage
    """
    entry_price = price_data.get("entry_price", 0)
    exit_price = price_data.get("exit_price", 0)
    prices = price_data.get("prices", [])
    holding_days = len(prices) - 1 if prices else 0

    # 基本收益计算
    return_info = compute_holding_return(
        entry_price, exit_price, holding_days,
        benchmark_data.get("entry_price", 0) if benchmark_data else 0,
        benchmark_data.get("exit_price", 0) if benchmark_data else 0,
    )

    # 最大回撤
    mdd, current_dd = compute_drawdown(prices) if prices else (0.0, 0.0)

    # 波动率
    if prices and len(prices) >= 2:
        daily_rets = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        vol = compute_volatility(daily_rets)
    else:
        vol = 0.0

    return {
        "snapshot_id": snapshot.get("snapshot_id", ""),
        "line_id": snapshot.get("line_id", ""),
        "term": snapshot.get("term", "short"),
        "horizon_days": holding_days,
        "outcome_date": price_data.get("end_date", ""),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "asset_return_pct": return_info["asset_return_pct"],
        "benchmark_return_pct": return_info["benchmark_return_pct"],
        "excess_return_pct": return_info["excess_return_pct"],
        "max_drawdown_pct": round(mdd * 100, 4),
        "volatility_pct": round(vol * 100, 4) if vol else 0.0,
        "is_valid": return_info["valid"],
        "settlement_notes": "",
        "meta_json": "",
    }
