"""
评测数据获取器 — 通过Tushare HTTP API获取真实市场数据。
接入现有 tushare_client.py 的轻量HTTP封装，不走MCP（简化批量查询）。
"""
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# 确保可以导入现有 tushare_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from tushare_client import _call


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def fetch_daily_prices(ts_code: str, start_date: str, end_date: str = "") -> List[Dict[str, Any]]:
    """
    获取日线行情数据。

    Args:
        ts_code: Tushare代码格式 (如 603871.SH, 000001.SZ)
        start_date: YYYYMMDD
        end_date: YYYYMMDD (默认今天)

    Returns:
        [{trade_date, open, high, low, close, pre_close, vol, amount}, ...]
    """
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")

    result = _call("daily", {
        "ts_code": ts_code,
        "start_date": start_date,
        "end_date": end_date,
    }, fields="trade_date,open,high,low,close,pre_close,vol,amount")

    if not result or "items" not in result:
        return []

    fields = result.get("fields", [])
    items = []
    for row in result["items"]:
        item = dict(zip(fields, row))
        items.append({
            "trade_date": item.get("trade_date", ""),
            "open": _safe_float(item.get("open")),
            "high": _safe_float(item.get("high")),
            "low": _safe_float(item.get("low")),
            "close": _safe_float(item.get("close")),
            "pre_close": _safe_float(item.get("pre_close")),
            "volume": _safe_float(item.get("vol")),
            "amount": _safe_float(item.get("amount")),
        })
    return items


def fetch_daily_basic(ts_code: str, trade_date: str) -> Dict[str, Any]:
    """获取单日个股指标（PE/PB/换手率/总市值）"""
    result = _call("daily_basic", {
        "ts_code": ts_code,
        "trade_date": trade_date.replace("-", ""),
    }, fields="ts_code,trade_date,pe,pb,turnover_rate,total_mv")

    if not result or "items" not in result or not result["items"]:
        return {}

    fields = result["fields"]
    item = dict(zip(fields, result["items"][0]))
    return {
        "pe": _safe_float(item.get("pe")),
        "pb": _safe_float(item.get("pb")),
        "turnover_rate": _safe_float(item.get("turnover_rate")),
        "total_mv": _safe_float(item.get("total_mv")),
    }


def fetch_index_daily(index_code: str, start_date: str, end_date: str = "") -> List[Dict[str, Any]]:
    """获取指数日线（CSI 300 = 000300.SH）"""
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")

    result = _call("index_daily", {
        "ts_code": index_code,
        "start_date": start_date,
        "end_date": end_date,
    }, fields="trade_date,close")

    if not result or "items" not in result:
        return []

    fields = result.get("fields", [])
    return [
        {"trade_date": dict(zip(fields, row)).get("trade_date", ""),
         "close": _safe_float(dict(zip(fields, row)).get("close"))}
        for row in result["items"]
    ]


def fetch_suspend_info(ts_code: str, trade_date: str) -> bool:
    """检查某只股票在某日是否停牌"""
    result = _call("suspend_d", {
        "ts_code": ts_code,
        "suspend_date": trade_date.replace("-", ""),
    }, fields="ts_code")

    return result is not None and len(result.get("items", [])) > 0


def fetch_adj_factor(ts_code: str, trade_date: str = "") -> Dict[str, Any]:
    """获取复权因子"""
    params = {"ts_code": ts_code}
    if trade_date:
        params["trade_date"] = trade_date.replace("-", "")
    result = _call("adj_factor", params, fields="ts_code,trade_date,adj_factor")
    if not result or "items" not in result or not result["items"]:
        return {"adj_factor": 1.0}
    fields = result["fields"]
    item = dict(zip(fields, result["items"][-1]))  # 最新
    return {"adj_factor": _safe_float(item.get("adj_factor"), 1.0)}


def convert_code_to_ts(stock_code: str) -> str:
    """将项目内部代码格式转换为Tushare格式
    sh.603871 → 603871.SH
    sz.300308 → 300308.SZ
    """
    code = stock_code.strip()
    if code.startswith("sh."):
        return code[3:] + ".SH"
    elif code.startswith("sz."):
        return code[3:] + ".SZ"
    return code


def convert_date_to_ts(date_str: str) -> str:
    """YYYY-MM-DD → YYYYMMDD"""
    return date_str.replace("-", "")


def fetch_realtime_prices(stock_codes: List[str], trade_date: str = "") -> Dict[str, float]:
    """
    批量获取股票收盘价（用于日常结算）。

    Args:
        stock_codes: [sh.603871, sz.300308, ...]
        trade_date: YYYY-MM-DD（默认最新交易日）

    Returns:
        {stock_code: close_price}
    """
    if not trade_date:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    ts_date = convert_date_to_ts(trade_date)

    prices = {}
    for code in stock_codes:
        ts_code = convert_code_to_ts(code)
        daily_data = fetch_daily_prices(ts_code, ts_date, ts_date)
        if daily_data:
            prices[code] = daily_data[0]["close"]
    return prices


def build_market_data_map(stock_codes: List[str], trade_date: str = "") -> Dict[str, Any]:
    """
    构建评测系统所需的 MarketData 映射（用于日常调仓）。

    Returns:
        {stock_code: MarketData}
    """
    from src.eval.market_simulator import MarketData

    if not trade_date:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    ts_date = convert_date_to_ts(trade_date)

    market_map = {}
    for code in stock_codes:
        ts_code = convert_code_to_ts(code)
        daily_data = fetch_daily_prices(ts_code, ts_date, ts_date)

        if not daily_data:
            continue

        d = daily_data[0]
        close = d["close"]
        pre_close = d["pre_close"]

        # 判断涨跌停
        is_limit_up = False
        is_limit_down = False
        if pre_close > 0:
            change_pct = (close - pre_close) / pre_close
            code_clean = code.replace("sh.", "").replace("sz.", "")
            if code_clean.startswith(("30", "68")):
                limit = 0.20
            elif code_clean.startswith("8"):
                limit = 0.30
            else:
                limit = 0.10
            is_limit_up = change_pct >= limit * 0.99
            is_limit_down = change_pct <= -limit * 0.99

        # 获取额外指标
        basic = fetch_daily_basic(ts_code, trade_date)

        market_map[code] = MarketData(
            stock_code=code,
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=close,
            pre_close=pre_close,
            volume=d["volume"],
            amount=d["amount"],
            turnover_rate=basic.get("turnover_rate", 0),
            is_suspended=False,  # Tushare daily返回 = 未停牌
            is_limit_up=is_limit_up,
            is_limit_down=is_limit_down,
            is_hs300=code.startswith("sh.60"),  # 近似判断
            market_cap=basic.get("total_mv", 0),
            pe_ratio=basic.get("pe", 0),
            pb_ratio=basic.get("pb", 0),
            price_to_ma_ratio=1.0,  # Will be populated by technical analysis when available
            risk_flags=None,  # Will be populated by risk analysis when available
        )

    return market_map


def fetch_benchmark_prices(benchmark: str, start_date: str, end_date: str) -> List[float]:
    """
    获取基准指数在持有期的价格序列。

    Args:
        benchmark: 000300.SH (CSI 300)
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD

    Returns:
        [close_price, ...]
    """
    ts_start = convert_date_to_ts(start_date)
    ts_end = convert_date_to_ts(end_date)
    data = fetch_index_daily(benchmark, ts_start, ts_end)
    return [d["close"] for d in data]
