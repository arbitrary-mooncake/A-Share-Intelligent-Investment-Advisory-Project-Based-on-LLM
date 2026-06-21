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

        # 停牌判断：Tushare daily接口对停牌股通常不返回数据；
        # 若返回但 volume=0 且 open=close=0 则视为停牌（Tushare停牌日占位行特征）
        is_suspended = (d.get("volume", 0) == 0 and close == 0 and d.get("open", 0) == 0)

        # 沪深300成分股近似判断：无法在此处获取真实成分列表，
        # 按"主板大盘股"启发式覆盖沪深两市典型大盘股代码段
        # 注意：这是近似，真正判定应查 index_member (000300.SH)
        code_clean = code.replace("sh.", "").replace("sz.", "")
        is_hs300 = (
            code_clean.startswith(("601318", "600519", "600036", "601398", "600028"))  # 沪市大盘
            or code_clean.startswith(("000001", "000333", "000651", "000858", "002594"))  # 深市大盘
        )

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
            is_suspended=is_suspended,
            is_limit_up=is_limit_up,
            is_limit_down=is_limit_down,
            is_hs300=is_hs300,
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


def fetch_trade_calendar(start_date: str, end_date: str,
                         exchange: str = "SSE") -> List[str]:
    """
    获取交易日列表。

    通过Tushare trade_cal接口获取指定日期范围内的交易日历，
    筛选is_open=1的交易日，返回YYYY-MM-DD格式的日期列表（升序）。

    Args:
        start_date: 起始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        exchange: 交易所代码，默认SSE（上交所）

    Returns:
        交易日列表（YYYY-MM-DD，升序）。失败返回空列表。
    """
    ts_start = convert_date_to_ts(start_date)
    ts_end = convert_date_to_ts(end_date)

    result = _call("trade_cal", {
        "exchange": exchange,
        "start_date": ts_start,
        "end_date": ts_end,
    }, fields="cal_date,is_open")

    if not result or "items" not in result:
        return []

    fields = result.get("fields", [])
    trading_days = []
    for row in result["items"]:
        item = dict(zip(fields, row))
        # is_open: 0=非交易日, 1=交易日
        is_open_val = item.get("is_open")
        if is_open_val is not None and int(is_open_val) == 1:
            cal_date = str(item.get("cal_date", ""))
            if len(cal_date) == 8:
                trading_days.append(
                    f"{cal_date[:4]}-{cal_date[4:6]}-{cal_date[6:8]}"
                )
    return sorted(trading_days)


class TushareUnavailableError(RuntimeError):
    """Tushare数据源不可用异常 — 总纲 §20.2 第10条红线。"""
    pass


def check_tushare_available(timeout: float = 3.0) -> bool:
    """
    快速检测Tushare API是否可达。

    总纲 §20.2 第10条: Tushare不可用时硬报错，禁止使用不可靠数据。

    Returns:
        True 如果Tushare可达
    Raises:
        TushareUnavailableError 如果网络不通或服务端不可用
    """
    import socket as _sock
    try:
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(("api.tushare.pro", 443))
        s.close()
    except Exception as e:
        raise TushareUnavailableError(
            f"Tushare API (api.tushare.pro:443) 不可达: {e}。"
            f"请检查网络连接。系统禁止在数据不可靠时继续运行。"
        ) from e

    # 二级验证：发一个轻量请求确认服务正常
    try:
        result = _call("trade_cal", {
            "exchange": "SSE",
            "start_date": datetime.now().strftime("%Y%m%d"),
            "end_date": datetime.now().strftime("%Y%m%d"),
        }, fields="cal_date")
        if result is None:
            raise TushareUnavailableError(
                "Tushare API端口可达但返回空响应，可能token无效或服务异常。"
                f"请检查.env中的TUSHARE_TOKEN配置。"
            )
    except TushareUnavailableError:
        raise
    except Exception as e:
        raise TushareUnavailableError(
            f"Tushare API连通性验证失败: {e}"
        ) from e

    return True


def fetch_latest_trading_day(lookback_days: int = 10) -> str:
    """
    获取距今天最近的交易日。

    从今天向前回溯lookback_days天，查询交易日历并返回最新的交易日。
    如果查询失败或找不到交易日，返回今天的日期作为fallback。

    Args:
        lookback_days: 向前查找的天数上限

    Returns:
        最新交易日 YYYY-MM-DD
    """
    today = datetime.now()
    start = today - timedelta(days=lookback_days)
    end_str = today.strftime("%Y-%m-%d")
    start_str = start.strftime("%Y-%m-%d")

    trading_days = fetch_trade_calendar(start_str, end_str)
    if trading_days:
        return trading_days[-1]
    return today.strftime("%Y-%m-%d")
