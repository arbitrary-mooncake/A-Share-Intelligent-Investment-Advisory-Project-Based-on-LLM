"""
Yahoo Finance MCP Server — International commodity, currency, and bond data.
Provides COMEX gold, DXY, US Treasury yields, SPDR GLD, and more.
"""
import sys
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
import threading
import time
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastMCP()

# 限制对 Yahoo Finance 的并发调用，防止速率限制
_yf_lock = threading.Lock()
_yf_min_interval = 1.5  # 两次调用之间的最小间隔（秒）
_last_yf_call = 0.0

COMMODITY_SYMBOL_MAP = {
    "gold": "GC=F",
    "silver": "SI=F",
    "oil": "CL=F",
    "crude": "CL=F",
    "wti": "CL=F",
    "brent": "BZ=F",
    "copper": "HG=F",
    "natural_gas": "NG=F",
    "gold_future": "GC=F",
    "silver_future": "SI=F",
}

GOLD_ETF_MAP = {
    "gld": "GLD",
    "iau": "IAU",
    "gdx": "GDX",
}

TREASURY_MAP = {
    "3m": "^IRX",
    "5y": "^FVX",
    "10y": "^TNX",
    "30y": "^TYX",
}


def _resolve_symbol(symbol: str) -> str:
    key = symbol.lower().strip()
    if key in COMMODITY_SYMBOL_MAP:
        return COMMODITY_SYMBOL_MAP[key]
    return symbol


def _get_yf_history(symbol: str, period: str = "6mo") -> str:
    """获取 Yahoo Finance 历史数据，含速率限制保护和重试"""
    import yfinance as yf
    global _last_yf_call

    with _yf_lock:
        elapsed = time.time() - _last_yf_call
        if elapsed < _yf_min_interval:
            time.sleep(_yf_min_interval - elapsed)

        for attempt in range(3):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period=period)
                _last_yf_call = time.time()

                if "Too Many Requests" in str(info.get("regularMarketPrice", "")):
                    if attempt < 2:
                        wait = (attempt + 1) * 3
                        logger.warning(f"YH rate limited for {symbol}, retry in {wait}s")
                        time.sleep(wait)
                        continue
                    return f"获取 {symbol} 数据失败: Yahoo Finance 请求过于频繁，请稍后重试"

                name = info.get("shortName") or info.get("longName") or symbol
                current_price = info.get("regularMarketPrice") or info.get("previousClose", "N/A")

                lines = [
                    f"## {name} ({symbol})",
                    f"**最新价格**: {current_price}",
                    f"**数据周期**: {period}",
                    "",
                ]

                if hist.empty:
                    lines.append("无历史数据")
                    return "\n".join(lines)

                recent = hist.tail(20)
                lines.append("| 日期 | 开盘 | 最高 | 最低 | 收盘 | 成交量 |")
                lines.append("| --- | --- | --- | --- | --- | --- |")
                for idx, row in recent.iterrows():
                    date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                    lines.append(
                        f"| {date_str} "
                        f"| {row.get('Open', 'N/A')} "
                        f"| {row.get('High', 'N/A')} "
                        f"| {row.get('Low', 'N/A')} "
                        f"| {row.get('Close', 'N/A')} "
                        f"| {int(row.get('Volume', 0)):,} |"
                    )
                return "\n".join(lines)

            except Exception as e:
                err_msg = str(e)
                if "rate limit" in err_msg.lower() or "too many requests" in err_msg.lower():
                    if attempt < 2:
                        wait = (attempt + 1) * 3
                        logger.warning(f"YH rate limited for {symbol}, retry in {wait}s")
                        time.sleep(wait)
                        continue
                logger.error(f"Yahoo Finance fetch failed for {symbol}: {e}")
                return f"获取 {symbol} 数据失败: {e}"


@app.tool()
def get_commodity_price(symbol: str = "GC=F") -> str:
    """
    获取国际大宗商品期货价格（COMEX/NYMEX等）。
    通过 Yahoo Finance 获取 OHLCV 历史数据。

    常用代码:
    - GC=F: COMEX黄金期货
    - SI=F: COMEX白银期货
    - CL=F: WTI原油期货
    - BZ=F: 布伦特原油期货
    - HG=F: COMEX铜期货
    - NG=F: 天然气期货

    参数:
        symbol: 商品代码，默认 GC=F (黄金期货)

    返回: Markdown 表格，包含最近20个交易日的 OHLCV 数据
    """
    symbol = _resolve_symbol(symbol)
    logger.info(f"Fetching commodity: {symbol}")
    return _get_yf_history(symbol, period="6mo")


@app.tool()
def get_us_treasury_yield(tenor: str = "10y") -> str:
    """
    获取美国国债收益率（CBOE利率指数）。
    国债收益率是无风险利率的基准，对黄金和全球资产定价有直接影响。
    - 收益率上升 → 美元走强 → 利空黄金
    - 收益率下降 → 利好黄金

    参数:
        tenor: 期限，'3m'=3个月, '5y'=5年, '10y'=10年, '30y'=30年

    返回: Markdown 表格
    """
    symbol = TREASURY_MAP.get(tenor.lower(), "^TNX")
    logger.info(f"Fetching US Treasury yield: {symbol} (tenor={tenor})")
    return _get_yf_history(symbol, period="6mo")


@app.tool()
def get_dollar_index() -> str:
    """
    获取美元指数(DXY)走势。
    DXY衡量美元对一篮子主要货币的强弱，与黄金价格呈负相关。
    - 美元走强 → 利空黄金
    - 美元走弱 → 利好黄金

    返回: Markdown 表格
    """
    logger.info("Fetching Dollar Index (DXY)")
    return _get_yf_history("DX-Y.NYB", period="6mo")


@app.tool()
def get_gold_etf(symbol: str = "GLD") -> str:
    """
    获取黄金ETF的持仓和价格数据。
    监控全球最大黄金ETF的持仓变化，是判断机构黄金配置的重要指标。

    参数:
        symbol: ETF代码，默认 GLD (SPDR Gold Trust)，也可用 IAU

    返回: Markdown 表格
    """
    resolved = GOLD_ETF_MAP.get(symbol.lower().strip(), symbol)
    logger.info(f"Fetching gold ETF: {resolved} (input={symbol})")
    return _get_yf_history(resolved, period="6mo")


if __name__ == "__main__":
    logger.info("Starting Yahoo Finance MCP Server via stdio...")
    app.run(transport="stdio")
