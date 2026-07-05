"""
批量打分编排器 — Excel 解析 + 并行轻量数据获取 + 批量LLM打分

Phase 1: Excel 解析 + 轻量并行数据获取（Light Tushare: 3 calls only）
Phase 2: 批量LLM打分（5 stocks/call, qwen3.6-flash, thinking disabled）
"""

import asyncio
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Any

# ──────────────────────────────────────────────
# 路径初始化（必须最早，确保 src 可导入）
# ──────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(override=True)

import openpyxl

from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)

# ──────────────────────────────────────────────
# 全局状态
# ──────────────────────────────────────────────
_batch_jobs: Dict[str, Dict[str, Any]] = {}
_batch_lock = threading.Lock()
# 大线程池（50 workers）防止僵尸连接占满导致新任务无线程可用
_batch_pool = ThreadPoolExecutor(max_workers=50, thread_name_prefix="batch")

# 全局 socket 超时兜底（只设一次，不在线程间反复 set/restore 避免竞态）
import socket as _socket
_socket.setdefaulttimeout(25)

# ──────────────────────────────────────────────
# Tushare 批量预取 (主线程顺序执行，无需锁)
# ──────────────────────────────────────────────

_BATCH_PREFETCH_ENABLED = True


def _prefetch_tushare_batch(
    stocks: List[Dict[str, str]],
    on_progress: callable = None,
) -> Dict[str, Dict]:
    """批量预取 Tushare 数据 (主线程顺序执行)。

    三步: stock_basic 全量 → daily_basic 逐只 → fina_indicator 批量(200只/批)
    返回: {ts_code: {industry, pe, pb, ps, roe, gross_margin, ...}}
    缺失字段留空字符串。

    on_progress(completed, total) 在 Step B 中每 100 只回调一次，
    确保前端进度条在数分钟的预取阶段也能看到推进。
    """
    from src.utils.tushare_client import (
        get_all_stocks, get_daily_basic, get_fina_indicator_batch,
    )

    if not stocks:
        return {}

    ts_codes = [_to_tushare_code(s["code"]) for s in stocks]
    cache: Dict[str, Dict] = {tc: {} for tc in ts_codes}
    total = len(stocks)

    # ── Step A: stock_basic 全量 (1次调用) ──
    logger.info(f"  预取 Step A: stock_basic 全量 (1 次调用)")
    try:
        all_info = get_all_stocks()
        if all_info:
            info_map = {d["ts_code"]: d for d in all_info if d.get("ts_code")}
            for tc in ts_codes:
                if tc in info_map:
                    info = info_map[tc]
                    cache[tc]["industry"] = info.get("industry", "") or ""
                else:
                    cache[tc]["industry"] = ""
            hit = sum(1 for tc in ts_codes if tc in info_map)
            logger.info(f"  预取 Step A 完成: 行业命中 {hit}/{total}")
        else:
            logger.warning("  预取 Step A: stock_basic 返回空")
    except Exception as e:
        logger.warning(f"  预取 Step A 失败: {e}")

    # ── Step B: daily_basic 批量 (按交易日查全市场, 替代逐只查询) ──
    logger.info(f"  预取 Step B: daily_basic 批量 (按交易日查全市场)")
    from src.utils.tushare_client import _call
    # 获取最近的交易日
    today = datetime.now().strftime("%Y%m%d")
    trade_cal = _call("trade_cal", {
        "exchange": "SSE",
        "start_date": (datetime.now() - timedelta(days=10)).strftime("%Y%m%d"),
        "end_date": today,
    }, fields="cal_date,is_open")
    latest_trade_date = today
    if trade_cal and "items" in trade_cal:
        fc = trade_cal["fields"]
        for row in reversed(trade_cal["items"]):
            item = dict(zip(fc, row))
            if item.get("is_open") == 1:
                latest_trade_date = item["cal_date"]
    logger.info(f"  最新交易日: {latest_trade_date}")

    # 批量查全市场 daily_basic (1次调用替代 {total} 次)
    try:
        bulk = _call("daily_basic", {
            "trade_date": latest_trade_date,
        }, "ts_code,pe_ttm,pb,ps_ttm")
        if bulk and "items" in bulk:
            fc = bulk["fields"]
            for row in bulk["items"]:
                item = dict(zip(fc, row))
                tc = item.get("ts_code", "")
                if tc in cache:
                    cache[tc]["pe"] = str(item.get("pe_ttm", "") or "")
                    cache[tc]["pb"] = str(item.get("pb", "") or "")
                    cache[tc]["ps"] = str(item.get("ps_ttm", "") or "")
    except Exception as e:
        logger.warning(f"  预取 Step B 批量查询失败: {e}")
    pe_hit = sum(1 for tc in ts_codes if cache[tc].get("pe"))
    logger.info(f"  预取 Step B 完成: PE 命中 {pe_hit}/{total} (批量查询模式)")

    # ── Step C: fina_indicator 批量 (每批200只) ──
    batch_count = (total + 199) // 200
    logger.info(
        f"  预取 Step C: fina_indicator 批量 ({batch_count} 批, 每批200只)"
    )
    try:
        fina_map = get_fina_indicator_batch(ts_codes, years=2)
        for tc in ts_codes:
            if tc in fina_map:
                f = fina_map[tc]
                cache[tc]["roe"] = str(f.get("roe", "") or "")
                cache[tc]["gross_margin"] = str(
                    f.get("grossprofit_margin", "") or ""
                )
                cache[tc]["net_margin"] = str(
                    f.get("netprofit_margin", "") or ""
                )
                cache[tc]["debt_ratio"] = str(
                    f.get("debt_to_assets", "") or ""
                )
                cache[tc]["revenue_growth"] = str(f.get("or_yoy", "") or "")
                cache[tc]["profit_growth"] = str(f.get("profit_yoy", "") or "")
    except Exception as e:
        logger.warning(f"  预取 Step C 失败: {e}")
    roe_hit = sum(1 for tc in ts_codes if cache[tc].get("roe"))
    logger.info(f"  预取 Step C 完成: ROE 命中 {roe_hit}/{total}")
    if on_progress:
        on_progress(total * 4 // 5, total)

    return cache


# ──────────────────────────────────────────────
# 工具函数（本地副本，避免从 app.py 导入产生副作用）
# ──────────────────────────────────────────────

def _is_bse_code(pure_code: str) -> bool:
    """检测是否为北交所(BSE)代码: 430xxx, 431xxx, 830-839xxx, 870-873xxx, 920xxx"""
    if len(pure_code) < 3:
        return False
    return (pure_code.startswith(("430", "431", "920")) or
            (len(pure_code) >= 3 and pure_code[:3] in
             ("830", "831", "832", "833", "834", "835", "836", "837", "838", "839",
              "870", "871", "872", "873")))


def _get_exchange_prefix(pure_code: str) -> str:
    """返回交易所前缀 (sh/sz/bj)，基于 A 股编码规则"""
    if _is_bse_code(pure_code):
        return "bj"
    if pure_code.startswith(("6", "688", "5")):
        return "sh"
    elif pure_code.startswith(("0", "3", "1", "4", "2")):
        return "sz"
    return "sz"


def normalize_code(code: str) -> str:
    """规范化股票代码为 sh.XXXXXX / sz.XXXXXX 标准格式。

    支持的所有输入格式:
    - 纯数字: "603871" → "sh.603871", "000001" → "sz.000001"
    - Tushare格式: "000001.SZ" → "sz.000001", "603871.SH" → "sh.603871"
    - 交易所后缀: "600519.XSHG" → "sh.600519", "000001.XSHE" → "sz.000001"
    - 已有前缀: "sh.603871" → "sh.603871", "sz.000001" → "sz.000001"
    - 前后缀叠加: "sh.603871.SH" → "sh.603871", "sz.000001.SZ" → "sz.000001"
    - Excel数值截断: "1" → "sz.000001" (零补齐)
    - 括号包裹: "贵州茅台(600519)" → "sh.600519"
    - 带空白: " 603871 " → "sh.603871"

    无法识别时返回原始输入（由调用方决定如何处理）。
    """
    code = str(code).strip().replace("'", "").replace('"', "")

    # 1. 从括号中提取: "贵州茅台(600519)" 或 "茅台（600519.SH）"
    paren = re.search(r'[（(](\d{5,6}(?:\.(?:SZ|SH|sz|sh|XSHG|XSHE))?)[）)]', code)
    if paren:
        code = paren.group(1)

    # 2. 剥离所有已知后缀 (.SZ/.SH/.BJ/.sz/.sh/.bj/.XSHG/.XSHE)
    code = re.sub(r'\.(SZ|SH|BJ|sz|sh|bj|XSHG|XSHE)$', '', code)

    # 3. 提取纯数字部分（应对 "sh.603871" / "stock603871" / Excel数值截断 "1" 等）
    digit_match = re.search(r'(\d{1,6})$', code)
    if digit_match:
        pure = digit_match.group(1)
        # 零补齐: Excel 会把 000001 读成 1
        if len(pure) < 6:
            pure = pure.zfill(6)
        prefix = _get_exchange_prefix(pure)
        return f"{prefix}.{pure}"

    # 4. 已有 sh./sz./bj. 前缀的情况（带非标准后缀已在上方剥离）
    if code.startswith("sh.") or code.startswith("sz.") or code.startswith("bj."):
        inner = code[3:]
        if inner.isdigit() and len(inner) >= 5:
            return code
        return code

    # 5. 无法识别，返回原值
    return code


def _to_tushare_code(stock_code: str) -> str:
    """将任意股票代码转为 603871.SH / 000001.SZ 格式（Tushare API 要求）。

    输入: "sh.603871" / "sz.000001" / "603871.SH" / "000001.SZ" / "sz.000001.SZ" 均可。
    """
    # 剥离所有后缀和前缀，得到纯数字
    pure = stock_code
    pure = re.sub(r'\.(SH|SZ|BJ|sh|sz|bj|XSHG|XSHE)$', '', pure)
    pure = pure.replace("sh.", "").replace("sz.", "").replace("bj.", "")
    # 剥离可能的残留
    pure = re.sub(r'\.(SH|SZ|BJ|sh|sz|bj)$', '', pure)

    # 零补齐（纯数字不足6位时）
    if pure.isdigit() and len(pure) < 6:
        pure = pure.zfill(6)

    if _is_bse_code(pure):
        return f"{pure}.BJ"
    if pure.startswith(("6", "688", "5")):
        return f"{pure}.SH"
    return f"{pure}.SZ"


# ──────────────────────────────────────────────
# Excel 解析
# ──────────────────────────────────────────────

def parse_excel(file_content: bytes) -> List[Dict[str, str]]:
    """解析上传的 Excel 文件，提取股票代码和名称。

    支持格式：
    - 第一行为表头（自动识别"股票代码"/"代码"/"code" 和 "股票名称"/"名称"/"name"列）
    - 股票代码列必须存在
    - 股票名称列为可选（缺失时后续通过 akshare 自动查询）

    返回: [{"code": "sh.603871", "name": "嘉友国际"}, ...]
    限制: 最多 1000 只股票
    """
    wb = openpyxl.load_workbook(BytesIO(file_content), read_only=True, data_only=True)
    ws = wb.active
    if ws is None:
        raise ValueError("Excel 文件中没有找到活动工作表")

    rows = list(ws.iter_rows(min_row=1, values_only=True))
    if len(rows) < 2:
        raise ValueError("Excel 文件至少需要 1 行表头 + 1 行数据")

    # 识别表头
    header = [str(c).strip().lower() if c else "" for c in rows[0]]
    code_col = None
    name_col = None

    code_keywords = ["股票代码", "代码", "code", "symbol", "ticker"]
    name_keywords = ["股票名称", "名称", "name", "company", "公司"]

    for idx, h in enumerate(header):
        if h in code_keywords or any(kw in h for kw in code_keywords):
            code_col = idx
        if h in name_keywords or any(kw in h for kw in name_keywords):
            name_col = idx

    if code_col is None:
        raise ValueError(
            f"未识别到股票代码列，请确保表头包含以下关键词之一: {', '.join(code_keywords)}"
        )

    stocks = []
    seen = set()

    for row in rows[1:]:
        code = str(row[code_col]).strip() if row[code_col] is not None else ""
        if not code or code == "None":
            continue

        # 去掉可能的小数（Excel 会把 603871 读成 603871.0）
        code = re.sub(r'\.0+$', '', code)
        # 零补齐: Excel 读取 000001 时可能变为 1
        if code.isdigit() and len(code) < 6:
            code = code.zfill(6)

        try:
            normalized = normalize_code(code)
        except Exception:
            logger.warning(f"无法规范化代码: {code}")
            continue

        if normalized in seen:
            continue
        seen.add(normalized)

        name = ""
        if name_col is not None and row[name_col] is not None:
            name = str(row[name_col]).strip()

        stocks.append({"code": normalized, "name": name})

    wb.close()

    if not stocks:
        raise ValueError("未从 Excel 中提取到有效股票代码")

    if len(stocks) > 1000:
        raise ValueError(f"单次批量打分最多支持 1000 只股票，当前 {len(stocks)} 只")

    logger.info(f"{SUCCESS_ICON} Excel 解析完成: {len(stocks)} 只股票")
    return stocks


# ──────────────────────────────────────────────
# 轻量数据获取（HTTP 并行 + Tushare 缓存合并）
# ──────────────────────────────────────────────

def _compute_price_changes(kline_data: List[Dict]) -> Dict[str, str]:
    """从K线数据计算各周期涨跌幅"""
    if not kline_data or len(kline_data) < 2:
        return {}

    def to_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    closes = [(row.get("date", ""), to_float(row.get("close"))) for row in kline_data]
    closes = [(d, c) for d, c in closes if c is not None]
    if len(closes) < 2:
        return {}

    latest_close = closes[-1][1]
    periods = {"1d": 1, "5d": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 250, "3y": 750}
    result = {}
    for key, days in periods.items():
        idx = max(0, len(closes) - days - 1)
        base_close = closes[idx][1]
        if base_close and base_close != 0:
            result[key] = f"{(latest_close - base_close) / base_close * 100:.1f}%"
        else:
            result[key] = "N/A"
    return result


def _format_market_cap(value_str: str) -> str:
    """格式化市值"""
    try:
        val = float(value_str)
        if val >= 1e8:
            yi = val / 1e8
        else:
            yi = val
        if yi >= 10000:
            return f"{yi / 10000:.2f}万亿"
        return f"{yi:.0f}亿"
    except (ValueError, TypeError):
        return value_str if value_str else ""


def _fetch_light_stock_data_sync(
    stock_code: str,
    tushare_cache: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """获取单只股票的轻量数据: HTTP 实时行情 + K线 + Tushare 缓存合并。

    Tushare 数据在批量预取阶段已完成，此处仅做内存合并。
    支持磁盘缓存: 命中时跳过 HTTP 请求 (fundamentals 7d, market_data 1d)。
    """
    from src.utils.stock_data_cache import (
        read_data_cache, write_data_cache,
        split_cached_result, merge_cached_into_result,
    )
    import requests as req

    pure_code = stock_code.replace("sh.", "").replace("sz.", "")
    tx_code = f"{_get_exchange_prefix(pure_code)}{pure_code}"
    result = {"code": stock_code, "name": "", "status": "fetched"}

    # ── 0. Disk cache check ──
    today = datetime.now().strftime("%Y-%m-%d")
    cached_fund = read_data_cache("fundamentals", stock_code, today)
    cached_mkt = read_data_cache("market_data", stock_code, today)
    cache_hit = cached_fund and cached_mkt
    if cache_hit:
        result.update(cached_fund)
        result.update(cached_mkt)
        cached_info = read_data_cache("company_info", stock_code, today)
        if cached_info:
            result.update(cached_info)
        result["_cache_hit"] = True
        return result

    # ── 1. Tencent 实时行情 ──
    try:
        resp = req.get(
            f"https://qt.gtimg.cn/q={tx_code}",
            timeout=(3, 12),  # connect=3s, read=12s
            headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        match = re.search(r'"(.+)"', resp.text)
        if match:
            fields = match.group(1).split("~")
            if len(fields) >= 47:
                result["name"] = fields[1]
                result["last_price"] = fields[3]
                result["pct_chg"] = fields[32]
                result["pe"] = fields[39]
                result["market_cap"] = _format_market_cap(fields[44]) if fields[44] else ""
                result["pb"] = fields[46]
                if fields[38] and fields[38] != "0.00":
                    result["turnover_rate"] = fields[38]
    except Exception:
        pass

    # 1b. 名称回退: akshare 缓存
    if not result.get("name"):
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            if df is not None and not df.empty:
                row = df[df["code"] == pure_code]
                if not row.empty:
                    result["name"] = str(row.iloc[0].get("name", ""))
        except Exception:
            pass

    # ── 2. K线数据: Tencent 优先 ──
    kline_data = []
    try:
        start = (datetime.now() - timedelta(days=1130)).strftime("%Y-%m-%d")
        k_url = (
            f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={tx_code},day,{start},,1100,qfq"
        )
        k_resp = req.get(k_url, timeout=(3, 20),
                        headers={"User-Agent": "Mozilla/5.0"})
        k_resp.raise_for_status()
        k_raw = k_resp.json()
        day_data = None
        if "data" in k_raw:
            stock_info = k_raw["data"]
            for key in [tx_code] + list(stock_info.keys()):
                if key in stock_info:
                    dd = stock_info[key]
                    day_data = dd.get("qfqday") or dd.get("day") or dd.get("qfq")
                    if day_data:
                        break
        if day_data:
            prev_close = None
            for item in day_data:
                close_val = float(item[2]) if len(item) > 2 else 0
                pct = ""
                if prev_close and prev_close > 0:
                    pct = f"{((close_val - prev_close) / prev_close * 100):.2f}"
                prev_close = close_val
                kline_data.append({"date": item[0], "close": str(close_val), "pctChg": pct})
    except Exception:
        pass

    # ── 3. 涨跌幅计算 ──
    result["price_changes"] = _compute_price_changes(kline_data)

    # 最近60日K线摘要
    recent = []
    for bar in kline_data[-60:]:
        recent.append({
            "date": bar.get("date", ""),
            "close": bar.get("close", ""),
            "pctChg": bar.get("pctChg", ""),
        })
    result["recent_kline"] = recent

    # ── 4. 从 Tushare 缓存合并数据 ──
    if tushare_cache:
        ts_code = _to_tushare_code(stock_code)
        cached = tushare_cache.get(ts_code, {})
        if cached:
            # 行业: Tushare 优先 (比 Tencent 可靠)
            if cached.get("industry"):
                result["industry"] = cached["industry"]
            # 估值/财务: 缓存优先 (Tushare 数据质量更高)
            for key in ("pe", "pb", "ps", "roe", "gross_margin",
                        "net_margin", "debt_ratio", "revenue_growth",
                        "profit_growth"):
                val = cached.get(key, "")
                if val:
                    result[key] = val
            result["_enrich_errors"] = []
        else:
            result["_enrich_errors"] = [
                f"ts_code {ts_code} 未在 Tushare 缓存命中"
            ]
    else:
        result["_enrich_errors"] = ["无 Tushare 缓存 (预取阶段可能失败)"]

    # ── 5. Write to disk cache ──
    buckets = split_cached_result(result)
    for data_type, data in buckets.items():
        write_data_cache(data_type, stock_code, today, data)

    return result


async def _fetch_round(
    stocks: List[Dict[str, str]],
    total: int,
    semaphore: int,
    on_progress: callable = None,
    only_failed: bool = False,
    round_label: str = "",
    base_completed: int = 0,
    tushare_cache: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    """单轮并行数据获取。返回处理后的 stocks 列表。"""
    targets = [s for s in stocks if not only_failed or s.get("status") != "fetched"]
    if not targets:
        return stocks

    sem = asyncio.Semaphore(semaphore)
    completed = 0
    target_count = len(targets)
    lock = asyncio.Lock()

    async def fetch_one(stock: Dict) -> Dict:
        nonlocal completed
        async with sem:
            try:
                data = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        _batch_pool,
                        _fetch_light_stock_data_sync,
                        stock["code"],
                        tushare_cache,
                    ),
                    timeout=20.0
                )
            except asyncio.TimeoutError:
                data = {"code": stock["code"], "name": stock.get("name", ""),
                        "status": "timeout", "error": "数据获取超时(20s)"}
            except Exception as e:
                data = {"code": stock["code"], "name": stock.get("name", ""),
                        "status": "error", "error": str(e)}

            stock["data"] = data
            stock["status"] = data.get("status", "fetched")
            async with lock:
                completed += 1
                current = base_completed + completed
                if completed % 50 == 0 or completed == target_count:
                    label = f"[{round_label}] " if round_label else ""
                    logger.info(f"  数据获取进度: {label}{current}/{total} 只")
            if on_progress:
                try:
                    on_progress(min(current, total), total)
                except Exception:
                    pass
            return stock

    tasks = [fetch_one(s) for s in targets]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            idx = stocks.index(targets[i])
            stocks[idx]["data"] = {"code": stocks[idx]["code"],
                                    "name": stocks[idx].get("name", ""),
                                    "status": "error", "error": str(r)}
            stocks[idx]["status"] = "error"

    return stocks


async def fetch_batch(
    stocks: List[Dict[str, str]],
    semaphore: int = 6,
    on_progress: callable = None,
    max_retry_rounds: int = 5,
) -> List[Dict]:
    """并行获取所有股票数据，失败自动重试。

    两阶段架构:
      1. Tushare 批量预取 (主线程顺序, 无锁): ~5-10分钟 (1000只)
      2. HTTP 并行获取 (6 线程): ~2-3分钟 (1000只)

    Args:
        stocks: [{code, name}, ...]
        semaphore: HTTP 并发数
        on_progress: 可选回调 on_progress(completed, total)
        max_retry_rounds: 失败重试最大轮数（默认 5）
    """
    total = len(stocks)

    # 立即触发进度回调，确保前端进度条一开始就出现
    if on_progress:
        try:
            on_progress(0, total)
        except Exception:
            pass

    # ── 阶段1: Tushare 批量预取 (线程池执行，不阻塞事件循环) ──
    # 预取进度权重 80%（占数据阶段绝大多数时间），HTTP 占 20%
    tushare_cache: Dict[str, Dict] = {}
    prefetch_base = 0
    if _BATCH_PREFETCH_ENABLED:
        logger.info(f"  Tushare 批量预取开始: {total} 只")
        t_start = time.time()
        try:
            tushare_cache = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    _batch_pool,
                    _prefetch_tushare_batch,
                    stocks,
                    on_progress,
                ),
                timeout=1800.0,  # 最多 30 分钟
            )
            elapsed = time.time() - t_start
            logger.info(
                f"  Tushare 批量预取完成: {len(tushare_cache)} 只, "
                f"耗时 {elapsed:.0f}s"
            )
        except asyncio.TimeoutError:
            logger.error("Tushare 批量预取超时 (30分钟)")
            tushare_cache = {}
        except Exception as e:
            logger.error(f"Tushare 批量预取异常: {e}")
            tushare_cache = {}
        prefetch_base = total * 4 // 5

    # ── 阶段2: HTTP 并行获取 ──
    # 从 prefetch_base 继续，让进度条平滑过渡不倒退
    stocks = await _fetch_round(stocks, total, semaphore, on_progress,
                                round_label="R1",
                                tushare_cache=tushare_cache,
                                base_completed=prefetch_base)

    # ── 重试轮次：仅重试失败股票，并发逐轮递减 ──
    for retry_idx in range(1, max_retry_rounds + 1):
        failed = [s for s in stocks if s.get("status") != "fetched"]
        if not failed:
            break

        retry_sem = max(2, semaphore // (retry_idx + 1))
        success_so_far = total - len(failed)
        logger.warning(
            f"  重试轮次 {retry_idx}/{max_retry_rounds}: "
            f"{len(failed)} 只数据获取失败，以并发={retry_sem} 重试"
        )

        stocks = await _fetch_round(
            stocks, total, retry_sem, on_progress,
            only_failed=True,
            round_label=f"R{retry_idx + 1}",
            base_completed=success_so_far,
            tushare_cache=tushare_cache,
        )

    final_success = sum(1 for s in stocks if s.get("status") == "fetched")
    if final_success < total:
        logger.warning(
            f"{WAIT_ICON} 数据获取完成: {final_success}/{total} 成功 "
            f"({total - final_success} 只最终失败)"
        )
    else:
        logger.info(
            f"{SUCCESS_ICON} 批量数据获取完成: {final_success}/{total} 成功"
        )
    return stocks


# ──────────────────────────────────────────────
# 批量任务管理
# ──────────────────────────────────────────────

def create_batch_job(stocks: List[Dict[str, str]], horizon: str = "medium") -> str:
    """创建新的批量打分任务，返回 batch_id。

    Args:
        stocks: [{code, name}, ...]
        horizon: 打分维度 (short/medium/long/all)
    """
    batch_id = uuid.uuid4().hex[:12]
    job = {
        "batch_id": batch_id,
        "status": "parsed",
        "horizon": horizon,
        "created_at": datetime.now().isoformat(),
        "total_stocks": len(stocks),
        "fetched_count": 0,
        "scored_count": 0,
        "stocks": stocks,
        "results": {},
        "scores": {},
        "error": None,
    }
    with _batch_lock:
        _batch_jobs[batch_id] = job
    logger.info(f"{SUCCESS_ICON} 批量任务创建: {batch_id} ({len(stocks)} 只, {horizon})")
    return batch_id


def get_job(batch_id: str) -> Optional[Dict]:
    """获取批量任务状态"""
    with _batch_lock:
        return _batch_jobs.get(batch_id)


def update_job_progress(batch_id: str, completed: int, total: int):
    """更新数据获取进度（不覆盖后续阶段的状态）"""
    with _batch_lock:
        job = _batch_jobs.get(batch_id)
        if job:
            job["fetched_count"] = completed
            # 只在确实在获取阶段时才设 fetching，避免覆盖 scoring/completed
            if job.get("status") in ("parsed", "fetching"):
                job["status"] = "fetching"


def finalize_fetch(batch_id: str):
    """标记数据获取阶段完成"""
    with _batch_lock:
        job = _batch_jobs.get(batch_id)
        if job:
            job["status"] = "fetched"
            # 将数据从 stocks 列表提取到 results 字典
            for s in job["stocks"]:
                job["results"][s["code"]] = s.get("data", {})


def finalize_batch(batch_id: str, error: str = None):
    """标记整个批量任务完成或失败"""
    with _batch_lock:
        job = _batch_jobs.get(batch_id)
        if job:
            if error:
                job["status"] = "failed"
                job["error"] = error
            else:
                job["status"] = "completed"


def start_scoring_phase(batch_id: str):
    """将任务状态切换为 scoring，重置 scored_count"""
    with _batch_lock:
        job = _batch_jobs.get(batch_id)
        if job:
            job["status"] = "scoring"
            job["scored_count"] = 0


def update_scoring_progress(batch_id: str, scored: int):
    """更新打分进度（供 app.py 调用，避免直接访问私有变量）"""
    with _batch_lock:
        job = _batch_jobs.get(batch_id)
        if job:
            job["scored_count"] = scored
            job["status"] = "scoring"


def cleanup_old_jobs(max_age_hours: int = 6):
    """清理超过指定时间的旧任务"""
    cutoff = datetime.now().timestamp() - max_age_hours * 3600
    with _batch_lock:
        stale = []
        for bid, job in _batch_jobs.items():
            try:
                created = datetime.fromisoformat(job["created_at"]).timestamp()
                if created < cutoff:
                    stale.append(bid)
            except Exception:
                stale.append(bid)
        for bid in stale:
            del _batch_jobs[bid]
    if stale:
        logger.info(f"{WAIT_ICON} 清理 {len(stale)} 个过期批量任务")


# ──────────────────────────────────────────────
# 名称自动查询（批量模式）
# ──────────────────────────────────────────────

def lookup_names(stocks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """为没有名称的股票通过 akshare 补全名称"""
    need_lookup = [s for s in stocks if not s.get("name")]
    if not need_lookup:
        return stocks

    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        if df is not None and not df.empty:
            name_map = {}
            for _, row in df.iterrows():
                c = str(row.get("code", "")).strip()
                n = str(row.get("name", "")).strip()
                if c and n:
                    name_map[c] = n

            for s in need_lookup:
                pure = s["code"].replace("sh.", "").replace("sz.", "")
                if pure in name_map:
                    s["name"] = name_map[pure]

            found = sum(1 for s in need_lookup if s.get("name"))
            logger.info(f"{SUCCESS_ICON} 名称补全: {found}/{len(need_lookup)}")
    except Exception as e:
        logger.warning(f"名称补全失败: {e}")

    return stocks


# ══════════════════════════════════════════════════════════════
# Phase 2: 批量 LLM 打分 (5 stocks/call, qwen3.6-flash, no thinking)
# ══════════════════════════════════════════════════════════════

# 5 级分类定义
LEVELS = ["强烈推荐", "推荐", "中性", "回避", "卖出"]
VALID_CODES = re.compile(r'^(?:sh|sz)\.\d{5,6}$')


def chunk_stocks(items: list, chunk_size: int = 5) -> list:
    """将股票列表拆分为固定大小的批次"""
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i:i + chunk_size])
    return chunks


_BATCH_SYSTEM_PROMPT = """你是 A 股批量筛选器。你的任务是对每只股票基于其数据给出 5 级分类和一句话理由。

## 分类标准

| 级别 | 含义 | 判定标准 |
|------|------|---------|
| 强烈推荐 | 绿 | 基本面优秀 + 估值合理或低估 + 无明显风险信号 |
| 推荐 | 浅绿 | 基本面良好 + 估值合理 + 有一定投资价值 |
| 中性 | 黄 | 多空因素交织 / 数据不足无法判断 / 估值合理但成长性一般 |
| 回避 | 橙 | 基本面偏弱 / 估值偏高 / 风险因素明显 |
| 卖出 | 红 | 基本面恶化 / 严重高估 / 重大风险 |

## 核心原则
1. **宁可保守不可激进**：数据不明确时选"中性"，绝不猜测
2. **数据引用规则**：只能引用下方股票数据中实际出现的数字，严禁编造
3. **跨行业可比**：参考行业估值基准，银行 PE 6 倍可能合理，科技 PE 60 倍可能低估
4. **ST/*ST 股票一律标"卖出"**

## 输出格式
严格输出 JSON 数组（不要 markdown 代码块标记）：

[{"code": "sh.603871", "level": "推荐", "confidence": "高",
  "reason": "低估值+高ROE+行业景气(30字内)", "risk": "原材料涨价(30字内)"},
 ...]
"""

_HORIZON_CONTEXT = {
    "short": """## 当前评分维度: 短线 (1-5 交易日)
重点关注: 量价关系、技术信号、短期资金情绪、近期涨跌幅。基本面权重降低。""",

    "medium": """## 当前评分维度: 中线 (1-3 个月)
重点关注: 基本面质量(ROE/毛利率/增速)、估值水平(vs行业)、技术趋势、风险评估。""",

    "long": """## 当前评分维度: 长线 (1-3 年)
重点关注: 商业护城河、行业景气度、ROE 持续性、估值安全边际。短线波动可忽略。""",
}


def build_batch_prompt(stocks_data: list, horizon: str = "medium", custom_levels: list = None) -> dict:
    """构建批量打分 prompt。

    Args:
        stocks_data: [{code, name, pe, pb, roe, ...}, ...] 最多 5 只
        horizon: 评分维度
        custom_levels: 自定义5级分类(默认用batch_scorer内置的"强烈推荐/推荐/中性/回避/卖出")

    Returns:
        {"system": str, "user": str}
    """
    # 自定义级别覆盖内置 LEVELS
    _levels = custom_levels if custom_levels is not None else LEVELS
    if not stocks_data:
        raise ValueError("stocks_data 不能为空")
    if len(stocks_data) > 5:
        raise ValueError(f"每批最多 5 只股票，当前 {len(stocks_data)} 只")

    from src.utils.industry_knowledge import identify_industry, get_industry_info, get_industry_scoring_guidance

    # 构建每只股票的数据摘要
    stock_blocks = []
    all_industries = set()

    for s in stocks_data:
        code = s.get("code", "")
        name = s.get("name", "")
        industry = s.get("industry", "")

        if industry:
            matched = identify_industry(industry) or industry
            all_industries.add(matched)

        # 提取关键字段
        fields = []
        def _f(label, key):
            v = s.get(key, "")
            if v and str(v).strip() not in ("", "None", "N/A"):
                fields.append(f"  {label}: {v}")

        _f("PE", "pe")
        _f("PB", "pb")
        _f("PS", "ps")
        _f("ROE(%)", "roe")
        _f("毛利率(%)", "gross_margin")
        _f("净利率(%)", "net_margin")
        _f("营收增速(%)", "revenue_growth")
        _f("利润增速(%)", "profit_growth")
        _f("负债率(%)", "debt_ratio")
        _f("换手率(%)", "turnover_rate")
        _f("市值", "market_cap")
        _f("行业", "industry")
        _f("最新价", "last_price")

        pc = s.get("price_changes", {}) or {}
        for period, label in [("1d", "近1日"), ("5d", "近5日"), ("1m", "近1月"),
                               ("3m", "近3月"), ("6m", "近6月"), ("1y", "近1年")]:
            v = pc.get(period, "")
            if v and v != "N/A":
                fields.append(f"  涨跌({label}): {v}")

        field_str = "\n".join(fields) if fields else "  (无数据)"

        block = f"""### {name} ({code})
{field_str}"""
        stock_blocks.append(block)

    # 行业基准
    industry_guidance = ""
    if all_industries:
        guidance_parts = []
        for ind in list(all_industries)[:3]:
            info = get_industry_info(ind)
            if info:
                g = get_industry_scoring_guidance(ind)
                if g:
                    guidance_parts.append(
                        f"**{ind}**: PE合理区间 {info.get('pe_reasonable_range', 'N/A')}, "
                        f"PB合理区间 {info.get('pb_reasonable_range', 'N/A')}"
                    )
        if guidance_parts:
            industry_guidance = "\n".join(guidance_parts)

    horizon_context = _HORIZON_CONTEXT.get(horizon, _HORIZON_CONTEXT["medium"])

    user_message = f"""{horizon_context}

## 行业估值基准
{industry_guidance or "（无行业数据，请根据通用标准判断）"}

## 股票数据

{chr(10).join(stock_blocks)}

## 输出要求
对上述 {len(stocks_data)} 只股票，返回 JSON 数组（不要 markdown 代码块）。
每只股票输出: code, level({'/'.join(_levels)}), confidence(高/中/低), reason(30字内), risk(30字内)

只返回 JSON 数组:"""

    return {"system": _BATCH_SYSTEM_PROMPT, "user": user_message}


def parse_batch_response(response_text: str, custom_levels: list = None) -> list:
    """解析 LLM 批量打分响应，提取有效的 [{code, level, ...}] 列表。

    Args:
        response_text: LLM 原始响应文本
        custom_levels: 自定义5级分类(默认用batch_scorer内置的"强烈推荐/推荐/中性/回避/卖出")

    Returns:
        [{code, level, confidence, reason, risk, ...}, ...]
    """
    if not response_text or not response_text.strip():
        return []

    # 提取 JSON 数组 (使用平衡括号匹配避免贪婪捕获)
    text = response_text.strip()
    start = text.find('[')
    if start == -1:
        return []

    # 从末尾向前尝试不同截断点，找到有效 JSON 数组
    raw = None
    for end in range(len(text), start, -1):
        try:
            candidate = text[start:end]
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                raw = parsed
                break
        except json.JSONDecodeError:
            continue
    if raw is None:
        return []

    if not isinstance(raw, list):
        return []

    results = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        code = item.get("code", "").strip()
        if not code or not VALID_CODES.match(code):
            continue

        level = item.get("level", "中性").strip()
        _valid_levels = custom_levels if custom_levels is not None else LEVELS
        if level not in _valid_levels:
            level = _valid_levels[-2] if len(_valid_levels) >= 2 else "中性"  # default to second-to-last (观望/回避)

        confidence = item.get("confidence", "中").strip()
        if confidence not in ("高", "中", "低"):
            confidence = "中"

        reason = item.get("reason", "").strip()
        risk = item.get("risk", "").strip()

        results.append({
            "code": code,
            "level": level,
            "confidence": confidence,
            "reason": reason[:150],
            "risk": risk[:150],
        })

    return results


async def score_batch(
    stocks: list,
    horizon: str = "medium",
    semaphore: int = 8,
    on_progress: callable = None,
    model_suffix: str = "_2",
    custom_levels: list = None,
) -> list:
    """批量 LLM 打分编排器。

    Args:
        stocks: [{code, name, data: {...}, status: ...}, ...]
        horizon: 打分维度
        semaphore: LLM 调用并发数
        on_progress: 可选回调 on_progress(scored_count, total)
        model_suffix: 模型后缀, ""=M1(MiMo-V2.5-Pro), "_2"=M2(Qwen3.6-Flash), "_3"=M3(Qwen3.7-Plus)
        custom_levels: 自定义5级分类(默认用batch_scorer内置的"强烈推荐/推荐/中性/回避/卖出")

    Returns:
        stocks 列表中添加 "score" 字段
    """
    # 过滤只保留数据获取成功的股票
    valid = [s for s in stocks if s.get("status") == "fetched" and s.get("data")]
    if not valid:
        logger.warning("没有可打分的股票（所有数据获取均失败）")
        return stocks

    # 分块
    chunks = chunk_stocks(valid, chunk_size=5)
    total_chunks = len(chunks)
    scored_count = 0
    total_valid = len(valid)

    logger.info(
        f"{WAIT_ICON} 批量LLM打分开始: {total_valid} 只股票, "
        f"{total_chunks} 批次, 并发={semaphore}, 模型后缀={model_suffix}"
    )

    from src.utils.llm_clients import OpenAICompatibleClient
    import os

    # 根据 model_suffix 选择模型配置
    api_key = os.getenv(f"OPENAI_COMPATIBLE_API_KEY{model_suffix}", "")
    base_url = os.getenv(f"OPENAI_COMPATIBLE_BASE_URL{model_suffix}", "")
    model = os.getenv(f"OPENAI_COMPATIBLE_MODEL{model_suffix}", "")

    # 回退: 如果指定模型未配置 → M1
    if not all([api_key, base_url, model]):
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY", "")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")
        model = os.getenv("OPENAI_COMPATIBLE_MODEL", "mimo-v2.5-pro")

    # M1/M3 启用 thinking, M2 禁用
    extra_body = {}
    if model_suffix in ("_2",):
        extra_body = {}
    else:
        from src.utils.model_config import get_thinking_body
        extra_body = get_thinking_body(base_url, enabled=True)

    llm = OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        env_prefix="",
        extra_body=extra_body,
        http_timeout=180,
        http_connect_timeout=10,
    )

    score_lock = asyncio.Lock()
    sem = asyncio.Semaphore(semaphore)

    async def score_chunk(chunk: list) -> list:
        nonlocal scored_count
        async with sem:
            try:
                stocks_data = [s["data"] for s in chunk]
                prompt = build_batch_prompt(stocks_data, horizon, custom_levels=custom_levels)

                response = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        _batch_pool,
                        lambda: llm.get_completion(
                            [
                                {"role": "system", "content": prompt["system"]},
                                {"role": "user", "content": prompt["user"]}
                            ],
                            max_retries=1,
                        )
                    ),
                    timeout=300.0
                )

                if response:
                    results = parse_batch_response(response, custom_levels=custom_levels)
                    score_map = {r["code"]: r for r in results}
                    for s in chunk:
                        code = s.get("code", "")
                        if code in score_map:
                            s["score"] = score_map[code]
                            s["scored_at"] = datetime.now().isoformat()
                        else:
                            s["score"] = {
                                "code": code, "level": "中性",
                                "confidence": "低",
                                "reason": "LLM未返回该股票结果",
                                "risk": ""
                            }
                        async with score_lock:
                            scored_count += 1
                            current = scored_count
                else:
                    for s in chunk:
                        s["score"] = {
                            "code": s.get("code", ""), "level": "中性",
                            "confidence": "低",
                            "reason": "LLM响应为空",
                            "risk": ""
                        }
                        async with score_lock:
                            scored_count += 1
                            current = scored_count

            except asyncio.TimeoutError:
                for s in chunk:
                    s["score"] = {
                        "code": s.get("code", ""), "level": "中性",
                        "confidence": "低",
                        "reason": "打分超时",
                        "risk": ""
                    }
                    async with score_lock:
                        scored_count += 1
                        current = scored_count
            except Exception as e:
                logger.error(f"批次打分失败: {e}")
                for s in chunk:
                    s["score"] = {
                        "code": s.get("code", ""), "level": "中性",
                        "confidence": "低",
                        "reason": f"打分异常: {str(e)[:50]}",
                        "risk": ""
                    }
                    async with score_lock:
                        scored_count += 1
                        current = scored_count

            if on_progress:
                try:
                    on_progress(current, total_valid)
                except Exception:
                    pass

    # 批次计数器用于日志
    batch_counter = 0
    batch_log_lock = asyncio.Lock()

    async def _score_with_log(chunk):
        nonlocal batch_counter
        result = await score_chunk(chunk)
        async with batch_log_lock:
            batch_counter += 1
            if batch_counter % 20 == 0 or batch_counter == total_chunks:
                done_approx = min(batch_counter * 5, total_valid)
                logger.info(
                    f"  打分进度: ~{done_approx}/{total_valid} 只"
                    f" ({batch_counter}/{total_chunks} 批次)"
                )
        return result

    await asyncio.gather(*[_score_with_log(c) for c in chunks])

    total_scored = sum(1 for s in stocks if s.get("score"))
    logger.info(
        f"{SUCCESS_ICON} 批量LLM打分完成: {total_scored}/{total_valid} 只, "
        f"共 {total_chunks} 批次"
    )

    return stocks
