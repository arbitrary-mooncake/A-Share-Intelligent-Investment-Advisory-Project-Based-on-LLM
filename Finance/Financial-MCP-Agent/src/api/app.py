"""
FastAPI 后端 — 8 REST 端点，连接现有 LangGraph 分析引擎
"""

# ──────────────────────────────────────────────
# 抑制第三方库冗余输出（必须在其他导入之前）
# ──────────────────────────────────────────────
import os
import sys

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio

# Windows: 使用 SelectorEventLoop 以支持子进程（MCP stdio 传输需要）
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import re
import threading
import time
import requests
import uuid
import glob as glob_module
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ──────────────────────────────────────────────
# 添加项目根目录到 Python 路径
# ──────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# src/api/ → src/
SRC_DIR = os.path.dirname(CURRENT_DIR)
# src/ → project root (Financial-MCP-Agent/)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(override=True)

# ──────────────────────────────────────────────
# 导入现有分析引擎组件
# ──────────────────────────────────────────────
from src.utils.state_definition import AgentState
from src.utils.llm_clients import LLMClientFactory, OpenAICompatibleClient
from src.stock_pool.stock_pool_manager import StockPoolManager
from src.stock_pool.scoring_engine import ScoringEngine
from src.tools.mcp_client import get_mcp_tools
from src.utils.industry_knowledge import identify_industry, get_industry_info, get_industry_scoring_guidance

# ──────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON
logger = setup_logger(__name__)

# ──────────────────────────────────────────────
# Pydantic 请求模型
# ──────────────────────────────────────────────

class StockQuery(BaseModel):
    stock_input: str  # 股票代码或名称

class PoolAddRequest(BaseModel):
    stock_code: str
    stock_name: str

# ──────────────────────────────────────────────
# 全局状态
# ──────────────────────────────────────────────
_task_results: Dict[str, Dict[str, Any]] = {}
_task_statuses: Dict[str, str] = {}
_task_progress: Dict[str, float] = {}  # task_id → 0.0~1.0
_scoring_engine: Optional[ScoringEngine] = None
_pool_manager: Optional[StockPoolManager] = None

# ──────────────────────────────────────────────
# 中间产物缓存管理
# ──────────────────────────────────────────────

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "intermediate_cache")
CACHE_RETENTION_DAYS = 7


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(agent_name: str, stock_code: str, date_str: str) -> str:
    return os.path.join(CACHE_DIR, f"{agent_name}_{stock_code}_{date_str}.json")


def read_cache(agent_name: str, stock_code: str, date_str: str) -> Optional[Dict]:
    path = get_cache_path(agent_name, stock_code, date_str)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def write_cache(agent_name: str, stock_code: str, date_str: str, data: Dict):
    ensure_cache_dir()
    path = get_cache_path(agent_name, stock_code, date_str)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "agent_name": agent_name,
            "stock_code": stock_code,
            "date": date_str,
            "content": data,
            "timestamp": datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)


def clean_old_cache():
    """删除超过 CACHE_RETENTION_DAYS 天的中间产物"""
    ensure_cache_dir()
    cutoff = datetime.now() - timedelta(days=CACHE_RETENTION_DAYS)
    for filepath in glob_module.glob(os.path.join(CACHE_DIR, "*.json")):
        try:
            file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_mtime < cutoff:
                os.remove(filepath)
                logger.info(f"{WAIT_ICON} 清理过期缓存: {os.path.basename(filepath)}")
        except Exception as e:
            logger.warning(f"{ERROR_ICON} 清理缓存失败: {e}")


def get_cache_status_for_stock(stock_code: str) -> Dict[str, Any]:
    """查看某只股票的中间产物缓存状态"""
    ensure_cache_dir()
    cached = {}
    for filepath in glob_module.glob(os.path.join(CACHE_DIR, f"*_{stock_code}_*.json")):
        filename = os.path.basename(filepath)
        # 解析文件名: {agent_name}_{stock_code}_{date}.json
        parts = filename.replace(".json", "").split("_")
        if len(parts) >= 3:
            agent_name = parts[0]
            date_str = parts[-1]
            cached[agent_name] = date_str
    return {"stock_code": stock_code, "cached_agents": cached}


# ──────────────────────────────────────────────
# 股票代码规范化
# ──────────────────────────────────────────────

def _get_exchange_prefix(pure_code: str) -> str:
    """返回交易所前缀 (sh/sz) 用于 Tencent/Sina API"""
    if pure_code.startswith(("6", "688", "5", "8")):
        return "sh"
    elif pure_code.startswith(("0", "3", "1", "4")):
        return "sz"
    return "sz"


def normalize_code(code: str) -> str:
    code = code.strip()
    if code.startswith("sh.") or code.startswith("sz."):
        return code
    prefix = _get_exchange_prefix(code)
    return f"{prefix}.{code}"


def extract_stock_info(user_input: str) -> tuple:
    """从用户输入提取股票代码和名称"""
    user_input = user_input.strip()

    # 提取股票代码（6位数字，可能带 sh./sz. 前缀）
    code_match = re.search(r'\b(\d{5,6})\b', user_input)
    stock_code = code_match.group(1) if code_match else None

    # 括号内提取: 嘉友国际(603871) 或 茅台（600519）
    paren_match = re.search(r'([^（(]+?)\s*[（(](\d{5,6})[)）]', user_input)
    if paren_match:
        company_name = paren_match.group(1).strip()
        stock_code = paren_match.group(2)
        return company_name, stock_code

    # 提取公司名称（关键词后）
    name_patterns = [
        r'分析\s*([^0-9（）()\s]+)',
        r'分析一下?\s*([^0-9（）()\s]+)',
        r'帮我看看\s*([^0-9（）()\s]+)',
        r'了解.*?([^0-9（）()\s]+)',
        r'([^0-9（）()\s]+)\s*(?:这只|这个)?\s*股票',
        r'给.*?分析.*?([^0-9（）()\s]+)',
    ]
    company_name = None
    for pattern in name_patterns:
        match = re.search(pattern, user_input)
        if match:
            candidate = match.group(1).strip()
            if len(candidate) >= 2:
                company_name = candidate
                break

    # 纯中文名称（不含数字，2-10个字符）→ 可能是股票名
    if not company_name and not stock_code:
        chinese_match = re.match(r'^[一-鿿\w·]{2,10}$', user_input)
        if chinese_match:
            company_name = user_input

    return company_name, stock_code


# ──────────────────────────────────────────────
# 快速查询：akshare 优先，baostock 作为备用（绕过 MCP 子进程）
# ──────────────────────────────────────────────

_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="stock_data")
_baostock_lock = threading.Lock()
_akshare_cache = None


def _get_akshare():
    """延迟导入并缓存 akshare 模块"""
    global _akshare_cache
    if _akshare_cache is None:
        import akshare
        _akshare_cache = akshare
    return _akshare_cache


def _fetch_tencent_sync(stock_code: str) -> Dict:
    """Tencent Finance API — 实时行情（快速、含PE/PB/市值/换手率）"""
    pure_code = stock_code.replace("sh.", "").replace("sz.", "")
    tx_code = f"{_get_exchange_prefix(pure_code)}{pure_code}"

    data = {}
    try:
        resp = requests.get(
            f"https://qt.gtimg.cn/q={tx_code}",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        # Parse Tencent format: var v_sh600519="1~name~code~..."
        match = re.search(r'"(.+)"', resp.text)
        if not match:
            logger.warning(f"Tencent API 返回格式异常: {resp.text[:100]}")
            return data

        fields = match.group(1).split("~")
        if len(fields) < 47:
            logger.warning(f"Tencent API 字段不足: {len(fields)}")
            return data

        data["code_name"] = fields[1]          # 名称
        data["last_price"] = fields[3]          # 最新价
        data["pct_chg"] = fields[32]            # 涨跌幅
        data["peTTM"] = fields[39]              # 市盈率(TTM)
        data["totalMarketValue"] = fields[44]   # 总市值(亿元)
        data["pbMRQ"] = fields[46]              # 市净率
        if fields[38] and fields[38] != "0.00":
            data["turnoverRate"] = fields[38]   # 换手率(%)
        logger.debug(f"[Tencent] {data['code_name']}: PE={data['peTTM']}, PB={data['pbMRQ']}, MC={data['totalMarketValue']}亿")
    except Exception as e:
        logger.warning(f"Tencent API 失败: {e}")
    return data


def _fetch_tencent_kline(stock_code: str, days: int = 1100) -> list:
    """Tencent K线 API（复权数据）"""
    pure_code = stock_code.replace("sh.", "").replace("sz.", "")
    tx_code = f"{_get_exchange_prefix(pure_code)}{pure_code}"

    from datetime import timedelta
    start = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")
    url = (
        f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={tx_code},day,{start},,{days},qfq"
    )
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        raw = resp.json()
        # Navigate: data → {code} → qfqday or day
        stock_key = tx_code
        day_data = None
        if "data" in raw and stock_key in raw["data"]:
            stock_info = raw["data"][stock_key]
            day_data = stock_info.get("qfqday") or stock_info.get("day") or stock_info.get("qfq")
        if not day_data:
            # Try list format
            if "data" in raw:
                for k, v in raw["data"].items():
                    dd = v.get("qfqday") or v.get("day") or v.get("qfq")
                    if dd:
                        day_data = dd
                        break
        if not day_data:
            logger.warning(f"Tencent K-line 无数据: {tx_code}")
            return []

        kline = []
        prev_close = None
        for item in day_data:
            close_val = float(item[2]) if len(item) > 2 else 0
            pct_chg = ""
            if prev_close and prev_close > 0:
                pct_chg = f"{((close_val - prev_close) / prev_close * 100):.2f}"
            prev_close = close_val
            kline.append({
                "date": item[0] if len(item) > 0 else "",
                "close": str(close_val),
                "pctChg": pct_chg,
            })
        return kline
    except Exception as e:
        logger.warning(f"Tencent K-line 失败: {e}")
        return []


def _fetch_sina_kline(stock_code: str, days: int = 1100) -> list:
    """Sina HTTP K线 API（备用）"""
    pure_code = stock_code.replace("sh.", "").replace("sz.", "")
    sina_code = f"{_get_exchange_prefix(pure_code)}{pure_code}"

    url = (
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={sina_code}&scale=240&ma=no&datalen={days}"
    )
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        raw = resp.text.strip()
        if not raw:
            raise ValueError("Sina K-line API 返回空")
        data = json.loads(raw)
        kline = []
        prev_close = None
        for item in data:
            close_val = float(item.get("close", 0))
            pct_chg = ""
            if prev_close and prev_close > 0:
                pct_chg = f"{((close_val - prev_close) / prev_close * 100):.2f}"
            prev_close = close_val
            kline.append({
                "date": item.get("day", ""),
                "close": str(close_val),
                "pctChg": pct_chg,
            })
        return kline
    except Exception as e:
        logger.warning(f"Sina K-line 失败: {e}")
        return []


def _fetch_akshare_sync(stock_code: str) -> Dict:
    """
    主数据源：Tencent Finance API（实时行情）+ Tencent/Sina K线
    East Money 和 Baostock 均不可用：
    - East Money 主动封锁 API 请求（RemoteDisconnected）
    - Baostock 服务器不可达（网络接收错误）
    """
    data = {}
    pure_code = stock_code.replace("sh.", "").replace("sz.", "")

    # 1. 实时行情：Tencent API（0.2s，含 PE/PB/市值/换手率）
    tx_data = _fetch_tencent_sync(stock_code)
    if tx_data:
        data.update(tx_data)

    # 1b. 名称回退：Sina code name 表
    if not data.get("code_name"):
        try:
            ak = _get_akshare()
            df = ak.stock_info_a_code_name()
            if df is not None and not df.empty:
                row = df[df["code"] == pure_code]
                if not row.empty:
                    data["code_name"] = str(row.iloc[0].get("name", ""))
        except Exception as e:
            logger.debug(f"code_name 回退失败: {e}")

    # 2. K线数据：Tencent 优先，Sina 备用
    kline_data = _fetch_tencent_kline(stock_code)
    if not kline_data:
        logger.info(f"{WAIT_ICON} Tencent K-line 无数据，尝试 Sina...")
        try:
            kline_data = _fetch_sina_kline(stock_code)
        except Exception as e:
            logger.warning(f"Sina K-line 也失败: {e}")

    if kline_data:
        data["kline"] = kline_data

    if data:
        logger.info(
            f"{SUCCESS_ICON} 数据获取成功: {data.get('code_name', pure_code)} "
            f"(PE={data.get('peTTM', 'N/A')}, PB={data.get('pbMRQ', 'N/A')}, "
            f"市值={data.get('totalMarketValue', 'N/A')}亿, K线{len(data.get('kline', []))}条)"
        )
    return data


def _fetch_baostock_sync(stock_code: str) -> Dict:
    """baostock 备用数据源 — 仅在 akshare 完全失败时使用"""
    import baostock as bs
    try:
        lg = bs.login()
        if lg.error_code != '0':
            logger.debug(f"baostock login failed: {lg.error_msg}")
            return {}
    except Exception as e:
        logger.debug(f"baostock login exception: {e}")
        return {}

    try:
        data = {}
        rs = bs.query_stock_basic(code=stock_code)
        if rs.error_code == '0':
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                cols = rs.fields
                row_dict = dict(zip(cols, rows[-1]))
                data["code_name"] = row_dict.get("code_name", "")
                data["industry"] = row_dict.get("industry", "")

        rs_ind = bs.query_stock_industry(code=stock_code)
        if rs_ind.error_code == '0':
            ind_rows = []
            while rs_ind.next():
                ind_rows.append(rs_ind.get_row_data())
            if ind_rows:
                ind_dict = dict(zip(rs_ind.fields, ind_rows[-1]))
                data["industry"] = ind_dict.get("industry", data.get("industry", ""))
                data["peTTM"] = ind_dict.get("peTTM", "")
                data["pbMRQ"] = ind_dict.get("pbMRQ", "")
                data["totalMarketValue"] = ind_dict.get("totalMarketValue", "")
                data["turnoverRate"] = ind_dict.get("turnoverRate", "")

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=3*365+30)).strftime("%Y-%m-%d")
        rs_k = bs.query_history_k_data_plus(
            stock_code, "date,close,pctChg",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"
        )
        if rs_k.error_code == '0':
            k_rows = []
            while rs_k.next():
                k_rows.append(rs_k.get_row_data())
            if k_rows:
                data["kline"] = [dict(zip(rs_k.fields, r)) for r in k_rows]

        if data:
            logger.info(f"baostock 数据获取成功: {data.get('code_name', stock_code)}")
        return data
    except Exception as e:
        logger.warning(f"baostock 数据获取失败: {e}")
        return {}
    finally:
        try:
            bs.logout()
        except:
            pass


def _fetch_stock_data_sync(stock_code: str) -> Dict:
    """获取股票数据：akshare 优先 → baostock 备用"""
    # 1. 先尝试 akshare（快速，无需登录）
    logger.info(f"{WAIT_ICON} 获取股票数据: {stock_code} (akshare优先)...")
    result = _fetch_akshare_sync(stock_code)

    # 2. akshare 完全无数据时才尝试 baostock
    if not result or not result.get("code_name"):
        logger.info(f"{WAIT_ICON} akshare 无数据，尝试 baostock 备用...")
        bs_result = _fetch_baostock_sync(stock_code)
        if bs_result:
            for k, v in bs_result.items():
                if not result.get(k):
                    result[k] = v

    return result


def _compute_price_changes(kline_data: List[Dict]) -> Dict[str, str]:
    """从K线数据计算各周期涨跌幅"""
    if not kline_data or len(kline_data) < 2:
        return {}

    result = {}
    # 将字符串值转为 float
    def to_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    closes = [(row.get("date", ""), to_float(row.get("close"))) for row in kline_data]
    # 过滤掉 None
    closes = [(d, c) for d, c in closes if c is not None]
    if len(closes) < 2:
        return {}

    latest_close = closes[-1][1]
    periods = {
        "1d": 1, "5d": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 250, "3y": 750
    }
    for key, days in periods.items():
        idx = max(0, len(closes) - days - 1)
        base_close = closes[idx][1]
        if base_close and base_close != 0:
            pct = (latest_close - base_close) / base_close * 100
            result[key] = f"{pct:.1f}%"
        else:
            result[key] = "N/A"

    return result


def _format_market_cap(value_str: str) -> str:
    """将总市值数值格式化为可读格式。
    - Tencent API: 直接返回亿元值（如 17341.31 → 1.73万亿）
    - Baostock: 返回元（如 1734100000000 → 1.73万亿）"""
    try:
        val = float(value_str)
        if val >= 1e8:           # Baostock 格式（元），转成亿
            yi = val / 1e8
        else:                     # Tencent 格式（已为亿）
            yi = val
        if yi >= 10000:
            return f"{yi / 10000:.2f}万亿"
        return f"{yi:.0f}亿"
    except (ValueError, TypeError):
        return value_str if value_str else ""


def _enrich_with_tushare(stock_code: str, existing: Dict) -> Dict:
    """使用 Tushare 补充核心字段: 行业分类、PS、PE分位、ROE等财务指标。
    顺序调用（Tushare 有全局速率限制，并行无意义），每个调用独立隔离异常。
    低优先级字段（十大股东、基金持仓等）由深度报告覆盖。"""
    try:
        from src.utils.tushare_client import (
            get_stock_info, get_daily_basic, get_fina_indicator, get_dividend,
            compute_pe_percentile, compute_ev_ebitda, get_stock_news_em,
        )
        ts_code = stock_code.replace("sh.", "").replace("sz.", "")
        if ts_code.startswith(("6", "688", "5", "8")):
            ts_code = f"{ts_code}.SH"
        else:
            ts_code = f"{ts_code}.SZ"

        # 1. 行业分类
        if not existing.get("industry"):
            try:
                info = get_stock_info(ts_code)
                if info:
                    existing["industry"] = info.get("industry", "") or ""
            except Exception:
                pass

        # 2. 估值补充: PS, PE/PB确认
        basics = None
        try:
            basics = get_daily_basic(ts_code, days=5)
        except Exception:
            pass

        if basics and isinstance(basics, list) and basics:
            latest = basics[0]
            if not existing.get("pe"):
                existing["pe"] = str(latest.get("pe_ttm", ""))
            if not existing.get("pb"):
                existing["pb"] = str(latest.get("pb", ""))
            existing["ps"] = str(latest.get("ps_ttm", ""))
            existing["total_mv_raw"] = str(latest.get("total_mv", ""))

            # PE分位（单独Tushare调用，拉5年数据）
            pe_val = latest.get("pe_ttm")
            if pe_val and pe_val != "None":
                try:
                    pe_pct = compute_pe_percentile(ts_code, float(pe_val), years=5)
                    if pe_pct:
                        existing["pe_percentile"] = (
                            f"近5年PE分位: {pe_pct['percentile']}%, "
                            f"最低{pe_pct['min_pe']:.1f}, 中位{pe_pct['median_pe']:.1f}, "
                            f"最高{pe_pct['max_pe']:.1f}"
                        )
                except Exception:
                    pass

        # 3. 财务指标摘要
        try:
            fina = get_fina_indicator(ts_code, years=2)
            if fina and isinstance(fina, list) and fina:
                latest_f = fina[0]
                existing["roe"] = str(latest_f.get("roe", ""))
                existing["gross_margin"] = str(latest_f.get("grossprofit_margin", ""))
                existing["net_margin"] = str(latest_f.get("netprofit_margin", ""))
                existing["debt_ratio"] = str(latest_f.get("debt_to_assets", ""))
                existing["revenue_growth"] = str(latest_f.get("or_yoy", ""))
                existing["profit_growth"] = str(latest_f.get("profit_yoy", ""))
        except Exception:
            pass

        # 4. 分红
        try:
            div = get_dividend(ts_code)
            if div and isinstance(div, list) and div:
                recent = [d for d in div if d.get("cash_div") and float(d["cash_div"]) > 0][:3]
                if recent:
                    existing["dividend_summary"] = "; ".join(
                        f"{d['end_date'][:4]}年: {d['cash_div']}元/股" for d in recent
                    )
        except Exception:
            pass

        # 5. EV/EBITDA（daily_basic 的 total_mv 单位是万元，需转为元）
        mv_raw = existing.get("total_mv_raw")
        if mv_raw:
            try:
                ev_result = compute_ev_ebitda(ts_code, float(mv_raw) * 1e4)
                if ev_result:
                    existing["ev_ebitda"] = (
                        f"EV={ev_result['ev']}亿, EBITDA={ev_result['ebitda']}亿, "
                        f"EV/EBITDA={ev_result['ev_ebitda']}倍"
                        + ("(估算)" if ev_result.get("ebitda_estimated") else "")
                    )
            except Exception:
                pass

        # 6. 近期新闻
        pure_code = stock_code.replace("sh.", "").replace("sz.", "")
        try:
            news = get_stock_news_em(pure_code)
            if news:
                existing["recent_news"] = [
                    f"[{n['time'][:10]}] {n['title']} ({n['source']})"
                    for n in news[:5]
                ]
        except Exception:
            pass

    except Exception as e:
        logger.debug(f"Tushare 补充数据失败: {e}")
    return existing


async def fetch_stock_data(stock_code: str) -> Dict:
    """异步包装：调用数据源获取股票数据（带总超时15秒）"""
    try:
        raw = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(_thread_pool, _fetch_stock_data_sync, stock_code),
            timeout=15.0
        )
    except asyncio.TimeoutError:
        logger.error(f"获取股票数据超时(15s): {stock_code}")
        return {}

    if not raw:
        return {}

    kline = raw.get("kline", [])
    price_changes = _compute_price_changes(kline)

    # 最近60个交易日K线摘要（供快筛打分技术分析使用）
    recent_kline = []
    for bar in kline[-60:]:
        recent_kline.append({
            "date": bar.get("date", ""),
            "close": bar.get("close", ""),
            "pctChg": bar.get("pctChg", ""),
        })

    result = {
        "company_name": raw.get("code_name", ""),
        "industry": raw.get("industry", ""),
        "industry_classification": raw.get("industryClassification", ""),
        "pe": raw.get("peTTM", ""),
        "pb": raw.get("pbMRQ", ""),
        "market_cap": _format_market_cap(raw.get("totalMarketValue", "")),
        "turnover_rate": raw.get("turnoverRate", ""),
        "price_changes": price_changes,
        "recent_kline": recent_kline,
    }

    # Tushare 补充: 行业分类、PS、PE分位、ROE、毛利率等（最多等8秒）
    try:
        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                _thread_pool, _enrich_with_tushare, stock_code, result
            ),
            timeout=8.0
        )
    except asyncio.TimeoutError:
        logger.debug(f"Tushare 补充超时(8s)，跳过补充数据")
    except Exception as e:
        logger.debug(f"Tushare 补充失败: {e}")

    return result


# ──────────────────────────────────────────────
# 快速查询用 LLM 调用（低延迟模型）
# ──────────────────────────────────────────────

async def quick_analysis_llm(stock_code: str, company_name: str, stock_data: Dict) -> Dict:
    """使用 LLM 对真实股票数据进行快速分析。
    核心原则：LLM 只做解读和判断，绝不编造/修改数据字段。使用 qwen3.6-flash 高速模型。"""
    llm = OpenAICompatibleClient(
        api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY_2"),
        base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL_2"),
        model=os.getenv("OPENAI_COMPATIBLE_MODEL_2", "qwen3.6-flash"),
        env_prefix="",  # 使用显式参数，不从环境变量自动读取
        extra_body={"thinking": {"type": "disabled"}},
    )

    data_json = json.dumps(stock_data, ensure_ascii=False, indent=2) if stock_data else "无数据"
    name_hint = company_name if company_name and not company_name.isdigit() else stock_code

    prompt = f"""请对以下 A 股股票进行快速分析。

股票: {name_hint} ({stock_code})
以下是系统从数据源获取的真实数据，请仅基于这些数据进行分析:
{data_json}

请以 JSON 格式返回，只返回 JSON，不要添加任何其他文字:
{{
  "company_name": "基于真实数据中的 company_name 字段填写，没有则填 null",
  "market_cap": "基于真实数据中的 market_cap 字段填写，没有则填 null",
  "pb": "基于真实数据中的 pb 字段填写，没有则填 null",
  "pe": "基于真实数据中的 pe 字段填写，没有则填 null",
  "turnover_rate": "基于真实数据中的 turnover_rate 字段填写，没有则填 null",
  "price_changes": {{
    "1d": "来自真实数据 price_changes.1d，没有则填 null",
    "5d": "来自真实数据 price_changes.5d，没有则填 null",
    "1m": "来自真实数据 price_changes.1m，没有则填 null",
    "3m": "来自真实数据 price_changes.3m，没有则填 null",
    "6m": "来自真实数据 price_changes.6m，没有则填 null",
    "1y": "来自真实数据 price_changes.1y，没有则填 null",
    "3y": "来自真实数据 price_changes.3y，没有则填 null"
  }},
  "industry": "基于真实数据中的 industry 字段填写，没有则填 null",
  "industry_intro": "一句行业介绍，若无法判断则填 null",
  "company_intro": "一句公司介绍，若无法判断则填 null"
}}

⛔ 最严格规则（违反任何一条即为严重错误）:
1. 数值字段(pe/pb/market_cap/turnover_rate/price_changes)必须原样照抄真实数据，不得推测、不得补充、不得四舍五入、不得修改任何数字。若数据中为空或不存在该字段，必须填 null。
2. company_name/industry 必须使用真实数据中的值，若为空则填 null，严禁猜测或杜撰。
3. 只做分析解读，不做数据生成。你没有权限创造任何新数据，你只能基于已有数据做出判断。
4. industry_intro 和 company_intro 若无法从真实数据中获取，可基于你的知识补充（仅这两个字段）。"""

    # 从真实数据预填所有值，LLM 只能覆盖 industry_intro/company_intro
    pc = stock_data.get("price_changes", {}) if stock_data else {}
    factual = {
        "company_name": stock_data.get("company_name") or None,
        "market_cap": stock_data.get("market_cap") or None,
        "pb": stock_data.get("pb") or None,
        "pe": stock_data.get("pe") or None,
        "ps": stock_data.get("ps") or None,
        "turnover_rate": stock_data.get("turnover_rate") or None,
        "price_changes": {k: pc.get(k) or None for k in ["1d","5d","1m","3m","6m","1y","3y"]},
        "industry": stock_data.get("industry") or None,
        "pe_percentile": stock_data.get("pe_percentile") or None,
        "roe": stock_data.get("roe") or None,
        "gross_margin": stock_data.get("gross_margin") or None,
        "net_margin": stock_data.get("net_margin") or None,
        "debt_ratio": stock_data.get("debt_ratio") or None,
        "revenue_growth": stock_data.get("revenue_growth") or None,
        "profit_growth": stock_data.get("profit_growth") or None,
        "dividend_summary": stock_data.get("dividend_summary") or None,
        "ev_ebitda": stock_data.get("ev_ebitda") or None,
        "fund_holders": stock_data.get("fund_holders") or None,
        "insider_trades": stock_data.get("insider_trades") or None,
        "recent_news": stock_data.get("recent_news") or None,
        "industry_intro": None,
        "company_intro": None,
    }

    try:
        # 在线程池中执行 LLM 调用，避免阻塞事件循环
        response = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                _thread_pool,
                lambda: llm.get_completion([
                    {"role": "system", "content": "你是一个严格遵循指令的股票数据分析助手。你的唯一数据来源是系统提供的 JSON 数据。你严禁编造、估算、修改任何数值数据。对于数据源未提供的字段，你必须返回 null。只返回纯 JSON，不要包含 markdown 代码块标记。保证信息真实准确是你的首要任务。"},
                    {"role": "user", "content": prompt}
                ], max_retries=1)
            ),
            timeout=30.0
        )

        if response:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                llm_result = json.loads(json_match.group())
                factual["industry_intro"] = llm_result.get("industry_intro") or None
                factual["company_intro"] = llm_result.get("company_intro") or None
                return factual
    except asyncio.TimeoutError:
        logger.warning(f"快速分析 LLM 超时(30s)，返回基础数据")
    except Exception as e:
        logger.error(f"{ERROR_ICON} 快速分析 LLM 调用失败: {e}")

    return factual


# ──────────────────────────────────────────────
# 深度报告生成（异步任务）
# ──────────────────────────────────────────────

async def generate_report_task(task_id: str, stock_code: str, company_name: str):
    """后台异步执行深度报告生成"""
    try:
        _task_statuses[task_id] = "running"

        # 1. 使用统一状态构建器（与打分pipeline保持一致）
        from src.stock_pool.scoring_engine import ScoringEngine
        initial_state = ScoringEngine._build_initial_state(stock_code, company_name)

        # 2. 构建并执行 LangGraph 工作流（超时保护）
        from langgraph.graph import StateGraph, END
        from src.agents.summary_agent import summary_agent
        from src.agents.value_agent import value_agent
        from src.agents.technical_agent import technical_agent
        from src.agents.fundamental_agent import fundamental_agent
        from src.agents.news_agent import news_agent

        workflow = StateGraph(AgentState)
        workflow.add_node("start_node", lambda state: state)
        workflow.add_node("fundamental_analyst", fundamental_agent)
        workflow.add_node("technical_analyst", technical_agent)
        workflow.add_node("value_analyst", value_agent)
        workflow.add_node("news_analyst", news_agent)
        workflow.add_node("summarizer", summary_agent)

        workflow.set_entry_point("start_node")
        workflow.add_edge("start_node", "fundamental_analyst")
        workflow.add_edge("start_node", "technical_analyst")
        workflow.add_edge("start_node", "value_analyst")
        workflow.add_edge("start_node", "news_analyst")
        workflow.add_edge("fundamental_analyst", "summarizer")
        workflow.add_edge("technical_analyst", "summarizer")
        workflow.add_edge("value_analyst", "summarizer")
        workflow.add_edge("news_analyst", "summarizer")
        workflow.add_edge("summarizer", END)

        app = workflow.compile()
        logger.info(f"{WAIT_ICON} 报告Pipeline开始: {company_name}({stock_code}), 超时=2100s")

        # 使用 astream 追踪真实进度（6个节点：start + 4分析 + summarizer）
        completed_nodes = set()
        total_nodes = 6
        final_state = None
        async for chunk in app.astream(initial_state, config={"recursion_limit": 30}):
            for node_name, node_state in chunk.items():
                completed_nodes.add(node_name)
                _task_progress[task_id] = min(len(completed_nodes) / total_nodes, 0.95)
                logger.info(
                    f"报告Pipeline节点完成: {node_name} "
                    f"({len(completed_nodes)}/{total_nodes}, "
                    f"进度={_task_progress[task_id]:.0%})"
                )
                # 保留summarizer的输出作为最终状态
                if node_name == "summarizer":
                    final_state = node_state

        _task_progress[task_id] = 1.0

        if final_state is None:
            raise ValueError("Pipeline未返回summarizer结果")

        # 3. 提取报告
        report_data = final_state.get("data", {})
        report_content = report_data.get("final_report", "")
        report_path = report_data.get("report_path", "")

        _task_results[task_id] = {
            "status": "completed",
            "report_content": report_content,
            "report_path": report_path,
            "report_pdf_path": report_data.get("report_pdf_path", ""),
            "stock_code": stock_code,
            "company_name": company_name,
            "generated_at": datetime.now().isoformat()
        }
        _task_statuses[task_id] = "completed"
        logger.info(f"{SUCCESS_ICON} 报告生成完成: {company_name}({stock_code})")

    except asyncio.TimeoutError:
        logger.error(f"{ERROR_ICON} 报告生成超时(2100s): {company_name}({stock_code})")
        _task_results[task_id] = {
            "status": "failed",
            "error": "报告生成超时（35分钟），请重试",
            "stock_code": stock_code,
            "company_name": company_name
        }
        _task_statuses[task_id] = "failed"
    except Exception as e:
        logger.error(f"{ERROR_ICON} 报告生成失败: {e}", exc_info=True)
        _task_results[task_id] = {
            "status": "failed",
            "error": str(e),
            "stock_code": stock_code,
            "company_name": company_name
        }
        _task_statuses[task_id] = "failed"


# ──────────────────────────────────────────────
# FastAPI 应用生命周期
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool_manager, _scoring_engine
    logger.info(f"{SUCCESS_ICON} 启动 FastAPI 后端...")
    _pool_manager = StockPoolManager()
    _scoring_engine = ScoringEngine(pool_manager=_pool_manager)
    clean_old_cache()
    # 后台预加载名称缓存
    asyncio.get_running_loop().run_in_executor(_thread_pool, _ensure_name_cache)
    yield
    logger.info(f"{WAIT_ICON} 关闭 FastAPI 后端...")


app = FastAPI(
    title="Stock Investment Advisor API",
    description="A-share stock analysis and scoring REST API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# API 端点
# ──────────────────────────────────────────────

_name_code_cache: Optional[dict] = None  # name → code 映射


def _ensure_name_cache():
    """加载并缓存股票名称→代码映射（首次调用约13秒）"""
    global _name_code_cache
    if _name_code_cache is not None:
        return
    try:
        ak = _get_akshare()
        df = ak.stock_info_a_code_name()
        if df is not None and not df.empty:
            _name_code_cache = {}
            for _, row in df.iterrows():
                n = str(row.get("name", "")).strip()
                c = str(row.get("code", "")).strip()
                if n and c:
                    _name_code_cache[n] = c
            logger.info(f"{SUCCESS_ICON} 名称缓存加载完成: {len(_name_code_cache)} 只股票")
    except Exception as e:
        logger.warning(f"名称缓存加载失败: {e}")
        _name_code_cache = {}


def _lookup_stock_code_by_name(name: str) -> Optional[str]:
    """通过公司名查找股票代码（从缓存）"""
    _ensure_name_cache()
    if not _name_code_cache:
        return None
    # 精确匹配
    if name in _name_code_cache:
        return _name_code_cache[name]
    # 模糊匹配（名称包含输入）
    for n, c in _name_code_cache.items():
        if name in n:
            return c
    return None


@app.post("/api/query")
async def quick_query(query: StockQuery):
    """快速查询 — 返回股票基本信息 + 投资建议"""
    try:
        company_name, stock_code = extract_stock_info(query.stock_input)

        if not stock_code and not company_name:
            raise HTTPException(status_code=400, detail="无法识别股票代码或名称")

        # 如果只有名称没有代码，尝试查找
        if not stock_code and company_name:
            found_code = _lookup_stock_code_by_name(company_name)
            if found_code:
                stock_code = normalize_code(found_code)
                logger.info(f"名称→代码: {company_name} → {stock_code}")

        if not stock_code:
            stock_code = query.stock_input.strip()
        else:
            stock_code = normalize_code(stock_code)

        if not company_name:
            company_name = query.stock_input.strip()

        # 获取真实股票数据（Tencent + Sina K-line）
        stock_data = await fetch_stock_data(stock_code)

        # 如果返回了公司名，用它覆盖
        if stock_data.get("company_name"):
            company_name = stock_data["company_name"]

        # 调用 LLM 进行快速分析（传入真实数据）
        result = await quick_analysis_llm(stock_code, company_name, stock_data)

        # 使用 LLM 返回的公司名作为显示名
        display_name = result.get("company_name") or company_name

        # 行业估值基准匹配
        industry_benchmark = None
        raw_industry = result.get("industry", "")
        if raw_industry:
            matched = identify_industry(raw_industry) or raw_industry
            info = get_industry_info(matched)
            if info:
                industry_benchmark = {
                    "industry_name": matched,
                    "pe_reasonable_range": info.get("pe_reasonable_range", ""),
                    "pb_reasonable_range": info.get("pb_reasonable_range", ""),
                    "pe_cheap_threshold": info.get("pe_cheap_threshold"),
                    "pe_expensive_threshold": info.get("pe_expensive_threshold"),
                    "pb_cheap_threshold": info.get("pb_cheap_threshold"),
                    "pb_expensive_threshold": info.get("pb_expensive_threshold"),
                    "primary_valuation": info.get("primary_valuation", ""),
                    "scoring_notes": info.get("scoring_notes", ""),
                }

        return {
            "stock_code": stock_code,
            "stock_name": display_name,
            "industry_benchmark": industry_benchmark,
            **result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{ERROR_ICON} 快速查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/report")
async def trigger_report(query: StockQuery):
    """触发深度报告生成（异步），返回任务 ID"""
    try:
        company_name, stock_code = extract_stock_info(query.stock_input)

        if not stock_code and not company_name:
            raise HTTPException(status_code=400, detail="无法识别股票代码或名称")

        # 如果只有名称没有代码，尝试从缓存查找
        if not stock_code and company_name:
            found_code = _lookup_stock_code_by_name(company_name)
            if found_code:
                stock_code = normalize_code(found_code)
                logger.info(f"名称→代码: {company_name} → {stock_code}")

        if not stock_code:
            stock_code = query.stock_input.strip()
        else:
            stock_code = normalize_code(stock_code)

        if not company_name:
            company_name = query.stock_input.strip()

        task_id = str(uuid.uuid4())[:8]
        _task_statuses[task_id] = "pending"

        # 后台执行报告生成
        asyncio.create_task(generate_report_task(task_id, stock_code, company_name))

        return {"task_id": task_id, "status": "pending"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{ERROR_ICON} 报告触发失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/report/{task_id}")
async def get_report_status(task_id: str):
    """查询报告生成状态"""
    if task_id not in _task_statuses:
        raise HTTPException(status_code=404, detail="任务不存在")

    status = _task_statuses[task_id]

    if status == "completed":
        result = _task_results[task_id]
        return {
            "status": "completed",
            "report_content": result.get("report_content", ""),
            "report_path": result.get("report_path", ""),
            "report_pdf_path": result.get("report_pdf_path", ""),
            "company_name": result.get("company_name", ""),
            "stock_code": result.get("stock_code", ""),
            "generated_at": result.get("generated_at", "")
        }
    elif status == "failed":
        result = _task_results[task_id]
        return {
            "status": "failed",
            "error": result.get("error", "未知错误"),
            "company_name": result.get("company_name", ""),
            "stock_code": result.get("stock_code", "")
        }
    else:
        progress = _task_progress.get(task_id, 0.0)
        return {"status": status, "progress": progress}


@app.get("/api/pool/{term}")
async def get_pool(term: str):
    """获取指定期限股票池内容（term: short/medium/long）"""
    if term not in ("short", "medium", "long", "quick_screen"):
        raise HTTPException(status_code=400, detail="期限必须为 short/medium/long/quick_screen")

    if not _pool_manager:
        raise HTTPException(status_code=500, detail="股票池管理器未初始化")

    stocks = _pool_manager.list_stocks_by_term(term)
    return stocks


@app.post("/api/pool/{term}")
async def add_to_pool(term: str, request: PoolAddRequest):
    """向指定期限股票池添加股票"""
    if term not in ("short", "medium", "long", "quick_screen"):
        raise HTTPException(status_code=400, detail="期限必须为 short/medium/long/quick_screen")

    if not _pool_manager:
        raise HTTPException(status_code=500, detail="股票池管理器未初始化")

    stock_code = normalize_code(request.stock_code)
    stock = _pool_manager.add_stock_to_term(term, stock_code, request.stock_name)
    return {"status": "ok", "stock": stock}


@app.delete("/api/pool/{term}/{stock_code}")
async def remove_from_pool(term: str, stock_code: str):
    """从指定期限股票池删除股票"""
    if term not in ("short", "medium", "long", "quick_screen"):
        raise HTTPException(status_code=400, detail="期限必须为 short/medium/long/quick_screen")

    if not _pool_manager:
        raise HTTPException(status_code=500, detail="股票池管理器未初始化")

    stock_code = normalize_code(stock_code)
    success = _pool_manager.remove_stock_from_term(term, stock_code)
    if not success:
        raise HTTPException(status_code=404, detail=f"股票 {stock_code} 不在{term}池中")
    return {"status": "ok"}


@app.post("/api/score/{term}/{stock_code}")
async def score_stock(term: str, stock_code: str):
    """对指定股票执行指定限期打分"""
    if term not in ("short", "medium", "long", "quick_screen"):
        raise HTTPException(status_code=400, detail="期限必须为 short/medium/long/quick_screen")

    if not _pool_manager or not _scoring_engine:
        raise HTTPException(status_code=500, detail="服务未初始化")

    stock_code = normalize_code(stock_code)
    stock = _pool_manager.get_stock_in_term(term, stock_code)
    if not stock:
        raise HTTPException(status_code=404, detail=f"股票 {stock_code} 不在{term}池中")

    # 运行完整分析+打分，但只存储当前期限的评分
    result = await _scoring_engine.score_stock_for_term(term, stock_code, stock["company_name"])

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    term_score = result["term_score"]

    return {
        "score": term_score.get("score"),
        "score_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "term": term,
        "stock_code": stock_code,
        "company_name": stock["company_name"],
    }


# ──────────────────────────────────────────────
# 快筛股票池直接打分（绕过 MCP ReAct，HTTP数据+LLM直调）
# ──────────────────────────────────────────────

_SCORING_PROMPTS = {
    "short": """你是一位资深A股短线交易专家，专注于1-5个交易日的短线操作。

请基于以下真实股票数据，进行短线量化打分（总分100分）。

## 评分体系
- 量价关系(30分)：成交量变化、量价配合、换手率活跃度
- 技术信号(25分)：均线形态、MACD/RSI信号、K线形态
- 趋势动量(20分)：短期涨跌幅趋势、突破/回踩信号
- 情绪资金(25分)：新闻情绪、资金流向、市场关注度

## 输出格式（严格JSON，不要markdown代码块）
{"score": 整数0-100, "sub_scores": {"量价关系": 整数, "技术信号": 整数, "趋势动量": 整数, "情绪资金": 整数}, "rating": "买入/增持/观望/减持/卖出", "reasoning": "基于数据的打分理由(100字内)", "risk_warning": "短线风险提示(50字内)", "suggested_action": "建议操作", "time_horizon": "1-5个交易日"}""",

    "medium": """你是一位资深A股中线投资专家，专注于1-3个月的中线持仓。

请基于以下真实股票数据，进行中线量化打分（总分100分）。

## 评分体系
- 基本面质量(25分)：ROE、毛利率、净利率、现金流质量
- 成长性(15分)：营收/利润增速、行业对比
- 估值水平(20分)：PE/PB/PS vs 行业、历史分位、安全边际
- 技术趋势(15分)：中期均线、MACD趋势、关键价位
- 情绪资金(10分)：新闻情绪、机构关注度
- 风险评估(15分)：财务风险、行业风险、市场风险

## 输出格式（严格JSON，不要markdown代码块）
{"score": 整数0-100, "sub_scores": {"基本面质量": 整数, "成长性": 整数, "估值水平": 整数, "技术趋势": 整数, "情绪资金": 整数, "风险评估": 整数}, "rating": "买入/增持/观望/减持/卖出", "reasoning": "基于数据的打分理由(150字内)", "risk_warning": "中线风险提示(80字内)", "suggested_action": "建议操作", "time_horizon": "1-3个月"}""",

    "long": """你是一位资深A股长线价值投资专家，参考巴菲特、芒格的投资理念，专注于1-3年的长线投资。

请基于以下真实股票数据，进行长线量化打分（总分100分）。

## 评分体系
- 商业护城河(30分)：品牌/技术壁垒、规模效应、客户粘性、ROE持续性
- 行业景气度(20分)：行业空间、竞争格局、政策支持、技术替代风险
- 估值水平(20分)：历史PE/PB分位、绝对估值、股息率
- 成长性(15分)：3-5年CAGR预期、第二增长曲线
- 技术面(5分)：月线级别大趋势（权重极低）
- 风险与治理(10分)：公司治理、大股东行为、ESG风险

## 输出格式（严格JSON，不要markdown代码块）
{"score": 整数0-100, "sub_scores": {"商业护城河": 整数, "行业景气度": 整数, "估值水平": 整数, "成长性": 整数, "技术面": 整数, "风险与治理": 整数}, "rating": "买入/增持/观望/减持/卖出", "reasoning": "基于数据的打分理由(150字内)", "risk_warning": "长线风险提示(80字内)", "suggested_action": "建议操作", "time_horizon": "1-3年", "moat_type": "护城河类型"}""",
}


async def _direct_llm_score(
    term: str, stock_code: str, company_name: str, stock_data: Dict
) -> Dict[str, Any]:
    """绕过 MCP ReAct，直接 HTTP 数据 + LLM 打分（快筛专用）"""
    llm = OpenAICompatibleClient(
        api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY_2"),
        base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL_2"),
        model=os.getenv("OPENAI_COMPATIBLE_MODEL_2", "qwen3.6-flash"),
        env_prefix="",
        extra_body={"thinking": {"type": "enabled"}},
    )

    # 构建数据摘要
    data_text = json.dumps(stock_data, ensure_ascii=False, indent=2)

    # 行业估值基准匹配
    industry_guidance = ""
    industry_pe_range = ""
    industry_pb_range = ""
    raw_industry = stock_data.get("industry", "")
    if raw_industry:
        matched = identify_industry(raw_industry) or raw_industry
        info = get_industry_info(matched)
        guidance = get_industry_scoring_guidance(matched) if info else ""
        if guidance:
            industry_guidance = f"""
## 行业估值基准（⚠️ 必须参考以下行业标准进行打分，保证跨行业可比性）

{guidance}

**核心要求**:
- PE/PB 打分必须相对于该行业正常区间，不得使用绝对数值判断高低
- 银行 PE 6倍可能是合理估值，科技 PE 60倍也可能被低估，请严格参照上述行业基准
- 各维度权重不变，但每个维度内的打分参考标准随行业调整
"""
            industry_pe_range = info.get("pe_reasonable_range", "")
            industry_pb_range = info.get("pb_reasonable_range", "")
            logger.info(f"{WAIT_ICON} QuickScreen: {company_name} 行业匹配={matched}")

    # 提取关键事实清单（可引用数字白名单）
    pc = stock_data.get("price_changes", {}) or {}
    facts = []
    def _add(label, val, suffix=""):
        v = (val or "").strip()
        if v and v != "None" and v != "N/A":
            facts.append(f"{label}: {v}{suffix}")

    _add("PE", stock_data.get("pe"))
    _add("PB", stock_data.get("pb"))
    _add("PS", stock_data.get("ps"))
    _add("ROE", stock_data.get("roe"))
    _add("毛利率", stock_data.get("gross_margin"))
    _add("净利率", stock_data.get("net_margin"))
    _add("营收增速", stock_data.get("revenue_growth"))
    _add("利润增速", stock_data.get("profit_growth"))
    _add("负债率", stock_data.get("debt_ratio"))
    _add("换手率", stock_data.get("turnover_rate"))
    _add("总市值", stock_data.get("market_cap"))
    _add("近1日涨跌", pc.get("1d"))
    _add("近5日涨跌", pc.get("5d"))
    _add("近1月涨跌", pc.get("1m"))
    _add("近3月涨跌", pc.get("3m"))
    _add("近6月涨跌", pc.get("6m"))
    _add("近1年涨跌", pc.get("1y"))
    _add("近3年涨跌", pc.get("3y"))
    pe_pct = stock_data.get("pe_percentile", "")
    if pe_pct:
        facts.append(f"PE分位: {pe_pct}")
    if industry_pe_range:
        facts.append(f"行业PE正常区间: {industry_pe_range}")
    if industry_pb_range:
        facts.append(f"行业PB正常区间: {industry_pb_range}")

    fact_checklist = "\n".join(f"  - {f}" for f in facts) if facts else "（无可用数据）"

    # 方案A：数据引用规则（幻觉控制）
    citation_rules = """## ⚠️ 数据引用规则 — 最高优先级

你是分析师而非数据源。下方「核心事实」和「真实股票数据」是你唯一的数据来源。

在 reasoning / risk_warning / suggested_action 文本中：
- ✅ 可以引用：上述数据中明确出现的数字（包括行业基准）
- ❌ 严禁编造：数据中不存在的具体数字、百分比、排名、年份、历史分位值
- ❌ 严禁计算：不要从已知数字推导未知数字（如已知PE和股价推算EPS）
- 若某维度数据缺失，使用定性语言（如"估值偏高"），严禁猜测具体数值
- 引用时使用原数值，不要自行四舍五入或修改
- 违反以上规则等于向用户传播虚假信息"""

    prompt = _SCORING_PROMPTS[term]
    user_message = f"""{citation_rules}

## 核心事实（reasoning/risk_warning 中只能引用以下数字）
{fact_checklist}

**股票**: {company_name} ({stock_code})
**当前日期**: {datetime.now().strftime('%Y-%m-%d')}

## 真实股票数据
{data_text}
{industry_guidance}
{prompt}"""

    try:
        response = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                _thread_pool,
                lambda: llm.get_completion([
                    {"role": "system", "content": "你是一个严格遵循指令的A股量化分析师。只返回纯JSON，不得包含markdown代码块标记，不得编造数据。"},
                    {"role": "user", "content": user_message}
                ], max_retries=1)
            ),
            timeout=300.0  # qwen3.6-flash + thinking 生成结构化JSON可能需60-180秒
        )

        if response:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                score_data = json.loads(json_match.group())
                raw_score = score_data.get("score")
                # 安全转为 float（LLM 可能返回 int/float/str）
                try:
                    score = float(raw_score) if raw_score is not None else None
                except (ValueError, TypeError):
                    score = None
                logger.info(
                    f"{SUCCESS_ICON} QuickScreen直接打分: {company_name} {term}="
                    f"{score}分 ({score_data.get('rating', 'N/A')})"
                )
                return {
                    "score": score,
                    "score_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "term": term,
                    "stock_code": stock_code,
                    "company_name": company_name,
                    "recommendation": score_data.get("rating", ""),
                    "suggested_action": score_data.get("suggested_action", ""),
                    "reasoning": score_data.get("reasoning", ""),
                    "risk_warning": score_data.get("risk_warning", ""),
                }

        raise ValueError("LLM返回无法解析")

    except asyncio.TimeoutError:
        raise ValueError(f"快筛打分超时(300s): {company_name}")
    except Exception:
        raise


@app.post("/api/quick-screen/score/{term}/{stock_code}")
async def quick_screen_score(term: str, stock_code: str):
    """快筛股票池打分 — 绕过MCP ReAct，直连HTTP数据+LLM打分（qwen3.6-flash）"""
    if term not in ("short", "medium", "long"):
        raise HTTPException(status_code=400, detail="期限必须为 short/medium/long")

    if not _pool_manager:
        raise HTTPException(status_code=500, detail="服务未初始化")

    stock_code = normalize_code(stock_code)
    stock = _pool_manager.get_stock_in_term("quick_screen", stock_code)
    if not stock:
        raise HTTPException(status_code=404, detail=f"股票 {stock_code} 不在快筛股票池中")

    company_name = stock["company_name"]

    try:
        # 1. 直连 HTTP 获取股票数据（1-3秒，复用快速查询路径）
        stock_data = await fetch_stock_data(stock_code)
        company_name = stock_data.get("company_name") or company_name

        # 2. 构建打分 prompt + 调用 qwen3.6-flash
        result = await _direct_llm_score(term, stock_code, company_name, stock_data)

        # 3. 持久化打分结果到股票池
        _pool_manager.update_quick_screen_score(stock_code, term, {
            "score": result.get("score"),
            "score_time": result.get("score_time", ""),
            "recommendation": result.get("recommendation", ""),
            "suggested_action": result.get("suggested_action", ""),
            "reasoning": result.get("reasoning", ""),
            "risk_warning": result.get("risk_warning", ""),
        })

        return result
    except Exception as e:
        logger.error(f"{ERROR_ICON} 快筛直接打分失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cache/{stock_code}")
async def cache_status(stock_code: str):
    """查看某只股票的中间产物缓存状态"""
    stock_code = normalize_code(stock_code)
    return get_cache_status_for_stock(stock_code)


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "cache_dir": CACHE_DIR,
        "cache_files": len(glob_module.glob(os.path.join(CACHE_DIR, "*.json"))) if os.path.exists(CACHE_DIR) else 0,
    }


# ══════════════════════════════════════════════════════════════
# 批量打分端点 (Phase 1: Excel 解析 + 并行数据获取)
# ══════════════════════════════════════════════════════════════

_BATCH_JOBS_DIR = os.path.join(PROJECT_ROOT, "data", "batch_jobs")


def _ensure_batch_dir():
    os.makedirs(_BATCH_JOBS_DIR, exist_ok=True)


async def _run_batch_fetch(batch_id: str):
    """后台任务: 执行批量数据获取 → LLM 打分（全流程）"""
    from src.api.batch_scorer import (
        fetch_batch, get_job, update_job_progress, finalize_fetch, finalize_batch,
        score_batch, start_scoring_phase, update_scoring_progress,
    )
    job = get_job(batch_id)
    if not job:
        logger.error(f"批量任务 {batch_id} 不存在")
        return

    stocks = job["stocks"]
    total = len(stocks)
    horizon = job.get("horizon", "medium")

    # ── Stage 1: 数据获取 ──
    def on_fetch_progress(completed, t):
        update_job_progress(batch_id, completed, t)

    try:
        stocks = await fetch_batch(stocks, semaphore=6, on_progress=on_fetch_progress)
        finalize_fetch(batch_id)
    except Exception as e:
        logger.error(f"{ERROR_ICON} 批量数据获取失败: {batch_id} - {e}", exc_info=True)
        finalize_batch(batch_id, error=str(e))
        return

    fetched_ok = sum(1 for s in stocks if s.get("status") == "fetched")
    logger.info(f"{SUCCESS_ICON} 批量数据获取完成: {batch_id}, {fetched_ok}/{total} 成功")

    if fetched_ok == 0:
        finalize_batch(batch_id, error="所有股票数据获取均失败")
        return

    # ── Stage 2: LLM 打分 ──
    start_scoring_phase(batch_id)

    def on_score_progress(scored, t):
        update_scoring_progress(batch_id, scored)

    scoring_failed = False
    try:
        if horizon == "all":
            for term, label in [("short", "短线"), ("medium", "中线"), ("long", "长线")]:
                logger.info(f"{WAIT_ICON} 批量打分 - {label}维度: {batch_id}")
                start_scoring_phase(batch_id)
                stocks = await score_batch(
                    stocks, horizon=term, semaphore=5,
                    on_progress=on_score_progress
                )
                for s in stocks:
                    if s.get("score"):
                        s.setdefault("all_scores", {})[term] = s["score"]
        else:
            stocks = await score_batch(
                stocks, horizon=horizon, semaphore=5,
                on_progress=on_score_progress
            )
    except Exception as e:
        logger.error(f"{ERROR_ICON} 批量打分失败: {batch_id} - {e}", exc_info=True)
        finalize_batch(batch_id, error=f"打分阶段失败: {str(e)}")
        scoring_failed = True

    # ── 完成 ──
    if not scoring_failed:
        finalize_batch(batch_id)

    # 持久化到磁盘 (job 详情 + results 供前端加载)
    try:
        _ensure_batch_dir()
        path = os.path.join(_BATCH_JOBS_DIR, f"{batch_id}.json")
        job_data = get_job(batch_id) or job
        with open(path, "w", encoding="utf-8") as f:
            json.dump(job_data, f, ensure_ascii=False, indent=2, default=str)

        # 同时写入 batch_results.json（前端直接读取，不再依赖前端轮询存盘）
        _results_path = os.path.join(PROJECT_ROOT, "data", "batch_results.json")
        results_out = []
        for s in stocks:
            data = s.get("data", {}) or {}
            score = s.get("score", {}) or {}
            results_out.append({
                "code": s.get("code", ""),
                "name": s.get("name", ""),
                "pe": data.get("pe", ""), "pb": data.get("pb", ""),
                "roe": data.get("roe", ""), "industry": data.get("industry", ""),
                "market_cap": data.get("market_cap", ""),
                "price_changes": data.get("price_changes", {}),
                "level": score.get("level", "中性"),
                "confidence": score.get("confidence", ""),
                "reason": score.get("reason", ""), "risk": score.get("risk", ""),
            })
        with open(_results_path, "w", encoding="utf-8") as f:
            json.dump(results_out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"持久化批量任务失败: {e}")

    scored_ok = sum(1 for s in stocks if s.get("score"))
    logger.info(
        f"{SUCCESS_ICON} 批量打分全流程完成: {batch_id}, "
        f"数据 {fetched_ok}/{total}, 打分 {scored_ok}/{fetched_ok}"
    )


@app.post("/api/batch-score/upload")
async def batch_score_upload(
    file: UploadFile = File(...),
    horizon: str = "medium",
):
    """上传 Excel 并启动批量打分任务。

    Args:
        file: .xlsx 文件 (股票代码, 股票名称)
        horizon: 打分维度 (short/medium/long/all)，默认 medium

    Returns:
        {batch_id, total_stocks, stocks: [{code, name}, ...]}
    """
    if horizon not in ("short", "medium", "long", "all"):
        raise HTTPException(status_code=400, detail="horizon 必须为 short/medium/long/all")

    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="仅支持 .xlsx 格式文件")

    try:
        from src.api.batch_scorer import (
            parse_excel, lookup_names, create_batch_job, finalize_batch,
        )

        # 1. 读取文件（限制 10MB）
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="文件大小超过 10MB 限制")

        stocks = parse_excel(content)

        # 2. 补全缺失的名称
        stocks = lookup_names(stocks)

        # 3. 验证 LLM 配置
        if not os.getenv("OPENAI_COMPATIBLE_API_KEY_2"):
            raise HTTPException(
                status_code=500,
                detail="批量打分需要配置 OPENAI_COMPATIBLE_API_KEY_2 环境变量"
            )

        # 4. 创建任务
        batch_id = create_batch_job(stocks, horizon=horizon)

        # 5. 安全启动后台任务（带总超时 + 防休眠保护）
        # 性能预算: 500只≈8分钟, 1000只≈16分钟, 超时按 3x 预算留足余量
        total_stocks = len(stocks)
        overall_timeout = max(3600, min(14400, total_stocks * 10))  # 最少1小时，最多4小时

        async def _safe_batch_task():
            # Windows: 阻止系统在批量打分期间自动休眠（合盖只关屏幕）
            _prevent_sleep = None
            if sys.platform == "win32":
                try:
                    import ctypes
                    ES_CONTINUOUS = 0x80000000
                    ES_SYSTEM_REQUIRED = 0x00000001
                    ctypes.windll.kernel32.SetThreadExecutionState(
                        ES_CONTINUOUS | ES_SYSTEM_REQUIRED
                    )
                    _prevent_sleep = True
                    logger.info(f"已启用防休眠保护 (batch: {batch_id})")
                except Exception:
                    pass

            try:
                await asyncio.wait_for(
                    _run_batch_fetch(batch_id),
                    timeout=overall_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(f"批量任务 {batch_id} 总超时 ({overall_timeout}s)，强制终止")
                finalize_batch(batch_id, error=f"任务总超时 ({overall_timeout // 60} 分钟)")
            except asyncio.CancelledError:
                logger.warning(f"批量任务 {batch_id} 被取消")
                finalize_batch(batch_id, error="任务被系统取消")
            except Exception as e:
                logger.error(f"批量任务 {batch_id} 未捕获异常: {e}", exc_info=True)
                finalize_batch(batch_id, error=f"系统错误: {str(e)}")
            finally:
                # 恢复系统默认休眠策略
                if _prevent_sleep:
                    try:
                        import ctypes
                        ES_CONTINUOUS = 0x80000000
                        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                        logger.info(f"已恢复系统休眠策略 (batch: {batch_id})")
                    except Exception:
                        pass

        asyncio.create_task(_safe_batch_task())

        logger.info(
            f"{SUCCESS_ICON} 批量打分任务已启动: {batch_id} "
            f"({len(stocks)} 只, {horizon})"
        )

        return {
            "batch_id": batch_id,
            "horizon": horizon,
            "total_stocks": len(stocks),
            "stocks": [{"code": s["code"], "name": s.get("name", "")}
                       for s in stocks],
            "status": "fetching",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"{ERROR_ICON} 批量打分上传失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"上传处理失败: {str(e)}")


@app.get("/api/batch-score/{batch_id}/progress")
async def batch_score_progress(batch_id: str):
    """查询批量打分任务进度。

    Returns:
        {batch_id, status, total_stocks, fetched_count, scored_count,
         progress_pct, elapsed_seconds, error (if failed)}
    """
    from src.api.batch_scorer import get_job

    job = get_job(batch_id)
    if not job:
        # 尝试从磁盘恢复
        _ensure_batch_dir()
        path = os.path.join(_BATCH_JOBS_DIR, f"{batch_id}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    job = json.load(f)
            except Exception:
                pass

    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {batch_id} 不存在")

    status = job.get("status", "unknown")
    total = job.get("total_stocks", 0)
    fetched = job.get("fetched_count", 0)
    scored = job.get("scored_count", 0)

    elapsed = 0
    try:
        created = datetime.fromisoformat(job["created_at"])
        elapsed = (datetime.now() - created).total_seconds()
    except Exception:
        pass

    # 进度估算 (Phase 1 数据获取占 60%, Phase 2 打分占 40%)
    # 用实际计数而非状态字段，避免状态过渡瞬间读到旧值导致进度"卡住"
    fetch_pct = round(fetched / max(total, 1) * 60, 1) if total else 0
    score_pct = round(scored / max(total, 1) * 40, 1) if total else 0

    if status == "completed":
        progress_pct = 100.0
    elif fetched >= total or status in ("fetched", "scoring"):
        # 数据获取已完成，按实际打分进度
        progress_pct = min(99.0, 60.0 + score_pct)
    else:
        progress_pct = min(60.0, fetch_pct)

    return {
        "batch_id": batch_id,
        "status": status,
        "total_stocks": total,
        "fetched_count": fetched,
        "scored_count": scored,
        "progress_pct": progress_pct,
        "elapsed_seconds": int(elapsed),
        "horizon": job.get("horizon", ""),
        "error": job.get("error"),
    }


@app.get("/api/batch-score/{batch_id}/results")
async def batch_score_results(batch_id: str):
    """查询批量打分结果（实时返回，不阻塞）。

    数据获取中 → 202 提示进度
    数据已获取 → 返回原始数据
    打分完成 → 返回数据 + 分数

    Returns:
        {batch_id, status, total_stocks, stocks: [{code, name, data, score}]}
    """
    from src.api.batch_scorer import get_job

    job = get_job(batch_id)
    if not job:
        _ensure_batch_dir()
        path = os.path.join(_BATCH_JOBS_DIR, f"{batch_id}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    job = json.load(f)
            except Exception:
                pass

    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {batch_id} 不存在")

    status = job.get("status", "unknown")
    horizon = job.get("horizon", "")
    total = job.get("total_stocks", 0)

    # 构建返回的股票列表
    stocks_out = []
    for s in job.get("stocks", []):
        entry = {
            "code": s.get("code", ""),
            "name": s.get("name", ""),
            "status": s.get("status", "unknown"),
        }
        data = s.get("data", {})
        if data:
            entry["pe"] = data.get("pe", "")
            entry["pb"] = data.get("pb", "")
            entry["roe"] = data.get("roe", "")
            entry["industry"] = data.get("industry", "")
            entry["market_cap"] = data.get("market_cap", "")
            entry["price_changes"] = data.get("price_changes", {})
        score = s.get("score")
        if score:
            entry["level"] = score.get("level", "中性")
            entry["confidence"] = score.get("confidence", "")
            entry["reason"] = score.get("reason", "")
            entry["risk"] = score.get("risk", "")
        if s.get("all_scores"):
            # 全维度模式
            entry["all_scores"] = {}
            for term, sc in s["all_scores"].items():
                entry["all_scores"][term] = {
                    "level": sc.get("level", "中性"),
                    "reason": sc.get("reason", ""),
                }
        stocks_out.append(entry)

    return {
        "batch_id": batch_id,
        "status": status,
        "horizon": horizon,
        "total_stocks": total,
        "fetched_count": job.get("fetched_count", 0),
        "scored_count": job.get("scored_count", 0),
        "stocks": stocks_out,
        "created_at": job.get("created_at", ""),
        "error": job.get("error"),
    }
