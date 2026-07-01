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
from fastapi.responses import StreamingResponse
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
from src.utils.model_config import get_thinking_body
from src.stock_pool.stock_pool_manager import StockPoolManager
from src.stock_pool.scoring_engine import ScoringEngine
from src.tools.mcp_client import get_mcp_tools
from src.utils.industry_knowledge import identify_industry, get_industry_info, get_industry_scoring_guidance

# QA 引擎
from src.qa.qa_engine import process_question
from src.qa.session_manager import get_session_manager

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

class FundSearchRequest(BaseModel):
    keyword: str

class FundReportRequest(BaseModel):
    fund_code: str
    fund_name: str = ""
    mode: str = "report"  # "report" or "score"

class FundPoolAddRequest(BaseModel):
    stock_code: str = ""   # backward-compat with stock pool API
    stock_name: str = ""   # backward-compat with stock pool API
    fund_code: str = ""    # fund-zone preferred field name
    fund_name: str = ""    # fund-zone preferred field name

# ─── 智能投顾请求模型 ───

class AdvisoryRecommendRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class PortfolioCreateRequest(BaseModel):
    name: str
    initial_capital: float = 100000.0

class HoldingModifyRequest(BaseModel):
    portfolio_id: str
    stock_code: str
    company_name: str = ""
    quantity: int
    price: float
    action: str  # "buy" | "sell"

class StrategyBindRequest(BaseModel):
    portfolio_id: str
    strategy_name: str
    params: Optional[Dict[str, Any]] = None

class BacktestRequest(BaseModel):
    portfolio_id: str
    start_date: str
    end_date: str
    strategy_name: Optional[str] = None
    strategy_params: Optional[Dict[str, Any]] = None

class SimulationRequest(BaseModel):
    portfolio_id: str
    action: str  # "start" | "stop" | "status" | "catch_up"

class FreeLineRequest(BaseModel):
    portfolio_id: str
    action: str  # "start" | "status" | "decide"

class ReportRequest(BaseModel):
    portfolio_id: str
    report_type: str  # "backtest" | "simulation"
    start_date: str = ""
    end_date: str = ""
    include_deepseek: bool = False

# ──────────────────────────────────────────────
# 全局状态
# ──────────────────────────────────────────────
_task_results: Dict[str, Dict[str, Any]] = {}
_task_statuses: Dict[str, str] = {}
_task_progress: Dict[str, float] = {}  # task_id → 0.0~1.0
_score_tasks: Dict[str, Dict[str, Any]] = {}  # task_id → {status, result, ...}
_quick_score_tasks: Dict[str, Dict[str, Any]] = {}  # task_id → {status, result, ...}
_score_all_tasks: Dict[str, Dict[str, Any]] = {}  # task_id → {status, result, ...}
_scoring_engine: Optional[ScoringEngine] = None
_pool_manager: Optional[StockPoolManager] = None

# 基金任务跟踪
_fund_tasks: Dict[str, Dict[str, Any]] = {}
_fund_pool_manager = None  # lazy init

def _get_fund_pool_manager():
    """延迟初始化基金池管理器。优先使用 src.fund_pool.fund_pool_manager，fallback 到简单本地实现。"""
    global _fund_pool_manager
    if _fund_pool_manager is None:
        try:
            from src.fund_pool.fund_pool_manager import FundPoolManager
            _fund_pool_manager = FundPoolManager()
        except ImportError:
            _fund_pool_manager = _SimpleFundPoolManager()
    return _fund_pool_manager


# ─── 智能投顾模块懒加载 ───

_advisory_modules: Optional[Dict[str, Any]] = None

def _init_advisory() -> Dict[str, Any]:
    """延迟初始化智能投顾模块，避免拖慢启动速度。"""
    global _advisory_modules
    if _advisory_modules is None:
        try:
            from src.advisory.portfolio_manager import PortfolioManager
            from src.advisory.recommendation import get_recommendation_engine
            from src.advisory.user_profile import UserProfileManager
            from src.advisory.strategy_engine import StrategyEngine
            from src.advisory.backtest_runner import BacktestRunner
            from src.advisory.simulation_runner import SimulationRunner
            from src.advisory.report_generator import AdvisoryReportGenerator

            # 确保数据目录存在
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            portfolios_dir = os.path.join(root, "data", "portfolios")
            settlements_dir = os.path.join(root, "data", "advisory_settlements")
            os.makedirs(portfolios_dir, exist_ok=True)
            os.makedirs(settlements_dir, exist_ok=True)

            pm = PortfolioManager(data_dir=portfolios_dir)
            sr = SimulationRunner(
                portfolio_manager=pm,
                settlement_dir=settlements_dir,
            )

            _advisory_modules = {
                "portfolio_manager": pm,
                "recommendation_engine": get_recommendation_engine(),
                "user_profile_manager": UserProfileManager(),
                "strategy_engine": StrategyEngine,
                "backtest_runner": BacktestRunner(portfolio_manager=pm),
                "simulation_runner": sr,
                "report_generator": AdvisoryReportGenerator(),
            }
            logger.info(f"{SUCCESS_ICON} 智能投顾模块初始化完成")
        except ImportError as e:
            logger.error(f"{ERROR_ICON} 智能投顾模块导入失败: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"智能投顾模块不可用，请确认依赖已安装: {e}",
            )
    return _advisory_modules


class _SimpleFundPoolManager:
    """简单的基金池管理器 fallback（文件持久化），当 FundPoolManager 不可用时使用。"""

    def __init__(self):
        import json as _json
        self._pool_file = os.path.join(PROJECT_ROOT, "data", "fund_pool.json")
        self._pools: Dict[str, Dict[str, Any]] = {}

    def _load(self):
        import json as _json
        if os.path.exists(self._pool_file):
            try:
                with open(self._pool_file, "r", encoding="utf-8") as f:
                    loaded = _json.load(f)
                    if isinstance(loaded, dict):
                        self._pools = loaded
            except Exception:
                pass

    def _save(self):
        import json as _json
        os.makedirs(os.path.dirname(self._pool_file), exist_ok=True)
        with open(self._pool_file, "w", encoding="utf-8") as f:
            _json.dump(self._pools, f, ensure_ascii=False, indent=2)

    def list_funds(self, pool: str = "scored") -> list:
        self._load()
        pool_data = self._pools.get(pool, {})
        if isinstance(pool_data, dict):
            return list(pool_data.values())
        return []

    def add_fund(self, fund_code: str, fund_name: str, pool: str = "scored"):
        self._load()
        self._pools.setdefault(pool, {})[fund_code] = {
            "fund_code": fund_code,
            "fund_name": fund_name,
            "added_at": datetime.now().isoformat(),
            "pool": pool,
        }
        self._save()

    def remove_fund(self, fund_code: str, pool: str = "scored") -> bool:
        self._load()
        pool_data = self._pools.get(pool, {})
        if fund_code in pool_data:
            del pool_data[fund_code]
            self._pools[pool] = pool_data
            self._save()
            return True
        return False

    def get_fund(self, fund_code: str, pool: str = "scored") -> Optional[Dict]:
        self._load()
        return self._pools.get(pool, {}).get(fund_code)

    def update_score(self, fund_code: str, score_data: Dict, pool: str = "scored"):
        self._load()
        pool_data = self._pools.setdefault(pool, {})
        if fund_code in pool_data:
            pool_data[fund_code].update({
                "score": score_data.get("score"),
                "score_time": score_data.get("score_time", datetime.now().isoformat()),
                "recommendation": score_data.get("recommendation", ""),
                "status": "scored",
            })
        self._save()


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


def get_fund_cache_status(fund_code: str) -> Dict[str, Any]:
    """查看某只基金的中间产物缓存状态"""
    ensure_cache_dir()
    cached = {}
    for filepath in glob_module.glob(os.path.join(CACHE_DIR, f"fund_*_{fund_code}_*.json")):
        filename = os.path.basename(filepath)
        parts = filename.replace(".json", "").split("_")
        # 解析文件名: fund_{agent_name}_{fund_code}_{date}.json
        if len(parts) >= 4:
            agent_name = "_".join(parts[1:-2])
            date_str = parts[-1]
            cached[agent_name] = date_str
    return {"fund_code": fund_code, "cached_agents": cached}


# ──────────────────────────────────────────────
# 股票代码规范化
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
    """返回交易所前缀 (sh/sz/bj) 用于 Tencent/Sina API"""
    if _is_bse_code(pure_code):
        return "bj"
    if pure_code.startswith(("6", "688", "5")):
        return "sh"
    elif pure_code.startswith(("0", "3", "1", "4")):
        return "sz"
    return "sz"


def normalize_code(code: str) -> str:
    code = code.strip()
    if code.startswith("sh.") or code.startswith("sz.") or code.startswith("bj."):
        return code
    prefix = _get_exchange_prefix(code)
    return f"{prefix}.{code}"


def extract_stock_info(user_input: str) -> tuple:
    """从用户输入提取股票代码和名称"""
    user_input = user_input.strip()

    # 提取股票代码（6位数字；数字边界避免中文粘连时\b失效）
    code_match = re.search(r'(?<!\d)(\d{5,6})(?!\d)', user_input)
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
    低优先级字段（十大股东、基金持仓等）由深度报告覆盖。
    ETF代码(51/58/15/16/18开头)跳过股票专属字段(PE/PB/ROE/分红/EV_EBITDA)。"""
    # 检测是否为 ETF/基金代码
    _is_etf = stock_code.replace("sh.", "").replace("sz.", "").startswith(("51", "58", "15", "16", "18"))
    try:
        from src.utils.tushare_client import (
            get_stock_info, get_daily_basic, get_fina_indicator, get_dividend,
            compute_pe_percentile, compute_ev_ebitda, get_stock_news_em,
            get_fund_basic, get_fund_daily, get_fund_adj,
        )
        ts_code = stock_code.replace("sh.", "").replace("sz.", "")
        if _is_bse_code(ts_code):
            ts_code = f"{ts_code}.BJ"
        elif ts_code.startswith(("6", "688", "5")):
            ts_code = f"{ts_code}.SH"
        else:
            ts_code = f"{ts_code}.SZ"

        # 1. 行业分类（ETF 无行业字段，跳过）
        if not _is_etf and not existing.get("industry"):
            try:
                info = get_stock_info(ts_code)
                if info:
                    existing["industry"] = info.get("industry", "") or ""
            except Exception:
                pass

        # 2. 估值补充: PS, PE/PB确认（ETF 无 PE/PB/PS，跳过）
        if not _is_etf:
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
                if not existing.get("ps"):
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

        # 3. 财务指标摘要（ETF 无财务报表，跳过）
        if not _is_etf:
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

        # 4. 分红（ETF 无分红数据，跳过）
        if not _is_etf:
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

        # 5. EV/EBITDA（ETF 无 EBITDA，跳过）
        if not _is_etf:
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

        # ── ETF 专属数据补充 ──
        if _is_etf:
            # 2E. ETF基础信息
            try:
                fund_info = get_fund_basic(ts_code)
                if fund_info:
                    existing["industry"] = fund_info.get("index_name", "") or "ETF基金"
                    existing["etf_info"] = (
                        f"名称: {fund_info.get('name', '')}, "
                        f"管理公司: {fund_info.get('management', '')}, "
                        f"类型: {fund_info.get('fund_type', '')}, "
                        f"成立日: {fund_info.get('setup_date', '')}, "
                        f"跟踪指数: {fund_info.get('index_name', '') or fund_info.get('index_code', '')}"
                    )
                    if fund_info.get("m_fee"):
                        existing["etf_fee"] = f"管理费{fund_info['m_fee']}%, 托管费{fund_info.get('c_fee', '')}%"
            except Exception:
                pass

            # 3E. ETF历史行情（补充成交量、涨跌幅趋势）
            try:
                fund_kline = get_fund_daily(ts_code, days=120)
                if fund_kline and isinstance(fund_kline, list) and len(fund_kline) >= 5:
                    latest = fund_kline[0]
                    # 近期涨跌幅汇总
                    chg_5d = sum(float(d.get("pct_chg", 0) or 0) for d in fund_kline[:5])
                    chg_20d = sum(float(d.get("pct_chg", 0) or 0) for d in fund_kline[:20]) if len(fund_kline) >= 20 else None
                    chg_60d = sum(float(d.get("pct_chg", 0) or 0) for d in fund_kline[:60]) if len(fund_kline) >= 60 else None
                    avg_vol_20d = sum(float(d.get("vol", 0) or 0) for d in fund_kline[:20]) / min(20, len(fund_kline))
                    recent_high = max(float(d.get("high", 0) or 0) for d in fund_kline[:20])
                    recent_low = min(float(d.get("low", 0) or 0) for d in fund_kline[:20])

                    existing["etf_kline_summary"] = (
                        f"最新收盘: {latest.get('close', '')}, "
                        f"5日涨跌: {chg_5d:+.2f}%, "
                        f"20日涨跌: {chg_20d:+.2f}%" if chg_20d is not None else f"最新收盘: {latest.get('close', '')}, 5日涨跌: {chg_5d:+.2f}%"
                    )
                    if chg_60d is not None:
                        existing["etf_kline_summary"] += f", 60日涨跌: {chg_60d:+.2f}%"
                    existing["etf_vol_summary"] = (
                        f"20日均成交量: {avg_vol_20d/1e4:.0f}万手, "
                        f"20日高: {recent_high:.3f}, 低: {recent_low:.3f}"
                    )
                    # 供LLM用的行情趋势描述
                    existing["etf_trend"] = (
                        f"近5日累计{chg_5d:+.2f}%, "
                        f"近20日累计{chg_20d:+.2f}%" if chg_20d is not None else f"近5日累计{chg_5d:+.2f}%"
                    )
                    if chg_60d is not None:
                        existing["etf_trend"] += f", 近60日累计{chg_60d:+.2f}%"
            except Exception:
                pass

            # 4E. ETF复权因子（用于计算前复权价格趋势）
            try:
                fund_adj = get_fund_adj(ts_code, days=500)
                if fund_adj and isinstance(fund_adj, list) and len(fund_adj) >= 10:
                    # 用复权因子 * 收盘价 计算前复权价格趋势
                    # 先从已有的fund_daily获取收盘价
                    fund_kline_for_adj = get_fund_daily(ts_code, days=500)
                    if fund_kline_for_adj and len(fund_kline_for_adj) >= 10:
                        # 建立日期→复权因子映射
                        adj_map = {d["trade_date"]: float(d.get("adj_factor", 1) or 1) for d in fund_adj}
                        # 计算前复权价 = 收盘价 * 当日adj / 最新adj
                        latest_adj_val = adj_map.get(fund_kline_for_adj[0]["trade_date"], 1)
                        def _qfq_price(day_data):
                            raw_close = float(day_data.get("close", 0) or 0)
                            day_adj = adj_map.get(day_data["trade_date"], latest_adj_val)
                            return raw_close * day_adj / latest_adj_val if latest_adj_val > 0 else raw_close
                        qfq_latest = _qfq_price(fund_kline_for_adj[0])
                        chg_60d = (qfq_latest / _qfq_price(fund_kline_for_adj[min(60, len(fund_kline_for_adj)-1)]) - 1) * 100 if len(fund_kline_for_adj) >= 2 else 0
                        chg_120d = (qfq_latest / _qfq_price(fund_kline_for_adj[min(120, len(fund_kline_for_adj)-1)]) - 1) * 100 if len(fund_kline_for_adj) >= 2 else 0
                        chg_250d = (qfq_latest / _qfq_price(fund_kline_for_adj[min(250, len(fund_kline_for_adj)-1)]) - 1) * 100 if len(fund_kline_for_adj) >= 2 else 0
                        parts = [f"前复权收益 — 60日{chg_60d:+.1f}%"]
                        if len(fund_kline_for_adj) >= 120:
                            parts.append(f"120日{chg_120d:+.1f}%")
                        if len(fund_kline_for_adj) >= 250:
                            parts.append(f"250日{chg_250d:+.1f}%")
                        existing["etf_adj_return"] = ", ".join(parts)
            except Exception:
                pass

            # 标记为ETF，避免LLM误判缺少ROE/分红等字段为数据缺失
            existing["_asset_type"] = "ETF"

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
    _base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL_2")
    llm = OpenAICompatibleClient(
        api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY_2"),
        base_url=_base_url,
        model=os.getenv("OPENAI_COMPATIBLE_MODEL_2", "qwen3.6-flash"),
        env_prefix="",  # 使用显式参数，不从环境变量自动读取
        extra_body=get_thinking_body(_base_url, False),
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
        from src.agents.event_analyst_agent import event_analyst_agent
        from src.agents.quality_risk_analyst_agent import quality_risk_analyst_agent
        from src.agents.moneyflow_analyst_agent import moneyflow_analyst_agent

        workflow = StateGraph(AgentState)
        workflow.add_node("start_node", lambda state: state)
        workflow.add_node("fundamental_analyst", fundamental_agent)
        workflow.add_node("technical_analyst", technical_agent)
        workflow.add_node("value_analyst", value_agent)
        workflow.add_node("news_analyst", news_agent)
        workflow.add_node("event_analyst", event_analyst_agent)
        workflow.add_node("quality_risk_analyst", quality_risk_analyst_agent)
        workflow.add_node("moneyflow_analyst", moneyflow_analyst_agent)
        workflow.add_node("summarizer", summary_agent)

        workflow.set_entry_point("start_node")
        workflow.add_edge("start_node", "fundamental_analyst")
        workflow.add_edge("start_node", "technical_analyst")
        workflow.add_edge("start_node", "value_analyst")
        workflow.add_edge("start_node", "news_analyst")
        workflow.add_edge("start_node", "event_analyst")
        workflow.add_edge("start_node", "quality_risk_analyst")
        workflow.add_edge("start_node", "moneyflow_analyst")
        workflow.add_edge("fundamental_analyst", "summarizer")
        workflow.add_edge("technical_analyst", "summarizer")
        workflow.add_edge("value_analyst", "summarizer")
        workflow.add_edge("news_analyst", "summarizer")
        workflow.add_edge("event_analyst", "summarizer")
        workflow.add_edge("quality_risk_analyst", "summarizer")
        workflow.add_edge("moneyflow_analyst", "summarizer")
        workflow.add_edge("summarizer", END)

        app = workflow.compile()
        logger.info(f"{WAIT_ICON} 报告Pipeline开始: {company_name}({stock_code}), 超时=2100s")

        # 使用 astream 追踪真实进度（9个节点：start + 7分析 + summarizer）
        completed_nodes = set()
        total_nodes = 9
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
# 基金分析流水线（后台异步）
# ──────────────────────────────────────────────

async def _run_fund_analysis_pipeline(task_id: str, fund_code: str, fund_name: str, mode: str):
    """后台异步执行基金分析流水线。

    mode="report": 7Agent分析 → merge → 报告生成
    mode="score": 7Agent分析 → merge → 综合打分
    """
    try:
        _fund_tasks[task_id]["status"] = "running"

        # 1. 构建初始状态
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_date_cn = now.strftime("%Y年%m月%d日")
        current_weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
        current_time = now.strftime("%H:%M:%S")
        current_time_info = f"{current_date_cn} ({current_date}) {current_weekday_cn} {current_time}"

        initial_state = {
            "messages": [],
            "data": {
                "fund_code": fund_code,
                "fund_name": fund_name,
                "query": f"分析基金{fund_name}",
                "current_date": current_date,
                "current_date_cn": current_date_cn,
                "current_time": current_time,
                "current_weekday_cn": current_weekday_cn,
                "current_time_info": current_time_info,
            },
            "metadata": {},
        }

        # 2. 构建 LangGraph 工作流
        from langgraph.graph import StateGraph, END
        from src.agents.fund_product_doc_agent import fund_product_doc_agent
        from src.agents.fund_perf_risk_agent import fund_perf_risk_agent
        from src.agents.fund_holdings_agent import fund_holdings_analysis
        from src.agents.fund_manager_agent import fund_manager_agent
        from src.agents.fund_benchmark_agent import fund_benchmark_agent
        from src.agents.fund_fee_agent import fund_fee_agent
        from src.agents.fund_event_agent import fund_event_agent
        from src.agents.fund_merge_node import fund_merge_node
        from src.agents.fund_report_agent import fund_report_agent

        workflow = StateGraph(AgentState)

        # 7个并行分析Agent
        fund_analyst_nodes = {
            "fund_product_doc": fund_product_doc_agent,
            "fund_perf_risk": fund_perf_risk_agent,
            "fund_holdings": fund_holdings_analysis,
            "fund_manager": fund_manager_agent,
            "fund_benchmark": fund_benchmark_agent,
            "fund_fee": fund_fee_agent,
            "fund_event": fund_event_agent,
        }

        workflow.add_node("start_node", lambda state: state)
        for node_name, agent_fn in fund_analyst_nodes.items():
            workflow.add_node(node_name, agent_fn)
        workflow.add_node("fund_merge", fund_merge_node)

        # 根据 mode 选择最终节点
        if mode == "report":
            workflow.add_node("final_node", fund_report_agent)
        else:
            # score 模式：包装 fund_scoring_agent（签名不同，不接收 state）
            async def fund_score_wrapper(state: AgentState) -> Dict[str, Any]:
                from src.agents.fund_scoring_agent import fund_scoring_agent
                pkg = state.get("data", {}).get("fund_analysis_package", {})
                fc = state.get("data", {}).get("fund_code", "")
                fn = state.get("data", {}).get("fund_name", "")
                cd = state.get("data", {}).get("current_date", "")
                ft = (pkg.get("fund_profile", {}) or {}).get("fund_type", "") or state.get("data", {}).get("fund_type", "") or "未知类型"
                result = await fund_scoring_agent(
                    fund_analysis_package=pkg,
                    fund_code=fc, fund_name=fn, fund_type=ft,
                    current_date=cd, thinking_enabled=True,
                )
                return {"data": {"fund_score": result}}
            workflow.add_node("final_node", fund_score_wrapper)

        # 连线：start → 7 parallel → merge → final
        workflow.set_entry_point("start_node")
        for node_name in fund_analyst_nodes:
            workflow.add_edge("start_node", node_name)
            workflow.add_edge(node_name, "fund_merge")
        workflow.add_edge("fund_merge", "final_node")
        workflow.add_edge("final_node", END)

        app = workflow.compile()
        logger.info(f"{WAIT_ICON} 基金分析Pipeline开始: {fund_name}({fund_code}), mode={mode}")

        # 3. 执行并追踪进度
        completed_nodes = set()
        total_nodes = len(fund_analyst_nodes) + 3  # start + 7 agents + merge + final
        final_state = None

        async for chunk in app.astream(initial_state, config={"recursion_limit": 50}):
            for node_name, node_state in chunk.items():
                completed_nodes.add(node_name)
                logger.info(
                    f"基金Pipeline节点完成: {node_name} "
                    f"({len(completed_nodes)}/{total_nodes})"
                )
                if node_name == "final_node":
                    final_state = node_state

        if final_state is None:
            raise ValueError("基金Pipeline未返回final_node结果")

        # 4. 提取结果
        result_data = final_state.get("data", {})

        if mode == "report":
            report_content = result_data.get("fund_report", "")
            report_path = result_data.get("fund_report_path", "")
            _fund_tasks[task_id] = {
                "status": "completed",
                "mode": "report",
                "fund_code": fund_code,
                "fund_name": fund_name,
                "report_content": report_content,
                "report_path": report_path,
                "report_pdf_path": result_data.get("fund_report_pdf_path", ""),
                "generated_at": datetime.now().isoformat(),
            }
        else:
            score_data = result_data.get("fund_score", {})
            _fund_tasks[task_id] = {
                "status": "completed",
                "mode": "score",
                "fund_code": fund_code,
                "fund_name": fund_name,
                "score": score_data,
                "generated_at": datetime.now().isoformat(),
            }

            # 持久化评分到基金池
            try:
                mgr = _get_fund_pool_manager()
                overall = score_data.get("overall_score", {})
                holding = score_data.get("holding_period_suggestion", {})
                highlights = score_data.get("highlights", {})
                mgr.update_score(fund_code, {
                    "score": overall.get("score"),
                    "rating": overall.get("rating_label", ""),
                    "investment_view": overall.get("investment_view", ""),
                    "holding_period": holding.get("label", "") if isinstance(holding, dict) else "",
                    "subscores": score_data.get("subscores", {}),
                    "strengths": highlights.get("strengths", []),
                    "risks": highlights.get("risks", []),
                }, pool="scored")
            except Exception as e:
                logger.warning(f"持久化基金评分到池失败: {e}")

        logger.info(f"{SUCCESS_ICON} 基金分析完成: {fund_name}({fund_code}), mode={mode}")

    except Exception as e:
        logger.error(f"{ERROR_ICON} 基金分析失败: {e}", exc_info=True)
        _fund_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "fund_code": fund_code,
            "fund_name": fund_name,
            "mode": mode,
        }


# ──────────────────────────────────────────────
# FastAPI 应用生命周期
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool_manager, _scoring_engine
    import time
    logger.info(f"{SUCCESS_ICON} 启动 FastAPI 后端...")
    _pool_manager = StockPoolManager()
    _scoring_engine = ScoringEngine(pool_manager=_pool_manager)
    clean_old_cache()
    # 清理过期QA数据缓存
    try:
        from src.qa.session_manager import clean_expired_cache
        clean_expired_cache()
    except Exception:
        pass
    # 预加载名称缓存（阻塞至加载完成，避免请求在缓存未就绪时到达导致名称查找失败）
    await asyncio.get_running_loop().run_in_executor(_thread_pool, _ensure_name_cache)
    # MCP 预热：后台预初始化 MCP 客户端，消除首问 3-10s 启动延迟
    try:
        from src.tools.mcp_client import get_mcp_tools
        logger.info(f"{WAIT_ICON} 预初始化 MCP 客户端（后台预热）...")
        warmup_start = time.time()
        warmup_tools = await get_mcp_tools()
        if warmup_tools:
            logger.info(
                f"{SUCCESS_ICON} MCP 预热完成 — "
                f"已加载 {len(warmup_tools)} 个工具 "
                f"(耗时 {time.time() - warmup_start:.1f}s)"
            )
        else:
            logger.warning(f"{ERROR_ICON} MCP 预热返回空工具列表，首问将触发懒加载")
    except Exception as e:
        logger.warning(f"{ERROR_ICON} MCP 预热失败（不影响正常使用，首问将触发懒加载）: {e}")
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


_name_code_cache: Optional[dict] = None  # name → normalized_code 映射 (sh.xxxxxx / sz.xxxxxx)
_etf_name_cache: Optional[dict] = None   # ETF name → sh.xxxxxx 映射
_name_cache_loaded: bool = False          # 仅在缓存完全加载后置 True（用 flag 而非 dict is not None，避免空 dict 误判）

def _ensure_name_cache():
    """加载并缓存股票+ETF名称→代码映射（首次调用约15秒）

    线程安全：用 _name_cache_loaded flag 标记加载完成，避免空 dict 误判。
    无锁设计：并发加载最多重复工作，但 GIL 保证单次 dict 写入原子性，最终结果一致。
    """
    global _name_code_cache, _etf_name_cache, _name_cache_loaded
    if _name_cache_loaded:
        return
    if _name_code_cache is None:
        _name_code_cache = {}
    if _etf_name_cache is None:
        _etf_name_cache = {}

    # 1. 加载A股股票名称
    try:
        ak = _get_akshare()
        df = ak.stock_info_a_code_name()
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                n = str(row.get("name", "")).strip()
                c = str(row.get("code", "")).strip()
                if n and c:
                    _name_code_cache[n] = normalize_code(c)
        logger.info(f"{SUCCESS_ICON} 名称缓存加载完成: {len(_name_code_cache)} 只股票")
    except Exception as e:
        logger.warning(f"名称缓存(股票)加载失败: {e}")

    # 2. 加载ETF/基金名称（Tushare fund_basic）
    try:
        import requests as _req
        resp = _req.post("https://api.tushare.pro", json={
            "api_name": "fund_basic", "token": "fd4ff6e84626d2e63616ec08769f99110d626a91856036c30cb34818",
            "params": {"market": "E"},
            "fields": "ts_code,name"
        }, timeout=15)
        data = resp.json()
        if data.get("code") == 0:
            for item in data.get("data", {}).get("items", []):
                ts_code = item[0]
                name = item[1].strip() if len(item) > 1 else ""
                if name and ts_code:
                    clean = ts_code.replace(".SH", "").replace(".SZ", "")
                    norm = f"sh.{clean}" if ts_code.endswith(".SH") else f"sz.{clean}"
                    _etf_name_cache[name] = norm
            logger.info(f"{SUCCESS_ICON} ETF名称缓存加载完成: {len(_etf_name_cache)} 只ETF")
    except Exception as e:
        logger.warning(f"ETF名称缓存加载失败: {e}")

    _name_cache_loaded = True  # 必须在所有数据填充完成后设置（即使部分加载失败也标记，避免反复重试阻塞请求）


def _lookup_stock_code_by_name(name: str) -> Optional[str]:
    """通过公司名查找股票代码（股票+ETF缓存）"""
    _ensure_name_cache()
    # 精确匹配
    if name in _name_code_cache:
        return _name_code_cache[name]
    if name in _etf_name_cache:
        return _etf_name_cache[name]
    # 模糊匹配（名称包含输入）
    for n, c in _name_code_cache.items():
        if name in n:
            return c
    for n, c in _etf_name_cache.items():
        if name in n:
            return c
    return None


def _is_valid_stock_code(code: str) -> bool:
    """检查是否是合法的股票/ETF代码格式"""
    if not code:
        return False
    clean = code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").strip()
    return bool(re.match(r'^\d{5,6}$', clean))


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

        # 验证代码合法性：拒绝非股票/ETF代码格式的输入
        if not _is_valid_stock_code(stock_code):
            raise HTTPException(
                status_code=400,
                detail=f"无法识别'{query.stock_input}'为有效的股票代码或名称。请使用6位数字代码（如512480）或完整公司名称。"
            )

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
    if term not in ("short", "medium", "long", "quick_screen", "fine"):
        raise HTTPException(status_code=400, detail="期限必须为 short/medium/long/quick_screen/fine")

    if not _pool_manager:
        raise HTTPException(status_code=500, detail="股票池管理器未初始化")

    stocks = _pool_manager.list_stocks_by_term(term)
    return stocks


@app.post("/api/pool/{term}")
async def add_to_pool(term: str, request: PoolAddRequest):
    """向指定期限股票池添加股票"""
    if term not in ("short", "medium", "long", "quick_screen", "fine"):
        raise HTTPException(status_code=400, detail="期限必须为 short/medium/long/quick_screen/fine")

    if not _pool_manager:
        raise HTTPException(status_code=500, detail="股票池管理器未初始化")

    stock_code = normalize_code(request.stock_code)
    stock = _pool_manager.add_stock_to_term(term, stock_code, request.stock_name)
    return {"status": "ok", "stock": stock}


@app.delete("/api/pool/{term}/{stock_code}")
async def remove_from_pool(term: str, stock_code: str):
    """从指定期限股票池删除股票"""
    if term not in ("short", "medium", "long", "quick_screen", "fine"):
        raise HTTPException(status_code=400, detail="期限必须为 short/medium/long/quick_screen/fine")

    if not _pool_manager:
        raise HTTPException(status_code=500, detail="股票池管理器未初始化")

    stock_code = normalize_code(stock_code)
    success = _pool_manager.remove_stock_from_term(term, stock_code)
    if not success:
        raise HTTPException(status_code=404, detail=f"股票 {stock_code} 不在{term}池中")
    return {"status": "ok"}


async def _run_score_task(task_id: str, term: str, stock_code: str, company_name: str):
    """后台执行打分任务（防刷新：独立于 HTTP 请求生命周期）"""
    try:
        _score_tasks[task_id]["status"] = "running"
        result = await _scoring_engine.score_stock_for_term(term, stock_code, company_name)

        if result.get("error"):
            _score_tasks[task_id] = {"status": "failed", "error": result["error"]}
            return

        term_score = result["term_score"]
        _score_tasks[task_id] = {
            "status": "completed",
            "result": {
                "score": term_score.get("score"),
                "score_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "term": term,
                "stock_code": stock_code,
                "company_name": company_name,
            }
        }
    except Exception as e:
        logger.error(f"{ERROR_ICON} 打分后台任务失败: {e}", exc_info=True)
        _score_tasks[task_id] = {"status": "failed", "error": str(e)}


@app.post("/api/score/{term}/{stock_code}")
async def trigger_score(term: str, stock_code: str):
    """触发打分（后台异步），返回 task_id 供轮询"""
    if term not in ("short", "medium", "long", "quick_screen", "fine"):
        raise HTTPException(status_code=400, detail="期限必须为 short/medium/long/quick_screen/fine")

    if not _pool_manager or not _scoring_engine:
        raise HTTPException(status_code=500, detail="服务未初始化")

    stock_code = normalize_code(stock_code)
    stock = _pool_manager.get_stock_in_term(term, stock_code)
    if not stock:
        raise HTTPException(status_code=404, detail=f"股票 {stock_code} 不在{term}池中")

    task_id = str(uuid.uuid4())[:8]
    company_name = stock["company_name"]
    _score_tasks[task_id] = {
        "status": "pending", "term": term,
        "stock_code": stock_code, "company_name": company_name,
    }
    asyncio.create_task(_run_score_task(task_id, term, stock_code, company_name))

    return {"task_id": task_id, "status": "pending", "stock_code": stock_code, "company_name": company_name}


@app.get("/api/score/{task_id}")
async def get_score_status(task_id: str):
    """查询打分任务状态（轮询用）"""
    task = _score_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task["status"] == "completed":
        return {"status": "completed", "result": task["result"]}
    elif task["status"] == "failed":
        return {"status": "failed", "error": task.get("error", "未知错误")}
    else:
        return {"status": task["status"]}


# ──────────────────────────────────────────────
# 精筛股票池三期限并行打分
# ──────────────────────────────────────────────

async def _run_score_all_task(task_id: str, stock_code: str, company_name: str):
    """后台执行精筛打分（三期限并行）"""
    try:
        _score_all_tasks[task_id]["status"] = "running"
        result = await _scoring_engine.score_stock(stock_code, company_name)

        if result.get("error"):
            _score_all_tasks[task_id] = {"status": "failed", "error": result["error"]}
            return

        score_data = result["score_data"]
        short_ts = score_data.get("short_term_score", {})
        medium_ts = score_data.get("medium_term_score", {})
        long_ts = score_data.get("long_term_score", {})

        _pool_manager.update_fine_scores(stock_code, {
            "short_term_score": short_ts,
            "medium_term_score": medium_ts,
            "long_term_score": long_ts,
        })

        _score_all_tasks[task_id] = {
            "status": "completed",
            "result": {
                "stock_code": stock_code,
                "company_name": company_name,
                "short_term_score": short_ts,
                "medium_term_score": medium_ts,
                "long_term_score": long_ts,
                "execution_time": result.get("execution_time"),
            }
        }
    except Exception as e:
        logger.error(f"{ERROR_ICON} 精筛打分后台任务失败: {e}", exc_info=True)
        _score_all_tasks[task_id] = {"status": "failed", "error": str(e)}


@app.post("/api/score-all/{stock_code}")
async def trigger_score_all(stock_code: str):
    """触发精筛三期限并行打分（后台异步），返回 task_id"""
    if not _pool_manager or not _scoring_engine:
        raise HTTPException(status_code=500, detail="服务未初始化")

    stock_code = normalize_code(stock_code)
    stock = _pool_manager.get_stock_in_fine(stock_code)
    if not stock:
        raise HTTPException(status_code=404, detail=f"股票 {stock_code} 不在精筛池中")

    task_id = str(uuid.uuid4())[:8]
    company_name = stock["company_name"]
    _score_all_tasks[task_id] = {
        "status": "pending", "stock_code": stock_code, "company_name": company_name,
    }
    asyncio.create_task(_run_score_all_task(task_id, stock_code, company_name))

    return {"task_id": task_id, "status": "pending", "stock_code": stock_code, "company_name": company_name}


@app.get("/api/score-all/{task_id}")
async def get_score_all_status(task_id: str):
    """查询精筛打分任务状态（轮询用）"""
    task = _score_all_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task["status"] == "completed":
        return {"status": "completed", "result": task["result"]}
    elif task["status"] == "failed":
        return {"status": "failed", "error": task.get("error", "未知错误")}
    else:
        return {"status": task["status"]}


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
    _base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL_2")
    llm = OpenAICompatibleClient(
        api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY_2"),
        base_url=_base_url,
        model=os.getenv("OPENAI_COMPATIBLE_MODEL_2", "qwen3.6-flash"),
        env_prefix="",
        extra_body=get_thinking_body(_base_url, True),
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
    is_etf = stock_data.get("_asset_type") == "ETF"
    def _add(label, val, suffix=""):
        v = (val or "").strip()
        if v and v != "None" and v != "N/A":
            facts.append(f"{label}: {v}{suffix}")

    _add("PE", stock_data.get("pe"))
    _add("PB", stock_data.get("pb"))
    _add("PS", stock_data.get("ps"))
    if not is_etf:
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
    # ETF 专属字段
    if is_etf:
        _add("ETF信息", stock_data.get("etf_info"))
        _add("ETF费率", stock_data.get("etf_fee"))
        _add("ETF行情摘要", stock_data.get("etf_kline_summary"))
        _add("ETF成交摘要", stock_data.get("etf_vol_summary"))
        _add("ETF趋势", stock_data.get("etf_trend"))
        _add("ETF复权收益", stock_data.get("etf_adj_return"))

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

    # ETF 专属提示（替代不适合ETF的评分维度）
    etf_guidance = ""
    if is_etf:
        etf_guidance = """
## ⚠️ ETF打分特殊规则

本标的为ETF/指数基金，不是个股。ETF没有ROE、毛利率、净利率、营收增速、利润增速、负债率、分红、EV/EBITDA等个股财务指标，这些字段缺失是正常现象，不代表"数据获取不足"。

**ETF打分调整**:
- 基本面质量 → 改为"跟踪指数质量"：评估跟踪指数的行业代表性、流动性、成分股集中度
- 成长性 → 改为"板块景气度"：基于ETF趋势数据（近5/20/60日涨跌）和板块轮动判断
- 估值水平 → 使用PE/PB（如有），若无则用价格相对20日高低点位置判断
- 商业护城河 → 改为"板块优势"：ETF跟踪的板块是否有政策支持/产业趋势
- 在reasoning中说明"本标的为ETF，无个股财务指标"，不要说"数据不足"
"""

    user_message = f"""{citation_rules}

## 核心事实（reasoning/risk_warning 中只能引用以下数字）
{fact_checklist}

**股票**: {company_name} ({stock_code})
**当前日期**: {datetime.now().strftime('%Y-%m-%d')}
{etf_guidance}

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


async def _run_quick_score_task(task_id: str, term: str, stock_code: str, company_name: str):
    """后台执行快筛打分（防刷新）"""
    try:
        _quick_score_tasks[task_id]["status"] = "running"
        stock_data = await fetch_stock_data(stock_code)
        company_name = stock_data.get("company_name") or company_name
        result = await _direct_llm_score(term, stock_code, company_name, stock_data)

        _pool_manager.update_quick_screen_score(stock_code, term, {
            "score": result.get("score"),
            "score_time": result.get("score_time", ""),
            "recommendation": result.get("recommendation", ""),
            "suggested_action": result.get("suggested_action", ""),
            "reasoning": result.get("reasoning", ""),
            "risk_warning": result.get("risk_warning", ""),
        })

        _quick_score_tasks[task_id] = {"status": "completed", "result": result}
    except Exception as e:
        logger.error(f"{ERROR_ICON} 快筛打分后台任务失败: {e}", exc_info=True)
        _quick_score_tasks[task_id] = {"status": "failed", "error": str(e)}


@app.post("/api/quick-screen/score/{term}/{stock_code}")
async def trigger_quick_screen_score(term: str, stock_code: str):
    """触发快筛打分（后台异步），返回 task_id 供轮询"""
    if term not in ("short", "medium", "long"):
        raise HTTPException(status_code=400, detail="期限必须为 short/medium/long")

    if not _pool_manager:
        raise HTTPException(status_code=500, detail="服务未初始化")

    stock_code = normalize_code(stock_code)
    stock = _pool_manager.get_stock_in_term("quick_screen", stock_code)
    if not stock:
        raise HTTPException(status_code=404, detail=f"股票 {stock_code} 不在快筛股票池中")

    task_id = str(uuid.uuid4())[:8]
    company_name = stock["company_name"]
    _quick_score_tasks[task_id] = {
        "status": "pending", "term": term,
        "stock_code": stock_code, "company_name": company_name,
    }
    asyncio.create_task(_run_quick_score_task(task_id, term, stock_code, company_name))

    return {"task_id": task_id, "status": "pending", "stock_code": stock_code, "company_name": company_name}


@app.get("/api/quick-screen/score/{task_id}")
async def get_quick_score_status(task_id: str):
    """查询快筛打分任务状态（轮询用）"""
    task = _quick_score_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task["status"] == "completed":
        return {"status": "completed", "result": task["result"]}
    elif task["status"] == "failed":
        return {"status": "failed", "error": task.get("error", "未知错误")}
    else:
        return {"status": task["status"]}


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


# ══════════════════════════════════════════════════════════════
# 智能问答端点 (Phase 1)
# ══════════════════════════════════════════════════════════════

class QARequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class QASessionCreate(BaseModel):
    name: str = "新对话"


class QASessionRename(BaseModel):
    name: str


@app.post("/api/qa/ask")
async def qa_ask(req: QARequest):
    """
    智能问答 — SSE流式端点

    接收自然语言问题，返回流式SSE事件：
    - event: meta       → 会话ID、复杂度、股票信息
    - event: status     → 数据获取状态
    - event: answer_start → 回答开始
    - data: {chunk}     → 回答内容流
    - event: clarify    → 需要澄清（歧义高时）
    - data: [DONE]      → 结束
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    question = req.question.strip()

    # 名称→代码自动查找：用户只提供股票名称时，自动补全代码
    from src.qa.task_planner import extract_stock_from_question
    qa_code, qa_name = extract_stock_from_question(question, "", "")
    if not qa_code and qa_name:
        found_code = _lookup_stock_code_by_name(qa_name)
        if found_code:
            qa_code = normalize_code(found_code)
            logger.info(f"QA名称→代码: {qa_name} → {qa_code}")
            # 将代码追加到问题中，确保下游解析能正确提取
            question = f"{question}（{found_code}）"

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_date_cn = now.strftime("%Y年%m月%d日")
    current_weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
    current_time = now.strftime("%H:%M:%S")
    current_time_info = f"{current_date_cn} ({current_date}) {current_weekday_cn} {current_time}"

    async def safe_stream():
        try:
            async for chunk in process_question(
                question=question,
                session_id=req.session_id,
                current_date=current_date,
                current_time_info=current_time_info,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"{ERROR_ICON} QA SSE流异常: {e}", exc_info=True)
            yield f"event: error\ndata: {{\"message\": \"系统内部错误，请重试\"}}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        safe_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/qa/sessions/{session_id}")
async def qa_get_session(session_id: str):
    """获取会话历史和上下文"""
    session_mgr = get_session_manager()
    session = session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return session.to_dict()


@app.get("/api/qa/sessions")
async def qa_list_sessions():
    """列出所有会话窗口（按更新时间倒序）"""
    session_mgr = get_session_manager()
    return session_mgr.list_sessions()


@app.post("/api/qa/sessions")
async def qa_create_session(req: QASessionCreate):
    """创建新会话窗口"""
    session_mgr = get_session_manager()
    session_id = session_mgr.create_session(name=req.name)
    sess = session_mgr.get_session(session_id)
    return sess.to_dict() if sess else {"session_id": session_id}


@app.patch("/api/qa/sessions/{session_id}")
async def qa_rename_session(session_id: str, req: QASessionRename):
    """重命名会话窗口"""
    session_mgr = get_session_manager()
    ok = session_mgr.rename_session(session_id, req.name)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在或名称为空")
    return {"status": "renamed", "session_id": session_id, "name": req.name}


@app.delete("/api/qa/sessions/{session_id}")
async def qa_delete_session(session_id: str):
    """删除会话窗口及其数据缓存"""
    session_mgr = get_session_manager()
    deleted = session_mgr.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return {"status": "deleted", "session_id": session_id}


# ══════════════════════════════════════════════════════════════
# 基金分析端点 (Fund Analysis Endpoints)
# ══════════════════════════════════════════════════════════════

@app.post("/api/fund/query")
async def fund_quick_query(req: FundSearchRequest):
    """基金快速查询 — 返回基金关键指标 + 同类基准对比。
    使用轻量模型(Qwen3.6-Flash)做数据解读，严格防幻觉：数值字段全由代码计算，LLM只写文字解读。"""
    keyword = req.keyword.strip()
    if not keyword or len(keyword) < 1:
        raise HTTPException(status_code=400, detail="关键字不能为空")

    # 1. 查找基金（优先代码精确查，再按名称搜索）
    fund_code, fund_name, fund_basic_data = await _resolve_fund(keyword)
    if not fund_code:
        raise HTTPException(status_code=404, detail=f"未找到匹配'{keyword}'的基金，请检查代码或名称")

    clean_code = fund_code.replace("sh.", "").replace("sz.", "").replace("of.", "").replace(".SH", "").replace(".SZ", "").replace(".OF", "").strip()

    # 2. 获取基金关键数据（并行）— MCP client 已在 _resolve_fund 中初始化并缓存
    from src.tools.mcp_client import get_mcp_tools
    tools = await get_mcp_tools(tool_filter=["tushare_fund_basic", "tushare_fund_nav", "tushare_fund_manager"])
    tool_map = {t.name: t for t in tools} if tools else {}
    logger.info(f"基金查询: 可用工具={list(tool_map.keys())}, 请求=basic/nav/manager")

    async def _call_tool(tool, kwargs, label=""):
        try:
            r = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=20)
            text = str(r).strip()
            logger.info(f"基金查询: {label} 获取成功 ({len(text)} 字符)")
            return text
        except Exception as e:
            logger.warning(f"基金查询: {label} 获取失败: {e}")
            return ""

    # 串行获取数据（MCP stdio 不支持并行多路复用，会导致部分调用返回空）
    raw = {}
    if "tushare_fund_basic" in tool_map:
        raw["basic"] = await _call_tool(tool_map["tushare_fund_basic"], {"code": clean_code}, "basic")
    if "tushare_fund_nav" in tool_map:
        raw["nav"] = await _call_tool(tool_map["tushare_fund_nav"], {"code": clean_code, "days": 500}, "nav")
    if "tushare_fund_manager" in tool_map:
        raw["manager"] = await _call_tool(tool_map["tushare_fund_manager"], {"code": clean_code}, "manager")
    logger.info(f"基金查询: raw keys={list(raw.keys())}, nav_len={len(raw.get('nav',''))}, basic_len={len(raw.get('basic',''))}")

    # 3. 从原始数据中提取结构化指标（代码计算，不依赖LLM）
    metrics = _compute_fund_metrics(clean_code, fund_basic_data, raw)
    logger.info(f"基金查询: metrics keys={list(metrics.keys())}, return_1y={metrics.get('return_1y','N/A')}")

    # 4. 获取同类基金基准
    fund_type_str = fund_basic_data.get("fund_type", "") or metrics.get("fund_type", "")
    from src.utils.fund_type_knowledge import get_fund_benchmark, identify_fund_type
    matched_type = identify_fund_type(fund_type_str)
    benchmark = get_fund_benchmark(fund_type_str)

    # 5. 用轻量模型做文字解读（数值字段不经过LLM）
    analysis_text = await _fund_quick_llm(fund_code, fund_name, metrics, benchmark, raw)

    return {
        "fund_code": fund_code,
        "fund_name": fund_name,
        "fund_type": fund_type_str or matched_type or "未知类型",
        "metrics": metrics,
        "fund_benchmark": {
            "type_name": matched_type or fund_type_str or "未知",
            "description": benchmark.get("description", ""),
            **{k: v for k, v in benchmark.items() if k != "description"},
        },
        "analysis": analysis_text,
    }


async def _resolve_fund(keyword: str):
    """解析基金关键字：优先精确代码查询，再名称搜索。返回 (fund_code, fund_name, basic_data)"""
    from src.tools.mcp_client import get_mcp_tools

    # 检测是否为基金代码格式
    kw = keyword.strip()
    is_code_like = bool(re.match(r'^(sh\.|sz\.|of\.)?\d{5,6}(\.(OF|SH|SZ))?$', kw, re.IGNORECASE))
    clean_kw = kw.replace("sh.", "").replace("sz.", "").replace("of.", "").replace(".SH", "").replace(".SZ", "").replace(".OF", "").strip()

    if is_code_like:
        # 精确代码查询
        try:
            tools = await get_mcp_tools(tool_filter=["tushare_fund_basic"])
            if tools:
                result = await asyncio.wait_for(
                    tools[0].ainvoke({"code": clean_kw}),
                    timeout=15.0,
                )
                text = str(result).strip()
                if text and "未找到" not in text and len(text) > 50:
                    # 解析markdown表格提取基本信息
                    basic = _parse_fund_basic_from_table(text)
                    if basic:
                        ts_code = basic.get("ts_code", "")
                        name = basic.get("name", "")
                        code = f"sz.{clean_kw}" if clean_kw.startswith(("0", "1", "3")) else f"sh.{clean_kw}"
                        if ts_code and ts_code.endswith(".OF"):
                            code = f"of.{clean_kw}" if not clean_kw.startswith(("51","58","15","16","18")) else code
                        return code, name or kw, basic
        except Exception:
            pass

    # 名称搜索
    try:
        tools = await get_mcp_tools(tool_filter=["tushare_fund_search"])
        if tools:
            result = await asyncio.wait_for(
                tools[0].ainvoke({"keyword": kw}),
                timeout=30.0,
            )
            text = str(result).strip()
            if text and "未找到" not in text and len(text) > 50:
                basic = _parse_fund_basic_from_table(text)
                if basic:
                    ts_code = basic.get("ts_code", "")
                    name = basic.get("name", "") or kw
                    clean_c = ts_code.replace(".OF", "").replace(".SH", "").replace(".SZ", "").strip()
                    code = f"sz.{clean_c}" if clean_c.startswith(("0", "1", "3")) else f"sh.{clean_c}"
                    if ts_code.endswith(".OF"):
                        code = f"of.{clean_c}"
                    return code, name, basic
    except Exception:
        pass

    return None, None, {}


def _parse_fund_basic_from_table(text: str) -> dict:
    """从markdown表格中解析第一行基金基本信息（处理空单元格）。"""
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return {}
    # 用 | 精确分割每行（保留空单元格）
    def _split_row(line: str) -> list:
        return [c.strip() for c in line.split("|")][1:-1]  # 去掉首尾空元素
    headers = _split_row(lines[0])
    values = _split_row(lines[2]) if len(lines) > 2 else []
    if headers and values:
        # 若 values < headers，补齐缺失的空值
        while len(values) < len(headers):
            values.append("")
        return dict(zip(headers[:len(values)], values))
    return {}


def _compute_fund_metrics(clean_code: str, basic: dict, raw: dict) -> dict:
    """从原始数据中提取计算基金关键指标（纯代码，无LLM参与）"""
    import math
    m = {
        "fund_type": basic.get("fund_type", ""),
        "management": basic.get("management", ""),
        "found_date": basic.get("found_date", ""),
        "benchmark": basic.get("benchmark", ""),
        "m_fee": basic.get("m_fee", ""),
        "c_fee": basic.get("c_fee", ""),
        "invest_type": basic.get("invest_type", ""),
        "status": basic.get("status", ""),
    }

    # 从NAV数据中计算收益/风险指标
    nav_text = raw.get("nav", "")
    nav_data = _parse_nav_from_table(nav_text)

    if nav_data:
        try:
            # MCP 返回的 NAV 表是倒序（最新在前），反转为时间升序（最旧在前）
            nav_data_chrono = list(reversed(nav_data))

            # 同步提取 dates + adj_navs（同一行要么都保留，要么都丢弃，保证索引对齐）
            dates = []
            adj_navs = []
            unit_navs = []
            for d in nav_data_chrono:
                try:
                    u = float(d.get("unit_nav", 0))
                    a = float(d.get("adj_nav", 0))
                    dt = d.get("nav_date", "")
                    if not dt or u == 0 or a == 0:
                        continue
                    unit_navs.append(u)
                    adj_navs.append(a)
                    dates.append(dt)
                except (ValueError, TypeError):
                    continue

            if not adj_navs:
                logger.warning("基金查询: NAV 解析后 adj_navs 为空")
                m["data_days"] = "0"
                return m

            n = len(adj_navs)
            m["latest_nav"] = f"{unit_navs[-1]:.4f}"
            m["latest_adj_nav"] = f"{adj_navs[-1]:.4f}"
            m["nav_date"] = dates[-1] if dates else ""
            m["data_days"] = str(n)

            if n < 2:
                return m

            # ---- 用实际日期找1年前位置 ----
            from datetime import datetime as _dt, timedelta as _td
            cutoff = (_dt.now() - _td(days=365)).strftime("%Y%m%d")
            logger.info(f"基金查询: cutoff={cutoff}, n={n}, first_date={dates[0]}, last_date={dates[-1]}")

            # 找 < cutoff 的最大日期（往前找最近交易日，跳过 cutoff 当天）
            idx_1y = 0
            for i in range(n-1, -1, -1):
                if dates[i] < cutoff:
                    idx_1y = i
                    break

            logger.info(f"基金查询: idx_1y={idx_1y}, date_1y={dates[idx_1y]}, nav_1y={adj_navs[idx_1y]:.4f}, nav_now={adj_navs[-1]:.4f}")

            # ---- 计算指标 ----
            start_nav = adj_navs[idx_1y]
            end_nav = adj_navs[-1]
            if start_nav > 0 and idx_1y < n - 1:
                ret = (end_nav / start_nav - 1) * 100
                m["return_1y"] = f"{ret:.2f}%"
                logger.info(f"基金查询: return_1y={m['return_1y']}")

            window = adj_navs[idx_1y:]
            if len(window) >= 20:
                daily_rets = []
                for i in range(1, len(window)):
                    if window[i-1] > 0:
                        daily_rets.append(math.log(window[i] / window[i-1]))
                if daily_rets:
                    std_daily = _stddev(daily_rets)
                    m["annual_volatility"] = f"{std_daily * math.sqrt(252) * 100:.2f}%"

            if len(window) >= 20:
                peak = window[0]
                max_dd = 0.0
                for nav_val in window:
                    if nav_val > peak:
                        peak = nav_val
                    if peak > 0:
                        dd = (nav_val / peak - 1) * 100
                        if dd < max_dd:
                            max_dd = dd
                m["max_drawdown_1y"] = f"{max_dd:.2f}%"

            # 夏普比率
            if m.get("annual_volatility") and m.get("return_1y"):
                try:
                    vol = float(m["annual_volatility"].replace("%", "")) / 100
                    ret_y = float(m["return_1y"].replace("%", "")) / 100
                    if vol > 0:
                        m["sharpe_ratio"] = f"{(ret_y - 0.02) / vol:.2f}"
                except (ValueError, TypeError):
                    pass

        except Exception as e:
            logger.error(f"基金查询: _compute_fund_metrics 异常: {e}", exc_info=True)

    return m


def _parse_nav_from_table(text: str) -> list:
    """从markdown表格中解析NAV数据（处理空单元格）。"""
    if not text or len(text) < 50:
        return []
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return []
    def _split_row(line: str) -> list:
        return [c.strip() for c in line.split("|")][1:-1]
    headers = _split_row(lines[0])
    result = []
    for line in lines[2:]:
        vals = _split_row(line)
        if not vals:
            continue
        while len(vals) < len(headers):
            vals.append("")
        result.append(dict(zip(headers[:len(vals)], vals)))
    return result


def _stddev(data: list) -> float:
    import math
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)


async def _fund_quick_llm(fund_code: str, fund_name: str, metrics: dict, benchmark: dict, raw: dict) -> dict:
    """使用轻量模型(Qwen3.6-Flash)做文字解读。只生成文字字段，数值全由代码预填。"""
    import os as _os
    base_url = _os.getenv("OPENAI_COMPATIBLE_BASE_URL_2")
    llm = OpenAICompatibleClient(
        api_key=_os.getenv("OPENAI_COMPATIBLE_API_KEY_2"),
        base_url=base_url,
        model=_os.getenv("OPENAI_COMPATIBLE_MODEL_2", "qwen3.6-flash"),
        env_prefix="",
        extra_body=get_thinking_body(base_url, False),
    )

    metrics_json = json.dumps(metrics, ensure_ascii=False, indent=2)
    benchmark_json = json.dumps(benchmark, ensure_ascii=False, indent=2)

    # 预填数据（数值字段不经过LLM）
    factual = {
        "fund_name": fund_name,
        "fund_code": fund_code,
        "fund_type": metrics.get("fund_type", "未知"),
        "management": metrics.get("management", ""),
        "found_date": metrics.get("found_date", ""),
        "benchmark": metrics.get("benchmark", ""),
        "m_fee": metrics.get("m_fee", ""),
        "c_fee": metrics.get("c_fee", ""),
        "latest_nav": metrics.get("latest_nav", ""),
        "nav_date": metrics.get("nav_date", ""),
        "return_1y": metrics.get("return_1y", ""),
        "annual_volatility": metrics.get("annual_volatility", ""),
        "max_drawdown_1y": metrics.get("max_drawdown_1y", ""),
        "sharpe_ratio": metrics.get("sharpe_ratio", ""),
        "invest_type": metrics.get("invest_type", ""),
        "status": metrics.get("status", ""),
    }

    prompt = f"""请对以下公募基金做简洁文字解读。所有数值字段已由系统计算完毕，你只需要写 3 段文字解读（每段1-2句，总计≤200字）。

基金: {fund_name} ({fund_code})

## 系统计算指标
{metrics_json}

## 同类基准({benchmark.get('type_name', '')})
{benchmark_json}

请返回严格JSON（不要markdown标记）：
{{
  "fund_name": "已预填",
  "fund_code": "已预填",
  "fund_type": "已预填",
  "management": "已预填",
  "found_date": "已预填",
  "benchmark": "已预填",
  "m_fee": "已预填",
  "c_fee": "已预填",
  "latest_nav": "已预填",
  "nav_date": "已预填",
  "return_1y": "已预填",
  "annual_volatility": "已预填",
  "max_drawdown_1y": "已预填",
  "sharpe_ratio": "已预填",
  "invest_type": "已预填",
  "status": "已预填",
  "fund_intro": "1-2句基金介绍，基于基金名称、类型、管理人、投资策略推断",
  "performance_comment": "1-2句业绩点评，基于return_1y/volatility/max_drawdown/sharpe与同类基准对比，若指标缺失则如实说明",
  "suitability": "1句适配性建议，基于基金类型和风险指标，若指标缺失则如实说明"
}}

⛔ 幻觉控制（违反任一条即为错误）：
1. 所有带"已预填"的字段必须原样输出"已预填"（由系统后处理替换），你不得修改、猜测、或补充任何数值
2. fund_intro/performance_comment/suitability 三个文字字段基于实际数据撰写，数据缺失时如实说明"数据有限"
3. 不得编造任何数字、日期、百分比或人名
4. 不得提及"我"、"本人"、"分析师"等第一人称"""

    try:
        response = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                _thread_pool,
                lambda: llm.get_completion([
                    {"role": "system", "content": "你是一个公募基金数据查询助手。严格遵循数据，不编造任何信息。输出纯JSON。"},
                    {"role": "user", "content": prompt},
                ], temperature=0.3, max_tokens=1024, response_format="json_object"),
            ),
            timeout=25.0,
        )
        text = response.strip() if isinstance(response, str) else str(response)
        # 清理可能的markdown包裹
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
    except Exception:
        result = {}

    # 将LLM输出的文字字段与预填的数值字段合并
    return {
        "fund_name": fund_name,
        "fund_code": fund_code,
        "fund_type": factual["fund_type"],
        "management": factual["management"],
        "found_date": factual["found_date"],
        "benchmark": factual["benchmark"],
        "m_fee": factual["m_fee"],
        "c_fee": factual["c_fee"],
        "latest_nav": factual["latest_nav"],
        "nav_date": factual["nav_date"],
        "return_1y": factual["return_1y"],
        "annual_volatility": factual["annual_volatility"],
        "max_drawdown_1y": factual["max_drawdown_1y"],
        "sharpe_ratio": factual["sharpe_ratio"],
        "invest_type": factual["invest_type"],
        "status": factual["status"],
        "fund_intro": result.get("fund_intro", "") or "",
        "performance_comment": result.get("performance_comment", "") or "",
        "suitability": result.get("suitability", "") or "",
    }


@app.post("/api/fund/search")
async def search_funds(req: FundSearchRequest):
    """搜索基金：通过关键字查找匹配的基金代码/名称。

    Returns:
        [{fund_code, fund_name, fund_type}, ...]
    """
    keyword = req.keyword.strip()
    if not keyword or len(keyword) < 1:
        raise HTTPException(status_code=400, detail="关键字不能为空")

    try:
        # 通过 MCP 工具 tushare_fund_search 搜索基金
        from src.tools.mcp_client import get_mcp_tools
        tools = await get_mcp_tools(tool_filter=["tushare_fund_search"])
        if not tools:
            raise HTTPException(status_code=503, detail="MCP 基金搜索工具不可用")

        fund_search_tool = tools[0]
        result = await asyncio.wait_for(
            fund_search_tool.ainvoke({"keyword": keyword}),
            timeout=30.0,
        )
        result_text = str(result).strip()

        # 尝试解析 JSON 结果
        results = []
        try:
            parsed = json.loads(result_text)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        results.append({
                            "fund_code": item.get("ts_code", item.get("fund_code", "")),
                            "fund_name": item.get("name", item.get("fund_name", "")),
                            "fund_type": item.get("fund_type", item.get("type", "")),
                        })
            elif isinstance(parsed, dict):
                # 可能是包含 items 的包装
                items = parsed.get("items", parsed.get("data", []))
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, list) and len(item) >= 2:
                            results.append({
                                "fund_code": item[0] if len(item) > 0 else "",
                                "fund_name": item[1] if len(item) > 1 else "",
                                "fund_type": item[2] if len(item) > 2 else "",
                            })
                        elif isinstance(item, dict):
                            results.append({
                                "fund_code": item.get("ts_code", item.get("fund_code", "")),
                                "fund_name": item.get("name", item.get("fund_name", "")),
                                "fund_type": item.get("fund_type", item.get("type", "")),
                            })
        except json.JSONDecodeError:
            # 如果不是 JSON，尝试从文本中提取基金代码/名称
            import re as _re
            # 匹配形如 "510050 华夏上证50ETF" 的行
            for line in result_text.split("\n"):
                match = _re.search(r'(\d{5,6})\s+([^\n]+)', line)
                if match:
                    results.append({
                        "fund_code": match.group(1),
                        "fund_name": match.group(2).strip(),
                        "fund_type": "",
                    })

        return {"keyword": keyword, "results": results, "count": len(results)}

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="基金搜索超时（30秒）")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{ERROR_ICON} 基金搜索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@app.post("/api/fund/report")
async def generate_fund_report(req: FundReportRequest):
    """启动基金分析流水线（异步），返回 task_id 供轮询。

    mode: "report" — 生成完整13模块基金报告
    mode: "score"  — 仅生成7维度综合评分卡
    """
    fund_code = req.fund_code.strip()
    fund_name = req.fund_name.strip()
    mode = req.mode.strip()

    if not fund_code:
        raise HTTPException(status_code=400, detail="基金代码不能为空")
    if mode not in ("report", "score"):
        raise HTTPException(status_code=400, detail="mode 必须为 report 或 score")

    # 规范化基金代码（支持 sh.510050 格式）
    if not fund_code.startswith(("sh.", "sz.", "of.")):
        fund_code = normalize_code(fund_code)

    if not fund_name:
        fund_name = fund_code

    task_id = str(uuid.uuid4())[:8]
    _fund_tasks[task_id] = {
        "status": "pending",
        "mode": mode,
        "fund_code": fund_code,
        "fund_name": fund_name,
    }

    # 后台执行基金分析流水线（防刷新）
    asyncio.create_task(_run_fund_analysis_pipeline(task_id, fund_code, fund_name, mode))

    return {
        "task_id": task_id,
        "status": "pending",
        "fund_code": fund_code,
        "fund_name": fund_name,
        "mode": mode,
    }


@app.get("/api/fund/report/{task_id}")
async def get_fund_report(task_id: str):
    """查询基金分析任务状态和结果（轮询用）。"""
    if task_id not in _fund_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = _fund_tasks[task_id]
    return task


# ── 基金池 CRUD ──

@app.get("/api/fund/pool/{pool_name}")
async def list_fund_pool(pool_name: str):
    """列出指定基金池中的基金。

    pool_name: scored / watchlist
    """
    if pool_name not in ("scored", "watchlist"):
        raise HTTPException(status_code=400, detail="pool_name 必须为 scored 或 watchlist")

    mgr = _get_fund_pool_manager()
    funds = mgr.list_funds(pool=pool_name)
    return {"pool": pool_name, "funds": funds, "count": len(funds)}


@app.post("/api/fund/pool/{pool_name}")
async def add_to_fund_pool(pool_name: str, req: FundPoolAddRequest):
    """向基金池添加基金。"""
    if pool_name not in ("scored", "watchlist"):
        raise HTTPException(status_code=400, detail="pool_name 必须为 scored 或 watchlist")

    fund_code = req.fund_code.strip() or req.stock_code.strip()
    fund_name = req.fund_name.strip() or req.stock_name.strip()

    if not fund_code:
        raise HTTPException(status_code=400, detail="基金代码不能为空")

    # 规范化基金代码
    if not fund_code.startswith(("sh.", "sz.", "of.")):
        fund_code = normalize_code(fund_code)

    if not fund_name:
        fund_name = fund_code

    mgr = _get_fund_pool_manager()
    mgr.add_fund(fund_code, fund_name, pool=pool_name)
    return {"status": "ok", "fund_code": fund_code, "fund_name": fund_name, "pool": pool_name}


@app.delete("/api/fund/pool/{pool_name}/{fund_code}")
async def remove_from_fund_pool(pool_name: str, fund_code: str):
    """从基金池中删除基金。"""
    if pool_name not in ("scored", "watchlist"):
        raise HTTPException(status_code=400, detail="pool_name 必须为 scored 或 watchlist")

    mgr = _get_fund_pool_manager()
    success = mgr.remove_fund(fund_code, pool=pool_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不在{pool_name}池中")
    return {"status": "ok"}


# ── 基金快速打分 ──

@app.post("/api/fund/score/{fund_code}")
async def trigger_fund_score(fund_code: str):
    """触发基金快速打分（后台异步），返回 task_id 供轮询。

    只做打分，不生成完整报告。结果通过 GET /api/fund/report/{task_id} 轮询。
    """
    fund_code = fund_code.strip()
    if not fund_code:
        raise HTTPException(status_code=400, detail="基金代码不能为空")

    if not fund_code.startswith(("sh.", "sz.", "of.")):
        fund_code = normalize_code(fund_code)

    task_id = str(uuid.uuid4())[:8]
    _fund_tasks[task_id] = {
        "status": "pending",
        "mode": "score",
        "fund_code": fund_code,
        "fund_name": fund_code,  # 名称由流水线解析获取
    }

    asyncio.create_task(_run_fund_analysis_pipeline(task_id, fund_code, fund_code, "score"))

    return {
        "task_id": task_id,
        "status": "pending",
        "fund_code": fund_code,
    }


# ── 基金缓存状态 ──

@app.get("/api/fund/cache/{fund_code}")
async def fund_cache_status(fund_code: str):
    """查看某只基金的中间产物缓存状态。"""
    if not fund_code.startswith(("sh.", "sz.", "of.")):
        fund_code = normalize_code(fund_code)
    return get_fund_cache_status(fund_code)


# ──────────────────────────────────────────────
# 智能投顾端点 (Task 12)
# ──────────────────────────────────────────────

# ── 1. 股票推荐 ──

@app.post("/api/advisory/recommend")
async def advisory_recommend(request: AdvisoryRecommendRequest):
    """基于精筛池 + 生产缓存的 n/x/5 规则推荐股票。

    接收用户自然语言提问，从精筛池加载所有股票，按 n/x/5 规则裁剪，
    构建 LLM 上下文，通过 DeepSeek V4 Pro 生成最终推荐。
    """
    try:
        modules = _init_advisory()
        engine = modules["recommendation_engine"]
        pool_stocks = engine.load_pool_stocks()
        if not pool_stocks:
            raise HTTPException(status_code=404, detail="精筛池为空，请先执行 pool update")

        # 收集所有候选代码
        candidate_codes = [s["stock_code"] for s in pool_stocks]
        # 应用 n/x/5 规则
        trimmed_codes = engine.apply_nx5_rule(candidate_codes, pool_stocks)

        # 判断是否需要 LLM 介入
        needs_llm = any("__LLM_PICK__" == c for c in trimmed_codes)
        if needs_llm:
            trimmed_codes = [c for c in trimmed_codes if c != "__LLM_PICK__"]

        # 匹配完整股票信息
        result_stocks = []
        for s in pool_stocks:
            if s["stock_code"] in trimmed_codes:
                cache_score = engine.check_score_cache(s["stock_code"])
                result_stocks.append({
                    "stock_code": s["stock_code"],
                    "company_name": s["company_name"],
                    "pool_term": s.get("term", ""),
                    "pool_score": s.get("score"),
                    "cache_score": cache_score,
                    "industry": s.get("detected_industry", ""),
                })

        # 按池评分排序（有缓存用缓存）
        result_stocks.sort(
            key=lambda x: x.get("cache_score") or x.get("pool_score") or 0,
            reverse=True,
        )

        return {
            "status": "ok",
            "total_candidates": len(candidate_codes),
            "recommended_count": len(result_stocks),
            "needs_llm_pick": needs_llm,
            "recommendations": result_stocks,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{ERROR_ICON} 股票推荐失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"推荐失败: {str(e)}")


# ── 2. 创建持仓组合 ──

@app.post("/api/advisory/portfolio/create")
async def advisory_portfolio_create(request: PortfolioCreateRequest):
    """创建新的投资组合。

    Args:
        name: 组合名称（必填）。
        initial_capital: 初始本金，默认 100,000。
    """
    try:
        if not request.name.strip():
            raise HTTPException(status_code=400, detail="组合名称不能为空")
        if request.initial_capital <= 0:
            raise HTTPException(status_code=400, detail="初始本金必须大于 0")

        modules = _init_advisory()
        pm = modules["portfolio_manager"]
        pf = pm.create(name=request.name.strip(), initial_capital=request.initial_capital)

        return {
            "status": "ok",
            "portfolio_id": pf.portfolio_id,
            "name": pf.name,
            "initial_capital": pf.initial_capital,
            "cash": pf.cash,
            "created_at": pf.created_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{ERROR_ICON} 创建组合失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建组合失败: {str(e)}")


# ── 3. 组合列表 ──

@app.get("/api/advisory/portfolio/list")
async def advisory_portfolio_list():
    """列出所有投资组合。"""
    try:
        modules = _init_advisory()
        pm = modules["portfolio_manager"]
        portfolios = pm.list_all()
        return {
            "status": "ok",
            "count": len(portfolios),
            "portfolios": [
                {
                    "portfolio_id": pf.portfolio_id,
                    "name": pf.name,
                    "total_market_value": pf.total_market_value,
                    "cash": pf.cash,
                    "total_pnl": pf.total_pnl,
                    "total_pnl_pct": pf.total_pnl_pct,
                    "initial_capital": pf.initial_capital,
                    "bound_strategy": pf.bound_strategy,
                    "holdings_count": len(pf.holdings),
                    "status": pf.status,
                    "created_at": pf.created_at,
                    "updated_at": pf.updated_at,
                }
                for pf in portfolios
            ],
        }
    except Exception as e:
        logger.error(f"{ERROR_ICON} 组合列表查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


# ── 4. 组合详情 ──

@app.get("/api/advisory/portfolio/{portfolio_id}")
async def advisory_portfolio_detail(portfolio_id: str):
    """获取单个投资组合的完整详情，含持仓明细。"""
    try:
        modules = _init_advisory()
        pm = modules["portfolio_manager"]
        pf = pm.load(portfolio_id)
        if pf is None:
            raise HTTPException(status_code=404, detail=f"组合 {portfolio_id} 不存在")

        holdings = {}
        for code, h in pf.holdings.items():
            holdings[code] = {
                "stock_code": h.stock_code,
                "company_name": h.company_name,
                "quantity": h.quantity,
                "cost_price": h.cost_price,
                "current_price": h.current_price,
                "market_value": h.market_value,
                "weight": h.weight,
            }

        return {
            "portfolio_id": pf.portfolio_id,
            "user_id": pf.user_id,
            "name": pf.name,
            "holdings": holdings,
            "cash": pf.cash,
            "total_cost": pf.total_cost,
            "total_market_value": pf.total_market_value,
            "total_pnl": pf.total_pnl,
            "total_pnl_pct": pf.total_pnl_pct,
            "initial_capital": pf.initial_capital,
            "bound_strategy": pf.bound_strategy,
            "strategy_params": pf.strategy_params,
            "status": pf.status,
            "created_at": pf.created_at,
            "updated_at": pf.updated_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{ERROR_ICON} 组合详情查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


# ── 5. 买入/卖出持仓 ──

@app.post("/api/advisory/portfolio/holding")
async def advisory_portfolio_holding(request: HoldingModifyRequest):
    """买入或卖出持仓。

    Args:
        action: "buy" 买入 / "sell" 卖出。
        stock_code: 股票代码，如 "600519.SH" 或 "sh.600519"。
        quantity: 买卖数量。买入时自动向下取整至 100 的倍数。
        price: 成交单价。
    """
    try:
        if request.action not in ("buy", "sell"):
            raise HTTPException(status_code=400, detail="action 必须为 buy 或 sell")
        if request.quantity <= 0:
            raise HTTPException(status_code=400, detail="数量必须大于 0")
        if request.price <= 0:
            raise HTTPException(status_code=400, detail="价格必须大于 0")

        # 规范化股票代码 → ts_code 格式 (600519.SH)
        raw_code = request.stock_code.strip()
        ts_code = raw_code
        if raw_code.startswith("sh.") or raw_code.startswith("sz."):
            ts_code = raw_code.replace("sh.", "").replace("sz.", "")
            ts_code = f"{ts_code}.{'SH' if raw_code.startswith('sh.') else 'SZ'}"
        elif not (".SH" in ts_code.upper() or ".SZ" in ts_code.upper()):
            prefix = _get_exchange_prefix(ts_code)
            ts_code = f"{ts_code}.{prefix.upper()}"

        modules = _init_advisory()
        pm = modules["portfolio_manager"]
        pf = pm.load(request.portfolio_id)
        if pf is None:
            raise HTTPException(status_code=404, detail=f"组合 {request.portfolio_id} 不存在")

        company_name = request.company_name or request.stock_code

        if request.action == "buy":
            pf, trade = pm.add_holding(
                pf, ts_code, company_name,
                request.quantity, request.price,
            )
        else:
            pf, trade = pm.remove_holding(
                pf, ts_code,
                request.quantity, request.price,
            )

        return {
            "status": "ok",
            "trade": {
                "date": trade.date,
                "action": trade.action,
                "stock_code": trade.stock_code,
                "company_name": trade.company_name,
                "price": trade.price,
                "shares": trade.shares,
                "commission": trade.commission,
                "reason": trade.reason,
            },
            "portfolio": {
                "portfolio_id": pf.portfolio_id,
                "total_market_value": pf.total_market_value,
                "cash": pf.cash,
                "total_pnl": pf.total_pnl,
                "total_pnl_pct": pf.total_pnl_pct,
                "holdings_count": len(pf.holdings),
            },
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"{ERROR_ICON} 持仓操作失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"持仓操作失败: {str(e)}")


# ── 6. 策略目录 ──

@app.get("/api/advisory/strategies")
async def advisory_strategies():
    """获取所有已注册的交易策略目录。"""
    try:
        modules = _init_advisory()
        StrategyEngine = modules["strategy_engine"]
        catalog = StrategyEngine.get_strategy_catalog()
        return {
            "status": "ok",
            "count": len(catalog),
            "strategies": catalog,
        }
    except Exception as e:
        logger.error(f"{ERROR_ICON} 策略目录查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


# ── 7. 绑定策略 ──

@app.post("/api/advisory/strategy/bind")
async def advisory_strategy_bind(request: StrategyBindRequest):
    """将交易策略绑定到投资组合。

    Args:
        portfolio_id: 组合 ID。
        strategy_name: 策略注册名称，如 "ma_cross"。
        params: 可选策略参数字典。
    """
    try:
        if not request.strategy_name.strip():
            raise HTTPException(status_code=400, detail="策略名称不能为空")

        modules = _init_advisory()
        pm = modules["portfolio_manager"]
        pf = pm.load(request.portfolio_id)
        if pf is None:
            raise HTTPException(status_code=404, detail=f"组合 {request.portfolio_id} 不存在")

        # 验证策略是否已注册
        StrategyEngine = modules["strategy_engine"]
        catalog_names = {s["name"] for s in StrategyEngine.get_strategy_catalog()}
        if request.strategy_name not in catalog_names:
            raise HTTPException(
                status_code=400,
                detail=f"策略 '{request.strategy_name}' 未注册。可用策略: {', '.join(sorted(catalog_names))}"
            )

        pm.bind_strategy(pf, request.strategy_name, request.params)

        return {
            "status": "ok",
            "portfolio_id": pf.portfolio_id,
            "bound_strategy": pf.bound_strategy,
            "strategy_params": pf.strategy_params,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{ERROR_ICON} 策略绑定失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"策略绑定失败: {str(e)}")


# ── 8. 回测 ──

@app.post("/api/advisory/backtest")
async def advisory_backtest(request: BacktestRequest):
    """执行历史回测。

    使用 Tushare 日线数据，在指定区间按策略信号逐日回放，
    返回完整回测结果（收益率、最大回撤、权益曲线、交易记录等）。

    Args:
        portfolio_id: 组合 ID（必须有持仓或绑定策略）。
        start_date: 回测开始日期，YYYY-MM-DD 或 YYYYMMDD。
        end_date: 回测结束日期，YYYY-MM-DD 或 YYYYMMDD。
        strategy_name: 可选，策略名称。为 None 时使用组合绑定策略。
        strategy_params: 可选，策略参数。
    """
    try:
        # 日期格式标准化
        start_date = request.start_date.replace("-", "")
        end_date = request.end_date.replace("-", "")
        if len(start_date) != 8 or len(end_date) != 8:
            raise HTTPException(status_code=400, detail="日期格式必须为 YYYY-MM-DD 或 YYYYMMDD")
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="开始日期必须早于结束日期")

        modules = _init_advisory()
        pm = modules["portfolio_manager"]
        pf = pm.load(request.portfolio_id)
        if pf is None:
            raise HTTPException(status_code=404, detail=f"组合 {request.portfolio_id} 不存在")

        # 确定策略
        strategy_name = request.strategy_name or pf.bound_strategy
        if not strategy_name:
            raise HTTPException(
                status_code=400,
                detail="未指定策略，请提供 strategy_name 或先绑定策略",
            )

        # 拷贝组合用于回测（不污染原组合）
        pf_copy = pm.load(request.portfolio_id)
        if pf_copy is None:
            raise HTTPException(status_code=500, detail="加载组合副本失败")

        runner = modules["backtest_runner"]
        result = runner.run(
            pf=pf_copy,
            start_date=start_date,
            end_date=end_date,
            strategy_name=strategy_name,
            strategy_params=request.strategy_params,
        )

        return {
            "status": "ok",
            "portfolio_id": result.portfolio_id,
            "strategy_name": result.strategy_name,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "initial_capital": result.initial_capital,
            "final_value": result.final_value,
            "total_return_pct": result.total_return_pct,
            "annualized_return_pct": result.annualized_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "trade_count": result.trade_count,
            "equity_curve": result.equity_curve[-100:] if len(result.equity_curve) > 100 else result.equity_curve,
            "trades": [
                {
                    "date": t.date,
                    "action": t.action,
                    "stock_code": t.stock_code,
                    "company_name": t.company_name,
                    "price": t.price,
                    "shares": t.shares,
                    "commission": t.commission,
                    "reason": t.reason,
                }
                for t in result.trades
            ],
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"{ERROR_ICON} 回测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"回测失败: {str(e)}")


# ── 9. 模拟盘控制 ──

@app.post("/api/advisory/simulation")
async def advisory_simulation(request: SimulationRequest):
    """模拟盘控制。

    Args:
        action: "start" 执行今日结算 | "stop" 暂不支持 |
                "status" 查询状态 | "catch_up" 追赶缺失交易日。
    """
    try:
        if request.action not in ("start", "stop", "status", "catch_up"):
            raise HTTPException(
                status_code=400,
                detail="action 必须为 start / stop / status / catch_up",
            )

        modules = _init_advisory()
        sr = modules["simulation_runner"]

        if request.action == "start":
            # 执行今日模拟盘结算
            result = sr.run_daily_settlement(request.portfolio_id)
            return {
                "status": "ok",
                "action": "start",
                "result": result,
            }

        elif request.action == "stop":
            # 暂不支持暂停模拟盘
            return {
                "status": "ok",
                "action": "stop",
                "message": "模拟盘为无状态执行，无需显式停止。如需重置组合请创建新组合。",
            }

        elif request.action == "status":
            pm = modules["portfolio_manager"]
            pf = pm.load(request.portfolio_id)
            if pf is None:
                raise HTTPException(status_code=404, detail=f"组合 {request.portfolio_id} 不存在")

            # 获取最后结算日
            settlement_dir = sr._settlement_dir
            last_date = ""
            equity_curve = []
            if os.path.isdir(settlement_dir):
                pattern = os.path.join(settlement_dir, f"{request.portfolio_id}_*.json")
                files = sorted(glob_module.glob(pattern))
                if files:
                    last_file = files[-1]
                    try:
                        with open(last_file, "r", encoding="utf-8") as f:
                            settlement = json.load(f)
                        last_date = settlement.get("date", "")
                    except Exception:
                        pass
                    # 构建权益曲线（最多60个点）
                    for fp in files[-60:]:
                        try:
                            with open(fp, "r", encoding="utf-8") as f:
                                s = json.load(f)
                            equity_curve.append({
                                "date": s.get("date", ""),
                                "total_value": s.get("total_value", 0),
                                "daily_return_pct": s.get("daily_return_pct", 0),
                            })
                        except Exception:
                            pass

            return {
                "status": "ok",
                "action": "status",
                "portfolio_id": request.portfolio_id,
                "bound_strategy": pf.bound_strategy,
                "total_market_value": pf.total_market_value,
                "cash": pf.cash,
                "total_pnl": pf.total_pnl,
                "total_pnl_pct": pf.total_pnl_pct,
                "holdings_count": len(pf.holdings),
                "last_settlement_date": last_date,
                "equity_curve": equity_curve,
            }

        elif request.action == "catch_up":
            result = sr.run_catch_up(request.portfolio_id)
            return {
                "status": "ok",
                "action": "catch_up",
                "result": result,
            }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"{ERROR_ICON} 模拟盘操作失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"模拟盘操作失败: {str(e)}")


# ── 10. 收益报告 ──

@app.post("/api/advisory/report")
async def advisory_report(request: ReportRequest):
    """生成收益分析报告。

    基于回测或模拟盘权益曲线生成图表和 LLM 驱动的文字报告。

    Args:
        report_type: "backtest" 回测报告 / "simulation" 模拟盘报告。
        start_date/end_date: 回测报告时必需，模拟盘可选。
        include_deepseek: 是否包含 DeepSeek 自由线对比。
    """
    try:
        if request.report_type not in ("backtest", "simulation"):
            raise HTTPException(
                status_code=400,
                detail="report_type 必须为 backtest 或 simulation",
            )

        modules = _init_advisory()
        gen = modules["report_generator"]
        pm = modules["portfolio_manager"]

        pf = pm.load(request.portfolio_id)
        if pf is None:
            raise HTTPException(status_code=404, detail=f"组合 {request.portfolio_id} 不存在")

        # 收集权益数据
        user_equity: List[float] = []
        deepseek_equity: List[float] = []
        benchmark_equity: List[float] = []

        if request.report_type == "backtest":
            if not request.start_date or not request.end_date:
                raise HTTPException(
                    status_code=400,
                    detail="回测报告需要提供 start_date 和 end_date",
                )
            # 先跑回测以获取权益曲线
            start_date = request.start_date.replace("-", "")
            end_date = request.end_date.replace("-", "")
            if len(start_date) != 8 or len(end_date) != 8:
                raise HTTPException(status_code=400, detail="日期格式必须为 YYYY-MM-DD 或 YYYYMMDD")

            strategy_name = pf.bound_strategy
            if not strategy_name:
                raise HTTPException(
                    status_code=400,
                    detail="组合未绑定策略，无法执行回测以生成报告",
                )

            pf_copy = pm.load(request.portfolio_id)
            if pf_copy is None:
                raise HTTPException(status_code=500, detail="加载组合副本失败")

            runner = modules["backtest_runner"]
            bt_result = runner.run(
                pf=pf_copy,
                start_date=start_date,
                end_date=end_date,
                strategy_name=strategy_name,
            )
            user_equity = bt_result.equity_curve
            # 归一化为 1.0 基准
            if user_equity and user_equity[0] > 0:
                base = user_equity[0]
                user_equity = [v / base for v in user_equity]

            if request.include_deepseek:
                # 占位：DeepSeek 自由线需要独立运行，暂时用合成模拟数据
                deepseek_equity = [1.0 + i * 0.001 * (1 if i % 3 else -0.5) for i in range(len(user_equity))]

            # 基准线：线性增长
            benchmark_equity = [1.0 + i * (user_equity[-1] - 1.0) / max(len(user_equity) - 1, 1) for i in range(len(user_equity))]

            backtest_summary = {
                "total_return_pct": bt_result.total_return_pct,
                "annualized_return_pct": bt_result.annualized_return_pct,
                "max_drawdown_pct": bt_result.max_drawdown_pct,
                "trade_count": bt_result.trade_count,
                "strategy_name": bt_result.strategy_name,
                "start_date": request.start_date,
                "end_date": request.end_date,
            }

        else:  # simulation
            sr = modules["simulation_runner"]
            settlement_dir = sr._settlement_dir
            pattern = os.path.join(settlement_dir, f"{request.portfolio_id}_*.json")
            files = sorted(glob_module.glob(pattern))
            for fp in files:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        s = json.load(f)
                    tv = s.get("total_value", 0)
                    if tv > 0:
                        user_equity.append(tv)
                except Exception:
                    pass
            if user_equity and user_equity[0] > 0:
                base = user_equity[0]
                user_equity = [v / base for v in user_equity]

            backtest_summary = {
                "total_return_pct": round((user_equity[-1] - 1) * 100, 2) if user_equity else 0,
                "max_drawdown_pct": 0,
                "trade_count": len(files),
                "strategy_name": pf.bound_strategy or "无",
                "start_date": files[0] if files else "",
                "end_date": files[-1] if files else "",
            }

        # 生成图表
        chart_path = ""
        if user_equity:
            try:
                chart_path = gen.generate_comparison_chart(
                    user_equity=user_equity,
                    deepseek_equity=deepseek_equity if deepseek_equity else None,
                    benchmark_equity=benchmark_equity if benchmark_equity else None,
                )
            except Exception as e:
                logger.warning(f"图表生成失败: {e}")
                chart_path = ""

        # 构建 LLM 提示词
        chart_paths = [chart_path] if chart_path else []
        deepseek_summary_data = None
        if request.include_deepseek:
            deepseek_summary_data = {"total_return_pct": 0, "max_drawdown_pct": 0, "strategy_name": "DeepSeek 自由线"}
        prompt = gen.build_report_prompt(
            report_type=request.report_type,
            user_summary=backtest_summary,
            deepseek_summary=deepseek_summary_data,
            chart_paths=chart_paths,
        )

        # 调用 LLM 生成报告
        try:
            from src.utils.model_config import get_thinking_body
            base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL_6", os.getenv("OPENAI_COMPATIBLE_BASE_URL", ""))
            llm_client = OpenAICompatibleClient(
                api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY_6", os.getenv("OPENAI_COMPATIBLE_API_KEY", "")),
                base_url=base_url,
                model=os.getenv("OPENAI_COMPATIBLE_MODEL_6", "deepseek-v4-pro"),
                env_prefix="",
                extra_body=get_thinking_body(base_url, True),
            )

            llm_response = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    _thread_pool,
                    lambda: llm_client.get_completion([
                        {"role": "system", "content": "你是一位专业的A股投资报告分析师。请基于提供的数据生成结构化报告。使用 [数据]/[判断] 标签区分事实与推理。"},
                        {"role": "user", "content": prompt},
                    ], max_retries=1),
                ),
                timeout=120.0,
            )

            report_content = llm_response or "LLM 报告生成返回空结果"
        except asyncio.TimeoutError:
            report_content = "报告生成超时 (120s)，请稍后重试"
        except Exception as e:
            logger.warning(f"LLM 报告生成失败: {e}")
            report_content = f"报告生成失败: {e}"

        # 保存报告
        title = f"{request.report_type}_{request.portfolio_id}"
        report_path = gen.save_report(report_content, title)

        return {
            "status": "ok",
            "report_type": request.report_type,
            "portfolio_id": request.portfolio_id,
            "report_content": report_content,
            "report_path": report_path,
            "chart_path": chart_path,
            "summary": backtest_summary,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{ERROR_ICON} 报告生成失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"报告生成失败: {str(e)}")


# ── 11. 用户画像 ──

@app.get("/api/advisory/user-profile")
async def advisory_user_profile():
    """获取当前用户投资画像。

    返回风险承受能力、投资周期偏好、投资风格、
    偏好/回避板块及自定义偏好等信息。
    """
    try:
        modules = _init_advisory()
        upm = modules["user_profile_manager"]
        profile = upm.get_profile()
        summary = upm.get_profile_summary()
        return {
            "status": "ok",
            "profile": profile,
            "summary": summary,
        }
    except Exception as e:
        logger.error(f"{ERROR_ICON} 用户画像查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")
