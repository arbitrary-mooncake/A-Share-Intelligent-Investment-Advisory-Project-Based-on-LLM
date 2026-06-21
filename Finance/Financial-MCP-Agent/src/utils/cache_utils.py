"""
中间产物缓存工具 — 避免短期内对同一股票重复运行分析Agent。
缓存 Key: {agent_name}_{stock_code}_{date}.json
缓存时效: 按 agent 类型区分 TTL（基本面15天/估值7天/技术新闻1天）

缓存命名空间（V2新增）:
  - 默认（生产环境 M1/M3）: data/intermediate_cache/
  - eval（评测环境 M5）: data/eval/cache/ + _eval 后缀
  精筛池筛选使用生产模型，产物可被生产环境复用；
  日常模拟盘/回测使用评测模型M5，产物严格隔离，绝不污染生产缓存。
"""
import os
import json
import threading
from datetime import datetime
from typing import Optional

# 缓存命名空间 — 线程安全
# None = 生产环境（data/intermediate_cache/）
# "eval" = 评测环境（data/eval/cache/，_eval后缀）
_cache_namespace: Optional[str] = None
_namespace_lock = threading.Lock()

# 基础缓存目录
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 生产缓存目录（默认）
_PROD_CACHE_DIR = os.path.join(_BASE_DIR, "data", "intermediate_cache")
# 评测缓存目录
_EVAL_CACHE_DIR = os.path.join(_BASE_DIR, "data", "eval", "cache")
os.makedirs(_PROD_CACHE_DIR, exist_ok=True)
os.makedirs(_EVAL_CACHE_DIR, exist_ok=True)


def set_cache_namespace(namespace: Optional[str]):
    """
    设置缓存命名空间。

    Args:
        namespace: None（生产环境）或 "eval"（评测环境）

    评测环境调用 set_cache_namespace("eval") 后，
    所有后续的 read_cache/write_cache 操作自动路由到
    data/eval/cache/ 目录，文件名追加 _eval 后缀，
    与生产缓存完全物理隔离。
    """
    global _cache_namespace
    with _namespace_lock:
        _cache_namespace = namespace


def get_cache_namespace() -> Optional[str]:
    """获取当前缓存命名空间"""
    return _cache_namespace


def _get_active_cache_dir() -> str:
    """根据当前命名空间返回活跃的缓存目录"""
    if _cache_namespace == "eval":
        return _EVAL_CACHE_DIR
    return _PROD_CACHE_DIR


def _get_filename_suffix() -> str:
    """评测环境返回 _eval 后缀，生产环境无后缀"""
    return "_eval" if _cache_namespace == "eval" else ""

# 各 agent 缓存 TTL（天数）
AGENT_CACHE_TTL = {
    # Stock agents
    "fundamental_analysis": 15,
    "value_analysis": 7,
    "technical_analysis": 1,
    "news_analysis": 1,
    "event_analysis": 1,            # Event/news data needs daily freshness
    "quality_risk_analysis": 7,     # Financial quality data is quarterly, 7-day window
    "moneyflow_analysis": 1,        # Daily market data
    # Fund agents
    "fund_product_doc": 30,      # Fund basic info rarely changes
    "fund_perf_risk": 3,         # NAV updates daily, analysis is heavy
    "fund_holdings": 30,         # Portfolio disclosed quarterly
    "fund_manager": 15,          # Manager changes infrequent
    "fund_benchmark": 7,         # Benchmark consistency medium-term
    "fund_fee": 15,              # Fee structure changes rarely
    "fund_event": 1,             # Events/news need daily freshness
}


def get_cache_path(agent_name: str, stock_code: str, date_str: str) -> str:
    """返回缓存文件路径（根据当前命名空间自动路由）"""
    safe_code = stock_code.replace(".", "_").replace("/", "_")
    suffix = _get_filename_suffix()
    cache_dir = _get_active_cache_dir()
    return os.path.join(cache_dir, f"{agent_name}_{safe_code}_{date_str}{suffix}.json")


def _list_cached_dates(agent_name: str, stock_code: str) -> list:
    """列出某 agent+stock 的所有缓存日期（最新在前）"""
    safe_code = stock_code.replace(".", "_").replace("/", "_")
    cache_dir = _get_active_cache_dir()
    suffix = _get_filename_suffix()
    prefix = f"{agent_name}_{safe_code}_"
    dates = []
    try:
        for fname in os.listdir(cache_dir):
            if fname.startswith(prefix) and fname.endswith(f"{suffix}.json"):
                date_part = fname[len(prefix):-len(f"{suffix}.json")]
                if len(date_part) == 10:  # YYYY-MM-DD
                    dates.append(date_part)
    except Exception:
        pass
    dates.sort(reverse=True)
    return dates


def read_cache(agent_name: str, stock_code: str, date_str: str) -> Optional[str]:
    """
    读取缓存的 agent 分析结果。
    在 TTL 窗口内查找最近的缓存文件，命中则返回内容，否则返回 None。
    """
    ttl = AGENT_CACHE_TTL.get(agent_name, 1)
    try:
        query_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        # date_str 格式异常时回退到精确匹配
        path = get_cache_path(agent_name, stock_code, date_str)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("content", "") or None
            except Exception:
                pass
        return None

    # 在 TTL 窗口内查找最近的缓存
    cached_dates = _list_cached_dates(agent_name, stock_code)
    for cached_date_str in cached_dates:
        try:
            cached_date = datetime.strptime(cached_date_str, "%Y-%m-%d")
        except ValueError:
            continue
        age_days = (query_date - cached_date).days
        if 0 <= age_days <= ttl:
            path = get_cache_path(agent_name, stock_code, cached_date_str)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                content = data.get("content", "")
                if content:
                    return content
            except Exception:
                continue

    return None


def write_cache(agent_name: str, stock_code: str, date_str: str, content: str):
    """将 agent 分析结果写入缓存"""
    path = get_cache_path(agent_name, stock_code, date_str)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "agent_name": agent_name,
                "stock_code": stock_code,
                "date": date_str,
                "cached_at": datetime.now().isoformat(),
                "content": content,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 缓存写入失败不应影响主流程


def read_signal_pack_cache(agent_name: str, stock_code: str, date_str: str) -> dict | None:
    """Read cached signal_pack JSON. Returns None if not found."""
    safe_code = stock_code.replace(".", "_").replace("/", "_")
    cache_dir = _get_active_cache_dir()
    suffix = _get_filename_suffix()
    cache_path = os.path.join(cache_dir, f"{agent_name}_signal_pack_{safe_code}_{date_str}{suffix}.json")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def write_signal_pack_cache(agent_name: str, stock_code: str, date_str: str, signal_pack: dict) -> None:
    """Cache signal_pack as JSON."""
    safe_code = stock_code.replace(".", "_").replace("/", "_")
    cache_dir = _get_active_cache_dir()
    suffix = _get_filename_suffix()
    cache_path = os.path.join(cache_dir, f"{agent_name}_signal_pack_{safe_code}_{date_str}{suffix}.json")
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(signal_pack, f, ensure_ascii=False, default=str)
    except Exception:
        pass  # cache write failure is non-fatal
