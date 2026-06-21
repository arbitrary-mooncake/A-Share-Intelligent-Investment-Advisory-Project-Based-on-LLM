"""
评测专用缓存层 — 内存L1 + 磁盘L2，跨线路跨期限共享。
与生产缓存(cache_utils.py)完全隔离（_eval后缀），互不污染。
"""
import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional


# 缓存目录（与生产缓存在同一父目录但key隔离）
_EVAL_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "eval", "cache"
)

# 内存级L1缓存: {key: {data, cached_at}}
_L1_CACHE: Dict[str, Dict[str, Any]] = {}
_L1_LOCK = threading.Lock()  # Streamlit多线程安全

# TTL（秒）- 沿用生产系统TTL
DEFAULT_TTL = {
    "fundamental_analysis": 15 * 86400,
    "value_analysis": 7 * 86400,
    "technical_analysis": 1 * 86400,
    "news_analysis": 1 * 86400,
    "event_analysis": 1 * 86400,
    "quality_risk_analysis": 7 * 86400,
    "moneyflow_analysis": 1 * 86400,
    # 完整分析结果（包含所有agent输出+scorer评分）
    "full_analysis": 1 * 86400,
    # Scorer
    "short_term_scorer": 1 * 86400,
    "medium_term_scorer": 1 * 86400,
    "long_term_scorer": 1 * 86400,
}

# 最大L1条目数
MAX_L1_ENTRIES = 2000


def _make_cache_key(agent_name: str, stock_code: str, as_of_date: str) -> str:
    """生成评测专用缓存key"""
    safe_code = stock_code.replace(".", "_").replace("/", "_")
    return f"{agent_name}_{safe_code}_{as_of_date}_eval"


def _get_cache_path(agent_name: str, stock_code: str, as_of_date: str) -> str:
    """获取磁盘缓存文件路径"""
    os.makedirs(_EVAL_CACHE_DIR, exist_ok=True)
    key = _make_cache_key(agent_name, stock_code, as_of_date)
    return os.path.join(_EVAL_CACHE_DIR, f"{key}.json")


def _is_fresh(cached_at_str: str, agent_name: str) -> bool:
    """检查缓存是否在TTL内"""
    try:
        cached_at = datetime.fromisoformat(cached_at_str)
        ttl = DEFAULT_TTL.get(agent_name, 86400)
        age = (datetime.now() - cached_at).total_seconds()
        return age <= ttl
    except Exception:
        return False


def read_cache(agent_name: str, stock_code: str, as_of_date: str) -> Optional[str]:
    """
    读取缓存：L1 → L2 → None。
    返回缓存内容字符串（通常是LLM分析文本），未命中返回None。
    """
    key = _make_cache_key(agent_name, stock_code, as_of_date)

    # L1: 内存
    with _L1_LOCK:
        entry = _L1_CACHE.get(key)
    if entry:
        if _is_fresh(entry.get("cached_at", ""), agent_name):
            return entry.get("content", "")
        else:
            with _L1_LOCK:
                _L1_CACHE.pop(key, None)  # 过期淘汰

    # L2: 磁盘
    cache_path = _get_cache_path(agent_name, stock_code, as_of_date)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_at = data.get("cached_at", "")
            if _is_fresh(cached_at, agent_name):
                content = data.get("content", "")
                # 提升到L1
                _L1_CACHE[key] = {"content": content, "cached_at": cached_at}
                _evict_l1_if_needed()
                return content
        except Exception:
            pass

    return None


def write_cache(agent_name: str, stock_code: str, as_of_date: str, content: str):
    """写入缓存：L1 + L2（异步写磁盘）"""
    key = _make_cache_key(agent_name, stock_code, as_of_date)
    now = datetime.now().isoformat()

    # L1: 立即写入内存
    _L1_CACHE[key] = {"content": content, "cached_at": now}
    _evict_l1_if_needed()

    # L2: 同步写磁盘（对性能影响小，simplify实现）
    cache_path = _get_cache_path(agent_name, stock_code, as_of_date)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "agent_name": agent_name,
                "stock_code": stock_code,
                "as_of_date": as_of_date,
                "cached_at": now,
                "content": content,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 磁盘写失败不影响主流程


def read_signal_pack_cache(agent_name: str, stock_code: str, as_of_date: str) -> Optional[dict]:
    """
    读取缓存的signal_pack JSON。
    先查L1，再查L2磁盘（独立文件，*_signal_pack_*_eval.json）。
    """
    key = f"sigpack_{_make_cache_key(agent_name, stock_code, as_of_date)}"

    with _L1_LOCK:
        entry = _L1_CACHE.get(key)
    if entry:
        if _is_fresh(entry.get("cached_at", ""), agent_name):
            return entry.get("data")
        else:
            del _L1_CACHE[key]

    safe_code = stock_code.replace(".", "_").replace("/", "_")
    cache_path = os.path.join(
        _EVAL_CACHE_DIR,
        f"{agent_name}_signal_pack_{safe_code}_{as_of_date}_eval.json"
    )
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if _is_fresh(data.get("cached_at", ""), agent_name):
                _L1_CACHE[key] = {"data": data, "cached_at": data.get("cached_at", "")}
                _evict_l1_if_needed()
                return data
        except Exception:
            pass

    return None


def write_signal_pack_cache(agent_name: str, stock_code: str, as_of_date: str,
                             signal_pack: dict):
    """缓存signal_pack JSON"""
    key = f"sigpack_{_make_cache_key(agent_name, stock_code, as_of_date)}"
    now = datetime.now().isoformat()

    signal_pack["cached_at"] = now
    _L1_CACHE[key] = {"data": signal_pack, "cached_at": now}
    _evict_l1_if_needed()

    safe_code = stock_code.replace(".", "_").replace("/", "_")
    cache_path = os.path.join(
        _EVAL_CACHE_DIR,
        f"{agent_name}_signal_pack_{safe_code}_{as_of_date}_eval.json"
    )
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(signal_pack, f, ensure_ascii=False, default=str, indent=2)
    except Exception:
        pass


def preload_from_disk():
    """启动时从磁盘预加载缓存到L1"""
    if not os.path.exists(_EVAL_CACHE_DIR):
        return
    try:
        loaded = 0
        for fname in os.listdir(_EVAL_CACHE_DIR):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(_EVAL_CACHE_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                agent_name = data.get("agent_name", "")
                cached_at = data.get("cached_at", "")
                if _is_fresh(cached_at, agent_name):
                    key = fname[:-5]  # remove .json
                    _L1_CACHE[key] = {"content": data.get("content", ""),
                                       "data": data,
                                       "cached_at": cached_at}
                    loaded += 1
            except Exception:
                pass
            if loaded >= MAX_L1_ENTRIES:
                break
    except Exception:
        pass


def _evict_l1_if_needed():
    """L1超过上限时淘汰最旧的条目"""
    if len(_L1_CACHE) <= MAX_L1_ENTRIES:
        return
    # 按缓存时间排序，淘汰最旧的20%
    sorted_entries = sorted(
        _L1_CACHE.items(),
        key=lambda x: x[1].get("cached_at", "2000-01-01")
    )
    to_remove = int(len(sorted_entries) * 0.2)
    for key, _ in sorted_entries[:to_remove]:
        del _L1_CACHE[key]


def clear_cache(agent_name: Optional[str] = None):
    """清除缓存：agent_name=None时清除全部"""
    if agent_name:
        to_remove = [k for k in _L1_CACHE if agent_name in k]
        for k in to_remove:
            del _L1_CACHE[k]
    else:
        _L1_CACHE.clear()


def get_cache_stats() -> Dict[str, Any]:
    """获取缓存统计"""
    return {
        "l1_size": len(_L1_CACHE),
        "l1_max": MAX_L1_ENTRIES,
        "l2_dir": _EVAL_CACHE_DIR,
        "l2_files": len(os.listdir(_EVAL_CACHE_DIR)) if os.path.exists(_EVAL_CACHE_DIR) else 0,
    }
