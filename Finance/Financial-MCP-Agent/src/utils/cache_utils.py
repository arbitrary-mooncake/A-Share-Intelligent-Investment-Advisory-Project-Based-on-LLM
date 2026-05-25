"""
中间产物缓存工具 — 避免短期内对同一股票重复运行分析Agent。
缓存 Key: {agent_name}_{stock_code}_{date}.json
缓存时效: 按 agent 类型区分 TTL（基本面15天/估值7天/技术新闻1天）
"""
import os
import json
from datetime import datetime
from typing import Optional

# 缓存目录（与 app.py 共用 data/intermediate_cache/）
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "intermediate_cache"
)
os.makedirs(_CACHE_DIR, exist_ok=True)

# 各 agent 缓存 TTL（天数）
AGENT_CACHE_TTL = {
    "fundamental_analysis": 15,
    "value_analysis": 7,
    "technical_analysis": 1,
    "news_analysis": 1,
}


def get_cache_path(agent_name: str, stock_code: str, date_str: str) -> str:
    """返回缓存文件路径"""
    safe_code = stock_code.replace(".", "_").replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{agent_name}_{safe_code}_{date_str}.json")


def _list_cached_dates(agent_name: str, stock_code: str) -> list:
    """列出某 agent+stock 的所有缓存日期（最新在前）"""
    safe_code = stock_code.replace(".", "_").replace("/", "_")
    prefix = f"{agent_name}_{safe_code}_"
    dates = []
    try:
        for fname in os.listdir(_CACHE_DIR):
            if fname.startswith(prefix) and fname.endswith(".json"):
                date_part = fname[len(prefix):-len(".json")]
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
