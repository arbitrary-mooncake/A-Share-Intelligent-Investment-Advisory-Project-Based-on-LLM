"""Stock data disk cache for L1/L2 batch scoring pipeline.

Caches raw market data (fundamentals, valuation, market_data, company_info)
to disk so that repeated runs within the TTL window skip Tushare/HTTP fetches.

This cache is shared across:
  - batch_scorer._fetch_light_stock_data_sync (L1/L2 data path)
  - Future agent-level reuse (not yet implemented)

NOT for LLM scoring results — those are intentionally not cached (user decision).
"""
import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "intermediate_cache", "stock_data")

DATA_TTL_DAYS = {
    "fundamentals": 7,
    "valuation": 7,
    "market_data": 1,
    "company_info": 30,
}


def _cache_path(data_type: str, code: str, date_str: str) -> str:
    safe_code = code.replace(".", "_").replace("/", "_")
    return os.path.join(DATA_CACHE_DIR, f"{data_type}_{safe_code}_{date_str}.json")


def read_data_cache(data_type: str, code: str, date_str: str) -> Optional[Dict[str, Any]]:
    """Read cached stock data if it exists and is within TTL.

    Returns None if cache miss or expired.
    """
    path = _cache_path(data_type, code, date_str)
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
        age_days = (datetime.now() - datetime.fromtimestamp(mtime)).days
        ttl = DATA_TTL_DAYS.get(data_type, 1)
        if age_days > ttl:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_data_cache(data_type: str, code: str, date_str: str, data: Dict[str, Any]):
    """Write stock data to disk cache."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    path = _cache_path(data_type, code, date_str)
    try:
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)
    except Exception as e:
        logger.debug("[StockDataCache] write failed for %s/%s: %s", data_type, code, e)


def split_cached_result(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Split a fetch result dict into cacheable data_type buckets.

    Args:
        result: The dict returned by _fetch_light_stock_data_sync

    Returns:
        {"fundamentals": {...}, "market_data": {...}, "company_info": {...}}
    """
    fundamentals_keys = {
        "pe", "pb", "ps", "roe", "gross_margin", "net_margin",
        "debt_ratio", "revenue_growth", "profit_growth",
    }
    market_data_keys = {
        "recent_kline", "price_changes", "last_price", "pct_chg",
        "market_cap", "turnover_rate",
    }
    company_info_keys = {
        "name", "industry", "code",
    }

    buckets = {}
    fund = {k: v for k, v in result.items() if k in fundamentals_keys}
    if fund:
        buckets["fundamentals"] = fund

    mkt = {k: v for k, v in result.items() if k in market_data_keys}
    if mkt:
        buckets["market_data"] = mkt

    info = {k: v for k, v in result.items() if k in company_info_keys}
    if info:
        buckets["company_info"] = info

    return buckets


def merge_cached_into_result(
    result: Dict[str, Any],
    cached_buckets: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge cached data buckets back into a result dict (cached wins for missing keys)."""
    for bucket in cached_buckets.values():
        for k, v in bucket.items():
            if k not in result or not result[k]:
                result[k] = v
    return result
