"""
AKShare 异步客户端：封装 AKShare 调用，提供限流 + 缓存 + 异常保护。

设计目的：
- Lite 模式下当 Tushare 积分不足时，自动回退到 AKShare 获取数据
- 内置限流（2 次/秒），防止被数据源封 IP
- 结果缓存（5 分钟 TTL），减少重复调用
- 异常封装，失败时返回 None 而不是抛出异常
"""
import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# AKShare 缓存
_cache: Dict[str, Any] = {}
_cache_timestamps: Dict[str, float] = {}
_CACHE_TTL = 300  # 5 分钟

# 限流
_semaphore: Optional[asyncio.Semaphore] = None
_last_call_time = 0.0
_MIN_INTERVAL = 0.5  # 2 次/秒


def _get_semaphore() -> asyncio.Semaphore:
    """获取限流信号量（懒初始化）"""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(2)
    return _semaphore


def _make_cache_key(func_name: str, kwargs: dict) -> str:
    """生成缓存键"""
    key_parts = [func_name]
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cached(cache_key: str) -> Optional[Any]:
    """获取缓存值，过期或不存在返回 None"""
    if cache_key not in _cache:
        return None
    if time.time() - _cache_timestamps.get(cache_key, 0) > _CACHE_TTL:
        del _cache[cache_key]
        del _cache_timestamps[cache_key]
        return None
    return _cache[cache_key]


def _set_cache(cache_key: str, value: Any):
    """设置缓存"""
    if len(_cache) >= 500:
        oldest_key = min(_cache_timestamps, key=_cache_timestamps.get)
        del _cache[oldest_key]
        del _cache_timestamps[oldest_key]
    _cache[cache_key] = value
    _cache_timestamps[cache_key] = time.time()


async def call_akshare(func_name: str, **kwargs) -> Optional[Any]:
    """
    调用 AKShare 函数（异步）。

    Args:
        func_name: AKShare 函数名（如 "stock_financial_analysis_indicator"）
        **kwargs: 传递给 AKShare 函数的参数

    Returns:
        函数返回值（通常是 DataFrame），失败返回 None
    """
    cache_key = _make_cache_key(func_name, kwargs)

    cached = _get_cached(cache_key)
    if cached is not None:
        logger.debug(f"AKShare cache hit: {func_name}")
        return cached

    sem = _get_semaphore()
    async with sem:
        global _last_call_time
        elapsed = time.time() - _last_call_time
        if elapsed < _MIN_INTERVAL:
            await asyncio.sleep(_MIN_INTERVAL - elapsed)
        _last_call_time = time.time()

        try:
            import akshare as ak

            func = getattr(ak, func_name, None)
            if func is None:
                logger.warning(f"AKShare function not found: {func_name}")
                return None

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: func(**kwargs)
            )

            _set_cache(cache_key, result)
            logger.info(f"AKShare call success: {func_name}")
            return result

        except ImportError:
            logger.error("AKShare not installed. Run: pip install akshare")
            return None
        except Exception as e:
            logger.warning(f"AKShare call failed: {func_name}({kwargs}) -> {e}")
            return None


def dataframe_to_dicts(df) -> list:
    """将 pandas DataFrame 转换为字典列表"""
    if df is None:
        return []
    if hasattr(df, "to_dict"):
        return df.to_dict(orient="records")
    return []


def dataframe_to_tushare_format(df) -> list:
    """将 pandas DataFrame 转换为 Tushare {fields, items} 兼容格式"""
    if df is None or not hasattr(df, "columns"):
        return []
    fields = list(df.columns)
    items = df.values.tolist()
    return {"fields": fields, "items": items}
