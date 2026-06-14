"""
MCP Tool-Level Cache: in-memory cache for MCP tool results.
Multiple agents often call the same Tushare tool with identical parameters
in a single scoring run. This cache avoids redundant API calls.

TTL: 5 minutes (scoped to a single scoring run)
Thread-safe: uses asyncio.Lock for concurrent access
"""
import asyncio
import hashlib
import json
import time
from typing import Any, Optional

# {cache_key: (timestamp, result)}
_cache: dict = {}
_lock = asyncio.Lock()
TTL_SECONDS = 300  # 5 minutes


def _make_cache_key(tool_name: str, kwargs: dict) -> str:
    """Generate a deterministic cache key from tool name + arguments."""
    raw = json.dumps({"tool": tool_name, "args": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


async def get_cached_tool_result(tool_name: str, kwargs: dict) -> Optional[str]:
    """Return cached result if available and not expired."""
    key = _make_cache_key(tool_name, kwargs)
    async with _lock:
        if key in _cache:
            ts, result = _cache[key]
            if time.time() - ts < TTL_SECONDS:
                return result
            del _cache[key]
    return None


async def set_cached_tool_result(tool_name: str, kwargs: dict, result: str) -> None:
    """Store tool result in cache."""
    key = _make_cache_key(tool_name, kwargs)
    async with _lock:
        _cache[key] = (time.time(), result)
        # Prevent unbounded growth: keep max 500 entries
        if len(_cache) > 500:
            oldest_key = min(_cache, key=lambda k: _cache[k][0])
            del _cache[oldest_key]


def clear_tool_cache() -> None:
    """Clear all cached entries (for testing)."""
    _cache.clear()
