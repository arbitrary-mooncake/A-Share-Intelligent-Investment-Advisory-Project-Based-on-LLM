"""
统一数据网关（4.1 定稿：DataGateway）。

工作方式（strangler 最小侵入集成）：
1. 打分运行开始前，prefetch_term_bundle() 按 TermDataSpec 一次性并行取齐
   该期限所需数据，结果写入与各 Agent 共享的工具缓存（tool_cache，相同 MD5 键）；
2. 各 Agent 现有的 _call_tool_safe 缓存检查自动命中预取结果——
   同一次运行内跨 Agent 的重复取数被消除（尤其并行启动时的重复调用），
   且所有 Agent 看到的数据快照完全一致；
3. 预取只缓存非空结果：空结果不预热，Agent 的空数据重试（retry_failed_fetches）
   行为与改造前完全一致；
4. 预取失败/部分失败不影响主流程：Agent 缓存未命中即走原有取数路径。

运行级 Bundle（ContextVar）同时保留一份结果副本，供需要绕过缓存的调用点
（如 technical 预计算的直接 ainvoke）以及未来的确定性特征层使用。
"""
import asyncio
import contextvars
import os
from typing import Any, Dict, List, Optional, Tuple

from src.data.term_data_spec import build_term_spec
from src.utils.logging_config import setup_logger
from src.utils.tool_cache import (
    _make_cache_key,
    get_cached_tool_result,
    set_cached_tool_result,
)

logger = setup_logger(__name__)

# 运行级 Bundle：{cache_key: result_text}，随打分请求生命周期存在
_run_bundle: contextvars.ContextVar[Optional[Dict[str, str]]] = contextvars.ContextVar(
    "data_gateway_run_bundle", default=None
)

# 与 Agent 包装器一致：只缓存非平凡结果（空结果留给 Agent 原有重试路径）
_MIN_RESULT_LEN = 20

# 预取并发上限（MCP stdio 单连接是瓶颈，过高并发无收益）
_PREFETCH_CONCURRENCY = int(os.getenv("DATA_PREFETCH_CONCURRENCY", "6"))
_PREFETCH_TIMEOUT = float(os.getenv("DATA_PREFETCH_TIMEOUT", "60"))


def prefetch_enabled() -> bool:
    return os.getenv("DATA_PREFETCH_ENABLED", "1").strip() != "0"


def get_prefetched(tool_name: str, kwargs: Dict[str, Any]) -> Optional[str]:
    """从运行级 Bundle 读取预取结果（供不走工具缓存的调用点使用）。"""
    bundle = _run_bundle.get()
    if not bundle:
        return None
    return bundle.get(_make_cache_key(tool_name, kwargs))


def get_run_bundle_stats() -> Dict[str, int]:
    bundle = _run_bundle.get()
    return {"bundle_size": len(bundle) if bundle else 0}


async def prefetch_term_bundle(
    term: str,
    stock_code: str,
    current_date: str = "",
    is_etf: bool = False,
) -> Optional[Dict[str, str]]:
    """按期限统一预取数据并预热共享工具缓存。失败返回 None，绝不抛出。"""
    if not prefetch_enabled():
        return None
    try:
        spec = build_term_spec(term, stock_code, current_date, is_etf)
        if not spec:
            return None

        from src.tools.mcp_client import get_mcp_tools
        tool_names = list({name for name, _ in spec})
        try:
            tools = await get_mcp_tools(tool_filter=tool_names)
        except Exception as e:
            logger.warning(f"DataGateway: 获取 MCP 工具失败，跳过预取: {e}")
            return None
        tool_map = {t.name: t for t in tools}

        bundle: Dict[str, str] = {}
        sem = asyncio.Semaphore(_PREFETCH_CONCURRENCY)

        async def _fetch_one(tool_name: str, kwargs: Dict[str, Any]) -> None:
            key = _make_cache_key(tool_name, kwargs)
            cached = await get_cached_tool_result(tool_name, kwargs)
            if cached:
                bundle[key] = cached
                return
            tool = tool_map.get(tool_name)
            if tool is None:
                return
            async with sem:
                try:
                    result = await asyncio.wait_for(
                        tool.ainvoke(kwargs), timeout=_PREFETCH_TIMEOUT
                    )
                    text = str(result).strip() if result is not None else ""
                    if len(text) > _MIN_RESULT_LEN:
                        await set_cached_tool_result(tool_name, kwargs, text)
                        bundle[key] = text
                except Exception:
                    # 单项失败不预热，Agent 走原有取数+重试路径
                    return

        await asyncio.gather(
            *[_fetch_one(name, kwargs) for name, kwargs in spec],
            return_exceptions=True,
        )
        _run_bundle.set(bundle)
        logger.info(
            f"DataGateway: 预取完成 term={term} code={stock_code} "
            f"spec={len(spec)} 项, 命中 {len(bundle)} 项"
        )
        return bundle
    except Exception as e:
        logger.warning(f"DataGateway: 预取异常（不影响主流程）: {e}")
        return None
