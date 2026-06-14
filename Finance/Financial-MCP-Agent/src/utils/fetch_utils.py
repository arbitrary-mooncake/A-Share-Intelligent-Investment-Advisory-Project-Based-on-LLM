"""
数据获取重试工具 — Phase 1 并行取数后，对返回空数据的工具自动重试（最多4轮），
重试并发逐轮递减以避免压垮已不稳定的上游数据源。
"""
import asyncio
import logging
from typing import List, Callable, Any, Coroutine

logger = logging.getLogger(__name__)

# 重试并发数递减表：第1→2→3轮重试的并发上限
_RETRY_SEMAPHORE_SCHEDULE = (8, 4, 2)


def is_empty_result(text: str) -> bool:
    """检查工具返回是否为空/无效（需要重试）"""
    if not text:
        return True
    t = str(text).strip()
    if len(t) < 30:
        return True
    markers = ["数据不可用", "工具不可用", "工具调用异常", "返回过短", "调用失败"]
    return any(m in t for m in markers)


def _is_unrecoverable(result) -> bool:
    """检测不可恢复错误 — MCP stdio 子进程已崩溃，重试必定失败。
    范围极窄，仅包含已确认不可恢复的错误模式。"""
    if isinstance(result, Exception):
        msg = str(result)
        if "TaskGroup" in msg:
            return True
    elif isinstance(result, str):
        if "TaskGroup" in result:
            return True
    return False


async def retry_failed_fetches(
    results: list,
    tool_infos: list,
    labels: list,
    call_fn,
    agent_label: str = "",
    max_extra_rounds: int = 3,
    alt_kwargs_list: list = None,
) -> list:
    """
    对 Phase 1 结果中空数据的工具进行最多 max_extra_rounds 轮重试。
    如果覆盖率已达 100%，提前跳出。重试并发逐轮递减（8→4→2）。
    不可恢复错误（如 MCP stdio TaskGroup 崩溃）自动跳过重试。

    Args:
        results: 初始 asyncio.gather 的结果列表
        tool_infos: 每个任务对应的 (tool, kwargs)，占位任务为 None
        labels: 每个任务的标签（用于日志）
        call_fn: async (tool, kwargs, label) -> str 的工具调用函数
        agent_label: Agent 名称（用于日志）
        max_extra_rounds: 最多额外重试轮数（总执行 = 1 + max_extra_rounds）
        alt_kwargs_list: 每个任务对应的备选 kwargs 列表，初始调用失败后首轮重试用备选参数。
                         长度须与 tool_infos 一致，无备选的任务填 None。

    Returns:
        更新后的 results 列表
    """
    results = list(results)  # 确保可变
    total = len([ti for ti in tool_infos if ti is not None])  # 可重试的工具总数
    if alt_kwargs_list is None:
        alt_kwargs_list = [None] * len(tool_infos)
    alt_used = set()  # 已使用备选参数的任务索引

    for retry_round in range(max_extra_rounds):
        # 找到空结果且可重试的索引
        retry_indices = []
        for i, r in enumerate(results):
            if i >= len(tool_infos) or tool_infos[i] is None:
                continue  # 占位任务，不可重试
            if _is_unrecoverable(r):
                if i not in alt_used:
                    logger.info(f"{agent_label}: {labels[i]} 不可恢复错误，跳过重试")
                continue  # MCP stdio 崩溃等不可恢复错误，跳过
            if isinstance(r, Exception):
                retry_indices.append(i)
            elif is_empty_result(str(r)):
                retry_indices.append(i)

        if not retry_indices:
            break  # 100% 覆盖率或全部不可恢复

        # 本轮并发上限（从递减表中取，超出表长则用最小值 2）
        if retry_round < len(_RETRY_SEMAPHORE_SCHEDULE):
            sem_limit = _RETRY_SEMAPHORE_SCHEDULE[retry_round]
        else:
            sem_limit = 2
        sem = asyncio.Semaphore(sem_limit)

        # 计算当前覆盖率
        success_count = sum(
            1 for i, r in enumerate(results)
            if not isinstance(r, Exception) and not is_empty_result(str(r))
        )
        prefix = f"{agent_label}: " if agent_label else ""
        print(f"{prefix}第{retry_round + 1}轮重试 {len(retry_indices)} 个空数据工具 "
              f"(当前覆盖率 {success_count}/{total}, 并发={sem_limit})...")

        # 重建失败工具的任务（带并发限制）
        async def _limited_call(tool, kwargs, label):
            async with sem:
                return await call_fn(tool, kwargs, label)

        retry_tasks = []
        for i in retry_indices:
            tool, orig_kwargs = tool_infos[i]
            # 首轮重试且有备选参数时使用备选 kwargs
            use_alt = (retry_round == 0 and i < len(alt_kwargs_list)
                       and alt_kwargs_list[i] is not None)
            if use_alt:
                kwargs = alt_kwargs_list[i]
                alt_used.add(i)
            else:
                kwargs = orig_kwargs
            retry_tasks.append(_limited_call(tool, kwargs, labels[i]))

        # 并行重试（受 semaphore 限制）
        try:
            retry_new = await asyncio.gather(*retry_tasks, return_exceptions=True)
        except Exception:
            retry_new = [Exception("gather failed")] * len(retry_indices)

        # 更新结果
        for idx, new_r in zip(retry_indices, retry_new):
            results[idx] = new_r

    return results
