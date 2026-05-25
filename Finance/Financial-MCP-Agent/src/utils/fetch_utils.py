"""
数据获取重试工具 — Phase 1 并行取数后，对返回空数据的工具自动重试（最多3轮）。
"""
import asyncio
import time
from typing import List, Callable, Any, Coroutine


def is_empty_result(text: str) -> bool:
    """检查工具返回是否为空/无效（需要重试）"""
    if not text:
        return True
    t = str(text).strip()
    if len(t) < 30:
        return True
    markers = ["数据不可用", "工具不可用", "工具调用异常", "返回过短", "调用失败"]
    return any(m in t for m in markers)


async def retry_failed_fetches(
    results: list,
    tool_infos: list,
    labels: list,
    call_fn,
    agent_label: str = "",
    max_extra_rounds: int = 2,
) -> list:
    """
    对 Phase 1 结果中空数据的工具进行最多 max_extra_rounds 轮重试。
    如果覆盖率已达 100%，提前跳出。

    Args:
        results: 初始 asyncio.gather 的结果列表
        tool_infos: 每个任务对应的 (tool, kwargs)，占位任务为 None
        labels: 每个任务的标签（用于日志）
        call_fn: async (tool, kwargs, label) -> str 的工具调用函数
        agent_label: Agent 名称（用于日志）
        max_extra_rounds: 最多额外重试轮数（总执行 = 1 + max_extra_rounds）

    Returns:
        更新后的 results 列表
    """
    results = list(results)  # 确保可变
    total = len([ti for ti in tool_infos if ti is not None])  # 可重试的工具总数

    for retry_round in range(max_extra_rounds):
        # 找到空结果且可重试的索引
        retry_indices = []
        for i, r in enumerate(results):
            if i >= len(tool_infos) or tool_infos[i] is None:
                continue  # 占位任务，不可重试
            if isinstance(r, Exception):
                retry_indices.append(i)
            elif is_empty_result(str(r)):
                retry_indices.append(i)

        if not retry_indices:
            break  # 100% 覆盖率

        # 计算当前覆盖率
        success_count = sum(
            1 for i, r in enumerate(results)
            if not isinstance(r, Exception) and not is_empty_result(str(r))
        )
        prefix = f"{agent_label}: " if agent_label else ""
        print(f"{prefix}第{retry_round + 1}轮重试 {len(retry_indices)} 个空数据工具 "
              f"(当前覆盖率 {success_count}/{total})...")

        # 重建失败工具的任务
        retry_tasks = []
        for i in retry_indices:
            tool, kwargs = tool_infos[i]
            retry_tasks.append(call_fn(tool, kwargs, labels[i]))

        # 并行重试
        try:
            retry_new = await asyncio.gather(*retry_tasks, return_exceptions=True)
        except Exception:
            retry_new = [Exception("gather failed")] * len(retry_indices)

        # 更新结果
        for idx, new_r in zip(retry_indices, retry_new):
            results[idx] = new_r

    return results
