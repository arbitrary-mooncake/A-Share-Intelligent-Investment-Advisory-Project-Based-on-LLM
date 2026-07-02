"""
Web API适配器 — 总纲 §17.1
为Streamlit前端提供统一的异步数据访问接口。
V3: 支持流式进度回调 + ETA + 阶段展示。
"""
import asyncio
from typing import Dict, Any


class WebAdapter:
    """Web前端适配器 — 封装评测系统的异步调用"""

    def __init__(self, orchestrator=None):
        self._orch = orchestrator

    def _run_async(self, coro):
        """Safely run async coroutine in any context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    def run_full_check(self) -> Dict[str, Any]:
        """一键完整检查"""
        if self._orch:
            return self._run_async(self._orch.run_full_check())
        return {"error": "orchestrator not initialized"}

    def run_rebalance(self, term: str = "short") -> Dict[str, Any]:
        """收盘前调仓"""
        if self._orch:
            return self._run_async(
                self._orch.run_daily_rebalance(term, "", {})
            )
        return {"error": "orchestrator not initialized"}

    def run_settlement(self) -> Dict[str, Any]:
        """收盘后结算"""
        if self._orch:
            self._orch.run_daily_settlement()
            return {"status": "completed"}
        return {"error": "orchestrator not initialized"}

    def run_pool_update(self, term: str = "short") -> Dict[str, Any]:
        """精筛池更新 (V3 流式管线)"""
        if self._orch:
            return self._run_async(self._orch.run_pool_update(term))
        return {"error": "orchestrator not initialized"}

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        if self._orch:
            return self._orch.get_status()
        return {"error": "orchestrator not initialized"}

