"""
Web API适配器 — 总纲 §17.1
为Streamlit前端提供统一的异步数据访问接口。
V3: 支持流式进度回调 + ETA + 阶段展示。
"""
import asyncio
import json
import threading
import time
from typing import Dict, Any, Optional, Callable


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

    # ── 带进度回调的池更新 (Streamlit 轮询模式) ──

    def run_pool_update_streaming(
        self,
        term: str = "short",
        on_progress: Optional[Callable] = None,
        on_stage: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """在后台线程运行精筛池更新, 期间实时回调进度到 Streamlit.

        使用 threading 将异步管线在后台运行, 主线程 (Streamlit) 通过
        on_progress 回调接收结构化进度数据 {overall_pct, eta_str, stages, ...}
        来更新进度条和 ETA 显示。

        Returns:
            更新结果字典 {"pool", "stats", ...} 或 {"error": ...}
        """
        if not self._orch:
            return {"error": "orchestrator not initialized"}

        result_holder = {"result": None, "error": None, "done": False}

        async def _run():
            try:
                r = await self._orch.run_pool_update(
                    term=term,
                    on_stage=on_stage,
                    on_progress=on_progress,
                )
                result_holder["result"] = r
            except Exception as e:
                result_holder["error"] = str(e)
            finally:
                result_holder["done"] = True

        def _thread_target():
            asyncio.run(_run())

        thread = threading.Thread(target=_thread_target, daemon=False)
        thread.start()

        return result_holder  # caller polls result_holder["done"] for completion
