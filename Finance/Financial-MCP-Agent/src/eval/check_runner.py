"""
检查执行器 — 面向CLI和UI的统一检查入口。
封装orchestrator调用，提供同步/异步双模式。
"""
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional


async def run_check_async(current_date: str = "",
                           market_data: Dict = None) -> Dict[str, Any]:
    """异步运行完整检查"""
    from src.eval.orchestrator import EvalOrchestrator
    orch = EvalOrchestrator()
    return await orch.run_full_check(current_date, market_data)


def run_check(current_date: str = "", market_data: Dict = None) -> Dict[str, Any]:
    """同步运行完整检查"""
    return asyncio.run(run_check_async(current_date, market_data))


async def run_rebalance_async(term: str = "short", current_date: str = "",
                               market_data: Dict = None) -> Dict[str, Any]:
    """异步运行调仓"""
    from src.eval.orchestrator import EvalOrchestrator
    orch = EvalOrchestrator()
    return await orch.run_daily_rebalance(term, current_date or datetime.now().strftime("%Y-%m-%d"), market_data or {})


async def run_settlement_async(current_date: str = "") -> Dict[str, Any]:
    """异步运行结算"""
    from src.eval.orchestrator import EvalOrchestrator
    orch = EvalOrchestrator()
    return await orch.settle_historical(current_date or datetime.now().strftime("%Y-%m-%d"))


def get_eval_status() -> Dict[str, Any]:
    """获取评测系统状态"""
    from src.eval.orchestrator import EvalOrchestrator
    orch = EvalOrchestrator()
    return orch.get_status()
