"""
FastAPI HTTP 客户端封装
- 封装所有后端 API 调用
- 含错误处理、超时管理
- 支持 Mock 模式（后端未启动时前端仍可开发）
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import asyncio
import httpx
from typing import Optional
from config import (
    API_BASE_URL,
    QUERY_TIMEOUT,
    REPORT_TIMEOUT,
    SCORE_TRIGGER_TIMEOUT,
    SCORE_POLL_INTERVAL,
    SCORE_POLL_MAX_ATTEMPTS,
    QUICK_SCREEN_TIMEOUT,
    BATCH_UPLOAD_TIMEOUT,
    BATCH_POLL_INTERVAL,
    BATCH_POLL_MAX_ATTEMPTS,
    POLL_INTERVAL,
    POLL_MAX_ATTEMPTS,
    QA_STREAM_TIMEOUT,
)

# ──────────────────────────────────────────────
# Mock 模式开关
# 当后端 FastAPI 未启动时，返回模拟数据供前端开发
# ──────────────────────────────────────────────
MOCK_MODE = False  # 设为 True 启用 Mock


class APIError(Exception):
    """API 调用异常"""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _mock_response(data: dict) -> dict:
    """返回模拟数据（开发阶段使用）"""
    return data


async def _check_backend_health() -> bool:
    """检查后端是否可达"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{API_BASE_URL}/docs")
            return resp.status_code == 200
    except Exception:
        return False


# ──────────────────────────────────────────────
# 股票查询 API
# ──────────────────────────────────────────────

async def quick_query(stock_input: str) -> dict:
    """快速查询 — 返回股票基本信息"""
    if MOCK_MODE:
        return {
            "stock_code": "688256",
            "stock_name": "寒武纪",
            "market_cap": "1200亿",
            "pb": "15.2",
            "pe": "120.5",
            "turnover_rate": "3.8%",
            "price_changes": {
                "1d": "+2.5%", "5d": "-1.2%", "1m": "+8.3%",
                "3m": "+15.6%", "6m": "+25.1%", "1y": "+45.2%", "3y": "+120.8%"
            },
            "industry": "半导体",
            "industry_intro": "半导体设计与制造行业",
            "company_intro": "寒武纪是全球领先的 AI 芯片设计公司",
            "industry_benchmark": {
                "industry_name": "电子",
                "pe_reasonable_range": "25-80倍",
                "pb_reasonable_range": "2.5-6.5倍",
                "pe_cheap_threshold": 25.0,
                "pe_expensive_threshold": 80.0,
                "pb_cheap_threshold": 2.5,
                "pb_expensive_threshold": 6.5,
                "primary_valuation": "PE",
                "scoring_notes": "半导体PE高但增速快，看PEG<1为低估；消费电子看产品周期",
            },
        }

    try:
        async with httpx.AsyncClient(timeout=QUERY_TIMEOUT) as client:
            resp = await client.post(
                f"{API_BASE_URL}/api/query",
                json={"stock_input": stock_input}
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"快速查询失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


# ──────────────────────────────────────────────
# 报告生成 API
# ──────────────────────────────────────────────

async def trigger_report(stock_input: str) -> str:
    """触发深度报告生成，返回 task_id"""
    if MOCK_MODE:
        return "mock-task-12345"

    try:
        async with httpx.AsyncClient(timeout=REPORT_TIMEOUT) as client:
            resp = await client.post(
                f"{API_BASE_URL}/api/report",
                json={"stock_input": stock_input}
            )
            resp.raise_for_status()
            return resp.json()["task_id"]
    except httpx.HTTPStatusError as e:
        raise APIError(f"报告生成失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def poll_report_status(task_id: str, progress_callback=None) -> dict:
    """轮询报告生成状态，完成后返回结果。
    progress_callback 接收 real_progress (0.0~1.0) 参数。
    """
    for attempt in range(POLL_MAX_ATTEMPTS):
        if MOCK_MODE:
            await asyncio.sleep(POLL_INTERVAL)
            if progress_callback:
                progress_callback((attempt + 1) / POLL_MAX_ATTEMPTS)
            if attempt >= 2:
                return {
                    "status": "completed",
                    "report_content": "# 寒武纪_分析报告_20260426\n\n## 摘要...",
                    "download_url": f"{API_BASE_URL}/api/report/{task_id}/download"
                }
            continue

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{API_BASE_URL}/api/report/{task_id}")
                resp.raise_for_status()
                data = resp.json()

                # 使用后端返回的真实进度（0.0~1.0）
                real_progress = data.get("progress", 0.0)
                if progress_callback:
                    progress_callback(real_progress)

                if data.get("status") in ("completed", "failed"):
                    return data

                await asyncio.sleep(POLL_INTERVAL)
        except httpx.RequestError as e:
            raise APIError(f"报告状态查询失败: {e}")

    raise APIError("报告生成超时")


# ──────────────────────────────────────────────
# 股票池 API
# ──────────────────────────────────────────────

async def get_pool(term: str) -> list:
    """获取指定期限股票池内容"""
    if MOCK_MODE:
        return [
            {"stock_code": "688256", "stock_name": "寒武纪",
             "score": 78.5, "score_time": "2026-04-26 14:30"},
            {"stock_code": "002594", "stock_name": "比亚迪",
             "score": 65.2, "score_time": "2026-04-25 09:15"},
        ]

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{API_BASE_URL}/api/pool/{term}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"获取股票池失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def add_to_pool(term: str, stock_code: str, stock_name: str) -> dict:
    """向指定期限股票池添加股票"""
    if MOCK_MODE:
        return {"status": "ok"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{API_BASE_URL}/api/pool/{term}",
                json={"stock_code": stock_code, "stock_name": stock_name}
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"添加股票失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def remove_from_pool(term: str, stock_code: str) -> dict:
    """从指定期限股票池删除股票"""
    if MOCK_MODE:
        return {"status": "ok"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.delete(f"{API_BASE_URL}/api/pool/{term}/{stock_code}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"删除股票失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


# ──────────────────────────────────────────────
# 打分 API（防刷新：触发 → 轮询模式）
# ──────────────────────────────────────────────

async def trigger_score(term: str, stock_code: str) -> str:
    """触发打分（后台异步），返回 task_id"""
    if MOCK_MODE:
        return "mock_task_12345"

    try:
        async with httpx.AsyncClient(timeout=SCORE_TRIGGER_TIMEOUT) as client:
            resp = await client.post(f"{API_BASE_URL}/api/score/{term}/{stock_code}")
            resp.raise_for_status()
            return resp.json()["task_id"]
    except httpx.HTTPStatusError as e:
        raise APIError(f"触发打分失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def poll_score_result(task_id: str) -> dict:
    """轮询打分结果直到完成或失败"""
    if MOCK_MODE:
        import random
        from datetime import datetime
        await asyncio.sleep(1)
        return {
            "status": "completed",
            "result": {
                "score": round(random.uniform(50, 90), 1),
                "score_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "term": "medium", "stock_code": "sh.603871", "company_name": "嘉友国际",
            }
        }

    try:
        for attempt in range(SCORE_POLL_MAX_ATTEMPTS):
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{API_BASE_URL}/api/score/{task_id}")
                resp.raise_for_status()
                data = resp.json()
                if data["status"] in ("completed", "failed"):
                    return data
            await asyncio.sleep(SCORE_POLL_INTERVAL)

        raise APIError(f"打分超时（{SCORE_POLL_MAX_ATTEMPTS * SCORE_POLL_INTERVAL}s）")
    except httpx.RequestError as e:
        raise APIError(f"查询打分状态失败: {e}")


async def score_stock(term: str, stock_code: str) -> dict:
    """便捷方法：触发 + 轮询，返回最终结果（向后兼容）"""
    task_id = await trigger_score(term, stock_code)
    data = await poll_score_result(task_id)
    if data["status"] == "completed":
        return data["result"]
    raise APIError(data.get("error", "打分失败"))


# ──────────────────────────────────────────────
# 缓存状态 API
# ──────────────────────────────────────────────

async def trigger_quick_score(term: str, stock_code: str) -> str:
    """触发快筛打分（后台异步），返回 task_id"""
    if MOCK_MODE:
        return "mock_qs_task_12345"

    try:
        async with httpx.AsyncClient(timeout=SCORE_TRIGGER_TIMEOUT) as client:
            resp = await client.post(f"{API_BASE_URL}/api/quick-screen/score/{term}/{stock_code}")
            resp.raise_for_status()
            return resp.json()["task_id"]
    except httpx.HTTPStatusError as e:
        raise APIError(f"触发快筛打分失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def poll_quick_score_result(task_id: str) -> dict:
    """轮询快筛打分结果"""
    if MOCK_MODE:
        import random
        from datetime import datetime
        await asyncio.sleep(0.5)
        return {
            "status": "completed",
            "result": {
                "score": round(random.uniform(50, 90), 1),
                "score_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "term": "medium", "stock_code": "sh.603871", "company_name": "嘉友国际",
            }
        }

    try:
        for attempt in range(SCORE_POLL_MAX_ATTEMPTS):
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{API_BASE_URL}/api/quick-screen/score/{task_id}")
                resp.raise_for_status()
                data = resp.json()
                if data["status"] in ("completed", "failed"):
                    return data
            await asyncio.sleep(SCORE_POLL_INTERVAL)
        raise APIError(f"快筛打分超时")
    except httpx.RequestError as e:
        raise APIError(f"查询快筛打分状态失败: {e}")


async def score_quick_screen(term: str, stock_code: str) -> dict:
    """便捷方法：触发 + 轮询，返回最终结果（向后兼容）"""
    task_id = await trigger_quick_score(term, stock_code)
    data = await poll_quick_score_result(task_id)
    if data["status"] == "completed":
        return data["result"]
    raise APIError(data.get("error", "快筛打分失败"))


# ──────────────────────────────────────────────
# 精筛股票池三期限并行打分 API
# ──────────────────────────────────────────────

async def trigger_score_all(stock_code: str) -> str:
    """触发精筛三期限并行打分，返回 task_id"""
    if MOCK_MODE:
        return "mock_fine_task_12345"

    try:
        async with httpx.AsyncClient(timeout=SCORE_TRIGGER_TIMEOUT) as client:
            resp = await client.post(f"{API_BASE_URL}/api/score-all/{stock_code}")
            resp.raise_for_status()
            return resp.json()["task_id"]
    except httpx.HTTPStatusError as e:
        raise APIError(f"触发精筛打分失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def poll_score_all_result(task_id: str) -> dict:
    """轮询精筛打分结果直到完成或失败"""
    if MOCK_MODE:
        import random
        from datetime import datetime
        await asyncio.sleep(1)
        return {
            "status": "completed",
            "result": {
                "stock_code": "sh.603871", "company_name": "嘉友国际",
                "short_term_score": {"score": 72, "rating": "买入"},
                "medium_term_score": {"score": 77, "rating": "推荐"},
                "long_term_score": {"score": 85, "rating": "强烈买入"},
                "execution_time": 94.5,
            }
        }

    try:
        for attempt in range(SCORE_POLL_MAX_ATTEMPTS):
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{API_BASE_URL}/api/score-all/{task_id}")
                resp.raise_for_status()
                data = resp.json()
                if data["status"] in ("completed", "failed"):
                    return data
            await asyncio.sleep(SCORE_POLL_INTERVAL)
        raise APIError(f"精筛打分超时")
    except httpx.RequestError as e:
        raise APIError(f"查询精筛打分状态失败: {e}")


async def get_cache_status(stock_code: str) -> dict:
    """查看某只股票的中间产物缓存状态"""
    if MOCK_MODE:
        return {"cached_agents": ["fundamental", "technical"], "date": "2026-04-26"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{API_BASE_URL}/api/cache/{stock_code}")
            resp.raise_for_status()
            return resp.json()
    except httpx.RequestError as e:
        raise APIError(f"缓存状态查询失败: {e}")


# ══════════════════════════════════════════════════════════════
# 批量打分 API
# ══════════════════════════════════════════════════════════════

async def upload_excel(file_content: bytes, filename: str, horizon: str = "medium") -> dict:
    """上传 Excel 文件并启动批量打分任务。

    Args:
        file_content: Excel 文件二进制内容
        filename: 原始文件名
        horizon: 打分维度 (short/medium/long/all)

    Returns:
        {batch_id, total_stocks, stocks, status}
    """
    if MOCK_MODE:
        return {
            "batch_id": "mock-batch-001",
            "horizon": horizon,
            "total_stocks": 3,
            "stocks": [
                {"code": "sh.688256", "name": "寒武纪"},
                {"code": "sz.002594", "name": "比亚迪"},
                {"code": "sh.600519", "name": "贵州茅台"},
            ],
            "status": "fetching",
        }

    try:
        async with httpx.AsyncClient(timeout=BATCH_UPLOAD_TIMEOUT) as client:
            resp = await client.post(
                f"{API_BASE_URL}/api/batch-score/upload",
                files={"file": (filename, file_content,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                data={"horizon": horizon},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"上传失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def get_batch_progress(batch_id: str) -> dict:
    """查询批量打分任务进度。

    Returns:
        {batch_id, status, total_stocks, fetched_count, scored_count,
         progress_pct, elapsed_seconds, error}
    """
    if MOCK_MODE:
        return {
            "batch_id": batch_id, "status": "fetching",
            "total_stocks": 100, "fetched_count": 45,
            "scored_count": 0, "progress_pct": 27.0,
            "elapsed_seconds": 120, "horizon": "medium",
        }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{API_BASE_URL}/api/batch-score/{batch_id}/progress")
            resp.raise_for_status()
            return resp.json()
    except httpx.RequestError as e:
        raise APIError(f"进度查询失败: {e}")


async def get_batch_results(batch_id: str) -> dict:
    """查询批量打分结果。

    Returns:
        {batch_id, status, total_stocks, fetched_count, scored_count,
         stocks: [{code, name, level, confidence, reason, risk, pe, pb, ...}]}
    """
    if MOCK_MODE:
        return {
            "batch_id": batch_id, "status": "completed",
            "total_stocks": 3, "fetched_count": 3, "scored_count": 3,
            "horizon": "medium",
            "stocks": [
                {"code": "sh.688256", "name": "寒武纪", "level": "推荐",
                 "confidence": "高", "reason": "AI芯片龙头", "risk": "估值偏高",
                 "pe": "120.5", "pb": "15.2", "roe": "8.5",
                 "industry": "半导体", "market_cap": "1200亿"},
                {"code": "sz.002594", "name": "比亚迪", "level": "强烈推荐",
                 "confidence": "高", "reason": "新能源整车龙头", "risk": "竞争加剧",
                 "pe": "25.0", "pb": "4.5", "roe": "18.0",
                 "industry": "汽车", "market_cap": "6000亿"},
                {"code": "sh.600519", "name": "贵州茅台", "level": "强烈推荐",
                 "confidence": "高", "reason": "白酒龙头护城河深", "risk": "政策风险",
                 "pe": "28.0", "pb": "9.5", "roe": "30.0",
                 "industry": "食品饮料", "market_cap": "2.1万亿"},
            ],
        }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{API_BASE_URL}/api/batch-score/{batch_id}/results")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 202:
            # 数据获取进行中，返回进度
            try:
                return e.response.json()
            except Exception:
                return {"status": "fetching", "detail": str(e.response.text)}
        raise APIError(f"结果查询失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def poll_batch_results(
    batch_id: str,
    progress_callback=None,
    poll_interval: float = BATCH_POLL_INTERVAL,
    max_attempts: int = BATCH_POLL_MAX_ATTEMPTS,
) -> dict:
    """轮询批量打分结果直到完成。

    Args:
        batch_id: 任务ID
        progress_callback: 可选, progress_callback(progress_pct, status, fetched, scored, total)
        poll_interval: 轮询间隔
        max_attempts: 最大轮询次数

    Returns:
        最终结果 dict (与 get_batch_results 相同)
    """
    for attempt in range(max_attempts):
        try:
            progress = await get_batch_progress(batch_id)
        except APIError:
            await asyncio.sleep(poll_interval)
            continue

        status = progress.get("status", "unknown")
        if progress_callback:
            try:
                progress_callback(
                    progress.get("progress_pct", 0),
                    status,
                    progress.get("fetched_count", 0),
                    progress.get("scored_count", 0),
                    progress.get("total_stocks", 0),
                )
            except Exception:
                pass

        if status in ("completed", "failed"):
            return await get_batch_results(batch_id)

        await asyncio.sleep(poll_interval)

    raise APIError("批量打分超时")


# ──────────────────────────────────────────────
# 智能问答 API
# ──────────────────────────────────────────────

async def qa_ask_stream(question: str, session_id: Optional[str] = None):
    """
    智能问答流式请求 — 异步生成器

    Yields:
        dict: {"type": "meta"|"status"|"answer"|"clarify"|"done"|"error", "data": ...}
    """
    if MOCK_MODE:
        yield {"type": "meta", "data": {"session_id": "mock-session", "complexity": "L1"}}
        yield {"type": "status", "data": {"message": "正在分析..."}}
        mock_answer = (
            "根据当前市场数据，该股票近期走势较为稳健。"
            "从估值角度看，PE处于历史中位数水平，具有一定投资价值。"
            "需要注意的是，市场整体波动较大，建议关注成交量变化。"
        )
        yield {"type": "answer", "data": mock_answer}
        yield {"type": "done", "data": None}
        return

    try:
        async with httpx.AsyncClient(timeout=QA_STREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/api/qa/ask",
                json={"question": question, "session_id": session_id},
            ) as response:
                response.raise_for_status()

                current_event = None
                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        current_event = line[7:].strip()
                    elif line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            yield {"type": "done", "data": None}
                            return
                        if data_str.startswith("[ERROR]"):
                            yield {"type": "error", "data": data_str[8:]}
                            return
                        if current_event in ("meta", "status", "clarify", "answer_start"):
                            try:
                                import json as _json
                                parsed = _json.loads(data_str)
                            except Exception:
                                parsed = {"message": data_str}
                            yield {"type": current_event, "data": parsed}
                            current_event = None
                        else:
                            try:
                                import json as _json
                                text = _json.loads(data_str)
                            except Exception:
                                text = data_str
                            yield {"type": "answer", "data": text}

    except httpx.HTTPStatusError as e:
        yield {"type": "error", "data": f"请求失败: {e.response.text}"}
    except httpx.RequestError as e:
        yield {"type": "error", "data": f"连接后端失败: {e}"}
    except Exception as e:
        yield {"type": "error", "data": f"未知错误: {e}"}


async def qa_get_session(session_id: str) -> dict:
    """获取会话历史"""
    if MOCK_MODE:
        return {
            "session_id": session_id,
            "history": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮您的？"},
            ],
        }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{API_BASE_URL}/api/qa/sessions/{session_id}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"获取会话失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def qa_delete_session(session_id: str) -> dict:
    """删除会话"""
    if MOCK_MODE:
        return {"status": "deleted", "session_id": session_id}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.delete(f"{API_BASE_URL}/api/qa/sessions/{session_id}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"删除会话失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def qa_list_sessions() -> list:
    """列出所有会话窗口"""
    if MOCK_MODE:
        return [
            {"session_id": "mock-1", "name": "茅台分析", "message_count": 3,
             "updated_at": 1700000000, "last_message": "根据当前数据..."},
            {"session_id": "mock-2", "name": "半导体行业", "message_count": 5,
             "updated_at": 1700000100, "last_message": "行业整体来看..."},
        ]

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{API_BASE_URL}/api/qa/sessions")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"获取会话列表失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def qa_create_session(name: str = "新对话") -> dict:
    """创建新会话窗口"""
    if MOCK_MODE:
        return {"session_id": f"mock-{hash(name) % 1000:03d}", "name": name}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{API_BASE_URL}/api/qa/sessions",
                json={"name": name}
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"创建会话失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def qa_rename_session(session_id: str, name: str) -> dict:
    """重命名会话窗口"""
    if MOCK_MODE:
        return {"status": "renamed", "session_id": session_id, "name": name}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.patch(
                f"{API_BASE_URL}/api/qa/sessions/{session_id}",
                json={"name": name}
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"重命名失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")
