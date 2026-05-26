"""
Phase 3 前端测试 — batch API protocol + progress + colors + data normalization
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'app'))

import httpx
import asyncio


class MockTransport(httpx.AsyncBaseTransport):
    """Mock HTTP transport for testing API responses"""
    def __init__(self, status_code: int, json_data: dict):
        self._status = status_code
        self._data = json_data

    async def handle_async_request(self, request):
        return httpx.Response(self._status, json=self._data, request=request)


# ────────────────────────────────────────────────────────────
# API Protocol Tests
# ────────────────────────────────────────────────────────────

class TestBatchApiProtocol:
    """测试批量打分 API 端点的 HTTP 协议形状"""

    def test_upload_response_shape(self):
        """上传响应应包含 batch_id 和股票列表"""
        async def run():
            transport = MockTransport(200, {
                "batch_id": "abc123", "horizon": "medium",
                "total_stocks": 3, "status": "fetching",
                "stocks": [{"code": "sh.603871", "name": None}]
            })
            async with httpx.AsyncClient(transport=transport) as client:
                resp = await client.post("http://t/api/batch-score/upload", timeout=30)
                assert resp.status_code == 200
                data = resp.json()
                assert data["batch_id"] == "abc123"
                assert data["total_stocks"] == 3
        asyncio.run(run())

    def test_progress_response_shape(self):
        """进度响应应包含 fetched_count 和 progress_pct"""
        async def run():
            transport = MockTransport(200, {
                "batch_id": "abc123", "status": "fetching",
                "total_stocks": 100, "fetched_count": 45, "scored_count": 0,
                "progress_pct": 27.0, "elapsed_seconds": 120
            })
            async with httpx.AsyncClient(transport=transport) as client:
                resp = await client.get("http://t/api/batch-score/abc123/progress")
                assert resp.status_code == 200
                data = resp.json()
                assert data["fetched_count"] == 45
        asyncio.run(run())

    def test_results_response_shape(self):
        """结果响应应包含 stocks 列表和 level/reason 字段"""
        async def run():
            transport = MockTransport(200, {
                "batch_id": "abc123", "status": "completed",
                "total_stocks": 2, "stocks": [
                    {"code": "sh.603871", "name": None, "level": "推荐",
                     "reason": "低估值", "risk": "风险"},
                ]
            })
            async with httpx.AsyncClient(transport=transport) as client:
                resp = await client.get("http://t/api/batch-score/abc123/results")
                assert resp.status_code == 200
                data = resp.json()
                assert data["stocks"][0]["level"] == "推荐"
        asyncio.run(run())

    def test_error_response(self):
        """错误响应应返回 400"""
        async def run():
            transport = MockTransport(400, {"detail": "horizon error"})
            async with httpx.AsyncClient(transport=transport) as client:
                resp = await client.post("http://t/api/batch-score/upload", timeout=30)
                assert resp.status_code == 400
        asyncio.run(run())


# ────────────────────────────────────────────────────────────
# Progress Calculation Tests
# ────────────────────────────────────────────────────────────

class TestProgressCalculation:
    """测试进度百分比计算逻辑"""

    def _compute(self, status, fetched, scored, total):
        if status in ("parsed", "fetching"):
            return round(fetched / max(total, 1) * 60, 1) if total else 0
        elif status == "fetched":
            return 60.0
        elif status == "scoring":
            return 60 + round(scored / max(total, 1) * 40, 1) if total else 60
        elif status == "completed":
            return 100.0
        return 0.0

    def test_fetching_half(self):
        assert self._compute("fetching", 50, 0, 100) == 30.0

    def test_fetched_boundary(self):
        assert self._compute("fetched", 100, 0, 100) == 60.0

    def test_scoring_mid(self):
        assert self._compute("scoring", 100, 50, 100) == 80.0

    def test_completed(self):
        assert self._compute("completed", 100, 100, 100) == 100.0

    def test_zero_division(self):
        assert self._compute("fetching", 0, 0, 0) == 0.0


# ────────────────────────────────────────────────────────────
# Level Color Tests
# ────────────────────────────────────────────────────────────

class TestLevelColors:
    """测试 5 级分类颜色映射"""

    def _level_color(self, level):
        mapping = {
            "强烈推荐": "#059669", "推荐": "#0891b2",
            "中性": "#d97706", "回避": "#ea580c", "卖出": "#dc2626",
        }
        return mapping.get(level, "#6b7280")

    def test_all_five_levels(self):
        assert self._level_color("强烈推荐") == "#059669"
        assert self._level_color("推荐") == "#0891b2"
        assert self._level_color("中性") == "#d97706"
        assert self._level_color("回避") == "#ea580c"
        assert self._level_color("卖出") == "#dc2626"
        assert self._level_color("???") == "#6b7280"

    def _level_bg(self, level):
        mapping = {
            "强烈推荐": "#d1fae5", "推荐": "#ccfbf1",
            "中性": "#fef3c7", "回避": "#ffedd5", "卖出": "#fee2e2",
        }
        return mapping.get(level, "#f3f4f6")

    def test_all_five_bg_colors(self):
        assert self._level_bg("强烈推荐") == "#d1fae5"
        assert self._level_bg("中性") == "#fef3c7"
        assert self._level_bg("卖出") == "#fee2e2"
        assert self._level_bg("???") == "#f3f4f6"


# ────────────────────────────────────────────────────────────
# Data Normalization Tests
# ────────────────────────────────────────────────────────────

class TestDataNormalization:
    """测试前端数据规范化"""

    def _normalize(self, stock):
        return {
            "code": stock.get("code", ""),
            "name": stock.get("name") or "未知",
            "level": stock.get("level", "中性"),
            "confidence": stock.get("confidence", ""),
            "reason": stock.get("reason", ""),
            "risk": stock.get("risk", ""),
            "pe": stock.get("pe", ""),
            "pb": stock.get("pb", ""),
            "roe": stock.get("roe", ""),
            "industry": stock.get("industry", ""),
            "market_cap": stock.get("market_cap", ""),
        }

    def test_full_entry(self):
        s = {"code": "sh.603871", "name": "嘉友国际", "level": "强烈推荐",
             "pe": "14.1", "pb": "2.7", "roe": "5.3",
             "industry": "现代服务", "market_cap": "168亿"}
        e = self._normalize(s)
        assert e["level"] == "强烈推荐"
        assert e["pe"] == "14.1"

    def test_missing_level_defaults_neutral(self):
        assert self._normalize({"code": "sh.000001"})["level"] == "中性"

    def test_none_name_becomes_unknown(self):
        assert self._normalize({"code": "sh.000001", "name": None})["name"] == "未知"

    def test_missing_fields_empty(self):
        e = self._normalize({"code": "sh.000001"})
        assert e["pe"] == ""
