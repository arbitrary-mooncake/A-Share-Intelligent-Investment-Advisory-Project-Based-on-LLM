"""
批量打分 Phase 2 测试 — batch scoring prompt builder + response parser
TDD: 测试先行，所有测试应先失败再通过
"""
import json
import pytest
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ────────────────────────────────────────────────────────────
# Tests for build_batch_prompt
# ────────────────────────────────────────────────────────────

class TestBuildBatchPrompt:
    """测试批量打分 prompt 构建器"""

    def _make_stock_data(self, code, name, **overrides):
        """Helper: create minimal valid stock data dict"""
        data = {
            "code": code,
            "name": name,
            "pe": "15.0",
            "pb": "2.5",
            "ps": "3.0",
            "roe": "12.0",
            "gross_margin": "30.0",
            "net_margin": "15.0",
            "revenue_growth": "10.0",
            "profit_growth": "8.0",
            "debt_ratio": "40.0",
            "turnover_rate": "2.0",
            "market_cap": "500亿",
            "industry": "食品饮料",
            "last_price": "25.00",
            "pct_chg": "1.5",
            "price_changes": {
                "1d": "1.5%", "5d": "-2.0%", "1m": "5.0%",
                "3m": "10.0%", "6m": "-5.0%", "1y": "20.0%", "3y": "50.0%"
            }
        }
        data.update(overrides)
        return data

    def test_empty_stocks_raises(self):
        """空股票列表应抛出 ValueError"""
        from src.api.batch_scorer import build_batch_prompt
        with pytest.raises(ValueError):
            build_batch_prompt([], "medium")

    def test_single_stock_returns_valid_prompt(self):
        """单只股票应生成包含必要字段的 prompt"""
        from src.api.batch_scorer import build_batch_prompt
        stocks = [self._make_stock_data("sh.603871", "嘉友国际")]
        result = build_batch_prompt(stocks, "medium")

        assert "system" in result
        assert "user" in result
        assert "sh.603871" in result["user"]
        assert "嘉友国际" in result["user"]
        # 应包含行业基准
        assert "食品饮料" in result["user"]

    def test_multiple_stocks_in_prompt(self):
        """多只股票应全部出现在 prompt 中"""
        from src.api.batch_scorer import build_batch_prompt
        stocks = [
            self._make_stock_data("sh.603871", "嘉友国际"),
            self._make_stock_data("sz.000858", "五粮液"),
            self._make_stock_data("sh.600519", "贵州茅台", pe="30.0"),
        ]
        result = build_batch_prompt(stocks, "medium")

        for s in stocks:
            assert s["code"] in result["user"]
            assert s["name"] in result["user"]

    def test_prompt_includes_output_schema(self):
        """prompt 应明确要求 JSON 数组输出格式"""
        from src.api.batch_scorer import build_batch_prompt
        stocks = [self._make_stock_data("sh.603871", "嘉友国际")]
        result = build_batch_prompt(stocks, "medium")

        assert "JSON" in result["user"] or "json" in result["user"]
        assert "数组" in result["user"] or "[" in result["user"]

    def test_horizon_affects_prompt(self):
        """不同时间维度应生成不同的评分标准"""
        from src.api.batch_scorer import build_batch_prompt
        stocks = [self._make_stock_data("sh.603871", "嘉友国际")]

        prompt_short = build_batch_prompt(stocks, "short")
        prompt_medium = build_batch_prompt(stocks, "medium")
        prompt_long = build_batch_prompt(stocks, "long")

        # 三个维度的 prompt 应互不相同
        assert prompt_short["user"] != prompt_medium["user"]
        assert prompt_medium["user"] != prompt_long["user"]

    def test_too_many_stocks_raises(self):
        """超过 5 只股票应抛出 ValueError"""
        from src.api.batch_scorer import build_batch_prompt
        stocks = [
            self._make_stock_data(f"sh.60{1000 + i}", f"股票{i}")
            for i in range(6)
        ]
        with pytest.raises(ValueError):
            build_batch_prompt(stocks, "medium")

    def test_stocks_with_missing_fields_handled(self):
        """缺失字段的股票数据不应导致崩溃"""
        from src.api.batch_scorer import build_batch_prompt
        stock = self._make_stock_data("sh.603871", "测试")
        # 只保留 code 和 name
        minimal = {"code": stock["code"], "name": stock["name"]}
        result = build_batch_prompt([minimal], "medium")

        assert "sh.603871" in result["user"]


# ────────────────────────────────────────────────────────────
# Tests for parse_batch_response
# ────────────────────────────────────────────────────────────

class TestParseBatchResponse:
    """测试批量 LLM 响应解析器"""

    def test_valid_response_parsed(self):
        """有效的 JSON 数组响应应正确解析"""
        from src.api.batch_scorer import parse_batch_response

        response = """
        ```json
        [
            {"code": "sh.603871", "level": "推荐", "confidence": "高",
             "reason": "低估值+高ROE", "risk": "原材料波动"},
            {"code": "sz.000858", "level": "强烈推荐", "confidence": "高",
             "reason": "品牌护城河深", "risk": "消费疲软"}
        ]
        ```
        """
        results = parse_batch_response(response)

        assert len(results) == 2
        assert results[0]["code"] == "sh.603871"
        assert results[0]["level"] == "推荐"
        assert results[1]["code"] == "sz.000858"
        assert results[1]["level"] == "强烈推荐"

    def test_json_without_markdown_parsed(self):
        """无 markdown 包裹的纯 JSON 应正确解析"""
        from src.api.batch_scorer import parse_batch_response

        response = """[{"code": "sh.600519", "level": "强烈推荐", "confidence": "高",
        "reason": "白酒龙头", "risk": "政策风险"}]"""
        results = parse_batch_response(response)

        assert len(results) == 1
        assert results[0]["code"] == "sh.600519"

    def test_invalid_response_returns_empty(self):
        """无有效 JSON 的响应不得静默吞掉，应抛出语法错误由调用方标记 invalid。"""
        from src.api.batch_scorer import (
            BatchResponseSyntaxError,
            parse_batch_response,
        )

        with pytest.raises(BatchResponseSyntaxError):
            parse_batch_response("这不是JSON，只是随便说说")

    def test_blank_response_returns_empty(self):
        """空白响应视为空输出（由调用方按 empty_llm_response 处理），不抛错。"""
        from src.api.batch_scorer import parse_batch_response

        assert parse_batch_response("   ") == []

    def test_partial_valid_response_extracted(self):
        """响应中嵌入有效 JSON 数组应被提取"""
        from src.api.batch_scorer import parse_batch_response

        response = "分析结果如下：\n[{\"code\": \"sh.603871\", \"level\": \"中性\", \"confidence\": \"中\", \"reason\": \"数据不足\", \"risk\": \"无\"}]\n以上仅供参考"
        results = parse_batch_response(response)

        assert len(results) == 1
        assert results[0]["code"] == "sh.603871"

    def test_missing_level_is_explicitly_invalid(self):
        """缺少 level 字段不得伪装为中性结论。"""
        from src.api.batch_scorer import parse_batch_response

        response = """[{"code": "sh.000001", "reason": "测试", "risk": "测试"}]"""
        results = parse_batch_response(response)

        assert len(results) == 1
        assert results[0]["level"] is None
        assert results[0]["validity"] == "invalid"
        assert results[0]["error_code"] == "invalid_level"

    def test_invalid_stock_entry_filtered(self):
        """code 无效的条目应被过滤"""
        from src.api.batch_scorer import parse_batch_response

        response = """[
            {"code": "sh.603871", "level": "推荐"},
            {"code": "INVALID_FORMAT", "level": "卖出"},
            {"code": "", "level": "持有"}
        ]"""
        results = parse_batch_response(response)

        # 只有合法 code 格式的条目被保留
        codes = [r["code"] for r in results]
        assert "sh.603871" in codes
        assert "INVALID_FORMAT" not in codes
        assert "" not in codes

    def test_valid_levels_accepted(self):
        """5级分类的有效值应被接受"""
        from src.api.batch_scorer import parse_batch_response
        valid_levels = ["强烈推荐", "推荐", "中性", "回避", "卖出"]

        for level in valid_levels:
            response = f'[{{"code": "sh.603871", "level": "{level}"}}]'
            results = parse_batch_response(response)
            assert results[0]["level"] == level, f"Level '{level}' not accepted"


# ────────────────────────────────────────────────────────────
# Tests for batch orchestration helpers
# ────────────────────────────────────────────────────────────

class TestBatchChunking:
    """测试股票分块逻辑"""

    def test_even_split(self):
        """偶数分组应正确分割"""
        from src.api.batch_scorer import chunk_stocks
        items = list(range(10))
        chunks = chunk_stocks(items, chunk_size=5)
        assert len(chunks) == 2
        assert len(chunks[0]) == 5
        assert len(chunks[1]) == 5

    def test_uneven_split(self):
        """奇数分组最后一块应包含剩余项"""
        from src.api.batch_scorer import chunk_stocks
        items = list(range(12))
        chunks = chunk_stocks(items, chunk_size=5)
        assert len(chunks) == 3
        assert len(chunks[0]) == 5
        assert len(chunks[1]) == 5
        assert len(chunks[2]) == 2
