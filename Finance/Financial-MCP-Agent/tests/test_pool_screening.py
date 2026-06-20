"""四层精筛管线测试 — 严格按照总纲 §4.1 验证"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestHardScreen:
    """Layer 0: 硬筛逻辑测试（无LLM调用）"""

    def _make_stock(self, ts_code, name="测试", list_date="20200101", industry="制造业"):
        return {"ts_code": ts_code, "name": name, "list_date": list_date, "industry": industry}

    def test_excludes_bj_stocks(self):
        from src.eval.pool_screening import _is_stock_excluded
        assert _is_stock_excluded(self._make_stock("830001.BJ")) is True

    def test_excludes_b_shares(self):
        from src.eval.pool_screening import _is_stock_excluded
        assert _is_stock_excluded(self._make_stock("200001.B")) is True

    def test_excludes_st_stocks(self):
        from src.eval.pool_screening import _is_st_name
        assert _is_st_name("*ST测试") is True
        assert _is_st_name("ST中孚") is True
        assert _is_st_name("贵州茅台") is False

    def test_excludes_recent_ipo(self):
        from src.eval.pool_screening import _is_recent_ipo
        from datetime import datetime, timedelta
        recent = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        assert _is_recent_ipo(recent, min_days=60) is True

    def test_normal_stock_not_excluded(self):
        from src.eval.pool_screening import _is_stock_excluded
        assert _is_stock_excluded(self._make_stock("603871.SH", "嘉友国际")) is False

    def test_low_volume_detected(self):
        from src.eval.pool_screening import _is_low_volume
        assert _is_low_volume([10000000, 15000000, 8000000], 20000000) is True
        assert _is_low_volume([30000000, 25000000, 20000000], 20000000) is False
        assert _is_low_volume([], 20000000) is False


class TestLayerClassifier:
    """Layer 1 分类逻辑: 总纲5级→3档 (只有'卖出'进黑名单)"""

    def test_strong_recommend_to_whitelist(self):
        from src.eval.pool_screening import classify_batch_result
        result = classify_batch_result({
            "code": "sh.603871", "level": "强烈推荐", "confidence": "高",
            "reason": "低估值高ROE", "risk": "无"
        })
        assert result == "whitelist"

    def test_buy_cautious_watch_to_initial(self):
        from src.eval.pool_screening import classify_batch_result
        assert classify_batch_result({"code": "sh.000001", "level": "买入"}) == "initial_pool"
        assert classify_batch_result({"code": "sh.000001", "level": "谨慎买入"}) == "initial_pool"
        assert classify_batch_result({"code": "sh.000001", "level": "观望"}) == "initial_pool"

    def test_sell_to_blacklist(self):
        from src.eval.pool_screening import classify_batch_result
        assert classify_batch_result({"code": "sh.000001", "level": "卖出"}) == "blacklist"

    def test_invalid_level_defaults_to_initial(self):
        from src.eval.pool_screening import classify_batch_result
        assert classify_batch_result({"code": "sh.000001", "level": "不存在的分类"}) == "initial_pool"


class TestRatioCalculator:
    """Layer 3: 1:1.2差额计算"""

    def test_whitelist_smaller_than_target(self):
        from src.eval.pool_screening import calculate_candidate_quota
        result = calculate_candidate_quota(10, 50, 100, 1.2)
        assert result["whitelist_slots"] == 10
        assert result["initial_slots"] <= 50
        assert result["total_candidates"] == result["whitelist_slots"] + result["initial_slots"]

    def test_whitelist_larger_than_target(self):
        from src.eval.pool_screening import calculate_candidate_quota
        result = calculate_candidate_quota(200, 50, 100, 1.2)
        assert result["whitelist_slots"] < 200
        assert result["total_candidates"] <= 160
        assert result["whitelist_slots"] + result["initial_slots"] == result["total_candidates"]

    def test_quota_when_initial_empty(self):
        from src.eval.pool_screening import calculate_candidate_quota
        result = calculate_candidate_quota(10, 0, 100, 1.2)
        assert result["whitelist_slots"] == 10
        assert result["initial_slots"] == 0


class TestCodeFormatConversion:
    """Tushare格式 → 内部格式 转换"""

    def test_code_format_conversion(self):
        from src.eval.pool_screening import _prepare_stocks_for_batch
        ts_stocks = [
            {"ts_code": "603871.SH", "name": "嘉友国际"},
            {"ts_code": "000001.SZ", "name": "平安银行"},
        ]
        result = _prepare_stocks_for_batch(ts_stocks)
        assert result[0]["code"] == "sh.603871"
        assert result[1]["code"] == "sz.000001"

    def test_bj_stock_skipped(self):
        from src.eval.pool_screening import _prepare_stocks_for_batch
        ts_stocks = [{"ts_code": "830001.BJ", "name": "北交所"}]
        result = _prepare_stocks_for_batch(ts_stocks)
        assert len(result) == 0


class TestDynamicThreshold:
    """LLM动态阈值回退测试"""

    def test_empty_scores_returns_zero(self):
        import asyncio
        from src.eval.pool_screening import _dynamic_threshold
        threshold = asyncio.run(_dynamic_threshold([], 100))
        assert threshold == 0.0

    def test_normal_scores_fallback(self):
        import asyncio
        from src.eval.pool_screening import _dynamic_threshold
        scores = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
        threshold = asyncio.run(_dynamic_threshold(scores, 8))
        assert 45 <= threshold <= 85
