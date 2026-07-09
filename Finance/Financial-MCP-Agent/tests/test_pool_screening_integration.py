"""四层管线集成测试 — 验证数据在各层之间的流转正确性，不调真实API"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestLayer1Classification:
    """Layer 1: 总纲5级→3档 (只有'卖出'进黑名单)"""

    def test_all_five_levels_mapped_correctly(self):
        """完整的5级→3档映射验证"""
        from src.eval.pool_screening import classify_batch_result

        # 强烈推荐 → whitelist
        assert classify_batch_result({"code": "sh.000001", "level": "强烈推荐"}) == "whitelist"

        # 买入/谨慎买入/观望 → initial_pool
        assert classify_batch_result({"code": "sh.000001", "level": "买入"}) == "initial_pool"
        assert classify_batch_result({"code": "sh.000001", "level": "谨慎买入"}) == "initial_pool"
        assert classify_batch_result({"code": "sh.000001", "level": "观望"}) == "initial_pool"

        # 卖出 → blacklist (唯一进黑名单的)
        assert classify_batch_result({"code": "sh.000001", "level": "卖出"}) == "blacklist"

    def test_missing_level_defaults_to_initial(self):
        """分类缺失时默认入初筛池（保守处理，不误杀）"""
        from src.eval.pool_screening import classify_batch_result
        assert classify_batch_result({"code": "sh.000001"}) == "initial_pool"
        assert classify_batch_result({}) == "initial_pool"

    def test_lay1_levels_are_correct_taxonomy(self):
        """验证 LAYER1_LEVELS 是总纲规定的分类标准"""
        from src.eval.pool_screening import LAYER1_LEVELS
        assert LAYER1_LEVELS == ["强烈推荐", "推荐", "中性", "回避", "卖出"]
        assert len(LAYER1_LEVELS) == 5


class TestQuotaCalculation:
    """Layer 3: 1:1.2差额配额计算"""

    def test_normal_ratio_total(self):
        """正常场景: 白名单少, 初筛多 → 初筛配额被上限截断, 总候选=白名单+min(缺口,初筛)"""
        from src.eval.pool_screening import calculate_candidate_quota
        q = calculate_candidate_quota(10, 80, 100, 1.2)
        assert q["total_candidates"] == q["whitelist_slots"] + q["initial_slots"]
        # 白名单10全部进, 目标总候选120, 但初筛只有80只 < 缺口110, 实际只取80
        assert q["total_candidates"] == 90

    def test_whitelist_exceeds_ideal(self):
        """白名单过多时合理截断，不全取"""
        from src.eval.pool_screening import calculate_candidate_quota
        q = calculate_candidate_quota(200, 50, 100, 1.2)
        # 白名单200只, 但不应该全取200只(超过target*ratio太多)
        assert q["whitelist_slots"] < 200
        assert q["total_candidates"] <= 160  # target*ratio*1.3 margin
        assert q["whitelist_slots"] + q["initial_slots"] == q["total_candidates"]

    def test_initial_empty(self):
        """初筛为空时白名单顶上去"""
        from src.eval.pool_screening import calculate_candidate_quota
        q = calculate_candidate_quota(10, 0, 100, 1.2)
        assert q["whitelist_slots"] == 10
        assert q["initial_slots"] == 0
        assert q["total_candidates"] == 10

    def test_both_plentiful(self):
        """白名单和初筛都充足"""
        from src.eval.pool_screening import calculate_candidate_quota
        q = calculate_candidate_quota(50, 100, 80, 1.2)
        assert q["total_candidates"] == int(80 * 1.2)  # 96
        assert q["whitelist_slots"] > 0
        assert q["initial_slots"] > 0


class TestCodeFormatConversion:
    """Tushare TS代码 → 内部 sh.xxx / sz.xxx 格式"""

    def test_sh_conversion(self):
        from src.eval.pool_screening import _prepare_stocks_for_batch
        ts = [{"ts_code": "603871.SH", "name": "嘉友国际"}]
        result = _prepare_stocks_for_batch(ts)
        assert result[0]["code"] == "sh.603871"
        assert result[0]["name"] == "嘉友国际"

    def test_sz_conversion(self):
        from src.eval.pool_screening import _prepare_stocks_for_batch
        ts = [{"ts_code": "000001.SZ", "name": "平安银行"}]
        result = _prepare_stocks_for_batch(ts)
        assert result[0]["code"] == "sz.000001"

    def test_bj_skipped(self):
        from src.eval.pool_screening import _prepare_stocks_for_batch
        ts = [{"ts_code": "830001.BJ", "name": "北交所"}]
        result = _prepare_stocks_for_batch(ts)
        assert len(result) == 0

    def test_mixed_batch(self):
        from src.eval.pool_screening import _prepare_stocks_for_batch
        ts = [
            {"ts_code": "603871.SH", "name": "A"},
            {"ts_code": "000001.SZ", "name": "B"},
            {"ts_code": "830001.BJ", "name": "C"},  # skipped
        ]
        result = _prepare_stocks_for_batch(ts)
        assert len(result) == 2
        assert result[0]["code"] == "sh.603871"
        assert result[1]["code"] == "sz.000001"


class TestDynamicThreshold:
    """LLM动态阈值回退行为"""

    def test_empty_scores(self):
        import asyncio
        from src.eval.pool_screening import _dynamic_threshold
        t = asyncio.run(_dynamic_threshold([], 100))
        assert t == 0.0

    def test_single_score(self):
        import asyncio
        from src.eval.pool_screening import _dynamic_threshold
        t = asyncio.run(_dynamic_threshold([75.0], 10))
        # 只有1个分数, 默认阈值应 <= 该分数
        assert t <= 75.0

    def test_fallback_within_range(self):
        import asyncio
        from src.eval.pool_screening import _dynamic_threshold
        scores = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
        t = asyncio.run(_dynamic_threshold(scores, 8))
        # 第8名分数=60, 默认阈值≈59, 在合理范围
        assert 45 <= t <= 85


class TestLevelToTierMapping:
    """LEVEL_TO_TIER 字典完整性"""

    def test_all_lay1_levels_have_mapping(self):
        from src.eval.pool_screening import LEVEL_TO_TIER, LAYER1_LEVELS
        for level in LAYER1_LEVELS:
            assert level in LEVEL_TO_TIER, f"Missing mapping for '{level}'"

    def test_only_sell_is_blacklist(self):
        from src.eval.pool_screening import LEVEL_TO_TIER
        blacklist_levels = [k for k, v in LEVEL_TO_TIER.items() if v == "blacklist"]
        assert blacklist_levels == ["卖出"], f"Expected only '卖出' in blacklist, got {blacklist_levels}"

    def test_whitelist_has_only_strong_recommend(self):
        from src.eval.pool_screening import LEVEL_TO_TIER
        whitelist_levels = [k for k, v in LEVEL_TO_TIER.items() if v == "whitelist"]
        assert whitelist_levels == ["强烈推荐"], f"Expected only '强烈推荐' in whitelist, got {whitelist_levels}"
