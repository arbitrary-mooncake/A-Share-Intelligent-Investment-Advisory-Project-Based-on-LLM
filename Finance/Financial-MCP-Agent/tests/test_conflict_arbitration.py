"""冲突仲裁层测试（4.3c）：默认折扣规则、输出解析、签名稳定性、兜底结构。"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.conflict_arbitration import (
    DEFAULT_DISCOUNT,
    _conflict_signature,
    _default_rule_discounts,
    _parse_arbitration_output,
    _result_to_discounts,
    arbitrate_conflicts,
)


def _conflict(bull_source="official_like", bear_source="news"):
    return {
        "category": "catalyst_event",
        "bullish": {"factor": "回购公告", "direction": 1, "strength": 80,
                    "confidence": 0.9, "source_level": bull_source, "note": "公司发布回购"},
        "bearish": {"factor": "资金链传闻", "direction": -1, "strength": 70,
                    "confidence": 0.8, "source_level": bear_source, "note": "传资金链紧张"},
    }


def test_default_rule_discounts_lower_priority_side():
    conflict = _conflict()
    discounts = _default_rule_discounts(conflict)
    # news 优先级低于 official_like → 空头信号（news）被折扣
    assert discounts == {id(conflict["bearish"]): DEFAULT_DISCOUNT}


def test_parse_valid_dominant_output():
    result = _parse_arbitration_output('{"dominant": "bullish", "discount": 0.3}')
    assert result == {"dominant": "bullish", "discount": 0.3}


def test_parse_no_conflict_output():
    result = _parse_arbitration_output('{"no_conflict": true}')
    assert result == {"no_conflict": True}


def test_parse_garbage_returns_none():
    assert _parse_arbitration_output("这不是JSON") is None
    assert _parse_arbitration_output('{"dominant": "双方都对"}') is None
    assert _parse_arbitration_output('{"dominant": "bullish", "discount": "很高"}') is None


def test_parse_discount_clamped():
    result = _parse_arbitration_output('{"dominant": "bearish", "discount": 1.7}')
    assert result["discount"] == 1.0


def test_result_to_discounts_no_conflict():
    assert _result_to_discounts(_conflict(), {"no_conflict": True}) == {}


def test_result_to_discounts_dominant_side_wins():
    conflict = _conflict()
    discounts = _result_to_discounts(conflict, {"dominant": "bullish", "discount": 0.2})
    assert discounts == {id(conflict["bearish"]): 0.2}


def test_signature_stable_and_content_based():
    c1, c2 = _conflict(), _conflict()
    assert _conflict_signature(c1) == _conflict_signature(c2)
    c2["bullish"]["note"] = "完全不同的公告内容"
    assert _conflict_signature(c1) != _conflict_signature(c2)


def test_arbitrate_disabled_falls_back_to_default_rule():
    conflict = _conflict()
    discounts = asyncio.run(arbitrate_conflicts([conflict], enabled=False))
    # 未启用 LLM → source_level 默认折扣（news 一侧）
    assert discounts == {id(conflict["bearish"]): DEFAULT_DISCOUNT}
