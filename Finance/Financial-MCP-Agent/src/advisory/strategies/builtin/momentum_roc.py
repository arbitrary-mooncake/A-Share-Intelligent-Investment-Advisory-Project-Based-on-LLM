"""N 日 ROC 过零策略。
参数: roc_window=20
适用: 动量趋势转折识别，ROC 从负转正买入（动能转强），从正转负卖出（动能转弱）。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class MomentumRoc(TradingStrategy):
    name = "momentum_roc"
    description = "ROC 动量过零：ROC 从负转正买入，从正转负卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        window = int(params.get("roc_window", 20))
        df = df.copy()
        shifted = df["close"].shift(window)
        df["roc"] = (df["close"] - shifted) / shifted.replace(0, float("nan")) * 100.0
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row["roc"], prev_row["roc"]):
            return 0
        if prev_row["roc"] <= 0 and row["roc"] > 0:
            return 1
        if prev_row["roc"] >= 0 and row["roc"] < 0:
            return -1
        return 0
