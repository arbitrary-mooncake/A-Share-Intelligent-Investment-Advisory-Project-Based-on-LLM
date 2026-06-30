"""Williams %R 超卖/超买区段策略。
参数: period=14, oversold=-80, overbought=-20
适用: 短线震荡市，%R 上穿超卖线买入，下穿超买线卖出。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class WilliamsR(TradingStrategy):
    name = "williams_r"
    description = "Williams %R 超卖超买：上穿 -80 买入，下穿 -20 卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        period = int(params.get("period", 14))
        df = df.copy()
        high_n = df["high"].rolling(window=period).max()
        low_n = df["low"].rolling(window=period).min()
        df["williams_r"] = -100.0 * (high_n - df["close"]) / (high_n - low_n).replace(0, float("nan"))
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        oversold = float(params.get("oversold", -80))
        overbought = float(params.get("overbought", -20))
        if self._na(row["williams_r"], prev_row["williams_r"]):
            return 0
        if prev_row["williams_r"] <= oversold and row["williams_r"] > oversold:
            return 1
        if prev_row["williams_r"] >= overbought and row["williams_r"] < overbought:
            return -1
        return 0
