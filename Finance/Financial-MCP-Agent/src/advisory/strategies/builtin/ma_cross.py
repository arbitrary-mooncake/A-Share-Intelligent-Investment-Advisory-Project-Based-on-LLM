"""双均线金叉/死叉策略。
参数: short_window=5, long_window=20
适用: 趋势市中捕捉中短期转折点，短均线上穿长均线买入，下穿卖出。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class MaCross(TradingStrategy):
    name = "ma_cross"
    description = "双均线金叉买/死叉卖：短均线上穿长均线买入，下穿卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        short = int(params.get("short_window", 5))
        long = int(params.get("long_window", 20))
        df = df.copy()
        df["ma_short"] = df["close"].rolling(window=short).mean()
        df["ma_long"] = df["close"].rolling(window=long).mean()
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row["ma_short"], row["ma_long"],
                     prev_row["ma_short"], prev_row["ma_long"]):
            return 0
        # 金叉：短均线上穿长均线
        if prev_row["ma_short"] <= prev_row["ma_long"] and row["ma_short"] > row["ma_long"]:
            return 1
        # 死叉：短均线下穿长均线
        if prev_row["ma_short"] >= prev_row["ma_long"] and row["ma_short"] < row["ma_long"]:
            return -1
        return 0
