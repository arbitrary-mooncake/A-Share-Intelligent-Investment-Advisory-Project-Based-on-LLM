"""双均线 + 固定仓位比例策略。
参数: short_window=5, long_window=20, equity_fraction=0.5
适用: 固定仓位控制风险，金叉买入（用 equity_fraction 仓位），死叉卖出。
仓位比例从 params 或 position_fraction() 获取。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class KellyMaCross(TradingStrategy):
    name = "kelly_ma_cross"
    description = "双均线 + 固定仓位：金叉买死叉卖，固定仓位比例"

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
        if prev_row["ma_short"] <= prev_row["ma_long"] and row["ma_short"] > row["ma_long"]:
            return 1
        if prev_row["ma_short"] >= prev_row["ma_long"] and row["ma_short"] < row["ma_long"]:
            return -1
        return 0

    def position_fraction(self, params: Mapping[str, Any], context=None) -> float:
        """返回固定仓位比例。"""
        return float(params.get("equity_fraction", 0.5))
