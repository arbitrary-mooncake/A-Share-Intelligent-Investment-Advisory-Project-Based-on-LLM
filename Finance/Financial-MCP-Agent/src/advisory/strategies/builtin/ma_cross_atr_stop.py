"""双均线 + ATR 硬止损策略。
参数: short_window=5, long_window=20, atr_stop_mult=2, atr_period=14
适用: 趋势跟踪 + 风险控制，金叉买入死叉卖出，持仓中跌破 entry_price - N*ATR 止损。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class MaCrossAtrStop(TradingStrategy):
    name = "ma_cross_atr_stop"
    description = "双均线 + ATR 止损：金叉买入，死叉或 ATR 止损卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        short = int(params.get("short_window", 5))
        long = int(params.get("long_window", 20))
        atr_period = int(params.get("atr_period", 14))
        df = df.copy()

        df["ma_short"] = df["close"].rolling(window=short).mean()
        df["ma_long"] = df["close"].rolling(window=long).mean()

        high_low = df["high"] - df["low"]
        high_prev = (df["high"] - df["close"].shift(1)).abs()
        low_prev = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

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

    def risk_exit(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        """ATR 倍数硬止损。"""
        if context is None:
            return 0
        entry_price = context.get("entry_price")
        if entry_price is None:
            return 0
        if self._na(row["atr"]):
            return 0
        atr_stop_mult = float(params.get("atr_stop_mult", 2))
        stop_price = entry_price - atr_stop_mult * row["atr"]
        if row["close"] < stop_price:
            return -1
        return 0
