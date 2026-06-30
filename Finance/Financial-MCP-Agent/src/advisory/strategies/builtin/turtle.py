"""完整海龟交易系统。
参数: entry=20, exit=10, atr=14, atr_stop=2
适用: 中长期趋势跟踪，突破 entry 日高点买入，跌破 exit 日低点卖出，
含 ATR 倍数硬止损保护。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class Turtle(TradingStrategy):
    name = "turtle"
    description = "海龟交易系统：突破 N 日高点买入、跌破 M 日低点卖出，ATR 止损"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        entry_period = int(params.get("entry", 20))
        exit_period = int(params.get("exit", 10))
        atr_period = int(params.get("atr", 14))
        df = df.copy()

        df["turtle_entry_high"] = df["high"].rolling(window=entry_period).max()
        df["turtle_exit_low"] = df["low"].rolling(window=exit_period).min()

        high_low = df["high"] - df["low"]
        high_prev = (df["high"] - df["close"].shift(1)).abs()
        low_prev = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row["turtle_entry_high"], row["turtle_exit_low"],
                     prev_row["turtle_entry_high"], prev_row["turtle_exit_low"]):
            return 0
        if prev_row["close"] <= prev_row["turtle_entry_high"] and row["close"] > row["turtle_entry_high"]:
            return 1
        if prev_row["close"] >= prev_row["turtle_exit_low"] and row["close"] < row["turtle_exit_low"]:
            return -1
        return 0

    def risk_exit(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        """ATR 倍数硬止损：当前价跌破 entry_price - atr_stop_mult * ATR 时强制退出。"""
        if context is None:
            return 0
        entry_price = context.get("entry_price")
        if entry_price is None:
            return 0
        if self._na(row["atr"]):
            return 0
        atr_stop_mult = float(params.get("atr_stop", 2))
        stop_price = entry_price - atr_stop_mult * row["atr"]
        if row["close"] < stop_price:
            return -1
        return 0
