"""MACD 金叉/死叉策略。
参数: fast=12, slow=26, signal=9
适用: 趋势跟踪，DIF 上穿 DEA 买入，下穿卖出，适合波段操作。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class Macd(TradingStrategy):
    name = "macd"
    description = "MACD 金叉死叉：DIF 上穿 DEA 买入，下穿卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal_period = int(params.get("signal", 9))
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
        df["dif"] = df["ema_fast"] - df["ema_slow"]
        df["dea"] = df["dif"].ewm(span=signal_period, adjust=False).mean()
        df["macd_hist"] = 2 * (df["dif"] - df["dea"])
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row["dif"], row["dea"], prev_row["dif"], prev_row["dea"]):
            return 0
        if prev_row["dif"] <= prev_row["dea"] and row["dif"] > row["dea"]:
            return 1
        if prev_row["dif"] >= prev_row["dea"] and row["dif"] < row["dea"]:
            return -1
        return 0
