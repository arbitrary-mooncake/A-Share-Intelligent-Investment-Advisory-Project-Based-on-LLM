"""唐奇安通道突破策略。
参数: channel_period=20
适用: 趋势突破交易，价格突破 N 日最高价买入，跌破 N 日最低价卖出。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class Donchian(TradingStrategy):
    name = "donchian"
    description = "唐奇安通道突破：突破 N 日最高价买入，跌破 N 日最低价卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        period = int(params.get("channel_period", 20))
        df = df.copy()
        df["dc_upper"] = df["high"].rolling(window=period).max()
        df["dc_lower"] = df["low"].rolling(window=period).min()
        df["dc_middle"] = (df["dc_upper"] + df["dc_lower"]) / 2.0
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row["dc_upper"], row["dc_lower"],
                     prev_row["dc_upper"], prev_row["dc_lower"],
                     prev_row["close"], row["close"]):
            return 0
        if prev_row["close"] <= prev_row["dc_upper"] and row["close"] > row["dc_upper"]:
            return 1
        if prev_row["close"] >= prev_row["dc_lower"] and row["close"] < row["dc_lower"]:
            return -1
        return 0
