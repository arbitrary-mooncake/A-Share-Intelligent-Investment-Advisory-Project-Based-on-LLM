"""放量突破策略。
参数: breakout=20, vol_ma=20, vol_mult=1.5, exit=10
适用: 量价配合突破，放量突破 N 日高点买入，跌破 M 日低点卖出。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class VolumeBreakout(TradingStrategy):
    name = "volume_breakout"
    description = "放量突破：放量破 N 日高买入，缩量破 M 日低卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        breakout = int(params.get("breakout", 20))
        vol_period = int(params.get("vol_ma", 20))
        exit_period = int(params.get("exit", 10))
        df = df.copy()
        df["breakout_high"] = df["high"].rolling(window=breakout).max()
        df["vol_ma"] = df["vol"].rolling(window=vol_period).mean()
        df["exit_low"] = df["low"].rolling(window=exit_period).min()
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        vol_mult = float(params.get("vol_mult", 1.5))
        if self._na(row["breakout_high"], row["vol_ma"], row["vol"],
                     row["exit_low"],
                     prev_row["breakout_high"], prev_row["exit_low"]):
            return 0
        # 放量突破买入
        volume_surge = row["vol"] > vol_mult * row["vol_ma"]
        breakout_up = (prev_row["close"] <= prev_row["breakout_high"]
                       and row["close"] > row["breakout_high"])
        if breakout_up and volume_surge:
            return 1
        # 跌破 exit 低点卖出
        if (prev_row["close"] >= prev_row["exit_low"]
                and row["close"] < row["exit_low"]):
            return -1
        return 0
