"""布林带策略 — 包含 boll_reversion 和 boll_breakout 两个子策略。
参数: period=20, std=2
boll_reversion: 均值回归，碰下轨反弹买入，碰上轨回落卖出。
boll_breakout: 动量突破，突破上轨追涨买入，跌破下轨杀跌卖出。
适用: reversion 适合震荡市；breakout 适合强趋势市。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


def _compute_bollinger(df: pd.DataFrame, period: int, std_mult: float) -> pd.DataFrame:
    df = df.copy()
    df["bb_middle"] = df["close"].rolling(window=period).mean()
    df["bb_std"] = df["close"].rolling(window=period).std()
    df["bb_upper"] = df["bb_middle"] + std_mult * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - std_mult * df["bb_std"]
    return df


@register_strategy
class BollReversion(TradingStrategy):
    name = "boll_reversion"
    description = "布林带均值回归：碰下轨反弹买入，碰上轨回落卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        period = int(params.get("period", 20))
        std_mult = float(params.get("std", 2))
        return _compute_bollinger(df, period, std_mult)

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row["bb_lower"], row["bb_upper"],
                     prev_row["close"], row["close"],
                     prev_row["bb_lower"], prev_row["bb_upper"]):
            return 0
        # 价格从上穿下轨下方回到下轨上方
        if prev_row["close"] <= prev_row["bb_lower"] and row["close"] > row["bb_lower"]:
            return 1
        # 价格从下穿上轨上方回到上轨下方
        if prev_row["close"] >= prev_row["bb_upper"] and row["close"] < row["bb_upper"]:
            return -1
        return 0


@register_strategy
class BollBreakout(TradingStrategy):
    name = "boll_breakout"
    description = "布林带动量突破：突破上轨买入，跌破下轨卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        period = int(params.get("period", 20))
        std_mult = float(params.get("std", 2))
        return _compute_bollinger(df, period, std_mult)

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row["bb_lower"], row["bb_upper"],
                     prev_row["close"], row["close"],
                     prev_row["bb_lower"], prev_row["bb_upper"]):
            return 0
        if prev_row["close"] <= prev_row["bb_upper"] and row["close"] > row["bb_upper"]:
            return 1
        if prev_row["close"] >= prev_row["bb_lower"] and row["close"] < row["bb_lower"]:
            return -1
        return 0
