"""三均线排列与短穿中策略。
参数: short=5, mid=10, long=30
适用: 趋势跟踪，三线多头排列(短>中>长)时短均线上穿中均线买入，
三线空头排列(短<中<长)时短均线下穿中均线卖出。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class TripleMa(TradingStrategy):
    name = "triple_ma"
    description = "三均线排列：多头排列+短穿中买入，空头排列+短穿中卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        short = int(params.get("short", 5))
        mid = int(params.get("mid", 10))
        long = int(params.get("long", 30))
        df = df.copy()
        df["ma_short"] = df["close"].rolling(window=short).mean()
        df["ma_mid"] = df["close"].rolling(window=mid).mean()
        df["ma_long"] = df["close"].rolling(window=long).mean()
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row["ma_short"], row["ma_mid"], row["ma_long"],
                     prev_row["ma_short"], prev_row["ma_mid"], prev_row["ma_long"]):
            return 0
        # 多头排列：短 > 中 > 长
        bullish_align = row["ma_short"] > row["ma_mid"] > row["ma_long"]
        # 空头排列：短 < 中 < 长
        bearish_align = row["ma_short"] < row["ma_mid"] < row["ma_long"]
        # 短穿中交叉
        short_cross_up_mid = (prev_row["ma_short"] <= prev_row["ma_mid"]
                              and row["ma_short"] > row["ma_mid"])
        short_cross_down_mid = (prev_row["ma_short"] >= prev_row["ma_mid"]
                                and row["ma_short"] < row["ma_mid"])

        if bullish_align and short_cross_up_mid:
            return 1
        if bearish_align and short_cross_down_mid:
            return -1
        return 0
