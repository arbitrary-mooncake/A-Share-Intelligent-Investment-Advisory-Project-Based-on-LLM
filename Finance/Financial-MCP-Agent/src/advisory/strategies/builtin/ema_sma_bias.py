"""价穿 EMA + EMA > SMA 乖离策略。
参数: ema=20, sma=50
适用: 中短期趋势确认，价格上穿 EMA 且 EMA 在 SMA 之上时买入，
价格下穿 EMA 且 EMA 在 SMA 之下时卖出。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class EmaSmaBias(TradingStrategy):
    name = "ema_sma_bias"
    description = "价穿EMA+乖离：上穿EMA且EMA>SMA买入，下穿EMA且EMA<SMA卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        ema_period = int(params.get("ema", 20))
        sma_period = int(params.get("sma", 50))
        df = df.copy()
        df["ema"] = df["close"].ewm(span=ema_period, adjust=False).mean()
        df["sma"] = df["close"].rolling(window=sma_period).mean()
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row["ema"], row["sma"], prev_row["ema"], prev_row["sma"],
                     prev_row["close"], row["close"]):
            return 0
        price_cross_up_ema = (prev_row["close"] <= prev_row["ema"]
                              and row["close"] > row["ema"])
        price_cross_down_ema = (prev_row["close"] >= prev_row["ema"]
                                and row["close"] < row["ema"])
        ema_above_sma = row["ema"] > row["sma"]
        ema_below_sma = row["ema"] < row["sma"]

        if price_cross_up_ema and ema_above_sma:
            return 1
        if price_cross_down_ema and ema_below_sma:
            return -1
        return 0
