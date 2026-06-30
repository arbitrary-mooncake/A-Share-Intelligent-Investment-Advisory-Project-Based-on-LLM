"""OBV 能量潮与均线交叉策略。
参数: obv_ma=20
适用: 量价背离识别，OBV 上穿其 MA 买入（量能转强），下穿卖出（量能转弱）。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class ObvCross(TradingStrategy):
    name = "obv_cross"
    description = "OBV 能量潮均线交叉：OBV 上穿 MA 买入，下穿 MA 卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        ma_period = int(params.get("obv_ma", 20))
        df = df.copy()

        # OBV: price up → +vol, price down → -vol, unchanged → 0
        close_diff = df["close"].diff().fillna(0)
        signed_vol = np.where(
            close_diff > 0, df["vol"],
            np.where(close_diff < 0, -df["vol"], 0)
        )
        df["obv"] = signed_vol.cumsum()
        df["obv_ma"] = df["obv"].rolling(window=ma_period).mean()
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row["obv"], row["obv_ma"], prev_row["obv"], prev_row["obv_ma"]):
            return 0
        if prev_row["obv"] <= prev_row["obv_ma"] and row["obv"] > row["obv_ma"]:
            return 1
        if prev_row["obv"] >= prev_row["obv_ma"] and row["obv"] < row["obv_ma"]:
            return -1
        return 0
