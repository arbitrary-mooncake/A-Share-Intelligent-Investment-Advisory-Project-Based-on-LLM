"""CCI 超卖/超买策略。
参数: period=20, oversold=-100, overbought=100
适用: 震荡市识别极端区域，CCI 上穿超卖线买入，下穿超买线卖出。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class Cci(TradingStrategy):
    name = "cci"
    description = "CCI 超卖超买：CCI 上穿 -100 买入，下穿 +100 卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        period = int(params.get("period", 20))
        df = df.copy()
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        df["cci"] = (tp - sma_tp) / (0.015 * mad.replace(0, float("nan")))
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        oversold = float(params.get("oversold", -100))
        overbought = float(params.get("overbought", 100))
        if self._na(row["cci"], prev_row["cci"]):
            return 0
        if prev_row["cci"] <= oversold and row["cci"] > oversold:
            return 1
        if prev_row["cci"] >= overbought and row["cci"] < overbought:
            return -1
        return 0
