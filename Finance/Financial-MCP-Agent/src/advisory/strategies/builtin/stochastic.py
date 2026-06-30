"""随机指标 %K/%D 区段策略。
参数: k=14, d=3, oversold=20, overbought=80
适用: 短线震荡市，%K 上穿超卖线或 %K 上穿 %D 且在超卖区买入，
%K 下穿超买线或 %K 下穿 %D 且在超买区卖出。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class Stochastic(TradingStrategy):
    name = "stochastic"
    description = "随机指标：%K 上穿 %D 在超卖区买入，下穿 %D 在超买区卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        k_period = int(params.get("k", 14))
        d_period = int(params.get("d", 3))
        df = df.copy()
        low_n = df["low"].rolling(window=k_period).min()
        high_n = df["high"].rolling(window=k_period).max()
        denom = (high_n - low_n).replace(0, float("nan"))
        df["stoch_k"] = 100.0 * (df["close"] - low_n) / denom
        df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        oversold = float(params.get("oversold", 20))
        overbought = float(params.get("overbought", 80))
        if self._na(row["stoch_k"], row["stoch_d"],
                     prev_row["stoch_k"], prev_row["stoch_d"]):
            return 0
        # K 上穿 D 且在超卖区
        if (prev_row["stoch_k"] <= prev_row["stoch_d"]
                and row["stoch_k"] > row["stoch_d"]
                and row["stoch_k"] < oversold):
            return 1
        # K 下穿 D 且在超买区
        if (prev_row["stoch_k"] >= prev_row["stoch_d"]
                and row["stoch_k"] < row["stoch_d"]
                and row["stoch_k"] > overbought):
            return -1
        return 0
