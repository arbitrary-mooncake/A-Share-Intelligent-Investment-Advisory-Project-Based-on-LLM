"""VWAP 偏离策略（日频近似）。
参数: period=20, deviation=0.02
适用: 均值回归，价格低于滚动 VWAP × (1 - deviation) 买入，
高于滚动 VWAP × (1 + deviation) 卖出。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class VwapDeviation(TradingStrategy):
    name = "vwap_deviation"
    description = "VWAP 偏离：低于 VWAP × (1-dev) 买入，高于 VWAP × (1+dev) 卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        period = int(params.get("period", 20))
        df = df.copy()
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        df["tp_x_vol"] = tp * df["vol"]
        df["cum_tp_vol"] = df["tp_x_vol"].rolling(window=period).sum()
        df["cum_vol"] = df["vol"].rolling(window=period).sum()
        df["vwap_approx"] = df["cum_tp_vol"] / df["cum_vol"].replace(0, float("nan"))
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        deviation = float(params.get("deviation", 0.02))
        if self._na(row["vwap_approx"], row["close"]):
            return 0
        buy_threshold = row["vwap_approx"] * (1.0 - deviation)
        sell_threshold = row["vwap_approx"] * (1.0 + deviation)
        if row["close"] < buy_threshold:
            return 1
        if row["close"] > sell_threshold:
            return -1
        return 0
