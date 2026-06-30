"""双均线 + ATR 动态缩仓策略。
参数: short_window=5, long_window=20, risk_budget=0.01, max_fraction=1.0, atr_period=14
适用: 波动率自适应仓位管理，金叉买入死叉卖出，
仓位 = min(risk_budget * price / ATR, max_fraction)。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class VolTargetMaCross(TradingStrategy):
    name = "vol_target_ma_cross"
    description = "双均线 + 波动率目标仓位：金叉买死叉卖，按 ATR 缩仓"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        short = int(params.get("short_window", 5))
        long = int(params.get("long_window", 20))
        atr_period = int(params.get("atr_period", 14))
        df = df.copy()

        df["ma_short"] = df["close"].rolling(window=short).mean()
        df["ma_long"] = df["close"].rolling(window=long).mean()

        high_low = df["high"] - df["low"]
        high_prev = (df["high"] - df["close"].shift(1)).abs()
        low_prev = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row["ma_short"], row["ma_long"],
                     prev_row["ma_short"], prev_row["ma_long"]):
            return 0
        if prev_row["ma_short"] <= prev_row["ma_long"] and row["ma_short"] > row["ma_long"]:
            return 1
        if prev_row["ma_short"] >= prev_row["ma_long"] and row["ma_short"] < row["ma_long"]:
            return -1
        return 0

    def position_fraction(self, params: Mapping[str, Any], context=None) -> float:
        """波动率目标仓位：风险预算 / (ATR/价格) = risk_budget * price / ATR。"""
        risk_budget = float(params.get("risk_budget", 0.01))
        max_fraction = float(params.get("max_fraction", 1.0))

        if context is None:
            return max_fraction
        current_row = context.get("current_row")
        if current_row is None:
            return max_fraction
        price = current_row.get("close")
        atr = current_row.get("atr")
        if price is None or atr is None or atr <= 0:
            return max_fraction

        fraction = risk_budget * price / atr
        return min(max(fraction, 0.0), max_fraction)
