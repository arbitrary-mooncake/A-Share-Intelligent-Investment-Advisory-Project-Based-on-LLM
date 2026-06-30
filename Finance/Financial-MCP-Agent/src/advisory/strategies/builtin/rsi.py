"""RSI 超卖/超买阈值策略。
参数: period=14, lower=30, upper=70
适用: 震荡市中捕捉超卖反弹和超买回落，RSI 上穿超卖线买入，下穿超买线卖出。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class Rsi(TradingStrategy):
    name = "rsi"
    description = "RSI 超卖超买：RSI 上穿超卖阈值买入，下穿超买阈值卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        period = int(params.get("period", 14))
        df = df.copy()
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        # Wilder's smoothing: alpha = 1/period
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        lower = float(params.get("lower", 30))
        upper = float(params.get("upper", 70))
        if self._na(row["rsi"], prev_row["rsi"]):
            return 0
        if prev_row["rsi"] <= lower and row["rsi"] > lower:
            return 1
        if prev_row["rsi"] >= upper and row["rsi"] < upper:
            return -1
        return 0
