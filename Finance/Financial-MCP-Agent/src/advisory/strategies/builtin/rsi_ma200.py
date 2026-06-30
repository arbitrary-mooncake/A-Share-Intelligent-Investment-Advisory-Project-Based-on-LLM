"""RSI + 长期均线过滤策略。
参数: ma=200, rsi_period=14
适用: 中长期持仓（均线上方只多不空），价格在 MA200 上方时 RSI 超卖买入；
价格在 MA200 下方时 RSI 超买卖出。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class RsiMa200(TradingStrategy):
    name = "rsi_ma200"
    description = "RSI + MA200 过滤：均线上方 RSI 超卖买入，下方 RSI 超买卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        ma_period = int(params.get("ma", 200))
        rsi_period = int(params.get("rsi_period", 14))
        df = df.copy()

        # MA200
        df["ma200"] = df["close"].rolling(window=ma_period).mean()

        # RSI (Wilder's)
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        lower = float(params.get("lower", 30))
        upper = float(params.get("upper", 70))
        if self._na(row["ma200"], row["rsi"], prev_row["rsi"]):
            return 0
        above_ma = row["close"] > row["ma200"]
        below_ma = row["close"] < row["ma200"]
        if above_ma and prev_row["rsi"] <= lower and row["rsi"] > lower:
            return 1
        if below_ma and prev_row["rsi"] >= upper and row["rsi"] < upper:
            return -1
        return 0
