"""ADX 趋势过滤 + MACD 策略。
参数: adx_period=14, min_adx=25, fast=12, slow=26, signal=9
适用: 仅在强趋势市场中交易（ADX > min_adx），用 MACD 金叉死叉进出场。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class AdxMacd(TradingStrategy):
    name = "adx_macd"
    description = "ADX 趋势过滤 + MACD：ADX>阈值时 MACD 金叉买入，死叉卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        adx_period = int(params.get("adx_period", 14))
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal_period = int(params.get("signal", 9))
        df = df.copy()

        # MACD
        df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
        df["dif"] = df["ema_fast"] - df["ema_slow"]
        df["dea"] = df["dif"].ewm(span=signal_period, adjust=False).mean()
        df["macd_hist"] = 2 * (df["dif"] - df["dea"])

        # ADX / DMI
        high_diff = df["high"].diff()
        low_diff = -df["low"].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr_smooth = tr.ewm(alpha=1.0 / adx_period, adjust=False).mean()
        plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(
            alpha=1.0 / adx_period, adjust=False).mean() / atr_smooth.replace(0, float("nan"))
        minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(
            alpha=1.0 / adx_period, adjust=False).mean() / atr_smooth.replace(0, float("nan"))

        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, float("nan"))
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        df["adx"] = dx.ewm(alpha=1.0 / adx_period, adjust=False).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        min_adx = float(params.get("min_adx", 25))
        if self._na(row["dif"], row["dea"], row["adx"],
                     prev_row["dif"], prev_row["dea"]):
            return 0
        if row["adx"] < min_adx:
            return 0
        if prev_row["dif"] <= prev_row["dea"] and row["dif"] > row["dea"]:
            return 1
        if prev_row["dif"] >= prev_row["dea"] and row["dif"] < row["dea"]:
            return -1
        return 0
