"""KDJ 金叉/死叉策略。
参数: k=9, d=3, j=3
适用: 短线震荡市，K 线上穿 D 线买入，下穿卖出，J 线辅助判断极端区域。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class Kdj(TradingStrategy):
    name = "kdj"
    description = "KDJ 金叉死叉：K 线上穿 D 线买入，下穿卖出"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        k_period = int(params.get("k", 9))
        d_period = int(params.get("d", 3))
        df = df.copy()

        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        rsv = np.where(
            (high_max - low_min) > 0,
            (df["close"] - low_min) / (high_max - low_min) * 100,
            50.0,
        )

        n = len(df)
        k_vals = np.full(n, 50.0)
        d_vals = np.full(n, 50.0)
        for i in range(1, n):
            if not np.isnan(rsv[i]):
                k_vals[i] = 2.0 / 3.0 * k_vals[i - 1] + 1.0 / 3.0 * rsv[i]
            else:
                k_vals[i] = k_vals[i - 1]
            d_vals[i] = 2.0 / 3.0 * d_vals[i - 1] + 1.0 / 3.0 * k_vals[i]

        df["k"] = k_vals
        df["d"] = d_vals
        df["j_val"] = 3.0 * k_vals - 2.0 * d_vals
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row["k"], row["d"], prev_row["k"], prev_row["d"]):
            return 0
        if prev_row["k"] <= prev_row["d"] and row["k"] > row["d"]:
            return 1
        if prev_row["k"] >= prev_row["d"] and row["k"] < row["d"]:
            return -1
        return 0
