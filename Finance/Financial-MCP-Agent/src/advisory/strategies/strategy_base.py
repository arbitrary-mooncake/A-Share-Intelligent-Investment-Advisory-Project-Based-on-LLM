"""交易策略抽象基类。

所有策略必须继承 TradingStrategy 并实现 enrich() + signal()。
可选实现 risk_exit() 和 position_fraction()。
策略执行纯 Python 代码，无 LLM 参与。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

import pandas as pd


class TradingStrategy(ABC):
    """量化交易策略基类。

    Attributes:
        name: 策略唯一标识名（由 @register_strategy 装饰器注入或子类定义）。
        description: 策略简短描述。
        requires_multi_asset: 是否需要多资产数据（跨股票比较）。
    """

    name: str = ""
    description: str = ""
    requires_multi_asset: bool = False

    @abstractmethod
    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        """在日线数据上计算并添加指标列。

        Args:
            df: OHLCV 日线数据，必须包含列 trade_date, close, open, high, low, vol, amount。
            params: 策略参数映射。

        Returns:
            添加了指标列的 DataFrame（可以是原 df 或副本）。
        """

    @abstractmethod
    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        """根据当前行与前一行数据生成交易信号。

        Args:
            row: 当前行（pandas Series），包含 enrich() 添加的所有列。
            prev_row: 前一行（pandas Series 或 None）。
            params: 策略参数映射。
            context: 可选上下文 dict，可包含 position, entry_price 等。

        Returns:
            1 (买入), -1 (卖出), 或 0 (不动)。
        """

    def risk_exit(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        """风险退出检查（在持仓时优先于 signal 执行）。

        Args:
            row: 当前行。
            prev_row: 前一行。
            params: 策略参数映射。
            context: 上下文 dict（建议包含 entry_price, position）。

        Returns:
            -1 (强制退出) 或 0 (不触发)。
        """
        return 0

    def position_fraction(self, params: Mapping[str, Any], context=None) -> float:
        """返回仓位比例 (0.0 ~ 1.0)。

        Args:
            params: 策略参数映射。
            context: 可选上下文。

        Returns:
            仓位比例，默认 1.0（满仓）。
        """
        return float(params.get("position_fraction", 1.0))

    @classmethod
    def _na(cls, *values) -> bool:
        """检查任意值是否为 NaN。"""
        return any(pd.isna(v) for v in values)
