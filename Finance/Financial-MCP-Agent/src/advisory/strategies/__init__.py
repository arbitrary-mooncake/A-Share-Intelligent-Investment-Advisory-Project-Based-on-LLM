"""策略系统 — 可插拔量化交易策略框架。

提供:
- TradingStrategy 基类
- StrategyRegistry 注册器 + @register_strategy 装饰器
- builtin 内置策略包 (20+ 策略)
"""
from .strategy_base import TradingStrategy
from .strategy_registry import (
    StrategyRegistry,
    register_strategy,
    get_strategy_class,
)

__all__ = [
    "TradingStrategy",
    "StrategyRegistry",
    "register_strategy",
    "get_strategy_class",
]
