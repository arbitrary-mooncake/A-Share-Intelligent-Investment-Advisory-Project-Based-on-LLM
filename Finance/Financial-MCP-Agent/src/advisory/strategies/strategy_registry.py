"""策略注册器 — 单例模式 + @register_strategy 装饰器。

提供策略类注册、查找、元数据查询功能。
"""
from __future__ import annotations

from typing import Dict, Optional, Type

from .strategy_base import TradingStrategy


class _RegistryMeta(type):
    """单例元类。"""

    _instance: Optional["StrategyRegistry"] = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class StrategyRegistry(metaclass=_RegistryMeta):
    """策略注册器单例。

    Usage:
        @register_strategy
        class MaCross(TradingStrategy):
            name = "ma_cross"
            ...

        cls = StrategyRegistry.get("ma_cross")
        names = StrategyRegistry.list_names()
    """

    def __init__(self):
        if not hasattr(self, "_strategies"):
            self._strategies: Dict[str, Type[TradingStrategy]] = {}

    @classmethod
    def register(cls, strategy_cls: Type[TradingStrategy]) -> Type[TradingStrategy]:
        """注册一个策略类。"""
        instance = cls()
        name = strategy_cls.name
        if not name:
            raise ValueError(f"策略类 {strategy_cls.__name__} 必须定义 name 属性")
        if name in instance._strategies:
            raise ValueError(f"策略名称 '{name}' 已被 {instance._strategies[name].__name__} 注册")
        instance._strategies[name] = strategy_cls
        return strategy_cls

    @classmethod
    def get(cls, name: str) -> Optional[Type[TradingStrategy]]:
        """根据名称获取策略类。"""
        instance = cls()
        return instance._strategies.get(name)

    @classmethod
    def list_names(cls) -> list:
        """列出所有已注册策略名称。"""
        instance = cls()
        return sorted(instance._strategies.keys())

    @classmethod
    def get_metadata(cls, name: str) -> Optional[dict]:
        """获取策略元数据: {name, description, requires_multi_asset}。"""
        strategy_cls = cls.get(name)
        if strategy_cls is None:
            return None
        return {
            "name": strategy_cls.name,
            "description": strategy_cls.description,
            "requires_multi_asset": strategy_cls.requires_multi_asset,
        }


def register_strategy(cls: Type[TradingStrategy]) -> Type[TradingStrategy]:
    """装饰器：自动将策略类注册到 StrategyRegistry。"""
    return StrategyRegistry.register(cls)


def get_strategy_class(name: str) -> Optional[Type[TradingStrategy]]:
    """便捷函数：根据名称获取策略类。"""
    return StrategyRegistry.get(name)
