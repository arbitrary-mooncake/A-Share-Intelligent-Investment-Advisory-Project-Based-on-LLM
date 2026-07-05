"""策略工厂"""
from typing import Dict, Any
from src.eval.strategies.base import BaseStrategy
from src.eval.strategies.short_ablation import ShortAblationStrategy
from src.eval.strategies.short_longhold import ShortLongHoldStrategy
from src.eval.strategies.medium_term import MediumTermStrategy
from src.eval.strategies.long_term import LongTermStrategy
from src.eval.strategies.llm_free import LLMFreeStrategy


STRATEGY_REGISTRY = {
    ("short", "ablation"): ShortAblationStrategy,
    ("short", "longhold"): ShortLongHoldStrategy,
    ("short", "default"): ShortAblationStrategy,
    ("short", "llm_free"): LLMFreeStrategy,
    ("medium", "default"): MediumTermStrategy,
    ("medium", "llm_free"): LLMFreeStrategy,
    ("long", "default"): LongTermStrategy,
    ("long", "llm_free"): LLMFreeStrategy,
}


def get_strategy(term: str, strategy_type: str = "default",
                 config: Dict[str, Any] = None) -> BaseStrategy:
    """
    获取策略实例。

    Args:
        term: short/medium/long
        strategy_type: ablation/longhold/default
        config: 策略配置参数（覆盖默认值）

    Returns:
        BaseStrategy实例
    """
    key = (term, strategy_type)
    strategy_cls = STRATEGY_REGISTRY.get(key)
    if strategy_cls is None:
        # 回退到对应期限的默认策略
        fallback_key = (term, "default")
        strategy_cls = STRATEGY_REGISTRY.get(fallback_key)
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy: term={term}, type={strategy_type}")

    return strategy_cls(config, term=term)


def list_strategies() -> Dict:
    """列出所有已注册策略"""
    return {str(k): v.__name__ for k, v in STRATEGY_REGISTRY.items()}
