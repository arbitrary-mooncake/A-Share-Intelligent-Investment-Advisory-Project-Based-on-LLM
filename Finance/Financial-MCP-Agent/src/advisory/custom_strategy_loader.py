"""用户自定义策略加载器 — 从 data/strategies/custom/ 动态加载 .py 策略文件。

设计 §6.5：用户通过自然语言生成的策略保存在此目录，
启动时和生成新策略后自动加载，通过 @register_strategy 注册到 StrategyRegistry。

Usage:
    from src.advisory.custom_strategy_loader import load_custom_strategies

    loaded = load_custom_strategies()  # 返回加载的策略名列表
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from typing import List, Optional

from src.advisory.strategies.strategy_registry import StrategyRegistry

logger = logging.getLogger(__name__)

_LOADED_MODULES: dict = {}


def _get_custom_dir() -> str:
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(root, "data", "strategies", "custom")


def load_custom_strategies(custom_dir: Optional[str] = None) -> List[str]:
    """扫描目录并动态加载所有 .py 策略文件。

    每个文件会被 import 为模块 `advisory_custom_{filename}`，
    文件内的 @register_strategy 装饰器会自动注册策略类。

    Args:
        custom_dir: 自定义策略目录。None 时使用默认 data/strategies/custom/。

    Returns:
        成功加载的策略名称列表（来自 StrategyRegistry 中新增的策略）。
    """
    target_dir = custom_dir or _get_custom_dir()
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return []

    before_names = set(StrategyRegistry.list_names())

    for fname in os.listdir(target_dir):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        fpath = os.path.join(target_dir, fname)
        mod_name = f"advisory_custom_{os.path.splitext(fname)[0]}"

        if mod_name in _LOADED_MODULES:
            # 已加载过，先移除以支持热重载
            try:
                old_mod = _LOADED_MODULES.pop(mod_name)
                # 从 sys.modules 移除以便重新导入
                sys.modules.pop(mod_name, None)
            except Exception:
                pass

        try:
            spec = importlib.util.spec_from_file_location(mod_name, fpath)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            _LOADED_MODULES[mod_name] = module
            logger.info("自定义策略加载成功: %s", fname)
        except Exception as e:
            logger.error("自定义策略加载失败 %s: %s", fname, e)
            sys.modules.pop(mod_name, None)

    after_names = set(StrategyRegistry.list_names())
    new_names = sorted(after_names - before_names)
    return new_names


def save_custom_strategy(name: str, code: str) -> str:
    """将策略代码保存为 .py 文件。

    Args:
        name: 策略名称（用作文件名，会清洗特殊字符）。
        code: 策略 Python 代码。

    Returns:
        保存的文件绝对路径。
    """
    import re

    safe_name = re.sub(r"[^\w\-_]", "_", name)
    target_dir = _get_custom_dir()
    os.makedirs(target_dir, exist_ok=True)
    fpath = os.path.join(target_dir, f"{safe_name}.py")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(code)
    logger.info("自定义策略代码已保存: %s", fpath)
    return fpath
