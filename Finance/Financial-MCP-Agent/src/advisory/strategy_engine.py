"""策略执行引擎 — 在日线数据上运行策略信号计算。

纯 Python 实现，无 LLM 调用。提供策略信号计算、仓位比例建议、
策略代码模板生成、已注册策略目录查询等能力。

Usage:
    from src.advisory.strategy_engine import StrategyEngine

    sig, reason = StrategyEngine.compute_signal("ma_cross", df_daily)
    frac = StrategyEngine.compute_position_fraction("ma_cross")
    catalog = StrategyEngine.get_strategy_catalog()
    code = StrategyEngine.generate_strategy_code("my_strat", "自定义策略")
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from src.advisory.strategies.strategy_base import TradingStrategy
from src.advisory.strategies.strategy_registry import StrategyRegistry, get_strategy_class

logger = logging.getLogger(__name__)


class StrategyEngine:
    """策略执行引擎 — 所有方法均为静态方法。"""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compute_signal(
        strategy_name: str,
        df_daily: pd.DataFrame,
        params: Optional[Mapping[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, str]:
        """在日线数据上计算策略的交易信号。

        Args:
            strategy_name: 策略注册名称，如 "ma_cross"。
            df_daily: 升序排列的 OHLCV 日线 DataFrame，必须包含
                      trade_date, close, open, high, low, vol 列。
                      应包含足够的预热数据（通常 > 最长参数窗口）。
            params: 策略参数字典。为 None 时使用策略默认参数。
            context: 上下文字典，可包含:
                     - position (int): 当前持仓股数，默认 0。
                     - entry_price (float): 持仓均价，默认 None。
                     - cash (float): 可用现金，默认 0。

        Returns:
            (signal, reason) 元组:
                signal: 1 (买入), -1 (卖出), 0 (不动)。
                reason: 中文文字说明。

        Raises:
            ValueError: 策略名称未注册、DataFrame 为空或未升序排列。
        """
        params = dict(params) if params else {}
        context = dict(context) if context else {}

        # 1. 获取策略类
        strategy_cls = get_strategy_class(strategy_name)
        if strategy_cls is None:
            raise ValueError(
                f"策略 '{strategy_name}' 未注册。可用策略: "
                f"{', '.join(StrategyRegistry.list_names())}"
            )
        strategy = strategy_cls()

        # 当策略覆盖了 risk_exit 但上下文为空时发出调试日志
        if (
            type(strategy).risk_exit is not TradingStrategy.risk_exit
            and not context.get("position")
        ):
            logger.debug(
                "策略 '%s' 覆盖了 risk_exit，但传入的 context 缺少 position，"
                "风险退出逻辑可能被静默跳过。",
                strategy_name,
            )

        # 2. 校验 DataFrame
        if df_daily is None or df_daily.empty:
            raise ValueError("df_daily 为空，无法计算信号")

        # 检查升序（如果数据足够多，对比首尾日期）
        if len(df_daily) >= 2:
            first_date = df_daily["trade_date"].iloc[0]
            last_date = df_daily["trade_date"].iloc[-1]
            if str(first_date) > str(last_date):
                raise ValueError("df_daily 必须是升序排列，当前为降序")

        # 3. 计算指标（enrich）
        enriched = strategy.enrich(df_daily, params)

        # 4. 取最后两行
        if len(enriched) < 2:
            return 0, "数据不足（少于2行），无法生成信号"

        curr_row = enriched.iloc[-1]
        prev_row = enriched.iloc[-2]

        # 5. 持仓风险退出（优先于开仓信号）
        position = context.get("position", 0)
        if position > 0:
            exit_signal = strategy.risk_exit(curr_row, prev_row, params, context)
            if exit_signal == -1:
                reason_params = dict(params)
                reason_params["_reason_risk_exit"] = True
                reason = _build_reason(strategy_name, -1, curr_row, prev_row, reason_params)
                return -1, reason

        # 6. 计算开仓/平仓信号
        sig = strategy.signal(curr_row, prev_row, params, context)

        # 7. 生成文字说明
        reason = _build_reason(strategy_name, sig, curr_row, prev_row, params)

        return sig, reason

    @staticmethod
    def compute_position_fraction(
        strategy_name: str,
        params: Optional[Mapping[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """返回策略建议的仓位比例 (0.0 ~ 1.0)。

        委托给策略实例的 position_fraction() 方法，各策略可自定义仓位管理。
        默认返回 1.0（满仓）。

        Args:
            strategy_name: 策略注册名称。
            params: 策略参数字典。为 None 时使用策略默认参数。
            context: 可选上下文字典。

        Returns:
            仓位比例 0.0 ~ 1.0。

        Raises:
            ValueError: 策略名称未注册。
        """
        params = dict(params) if params else {}
        strategy_cls = get_strategy_class(strategy_name)
        if strategy_cls is None:
            raise ValueError(
                f"策略 '{strategy_name}' 未注册。可用策略: "
                f"{', '.join(StrategyRegistry.list_names())}"
            )
        strategy = strategy_cls()
        return float(strategy.position_fraction(params, context))

    @staticmethod
    def generate_strategy_code(
        strategy_name: str,
        user_description: str,
        base_strategy: Optional[str] = None,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """生成策略 Python 代码模板（供 LLM 填充）。

        模板包含 @register_strategy 装饰器、类骨架、enrich() 和 signal()
        方法签名，以及占位注释。

        Args:
            strategy_name: 策略唯一标识名（会用作类名和 name 属性）。
            user_description: 策略用途描述（会写入类文档字符串和 description）。
            base_strategy: 可选，基于的已有策略名称。提供后会在注释中列出参考。
            custom_params: 可选，自定义参数列表，会生成对应的默认参数字典。

        Returns:
            包含完整策略类骨架的 Python 代码字符串。
        """
        # 安全地生成类名：策略名转 PascalCase
        class_name = _name_to_pascal(strategy_name)
        params_code = _format_params(custom_params)
        base_ref = ""
        if base_strategy:
            base_cls = get_strategy_class(base_strategy)
            if base_cls:
                base_ref = (
                    f"    # 参考策略: {base_strategy} — {base_cls.description}\n"
                    f"    # 建议复用了 enrich 逻辑后，在 signal() 中实现差异化逻辑\n"
                )

        code = f'''"""
{user_description}

Generated by StrategyEngine.generate_strategy_code().
Fill in the TODOs below to complete the strategy.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..strategy_base import TradingStrategy
from ..strategy_registry import register_strategy


@register_strategy
class {class_name}(TradingStrategy):
    """TODO: 添加策略描述。"""

    name = "{strategy_name}"
    description = "{user_description}"
{base_ref}
    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        """计算技术指标。
{params_code}
        TODO: 在 df 上添加指标列。

        Args:
            df: OHLCV 日线数据，包含 trade_date, close, open, high, low, vol。
            params: 策略参数字典。

        Returns:
            添加了指标列的 DataFrame。
        """
        # TODO: 实现指标计算逻辑
        # 示例:
        # df = df.copy()
        # df["my_indicator"] = df["close"].rolling(window=5).mean()
        # return df
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        """生成交易信号。

        TODO: 实现信号生成逻辑。

        Args:
            row: 当前行（含 enrich() 添加的列）。
            prev_row: 前一行。
            params: 策略参数字典。
            context: 上下文（可含 position, entry_price, cash 等）。

        Returns:
            1 (买入), -1 (卖出), 或 0 (不动)。
        """
        # TODO: 实现信号逻辑
        # 使用 self._na() 检查 NaN 值:
        #   if self._na(row["my_indicator"], prev_row["my_indicator"]):
        #       return 0
        return 0

    def risk_exit(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        """风险退出检查（可选覆盖）。

        Args:
            row: 当前行。
            prev_row: 前一行。
            params: 策略参数字典。
            context: 上下文（建议包含 entry_price, position）。

        Returns:
            -1 (强制退出) 或 0 (不触发)。
        """
        return 0

    def position_fraction(self, params: Mapping[str, Any], context=None) -> float:
        """仓位比例（可选覆盖）。
{params_code}
        Returns:
            仓位比例 0.0 ~ 1.0，默认 1.0。
        """
        return float(params.get("position_fraction", 1.0))
'''
        return code

    @staticmethod
    def get_strategy_catalog() -> List[Dict[str, str]]:
        """返回所有已注册策略的元数据列表。

        每个元素包含:
            - name: 策略名称
            - description: 策略描述
            - category: 自动猜分类

        Returns:
            策略元数据字典列表。
        """
        catalog = []
        for name in StrategyRegistry.list_names():
            meta = StrategyRegistry.get_metadata(name)
            if meta is None:
                continue
            catalog.append({
                "name": name,
                "description": meta.get("description", ""),
                "category": _guess_category(name),
            })
        return catalog


# ------------------------------------------------------------------
# 内部辅助函数
# ------------------------------------------------------------------


def _guess_category(name: str) -> str:
    """按策略名称猜测分类。"""
    name_lower = name.lower()

    # 均线类
    if any(kw in name_lower for kw in ("ma_", "ema", "sma", "均线", "triple_ma")):
        return "均线"

    # 布林带（放在指标前，避免 "rsi" 误匹配 "boll_reversion" 中的 "version"）
    if any(kw in name_lower for kw in ("boll", "布林")):
        return "布林"

    # 指标类（通用技术指标）
    if any(kw in name_lower for kw in ("macd", "rsi", "kdj", "cci", "stochastic",
                                        "williams", "adx")):
        return "指标"

    # 通道类
    if any(kw in name_lower for kw in ("通道", "donchian", "turtle")):
        return "通道"

    # 震荡类
    if any(kw in name_lower for kw in ("震荡", "oscillator")):
        return "震荡"

    # 复合类
    if any(kw in name_lower for kw in ("复合", "atr_stop", "vol_target", "kelly",
                                        "bias")):
        return "复合"

    # 量价类
    if any(kw in name_lower for kw in ("volume", "obv", "vwap", "量价", "量")):
        return "量价"

    # 动量类
    if any(kw in name_lower for kw in ("动量", "momentum", "roc", "mome")):
        return "动量"

    # 风险类
    if any(kw in name_lower for kw in ("风险", "risk", "止损", "stop")):
        return "风险"

    # 默认
    return "其他"


def _name_to_pascal(name: str) -> str:
    """将策略名（如 ma_cross）转为 PascalCase（如 MaCross）。"""
    return "".join(word.capitalize() for word in name.replace("-", "_").split("_"))


def _format_params(custom_params: Optional[Dict[str, Any]]) -> str:
    """将自定义参数字典格式化为注释。"""
    if not custom_params:
        return ""
    lines = []
    for k, v in custom_params.items():
        lines.append(f"    #   {k}: {v} (默认值)")
    return "\n" + "\n".join(lines)


def _build_reason(
    strategy_name: str,
    signal: int,
    curr_row: pd.Series,
    prev_row: pd.Series,
    params: Mapping[str, Any],
) -> str:
    """生成信号的中文文字说明。"""
    date_str = str(curr_row.get("trade_date", "?"))
    close = curr_row.get("close", None)
    close_str = f"{close:.2f}" if close is not None and not pd.isna(close) else "N/A"

    if signal == 1:
        return f"{strategy_name} 于 {date_str} 触发买入信号（收盘价 {close_str}）"
    elif signal == -1:
        if params.get("_reason_risk_exit"):
            return f"{strategy_name} 于 {date_str} 触发风险退出（收盘价 {close_str}）"
        return f"{strategy_name} 于 {date_str} 触发卖出信号（收盘价 {close_str}）"
    else:
        return f"{strategy_name} 于 {date_str} 无操作（收盘价 {close_str}）"
