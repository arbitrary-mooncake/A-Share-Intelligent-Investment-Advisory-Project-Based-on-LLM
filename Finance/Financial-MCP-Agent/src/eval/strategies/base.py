"""策略基类"""
from typing import Dict, Any, List, Tuple


class BuyOrder:
    """买入订单"""
    def __init__(self, stock_code: str, target_value: float, score: float = 0):
        self.stock_code = stock_code
        self.target_value = target_value
        self.score = score
        self.reason = ""


class SellOrder:
    """卖出订单"""
    def __init__(self, stock_code: str, sell_ratio: float, reason: str = ""):
        self.stock_code = stock_code
        self.sell_ratio = sell_ratio  # 0.0-1.0
        self.reason = reason


class BaseStrategy:
    """策略基类 — 所有交易策略继承此类"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._load_config()

    def _load_config(self):
        """从config加载策略参数，子类可覆盖"""
        pass

    def select_stocks(self, pool: List[str], scores: Dict[str, float],
                      holdings: Dict[str, int], cash: float,
                      market_data_map: Dict[str, Any] = None,
                      total_capital: float = 0.0,
                      **kwargs) -> List[BuyOrder]:
        """
        选股逻辑 — 子类必须实现。

        Args:
            pool: 精筛池股票代码列表
            scores: {stock_code: score}
            holdings: {stock_code: shares}
            cash: 可用现金
            market_data_map: {stock_code: MarketData}
            total_capital: 总资金（用于加仓/补仓计算，v2新增）
            **kwargs: 跨期限评分等扩展参数 (medium_scores, long_scores 等)

        Returns:
            List[BuyOrder]: 按优先级排序的买入订单列表
        """
        raise NotImplementedError

    def generate_sell_orders(self, holdings: Dict[str, int],
                             scores: Dict[str, float],
                             market_data_map: Dict[str, Any] = None,
                             purchase_prices: Dict[str, float] = None,
                             hold_days: Dict[str, int] = None,
                             **kwargs) -> List[SellOrder]:
        """
        卖出判定 — 子类必须实现。

        Args:
            holdings: {stock_code: shares}
            scores: {stock_code: score}
            market_data_map: {stock_code: MarketData}
            purchase_prices: {stock_code: avg_purchase_price}
            hold_days: {stock_code: days_held}
            **kwargs: 跨期限评分等扩展参数 (medium_scores, long_scores, total_capital 等)

        Returns:
            List[SellOrder]: 卖出订单列表
        """
        raise NotImplementedError

    def size_positions(self, buy_orders: List[BuyOrder],
                       total_capital: float, current_positions: int,
                       max_positions: int, single_weight_limit: float,
                       min_cash_ratio: float,
                       holdings: Dict[str, int] = None,
                       market_data_map: Dict[str, Any] = None,
                       **kwargs) -> List[BuyOrder]:
        """
        仓位调整 — 基类提供默认实现。

        v2新增:
          - holdings/market_data_map: 用于行业上限检查
          - **kwargs: 跨期限评分等扩展参数

        返回调整后的买入订单（含target_value）。
        """
        available_cash = total_capital * (1 - min_cash_ratio)
        slots = max(0, max_positions - current_positions)

        if slots <= 0 or not buy_orders:
            return []

        sized = []
        for i, order in enumerate(buy_orders[:slots]):
            weight = min(1.0 / max(slots, 1), single_weight_limit)
            target = total_capital * weight
            target = min(target, available_cash / max(slots - i, 1))
            order.target_value = target
            sized.append(order)

        return sized

    def get_max_positions(self) -> int:
        """最大持仓数"""
        return 10

    def get_single_weight_limit(self) -> float:
        """单只股票仓位上限"""
        return 0.10

    def get_min_cash_ratio(self) -> float:
        """最低现金比例"""
        return 0.0

    def get_score_buy_threshold(self) -> float:
        """买入评分阈值"""
        return 60.0

    def get_score_sell_threshold(self) -> float:
        """卖出评分阈值"""
        return 35.0
