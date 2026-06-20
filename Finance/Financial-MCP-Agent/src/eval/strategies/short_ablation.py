"""极简日清策略 — 短线消融线专用（S-L0~S-L7）"""
from typing import Dict, Any, List
from src.eval.strategies.base import BaseStrategy, BuyOrder, SellOrder


class ShortAblationStrategy(BaseStrategy):
    """
    极简策略：每天清仓，从精筛池中取score最高的前N只等权买入。
    策略差异最小化，评分差异成为唯一的收益驱动因素。
    用于消融实验的交叉截面设计。
    """

    def _load_config(self):
        self.max_positions = self.config.get("max_positions_short_ablation", 10)
        self.single_weight_limit = self.config.get("single_weight_limit_short_ablation", 0.10)
        self.min_cash_ratio = self.config.get("min_cash_ratio_short", 0.0)
        self.buy_threshold = self.config.get("score_buy_threshold_short", 60)

    def get_max_positions(self) -> int:
        return self.max_positions

    def get_single_weight_limit(self) -> float:
        return self.single_weight_limit

    def get_min_cash_ratio(self) -> float:
        return self.min_cash_ratio

    def get_score_buy_threshold(self) -> float:
        return self.buy_threshold

    def select_stocks(self, pool, scores, holdings, cash, market_data_map=None):
        """取评分最高的前N只，score需≥阈值"""
        candidates = [
            (code, scores[code]) for code in pool
            if code in scores and scores[code] >= self.buy_threshold
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        orders = []
        for code, score in candidates[:self.max_positions]:
            order = BuyOrder(code, 0, score)  # target_value在size_positions中计算
            order.reason = f"极简策略 Top{len(orders)+1}: score={score}"
            orders.append(order)

        return orders

    def generate_sell_orders(self, holdings, scores, market_data_map=None,
                             purchase_prices=None, hold_days=None):
        """清仓所有持仓"""
        orders = []
        for code in holdings:
            if holdings[code] > 0:
                orders.append(SellOrder(code, 1.0, "日清策略：全部卖出"))
        return orders
