"""长线深度价值策略 — L-L0"""
from typing import Dict, Any, List
from src.eval.strategies.base import BaseStrategy, BuyOrder, SellOrder


class LongTermStrategy(BaseStrategy):
    """长线策略：深度价值，集中持仓，每月调仓。买了不轻易卖。

    总纲 §5.5 规则：
      选股 — 高确定性+高评分（score >= 70），纯评分排序
      卖出 — 低分持续确认(score<35持续≥40交易日全卖)
            + ROE连续下降(stub) + PE极端(stub) + 黑天鹅(立即清仓)
    """

    def _load_config(self):
        self.max_positions = self.config.get("max_positions_long", 5)
        self.single_weight_limit = self.config.get("single_weight_limit_long", 0.25)
        self.min_cash_ratio = self.config.get("min_cash_ratio_long", 0.20)
        self.buy_threshold = self.config.get("score_buy_threshold_long", 70)
        self.sell_hard = self.config.get("score_sell_hard_long", 35)
        # 低评分持续天数追踪（总纲 §5.5：score<35持续2个月以上才卖出）
        self._low_score_days: Dict[str, int] = {}

    def get_max_positions(self) -> int:
        return self.max_positions

    def get_single_weight_limit(self) -> float:
        return self.single_weight_limit

    def get_min_cash_ratio(self) -> float:
        return self.min_cash_ratio

    def get_score_buy_threshold(self) -> float:
        return self.buy_threshold

    def select_stocks(self, pool, scores, holdings, cash, market_data_map=None):
        """长线选股：高确定性+高评分"""
        candidates = []
        for code in pool:
            if code in holdings and holdings[code] > 0:
                continue
            score = scores.get(code, 0)
            if score >= self.buy_threshold:
                candidates.append((code, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        orders = []
        for code, score in candidates[:self.max_positions]:
            order = BuyOrder(code, 0, score)
            order.reason = f"长线价值: score={score:.0f}"
            orders.append(order)

        return orders

    def generate_sell_orders(self, holdings, scores, market_data_map=None,
                             purchase_prices=None, hold_days=None):
        """长线卖出：非常克制，需要强证据。

        总纲 §5.5 卖出规则（优先级从高到低）：
          1. 黑天鹅 — delist_risk/audit_risk → 立即清仓100%
          2. 低分持续 — score<35持续≥40交易日(≈2个月) → 全卖
          3. ROE连续下降 — ROE连续2季下降>30% → 卖出50%（STUB：需季度数据）
          4. PE极端 — PE突破历史95分位 → 卖出30%（STUB：需历史PE数据）
        """
        sell_orders = []

        for code, shares in holdings.items():
            if shares <= 0:
                continue
            score = scores.get(code, 0)
            reasons = []
            sell_ratio = 0.0

            md = market_data_map.get(code) if market_data_map else None

            # ── 规则1：黑天鹅 delist_risk/audit_risk → 立即清仓100% ──
            if md and hasattr(md, 'risk_flags') and md.risk_flags:
                flags = list(md.risk_flags) if isinstance(md.risk_flags, (list, tuple)) else [str(md.risk_flags)]
                critical = [f for f in flags if 'delist' in str(f).lower() or 'audit' in str(f).lower()]
                if critical:
                    sell_ratio = 1.0
                    reasons.append(f"黑天鹅风险: {', '.join(critical)}")

            # ── 规则2：score<35持续≥40交易日 → 全卖 ──
            # 不与规则1叠加（规则1已清仓）
            if sell_ratio < 1.0 and score < self.sell_hard:
                self._low_score_days[code] = self._low_score_days.get(code, 0) + 1
                if self._low_score_days[code] >= 40:
                    sell_ratio = 1.0
                    reasons.append(
                        f"长线基本面恶化 score={score:.0f}<{self.sell_hard}"
                        f" 持续{self._low_score_days[code]}个交易日"
                    )
                # 低于40天：不卖出但保留计数器，reasons为空则跳过
            elif score >= self.sell_hard:
                # 评分恢复到阈值以上，重置计数器
                if code in self._low_score_days:
                    self._low_score_days[code] = 0

            # ── 规则3：ROE连续2季下降>30% → 卖出50% ──
            # STUB：需要季度财报数据（fina_indicator），当前MarketData不含ROE历史。
            # 实现路径：从eval runner注入roe_q1, roe_q0到MarketData，
            # 检查 roe_q1/roe_q0 < 0.7（即下降>30%）且roe_q2/roe_q1 < 0.7。
            #
            # if (md and hasattr(md, 'roe_q0') and hasattr(md, 'roe_q1')
            #     and hasattr(md, 'roe_q2') and md.roe_q1 > 0 and md.roe_q2 > 0
            #     and md.roe_q1 / md.roe_q2 < 0.7):
            #     sell_ratio = max(sell_ratio, 0.5)
            #     reasons.append("ROE连续2季下降>30%")

            # ── 规则4：PE突破历史95分位 → 卖出30% ──
            # STUB：需要历史PE分位数数据（daily_basic长期历史），
            # 当前MarketData不含PE分位。
            # 实现路径：从eval runner注入pe_percentile到MarketData，
            # 检查 pe_percentile > 95。
            #
            # if md and hasattr(md, 'pe_percentile') and md.pe_percentile > 95:
            #     sell_ratio = max(sell_ratio, 0.3)
            #     reasons.append(f"PE突破历史95分位({md.pe_percentile:.0f})")

            # ── 清理已触发清仓的追踪 ──
            if sell_ratio >= 1.0:
                self._low_score_days.pop(code, None)

            if sell_ratio > 0:
                sell_orders.append(SellOrder(code, sell_ratio, "; ".join(reasons)))

        # ── 清理已不在持仓中的股票追踪 ──
        for code in list(self._low_score_days.keys()):
            if code not in holdings:
                del self._low_score_days[code]

        return sell_orders
