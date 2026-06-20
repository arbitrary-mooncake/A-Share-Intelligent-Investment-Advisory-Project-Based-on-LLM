"""中线价值+趋势策略 — M-L0"""
from typing import Dict, Any, List
from src.eval.strategies.base import BaseStrategy, BuyOrder, SellOrder


class MediumTermStrategy(BaseStrategy):
    """中线策略：基本面驱动，技术面确认入场时机。每周调仓。

    总纲 §5.4 规则：
      选股 — 复合排名公式 score×0.6 + value_adjustment×0.4（简化自
            原规约 score×0.5+ROE分位×0.2+1/PE分位×0.2+20日动量×0.1）
      卖出 — 基本面恶化(score<45全卖) + 最大回撤(>15%卖50%)
            + 趋势破位(跌破MA60代理卖50%)
    """

    def _load_config(self):
        self.max_positions = self.config.get("max_positions_medium", 8)
        self.single_weight_limit = self.config.get("single_weight_limit_medium", 0.18)
        self.min_cash_ratio = self.config.get("min_cash_ratio_medium", 0.15)
        self.buy_threshold = self.config.get("score_buy_threshold_medium", 65)
        self.sell_hard = self.config.get("score_sell_hard_medium", 45)

    def get_max_positions(self) -> int:
        return self.max_positions

    def get_single_weight_limit(self) -> float:
        return self.single_weight_limit

    def get_min_cash_ratio(self) -> float:
        return self.min_cash_ratio

    def get_score_buy_threshold(self) -> float:
        return self.buy_threshold

    # ── 估值调整分计算 ──────────────────────────────────────────────
    def _compute_value_adjustment(self, md) -> float:
        """计算估值调整分（0-100），低PE/PB获得更高分。

        简化说明：MarketData 不含PE/PB字段时，以市值+沪深300成分股
        作为估值代理——大市值蓝筹获得更高估值调整分。
        当PE/PB数据可用时，使用PE倒数映射（PE<10→90, PE=20→50, PE>50→20）。
        """
        if hasattr(md, 'pe_ratio') and md.pe_ratio and md.pe_ratio > 0:
            # PE越低→调整分越高：PE<10→90, PE=20→50, PE>50→20
            return max(10.0, min(90.0, 100.0 - md.pe_ratio * 1.6))
        if hasattr(md, 'pb_ratio') and md.pb_ratio and md.pb_ratio > 0:
            # PB越低→调整分越高：PB<1→90, PB=3→50, PB>6→20
            return max(10.0, min(90.0, 100.0 - md.pb_ratio * 15.0))
        if hasattr(md, 'is_hs300') and md.is_hs300:
            return 60.0  # 蓝筹估值分略高
        return 50.0  # 默认中性

    # ── 选股 ────────────────────────────────────────────────────────
    def select_stocks(self, pool, scores, holdings, cash, market_data_map=None):
        """中线选股：高确定性，注重质量和估值。

        总纲 §5.4 选股排序公式（简化）：
          rank_score = score × 0.6 + value_adjustment × 0.4
        其中 value_adjustment 由PE/PB/蓝筹属性映射为0-100估值分。
        当 market_data_map 缺失时回退为纯评分排序（兼容测试）。
        """
        candidates = []
        for code in pool:
            if code in holdings and holdings[code] > 0:
                continue
            score = scores.get(code, 0)
            if score >= self.buy_threshold:
                # 复合排名：score×0.6 + value_adjustment×0.4
                rank_score = float(score)
                if market_data_map:
                    md = market_data_map.get(code)
                    if md:
                        value_adj = self._compute_value_adjustment(md)
                        rank_score = score * 0.6 + value_adj * 0.4
                candidates.append((code, score, rank_score))

        # 按复合排名降序
        candidates.sort(key=lambda x: x[2], reverse=True)

        orders = []
        for code, score, _ in candidates[:self.max_positions]:
            order = BuyOrder(code, 0, score)
            order.reason = f"中线价值: score={score:.0f}"
            orders.append(order)

        return orders

    # ── 卖出 ────────────────────────────────────────────────────────
    def generate_sell_orders(self, holdings, scores, market_data_map=None,
                             purchase_prices=None, hold_days=None):
        """中线卖出：基本面恶化+估值回撤+趋势破位。

        总纲 §5.4 卖出规则（优先级从高到低）：
          1. 基本面恶化 — score < 45 → 全卖
          2. 最大回撤 — 从买入价回撤 > 15% → 卖出50%
          3. 趋势破位 — 收盘价/MA60 < 0.95（跌破MA60代理）→ 卖出50%
        """
        sell_orders = []
        purchase_prices = purchase_prices or {}

        for code, shares in holdings.items():
            if shares <= 0:
                continue
            score = scores.get(code, 0)
            reasons = []
            sell_ratio = 0.0

            # ── 规则1：基本面恶化 score < 45 → 全卖 ──
            if score < self.sell_hard:
                sell_ratio = 1.0
                reasons.append(f"中线基本面恶化 score={score:.0f}<{self.sell_hard}")

            # ── 规则2：最大回撤 > 15% → 卖出50% ──
            md = market_data_map.get(code) if market_data_map else None
            if md and hasattr(md, 'close') and code in purchase_prices and purchase_prices[code] > 0:
                drawdown = (purchase_prices[code] - md.close) / purchase_prices[code]
                if drawdown > 0.15:
                    sell_ratio = max(sell_ratio, 0.5)
                    reasons.append(f"最大回撤 {drawdown:.1%}>15%")

            # ── 规则3：趋势破位 跌破MA60代理 → 卖出50% ──
            # 简化说明：MarketData不含MA60数据，price_to_ma_ratio需由外部注入
            # （从前一日技术分析结果提取MA60计算收盘价/MA60比率）。
            # 当比率<0.95时视为有效跌破MA60且3日未收回，触发减仓50%。
            if md and hasattr(md, 'price_to_ma_ratio') and md.price_to_ma_ratio and md.price_to_ma_ratio > 0:
                if md.price_to_ma_ratio < 0.95:
                    sell_ratio = max(sell_ratio, 0.5)
                    reasons.append(f"趋势破位 MA60代理={md.price_to_ma_ratio:.3f}<0.95")

            if sell_ratio > 0:
                sell_orders.append(SellOrder(code, sell_ratio, "; ".join(reasons)))

        return sell_orders
