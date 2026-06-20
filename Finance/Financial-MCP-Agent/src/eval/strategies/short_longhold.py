"""成熟短线策略 — 短线长持线专用（S-L8）

实现总纲 §5.3 ShortLongHold 策略的完整交易规则：
  - 选股：流动性过滤 + 不接飞刀 + 不追高 + 评分-动量复合排序
  - 卖出：硬止损/软止损/时间止损/换手率异常/利润保护
  - 仓位：评分调整仓位（base=1/N, ±50%调整, 上限2.0x）
"""
from typing import Dict, Any, List
from src.eval.strategies.base import BaseStrategy, BuyOrder, SellOrder


class ShortLongHoldStrategy(BaseStrategy):
    """
    成熟短线策略：连续持仓运作，多信号保护。
    模拟一个成熟的短线交易者的完整决策框架。

    总纲 §5.3 规则：
      选股维度 — 流动性(换手率≥0.5%) + 不接飞刀(非跌停) + 不追高(日涨≤8%)
                + 排名公式(score×0.7 + momentum×0.3)
      卖出维度 — 硬止损(score<35全卖) + 软止损(35~45卖60%) + 时间止损(≥5天偏低)
                + 换手率异常(>25%卖70%) + 利润保护(高点回撤>8%卖50%)
      仓位维度 — base=1/N, 评分调整(score-50)/50, 上限base×2.0
    """

    def _load_config(self):
        self.max_positions = self.config.get("max_positions_short_longhold", 12)
        self.single_weight_limit = self.config.get("single_weight_limit_short_longhold", 0.12)
        self.min_cash_ratio = self.config.get("min_cash_ratio_short", 0.10)
        self.buy_threshold = self.config.get("score_buy_threshold_short", 60)
        self.sell_hard = self.config.get("score_sell_hard_short", 35)
        self.sell_soft_low = self.config.get("score_sell_soft_low_short", 35)
        self.sell_soft_high = self.config.get("score_sell_soft_high_short", 45)
        # 利润保护：追踪每只持仓股买入以来的最高收盘价（回撤>8%减半仓）
        self._high_water_marks: Dict[str, float] = {}

    def get_max_positions(self) -> int:
        return self.max_positions

    def get_single_weight_limit(self) -> float:
        return self.single_weight_limit

    def get_min_cash_ratio(self) -> float:
        return self.min_cash_ratio

    def get_score_buy_threshold(self) -> float:
        return self.buy_threshold

    def select_stocks(self, pool, scores, holdings, cash, market_data_map=None):
        """多信号选股：评分+流动性+动量+飞刀过滤

        总纲 §5.3 选股规则:
          1. 流动性过滤 — 近5日日均换手率 >= 0.5%
             简化说明：MarketData 不含5日历史，以单日换手率作为最近似代理。
          2. 不接飞刀 — 非连续2日跌停后首日
             简化说明：MarketData 不含历史跌停数据，当日跌停即跳过。
          3. 不追高 — 日涨幅 > 8% 跳过
          4. 排序公式 — score×0.7 + momentum×0.3
             原规约为 score×0.6 + 5日动量×0.3 + 换手率分位数×0.1，
             简化处理：5日动量不可得，以当日涨跌幅映射[0,100]替代。
        """
        candidates = []
        for code in pool:
            if code in holdings and holdings[code] > 0:
                continue  # 已持有，不重复买入
            score = scores.get(code, 0)
            if score < self.buy_threshold:
                continue

            md = market_data_map.get(code) if market_data_map else None

            # ── 规则1: 流动性过滤（近5日日均换手率 >= 0.5%）──
            # 简化：MarketData不含5日历史，单日换手率作为最近似代理
            if md and hasattr(md, 'turnover_rate') and md.turnover_rate < 0.005:
                continue  # 换手率太低，流动性差

            # ── 规则2: 不接飞刀（非连续2日跌停后首日）──
            # 简化：MarketData不含历史跌停数据，当日跌停即视为飞刀风险，跳过
            if md and hasattr(md, 'is_limit_down') and md.is_limit_down:
                continue

            # ── 规则3: 不追高（日涨幅 > 8%）──
            if md and hasattr(md, 'close') and hasattr(md, 'pre_close') and md.pre_close > 0:
                daily_change = (md.close - md.pre_close) / md.pre_close
                if daily_change > 0.08:
                    continue

            # ── 规则4: 排名公式 ──
            # 原规约: score×0.6 + 5日动量×0.3 + 换手率分位数×0.1
            # 简化: score×0.7 + momentum_adjustment×0.3
            # momentum_adjustment 将当日涨跌幅[-0.10, +0.10]映射到[0, 100]
            momentum_adjustment = 50  # neutral default (0% change → 50)
            if md and hasattr(md, 'close') and hasattr(md, 'pre_close') and md.pre_close > 0:
                daily_change = (md.close - md.pre_close) / md.pre_close
                # 映射公式：(-0.10→0, 0.00→50, +0.10→100)，clamp到[0, 100]
                momentum_adjustment = max(0, min(100, (daily_change + 0.10) * 500))

            rank_score = score * 0.7 + momentum_adjustment * 0.3
            candidates.append((code, score, rank_score))

        # 按rank_score降序
        candidates.sort(key=lambda x: x[2], reverse=True)

        orders = []
        for code, score, _ in candidates[:self.max_positions]:
            order = BuyOrder(code, 0, score)
            order.reason = f"成熟短线: score={score:.0f}"
            orders.append(order)

        return orders

    def generate_sell_orders(self, holdings, scores, market_data_map=None,
                             purchase_prices=None, hold_days=None):
        """多层卖出判定：硬止损/软止损/时间止损/换手率异常/利润保护

        总纲 §5.3 卖出规则（优先级从高到低）:
          1. 硬止损 — score < 35 → 全卖
          2. 软止损 — score 35~45 → 卖60%
          3. 时间止损 — 连续≥5天score < buy_threshold-10 → 全卖
          4. 换手率异常 — 单日换手率 > 25% → 卖70%
          5. 利润保护 — 从买入以来最高点回撤 > 8% → 卖50%
        """
        sell_orders = []
        purchase_prices = purchase_prices or {}
        hold_days = hold_days or {}

        for code, shares in holdings.items():
            if shares <= 0:
                continue

            if code not in scores:
                continue  # 缺少评分数据时跳过，不做卖出决策
            score = scores[code]
            reasons = []
            sell_ratio = 0.0

            # ── 硬止损：score < 35 → 全卖 ──
            if score < self.sell_hard:
                sell_ratio = 1.0
                reasons.append(f"硬止损 score={score:.0f}<{self.sell_hard}")

            # ── 软止损：score 35-45 → 卖出60% ──
            elif self.sell_soft_low <= score < self.sell_soft_high:
                sell_ratio = max(sell_ratio, 0.6)
                reasons.append(f"软止损 score={score:.0f}")

            # ── 时间止损：连续5天score低于买入阈值-10 → 全卖 ──
            if code in hold_days and hold_days[code] >= 5:
                if score < self.buy_threshold - 10:
                    sell_ratio = max(sell_ratio, 1.0)
                    reasons.append(f"时间止损 持有{hold_days[code]}天 score持续偏低")

            # ── 换手率异常：单日换手率 > 25% → 卖70% ──
            md = market_data_map.get(code) if market_data_map else None
            if md and hasattr(md, 'turnover_rate') and md.turnover_rate > 0.25:
                sell_ratio = max(sell_ratio, 0.7)
                reasons.append(f"换手率异常 {md.turnover_rate:.1%}")

            # ── 利润保护：从持仓以来最高点回撤 > 8% → 卖出50% ──
            # 规约 §5.3: 追踪买入以来最高收盘价，回撤超8%时减半仓保护利润
            if md and hasattr(md, 'close') and md.close > 0:
                # 更新高水位
                prev_high = self._high_water_marks.get(code)
                if prev_high is None:
                    # 首次：优先用买入均价初始化，其次用当日收盘价
                    prev_high = purchase_prices.get(code, md.close)
                current_high = max(prev_high, md.close)
                self._high_water_marks[code] = current_high

                # 检查回撤
                if current_high > 0:
                    drawdown = (current_high - md.close) / current_high
                    if drawdown > 0.08:
                        sell_ratio = max(sell_ratio, 0.5)
                        reasons.append(f"利润保护 高点回撤{drawdown:.1%}>8%")

            if sell_ratio > 0:
                sell_orders.append(SellOrder(code, sell_ratio, "; ".join(reasons)))

        # ── 清理已清仓股票的高水位记录 ──
        for code in list(self._high_water_marks.keys()):
            if code not in holdings:
                del self._high_water_marks[code]

        return sell_orders

    def size_positions(self, buy_orders, total_capital, current_positions,
                       max_positions, single_weight_limit, min_cash_ratio):
        """仓位调整 — 总纲 §5.3 评分调整仓位规则

        base = 1 / max_positions
        adjustment = (score - 50) / 50         # 范围 [-1.0, +1.0]
        final_weight = base × (1 + adjustment)  # 范围 [0, base×2.0]
        capped at base × 2.0 and single_weight_limit
        """
        available_cash = total_capital * (1 - min_cash_ratio)
        slots = max(0, max_positions - current_positions)

        if slots <= 0 or not buy_orders:
            return []

        sized = []
        for i, order in enumerate(buy_orders[:slots]):
            # 规则: base = 1 / max_positions
            base_weight = 1.0 / max_positions
            # 规则: adjustment = (score - 50) / 50
            adjustment = (order.score - 50) / 50
            # 规则: final_weight = base × (1 + adjustment), 上限 base × 2.0
            final_weight = base_weight * (1 + adjustment)
            final_weight = max(0.0, min(base_weight * 2.0, final_weight))
            # 额外约束：单只股票权重上限
            final_weight = min(final_weight, single_weight_limit)

            target = total_capital * final_weight
            # 确保剩余现金足够分配
            target = min(target, available_cash / max(slots - i, 1))
            order.target_value = target
            sized.append(order)

        return sized
