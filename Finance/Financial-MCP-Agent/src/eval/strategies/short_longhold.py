"""成熟短线策略 — 短线长持线专用（S-L8）

实现总纲 §5.3 ShortLongHold 策略的完整交易规则：
  - 选股：流动性过滤 + 不接飞刀 + 不追高 + 评分-动量-换手率复合排序
  - 买入：分2天建仓 + 加仓 + 补仓
  - 卖出：硬止损/软止损/时间止损/换手率异常/利润保护
  - 仓位：评分调整仓位（base=1/N, ±50%调整, 上限2.0x）
  - 风控：同行业上限35% + 收盘前30分钟异常波动标记 + 短中矛盾减半仓
"""
from typing import Dict, Any, List, Optional
from src.eval.strategies.base import BaseStrategy, BuyOrder, SellOrder


class ShortLongHoldStrategy(BaseStrategy):
    """成熟短线策略：连续持仓运作，多信号保护。

    总纲 §5.3 完整规则：
      选股维度 — 流动性(近5日日均换手率≥0.5%) + 不接飞刀(非连续2日跌停后首日)
                + 不追高(日涨≤8%)
                + 排名公式(score×0.6 + 5d_momentum×0.3 + turnover_percentile×0.1)
      买入维度 — 新开仓分2天建仓(T日50%, T+1日50%)
                + 加仓(score>75且近3日涨幅<8%) + 补仓(score>55且近3日跌幅>5%)
      卖出维度 — 硬止损(score<35全卖) + 软止损(35~45卖60%) + 时间止损(≥5天偏低)
                + 换手率异常(>25%卖70%) + 利润保护(高点回撤>8%卖50%)
      仓位维度 — base=1/N, 评分调整(score-50)/50, 上限base×2.0
      风控维度 — 同行业≤35%总资金 + 收盘前30分钟异常波动>3%标记 + 短中矛盾减半仓
    """

    def _load_config(self):
        self.max_positions = self.config.get("max_positions_short_longhold", 12)
        self.single_weight_limit = self.config.get("single_weight_limit_short_longhold", 0.12)
        self.min_cash_ratio = self.config.get("min_cash_ratio_short", 0.10)
        self.buy_threshold = self.config.get("score_buy_threshold_short", 60)
        self.sell_hard = self.config.get("score_sell_hard_short", 35)
        self.sell_soft_low = self.config.get("score_sell_soft_low_short", 35)
        self.sell_soft_high = self.config.get("score_sell_soft_high_short", 45)
        # ── 长效状态追踪 ──
        self._high_water_marks: Dict[str, float] = {}   # 利润保护：{code: highest_close}
        self._pending_builds: Dict[str, dict] = {}      # 分2天建仓：{code: {target_value, day, first_filled}}
        self._add_cooldown: Dict[str, int] = {}         # 加仓冷却期：{code: days_since_last_add}
        # SPEC: 近3日涨幅追踪 — MarketData单日快照不含历史序列，
        # 由 orchestrator 侧注入 _price_history 后计算近似值；无历史时用当日涨跌幅近似。
        self._price_history: Dict[str, List[float]] = {}  # {code: [close_t-2, close_t-1, close_t]}

    # ── 公共接口 ──────────────────────────────────────────────────────

    def get_max_positions(self) -> int:
        return self.max_positions

    def get_single_weight_limit(self) -> float:
        return self.single_weight_limit

    def get_min_cash_ratio(self) -> float:
        return self.min_cash_ratio

    def get_score_buy_threshold(self) -> float:
        return self.buy_threshold

    def update_price_history(self, code: str, close: float):
        """由 orchestrator 在每日数据加载后调用，维护近3日收盘价序列。"""
        if code not in self._price_history:
            self._price_history[code] = []
        self._price_history[code].append(close)
        # 仅保留最近3个
        if len(self._price_history[code]) > 3:
            self._price_history[code] = self._price_history[code][-3:]

    def _compute_3d_change(self, code: str, md) -> float:
        """计算近3日涨跌幅。

        SPEC: 需要近3日收盘价序列。MarketData不含历史序列，
        从 self._price_history 取；若不足3日，使用 (close-pre_close)/pre_close 作为近似。
        """
        history = self._price_history.get(code, [])
        if len(history) >= 3:
            return (history[-1] - history[0]) / max(history[0], 1e-8)
        # SPEC: requires 3-day series, using 1-day approximation
        if md and hasattr(md, 'close') and hasattr(md, 'pre_close') and md.pre_close > 0:
            return (md.close - md.pre_close) / md.pre_close
        return 0.0

    # ── 选股 ──────────────────────────────────────────────────────────

    def select_stocks(self, pool, scores, holdings, cash, market_data_map=None,
                      total_capital: float = 0.0, medium_scores: Dict[str, float] = None):
        """多信号选股：评分+流动性+动量+飞刀过滤 + 加仓/补仓

        总纲 §5.3 选股规则：
          1. 流动性过滤 — 近5日日均换手率 >= 0.5%
             简化说明：MarketData 不含5日历史，以单日换手率作为最近似代理。
          2. 不接飞刀 — 非连续2日跌停后首日
             简化说明：MarketData 不含历史跌停数据，当日跌停即跳过。
          3. 不追高 — 日涨幅 > 8% 跳过
          4. 排序公式 — score×0.6 + 5d_momentum×0.3 + turnover_percentile×0.1
             SPEC: 5d_momentum 以 (close-pre_close)/pre_close 作为近似（单日无5日序列）
             SPEC: turnover_percentile 在候选池内计算排名分位数
        """
        market_data_map = market_data_map or {}

        # ── 预处理：收集换手率用于分位数计算 ──
        # SPEC: turnover_percentile = rank(turnover_rate among pool) / pool_size * 100
        pool_turnovers = []
        for code in pool:
            md = market_data_map.get(code)
            if md and hasattr(md, 'turnover_rate'):
                pool_turnovers.append((code, md.turnover_rate))
        pool_turnovers.sort(key=lambda x: x[1])
        turnover_ranks = {}
        n = len(pool_turnovers)
        for rank, (code, _) in enumerate(pool_turnovers):
            turnover_ranks[code] = (rank / max(n - 1, 1)) * 100.0  # percentile [0, 100]

        # ── 候选收集 ──
        candidates = []       # 新开仓候选
        add_candidates = []   # 加仓/补仓候选（已持有）

        for code in pool:
            score = scores.get(code, 0)
            md = market_data_map.get(code)

            already_held = code in holdings and holdings[code] > 0

            if not already_held:
                # ── 新开仓筛选 ──
                if score < self.buy_threshold:
                    continue

                # 规则1: 流动性过滤（近5日日均换手率 >= 0.5%）
                # SPEC: requires 5-day average turnover; using single-day as approximation
                if md and hasattr(md, 'turnover_rate') and md.turnover_rate < 0.005:
                    continue

                # 规则2: 不接飞刀（非连续2日跌停后首日）
                # SPEC: requires 2-day limit-down history; using single-day as approximation
                if md and hasattr(md, 'is_limit_down') and md.is_limit_down:
                    continue

                # 规则3: 不追高（日涨幅 > 8%）
                if md and hasattr(md, 'close') and hasattr(md, 'pre_close') and md.pre_close > 0:
                    daily_change = (md.close - md.pre_close) / md.pre_close
                    if daily_change > 0.08:
                        continue

                # 规则4: 排名公式 score×0.6 + 5d_momentum×0.3 + turnover_percentile×0.1
                momentum_5d = 50.0  # neutral default
                if md and hasattr(md, 'close') and hasattr(md, 'pre_close') and md.pre_close > 0:
                    daily_change = (md.close - md.pre_close) / md.pre_close
                    # SPEC: 5d_momentum — requires 5-day price series; using 1-day as approximation
                    # Map daily_change to [0, 100]: (-0.10→0, 0.00→50, +0.10→100)
                    momentum_5d = max(0.0, min(100.0, (daily_change + 0.10) * 500.0))

                turnover_pct = turnover_ranks.get(code, 50.0)
                rank_score = score * 0.6 + momentum_5d * 0.3 + turnover_pct * 0.1
                candidates.append((code, score, rank_score))

            else:
                # ── 加仓/补仓判定（已持有股票） ──
                # SPEC: 加仓 — score>75 and 近3日涨幅<8% → add 2% total_capital
                # SPEC: 补仓 — score>55 and 近3日跌幅>5% → add 1% total_capital
                if total_capital <= 0:
                    continue
                change_3d = self._compute_3d_change(code, md)

                add_reason = None
                if score > 75 and change_3d < 0.08:
                    add_reason = f"加仓 score={score:.0f}>75 近3日涨幅={change_3d:.1%}<8%"
                    add_candidates.append((code, score, total_capital * 0.02, add_reason))
                elif score > 55 and change_3d < -0.05:
                    add_reason = f"补仓 score={score:.0f}>55 近3日跌幅={change_3d:.1%}>5%"
                    add_candidates.append((code, score, total_capital * 0.01, add_reason))

        # ── 按 rank_score 降序 ──
        candidates.sort(key=lambda x: x[2], reverse=True)
        # 加仓/补仓按 score 降序
        add_candidates.sort(key=lambda x: x[1], reverse=True)

        orders = []

        # ── 新开仓：分2天建仓（T日50%，T+1日50%）──
        for code, score, _ in candidates[:self.max_positions]:
            # 检查是否在 pending_builds 中（第2天补仓）
            pending = self._pending_builds.get(code)
            if pending:
                # Day 2: buy remaining 50%
                order = BuyOrder(code, pending["target_value"] * 0.5, score)
                order.reason = f"成熟短线 分2天建仓Day2: score={score:.0f}"
                orders.append(order)
                del self._pending_builds[code]
            else:
                # Day 1: buy 50%
                order = BuyOrder(code, 0, score)  # target_value will be set in size_positions
                order.reason = f"成熟短线 分2天建仓Day1: score={score:.0f}"
                # SPEC: 50% now, 50% tomorrow — track in _pending_builds immediately
                self._pending_builds[code] = {
                    "target_value": 0,  # will be filled by size_positions
                    "day": 1,
                }
                orders.append(order)

        # ── 加仓/补仓 ──
        for code, score, amount, reason in add_candidates:
            # 冷却检查：上次加仓后至少间隔3天
            cooldown = self._add_cooldown.get(code, 999)
            if cooldown < 3:
                continue
            order = BuyOrder(code, amount, score)
            order.reason = reason
            orders.append(order)
            self._add_cooldown[code] = 0  # reset cooldown counter

        # ── 冷却期递增 ──
        for code in list(self._add_cooldown.keys()):
            if code in holdings:
                self._add_cooldown[code] += 1

        # ── 清理不在池子中的 pending ──
        for code in list(self._pending_builds.keys()):
            if code not in pool:
                del self._pending_builds[code]

        return orders

    # ── 卖出 ──────────────────────────────────────────────────────────

    def generate_sell_orders(self, holdings, scores, market_data_map=None,
                             purchase_prices=None, hold_days=None,
                             medium_scores: Dict[str, float] = None,
                             total_capital: float = 0.0):
        """多层卖出判定：硬止损/软止损/时间止损/换手率异常/利润保护/短中矛盾

        总纲 §5.3 卖出规则（优先级从高到低）：
          1. 硬止损 — score < 35 → 全卖
          2. 软止损 — score 35~45 → 卖60%
          3. 时间止损 — 连续≥5天score < buy_threshold → 全卖
             SPEC 原文: 连续5天score低于买入时score
             简化: hold_days>=5 且 score<buy_threshold 时触发
          4. 换手率异常 — 单日换手率 > 25% → 卖70%
          5. 利润保护 — 从买入以来最高点回撤 > 8% → 卖50%
          6. 短中矛盾 — 短线score>70 且 中线score<40 → 仓位减半
             SPEC: short>70 AND medium<40 → halve position (sell_ratio=0.5)
        """
        sell_orders = []
        purchase_prices = purchase_prices or {}
        hold_days = hold_days or {}
        medium_scores = medium_scores or {}

        for code, shares in holdings.items():
            if shares <= 0:
                continue

            if code not in scores:
                continue
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

            # ── 时间止损：连续5天score低于买入阈值 → 全卖 ──
            if code in hold_days and hold_days[code] >= 5:
                if score < self.buy_threshold:
                    sell_ratio = max(sell_ratio, 1.0)
                    reasons.append(f"时间止损 持有{hold_days[code]}天 score持续偏低")

            # ── 换手率异常：单日换手率 > 25% → 卖70% ──
            md = market_data_map.get(code) if market_data_map else None
            if md and hasattr(md, 'turnover_rate') and md.turnover_rate > 0.25:
                sell_ratio = max(sell_ratio, 0.7)
                reasons.append(f"换手率异常 {md.turnover_rate:.1%}")

            # ── 利润保护：从持仓以来最高点回撤 > 8% → 卖出50% ──
            if md and hasattr(md, 'close') and md.close > 0:
                prev_high = self._high_water_marks.get(code)
                if prev_high is None:
                    prev_high = purchase_prices.get(code, md.close)
                current_high = max(prev_high, md.close)
                self._high_water_marks[code] = current_high

                if current_high > 0:
                    drawdown = (current_high - md.close) / current_high
                    if drawdown > 0.08:
                        sell_ratio = max(sell_ratio, 0.5)
                        reasons.append(f"利润保护 高点回撤{drawdown:.1%}>8%")

            # ── 短中矛盾：短线score>70 且 中线score<40 → 仓位减半 ──
            if score > 70 and code in medium_scores and medium_scores[code] < 40:
                sell_ratio = max(sell_ratio, 0.5)
                reasons.append(
                    f"短中矛盾 short={score:.0f}>70 medium={medium_scores[code]:.0f}<40 仓位减半"
                )

            # ── 收盘前30分钟异常波动标记（不自动卖出）──
            # SPEC: 收盘前30分钟异常波动>3%触发标记，标记但不自动卖出
            if md and hasattr(md, 'close_30min_spike') and md.close_30min_spike > 0.03:
                reasons.append(f"⚠收盘前30min异常波动 {md.close_30min_spike:.1%}（标记，不卖出）")

            if sell_ratio > 0:
                sell_orders.append(SellOrder(code, sell_ratio, "; ".join(reasons)))

        # ── 清理已清仓股票的高水位记录 ──
        for code in list(self._high_water_marks.keys()):
            if code not in holdings:
                del self._high_water_marks[code]

        # ── 清理冷却/历史 ──
        for code in list(self._add_cooldown.keys()):
            if code not in holdings:
                del self._add_cooldown[code]
        for code in list(self._price_history.keys()):
            if code not in holdings and code not in self._pending_builds:
                del self._price_history[code]

        return sell_orders

    # ── 仓位调整 ──────────────────────────────────────────────────────

    def size_positions(self, buy_orders, total_capital, current_positions,
                       max_positions, single_weight_limit, min_cash_ratio,
                       holdings: Dict[str, int] = None,
                       market_data_map: Dict[str, Any] = None):
        """仓位调整 — 总纲 §5.3 评分调整仓位规则 + 行业上限

        base = 1 / max_positions
        adjustment = (score - 50) / 50         # 范围 [-1.0, +1.0]
        final_weight = base * (1 + adjustment)  # 范围 [0, base*2.0]
        capped at base*2.0, single_weight_limit, and sector_cap (35%)
        """
        available_cash = total_capital * (1 - min_cash_ratio)
        slots = max(0, max_positions - current_positions)

        if slots <= 0 or not buy_orders:
            return []

        holdings = holdings or {}
        market_data_map = market_data_map or {}

        # ── SPEC: 同行业不超过总资金35% ──
        # 计算当前各行业已占用资金
        industry_used: Dict[str, float] = {}
        for code, shares in holdings.items():
            md = market_data_map.get(code)
            if md and hasattr(md, 'industry') and md.industry:
                ind = md.industry
                md2 = market_data_map.get(code)
                price = (md2.close if md2 and md2.close > 0 else
                         (md2.open if md2 and md2.open > 0 else 0))
                industry_used[ind] = industry_used.get(ind, 0) + shares * price

        sized = []
        for i, order in enumerate(buy_orders[:slots]):
            base_weight = 1.0 / max_positions
            adjustment = (order.score - 50) / 50.0
            final_weight = base_weight * (1.0 + adjustment)
            final_weight = max(0.0, min(base_weight * 2.0, final_weight))
            final_weight = min(final_weight, single_weight_limit)

            target = total_capital * final_weight

            # ── 行业上限检查 ──
            md = market_data_map.get(order.stock_code)
            if md and hasattr(md, 'industry') and md.industry:
                ind = md.industry
                current_ind_used = industry_used.get(ind, 0)
                ind_cap = total_capital * 0.35
                max_ind_available = max(0, ind_cap - current_ind_used)
                if target > max_ind_available:
                    # SPEC: 同一行业不超过总资金35%
                    target = max_ind_available
                    if not hasattr(order, 'reason') or not order.reason:
                        order.reason = ""
                    order.reason += f" 行业上限调整(>{ind_cap/total_capital*100:.0f}%)"

            # SPEC: 分2天建仓 — Day1 仅建50%
            # Day 1 if pending entry exists and target_value not yet set (==0)
            is_add_order = order.reason and ("加仓" in order.reason or "补仓" in order.reason)
            pending = self._pending_builds.get(order.stock_code)
            if not is_add_order and pending is not None and pending.get("target_value", 0) == 0:
                # Day 1: target = 50% of full position, update the pending record
                target_day1 = target * 0.5
                pending["target_value"] = target  # full target for Day 2 reference
                pending["first_filled"] = target_day1
                target = target_day1

            # 确保剩余现金足够
            target = min(target, available_cash / max(slots - i, 1))
            target = max(0, target)

            if target > 0:
                order.target_value = target
                sized.append(order)

            # 更新行业跟踪
            if md and hasattr(md, 'industry') and md.industry:
                industry_used[md.industry] = industry_used.get(md.industry, 0) + target

        return sized
