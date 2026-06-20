"""中线价值+趋势策略 — M-L0

实现总纲 §5.4 MediumTerm 策略的完整交易规则：
  - 选股：PE/ROE/营收增速/MA60/风险 五维买入条件 + 四因子复合排序
  - 买入：分4周建仓(25% each) + 加仓 + 越跌越买
  - 卖出：基本面恶化 + PE估值卖出 + 趋势破位3日确认 + 最大回撤
  - 跨期限协同：与长线score联动调整仓位上限
"""
from typing import Dict, Any, List, Optional
from src.eval.strategies.base import BaseStrategy, BuyOrder, SellOrder


class MediumTermStrategy(BaseStrategy):
    """中线策略：基本面驱动，技术面确认入场时机。每周调仓。

    总纲 §5.4 完整规则：
      选股 — PE>0且<行业75分位, ROE>5%或营收增速>10%, 股价>MA60,
             无delist/regulatory风险, 四因子排序公式
      买入 — 分4周建仓(25% each) + 加仓(score升≥5且PE分位降≥10%)
            + 越跌越买(score>70且跌>10%)
      卖出 — 基本面恶化(score<45) + PE估值卖出(>行业90分位)
            + 趋势破位3日未收回 + 最大回撤>15%
      协同 — 中线>70且长线>70 → 仓位上限22%
            中线>60且长线<35 → 仓位上限10%
    """

    def _load_config(self):
        self.max_positions = self.config.get("max_positions_medium", 8)
        self.single_weight_limit = self.config.get("single_weight_limit_medium", 0.18)
        self.min_cash_ratio = self.config.get("min_cash_ratio_medium", 0.15)
        self.buy_threshold = self.config.get("score_buy_threshold_medium", 65)
        self.sell_hard = self.config.get("score_sell_hard_medium", 45)
        # ── 长效状态追踪 ──
        self._pending_builds: Dict[str, dict] = {}       # {code: {target_value, tranche, remaining_tranches}}
        self._high_water_marks: Dict[str, float] = {}    # 最高收盘价追踪
        self._below_ma60_days: Dict[str, int] = {}       # 连续低于MA60天数
        self._prev_scores: Dict[str, float] = {}         # 上次评分（用于加仓判定）
        self._prev_pe_percentile: Dict[str, float] = {}  # 上次PE分位（用于加仓判定）
        self._build_prices: Dict[str, float] = {}        # 建仓均价（用于越跌越买判定）

    # ── 公共接口 ──────────────────────────────────────────────────────

    def get_max_positions(self) -> int:
        return self.max_positions

    def get_single_weight_limit(self) -> float:
        return self.single_weight_limit

    def get_min_cash_ratio(self) -> float:
        return self.min_cash_ratio

    def get_score_buy_threshold(self) -> float:
        return self.buy_threshold

    # ── 辅助计算 ──────────────────────────────────────────────────────

    @staticmethod
    def _get_pe_percentile(md) -> float:
        """获取PE在行业的分位数（简化：从MarketData读，无数据则-1）"""
        # SPEC: requires industry PE percentile data; when unavailable returns -1
        if md and hasattr(md, 'pe_ratio') and md.pe_ratio and md.pe_ratio > 0:
            if hasattr(md, 'industry_pe_75th') and md.industry_pe_75th > 0:
                # 近似：PE / 行业75分位 的比值映射到0-100
                ratio = md.pe_ratio / md.industry_pe_75th
                pe_pct = max(0.0, min(100.0, ratio * 50.0))
                return pe_pct
            # 无行业数据时用通用PE倒数映射
            # PE<10→低分位(~20), PE=20→中分位(~50), PE>50→高分位(~80)
            return max(5.0, min(95.0, md.pe_ratio * 1.8))
        return -1.0

    @staticmethod
    def _compute_20d_momentum(md) -> float:
        """计算近20日动量（0-100）。

        SPEC: requires 20-day price series; using single-day as approximation.
        """
        if md and hasattr(md, 'close') and hasattr(md, 'pre_close') and md.pre_close > 0:
            daily = (md.close - md.pre_close) / md.pre_close
            # Map [-0.20, +0.20] to [0, 100]
            return max(0.0, min(100.0, (daily + 0.20) * 250.0))
        return 50.0

    @staticmethod
    def _check_buy_conditions(code: str, score: float, md,
                              buy_threshold: float) -> tuple:
        """检查中线买入条件（全部满足）。

        SPEC §5.4 买入条件:
          1. score >= 65
          2. PE(TTM) > 0 且 PE < 行业75分位
          3. ROE > 5% 或 营收增速 > 10%
          4. 股价 > MA60
          5. 无 delist_risk / regulatory_risk

        Returns: (passed: bool, failed_reasons: list)
        """
        failed = []

        if score < buy_threshold:
            failed.append(f"score={score:.0f}<{buy_threshold}")
            return False, failed

        if md is None:
            failed.append("无市场数据")
            return False, failed

        # 条件2: PE > 0 且 PE < 行业75分位
        spec2_ok = False
        if hasattr(md, 'pe_ratio') and md.pe_ratio and md.pe_ratio > 0:
            if hasattr(md, 'industry_pe_75th') and md.industry_pe_75th > 0:
                if md.pe_ratio < md.industry_pe_75th:
                    spec2_ok = True
                else:
                    failed.append(f"PE={md.pe_ratio:.1f}>=行业75分位{md.industry_pe_75th:.1f}")
            else:
                # SPEC: industry_pe_75th unavailable — skip this sub-check
                # 仅检查 PE>0
                spec2_ok = True
        else:
            # SPEC: PE data unavailable — skip, do not reject
            spec2_ok = True

        if not spec2_ok:
            return False, failed

        # 条件3: ROE > 5% 或 营收增速 > 10%
        spec3_ok = False
        roe_ok = (hasattr(md, 'roe') and md.roe > 0.05)
        rg_ok = (hasattr(md, 'revenue_growth') and md.revenue_growth > 0.10)
        if roe_ok or rg_ok:
            spec3_ok = True
        elif not hasattr(md, 'roe') and not hasattr(md, 'revenue_growth'):
            # SPEC: both ROE and revenue_growth unavailable — skip, do not reject
            spec3_ok = True
        else:
            failed.append(f"ROE={getattr(md,'roe',0):.1%}≤5% 且 营收增速={getattr(md,'revenue_growth',0):.1%}≤10%")

        if not spec3_ok:
            return False, failed

        # 条件4: 股价 > MA60
        if hasattr(md, 'price_to_ma_ratio') and md.price_to_ma_ratio > 0:
            if md.price_to_ma_ratio <= 1.0:
                failed.append(f"股价/MA60={md.price_to_ma_ratio:.3f}≤1.0")
                return False, failed
        # SPEC: price_to_ma_ratio unavailable — skip, do not reject

        # 条件5: 无 delist_risk / regulatory_risk
        if md and hasattr(md, 'risk_flags') and md.risk_flags:
            flags = list(md.risk_flags) if isinstance(md.risk_flags, (list, tuple)) else [str(md.risk_flags)]
            risky = [f for f in flags if any(kw in str(f).lower() for kw in ('delist', 'regulatory'))]
            if risky:
                failed.append(f"风险标记: {', '.join(risky)}")
                return False, failed

        return True, []

    # ── 选股 ──────────────────────────────────────────────────────────

    def select_stocks(self, pool, scores, holdings, cash, market_data_map=None,
                      total_capital: float = 0.0, long_scores: Dict[str, float] = None):
        """中线选股：五维买入条件 + 四因子复合排序

        总纲 §5.4 排序公式：
          rank_score = score×0.5 + ROE_percentile×0.2 + (1/PE_percentile)×0.2 + 20d_momentum×0.1

        其中 ROE_percentile 和 PE_percentile 相对池子计算。
        """
        candidates = []
        market_data_map = market_data_map or {}

        # ── 预处理：计算池内 ROE/PE 分位数 ──
        pool_roe = []
        pool_pe_inv = []  # 1/PE
        for code in pool:
            md = market_data_map.get(code)
            if md:
                if hasattr(md, 'roe') and md.roe:
                    pool_roe.append((code, md.roe))
                if hasattr(md, 'pe_ratio') and md.pe_ratio and md.pe_ratio > 0:
                    pool_pe_inv.append((code, 1.0 / md.pe_ratio))

        pool_roe.sort(key=lambda x: x[1])
        pool_pe_inv.sort(key=lambda x: x[1])
        roe_ranks = {code: (rank / max(len(pool_roe) - 1, 1)) * 100.0
                     for rank, (code, _) in enumerate(pool_roe)}
        pe_inv_ranks = {code: (rank / max(len(pool_pe_inv) - 1, 1)) * 100.0
                        for rank, (code, _) in enumerate(pool_pe_inv)}

        # ── 筛选 ──
        for code in pool:
            if code in holdings and holdings[code] > 0:
                continue

            score = scores.get(code, 0)
            md = market_data_map.get(code)

            # 五维买入条件检查
            passed, failed_reasons = self._check_buy_conditions(
                code, score, md, self.buy_threshold
            )
            if not passed:
                continue

            # 四因子复合排序
            roe_pct = roe_ranks.get(code, 50.0)
            pe_inv_pct = pe_inv_ranks.get(code, 50.0)
            momentum_20d = self._compute_20d_momentum(md)

            rank_score = (score * 0.5 +
                         roe_pct * 0.2 +
                         pe_inv_pct * 0.2 +
                         momentum_20d * 0.1)
            candidates.append((code, score, rank_score))

        candidates.sort(key=lambda x: x[2], reverse=True)

        orders = []
        for code, score, _ in candidates[:self.max_positions]:
            # SPEC: 分4周建仓（每周调仓日建25%）
            # 首次建仓: 25%
            pending = self._pending_builds.get(code)
            if pending:
                tranche = pending.get("tranche", 0)
                remaining = pending.get("remaining_tranches", 0)
                if remaining > 0:
                    order = BuyOrder(code, pending["target_value"] * 0.25, score)
                    order.reason = f"中线 分4周建仓 第{tranche+1}/4步: score={score:.0f}"
                    orders.append(order)
                    pending["tranche"] = tranche + 1
                    pending["remaining_tranches"] = remaining - 1
                    if pending["remaining_tranches"] <= 0:
                        del self._pending_builds[code]
            else:
                order = BuyOrder(code, 0, score)
                order.reason = f"中线 分4周建仓 第1/4步: score={score:.0f}"
                orders.append(order)
                # SPEC: 分4周建仓 — 第1步已创建，剩余3步
                self._pending_builds[code] = {
                    "target_value": 0,  # will be set by size_positions
                    "tranche": 1,  # step 1 done, next = step 2
                    "remaining_tranches": 3,
                }
                # 记录建仓价用于越跌越买
                if md and hasattr(md, 'close') and md.close > 0:
                    self._build_prices[code] = md.close

        # ── 加仓判定（已持有股票）──
        for code in pool:
            if code not in holdings or holdings[code] <= 0:
                continue
            score = scores.get(code, 0)
            md = market_data_map.get(code)
            if not md or total_capital <= 0:
                continue

            prev_score = self._prev_scores.get(code, score)
            pe_pct = self._get_pe_percentile(md)
            prev_pe_pct = self._prev_pe_percentile.get(code, pe_pct)

            # SPEC: 加仓 — score上升≥5分 且 PE分位下降≥10% → 加仓到上限
            if (score - prev_score >= 5 and
                    prev_pe_pct > 0 and pe_pct > 0 and
                    prev_pe_pct - pe_pct >= 10.0):
                order = BuyOrder(code, 0, score)
                max_add = total_capital * self.single_weight_limit
                # 当前持仓市值
                price = md.close if md.close > 0 else md.open
                current_value = holdings[code] * price
                add_amount = max(0, max_add - current_value)
                if add_amount > total_capital * 0.01:  # 至少1%
                    order.target_value = add_amount
                    order.reason = (
                        f"中线加仓 score↑{score-prev_score:.0f}≥5 "
                        f"PE分位↓{prev_pe_pct-pe_pct:.0f}%≥10%"
                    )
                    orders.append(order)

            # SPEC: 越跌越买（有限）— score>70 且股价较建仓价跌>10%
            # 前提：基本面没有恶化 (由调用方保证)
            build_price = self._build_prices.get(code)
            if (score > 70 and build_price and build_price > 0 and
                    md.close > 0 and md.close < build_price * 0.90):
                add_amount = total_capital * self.single_weight_limit * 0.30
                if add_amount > total_capital * 0.005:
                    order = BuyOrder(code, add_amount, score)
                    order.reason = (
                        f"中线越跌越买 score={score:.0f}>70 "
                        f"跌幅={(build_price-md.close)/build_price:.1%}>10%"
                    )
                    orders.append(order)

        # ── 更新历史 ──
        for code, score in scores.items():
            self._prev_scores[code] = score
        for code in pool:
            md = market_data_map.get(code)
            if md:
                pe_pct = self._get_pe_percentile(md)
                if pe_pct >= 0:
                    self._prev_pe_percentile[code] = pe_pct

        # ── 清理不在池子中的 pending ──
        for code in list(self._pending_builds.keys()):
            if code not in pool:
                del self._pending_builds[code]

        return orders

    # ── 卖出 ──────────────────────────────────────────────────────────

    def generate_sell_orders(self, holdings, scores, market_data_map=None,
                             purchase_prices=None, hold_days=None,
                             long_scores: Dict[str, float] = None,
                             total_capital: float = 0.0):
        """中线卖出：基本面恶化 + PE估值卖出 + 趋势破位3日确认 + 最大回撤

        总纲 §5.4 卖出规则（优先级从高到低）：
          1. 基本面恶化 — score < 45 → 全卖
          2. PE估值卖出 — PE超过行业90分位 且 score < 55
          3. 趋势卖出 — 股价跌破MA60且3日未收回
          4. 最大回撤 — 从最高点回撤 > 15% → 卖出50%
        """
        sell_orders = []
        purchase_prices = purchase_prices or {}
        long_scores = long_scores or {}

        for code, shares in holdings.items():
            if shares <= 0:
                continue
            score = scores.get(code, 0)
            reasons = []
            sell_ratio = 0.0

            md = market_data_map.get(code) if market_data_map else None

            # ── 规则1：基本面恶化 score < 45 → 全卖 ──
            if score < self.sell_hard:
                sell_ratio = 1.0
                reasons.append(f"中线基本面恶化 score={score:.0f}<{self.sell_hard}")

            # ── 规则2：PE估值卖出 — PE超过行业90分位 且 score<55 ──
            if sell_ratio < 1.0 and md and score < 55:
                if (hasattr(md, 'pe_ratio') and md.pe_ratio > 0 and
                        hasattr(md, 'industry_pe_90th') and md.industry_pe_90th > 0 and
                        md.pe_ratio > md.industry_pe_90th):
                    sell_ratio = max(sell_ratio, 1.0)
                    reasons.append(
                        f"PE估值卖出 PE={md.pe_ratio:.1f}>行业90分位{md.industry_pe_90th:.1f}"
                        f" score={score:.0f}<55"
                    )

            # ── 规则3：趋势破位 — 跌破MA60且3日未收回 ──
            if sell_ratio < 1.0 and md and hasattr(md, 'price_to_ma_ratio') and md.price_to_ma_ratio > 0:
                if md.price_to_ma_ratio < 1.0:
                    self._below_ma60_days[code] = self._below_ma60_days.get(code, 0) + 1
                    if self._below_ma60_days[code] >= 3:
                        # SPEC: below MA60 for 3 consecutive days → sell 50%
                        sell_ratio = max(sell_ratio, 0.5)
                        reasons.append(
                            f"趋势破位 连续{self._below_ma60_days[code]}日低于MA60"
                        )
                else:
                    # Recovered above MA60 → reset counter
                    self._below_ma60_days[code] = 0

            # ── 规则4：最大回撤 > 15% → 卖出50% ──
            if sell_ratio < 1.0 and md and hasattr(md, 'close') and md.close > 0:
                # 更新高水位
                prev_high = self._high_water_marks.get(code)
                if prev_high is None:
                    prev_high = purchase_prices.get(code, md.close)
                current_high = max(prev_high, md.close)
                self._high_water_marks[code] = current_high

                if current_high > 0:
                    drawdown = (current_high - md.close) / current_high
                    if drawdown > 0.15:
                        sell_ratio = max(sell_ratio, 0.5)
                        reasons.append(f"最大回撤 {drawdown:.1%}>15%")

            # ── 清理全卖后的追踪 ──
            if sell_ratio >= 1.0:
                self._below_ma60_days.pop(code, None)

            if sell_ratio > 0:
                sell_orders.append(SellOrder(code, sell_ratio, "; ".join(reasons)))

        # ── 清理不在持仓中的追踪 ──
        for code in list(self._high_water_marks.keys()):
            if code not in holdings:
                del self._high_water_marks[code]
        for code in list(self._below_ma60_days.keys()):
            if code not in holdings:
                del self._below_ma60_days[code]
        for code in list(self._build_prices.keys()):
            if code not in holdings:
                del self._build_prices[code]

        return sell_orders

    # ── 仓位调整 ──────────────────────────────────────────────────────

    def size_positions(self, buy_orders, total_capital, current_positions,
                       max_positions, single_weight_limit, min_cash_ratio,
                       holdings: Dict[str, int] = None,
                       market_data_map: Dict[str, Any] = None,
                       long_scores: Dict[str, float] = None):
        """仓位调整 — 总纲 §5.4 + 跨期限协同

        SPEC 跨期限协同：
          - 中线score>70且长线score>70 → "双重确认"，仓位上限提到22%
          - 中线score>60但长线score<35 → "短中期看好但长期堪忧"，仓位上限降到10%
        """
        available_cash = total_capital * (1 - min_cash_ratio)
        slots = max(0, max_positions - current_positions)

        if slots <= 0 or not buy_orders:
            return []

        long_scores = long_scores or {}

        sized = []
        for i, order in enumerate(buy_orders[:slots]):
            # ── 跨期限协同：动态仓位上限 ──
            effective_weight_limit = single_weight_limit
            code = order.stock_code
            score = order.score
            long_score = long_scores.get(code, 50)

            if score > 70 and long_score > 70:
                # SPEC: 双重确认 → 仓位上限 22%
                effective_weight_limit = max(effective_weight_limit, 0.22)
            elif score > 60 and long_score < 35:
                # SPEC: 长期堪忧 → 仓位上限 10%
                effective_weight_limit = min(effective_weight_limit, 0.10)

            weight = min(1.0 / max(slots, 1), effective_weight_limit)
            target = total_capital * weight
            target = min(target, available_cash / max(slots - i, 1))

            # ── 分4周建仓: 仅建25% ──
            pending = self._pending_builds.get(code)
            if pending and pending.get("remaining_tranches", 0) >= 0:
                # Week 1-4: 25% each
                full_target = target
                target = full_target * 0.25
                # Update pending target_value for future tranches
                pending["target_value"] = full_target

            target = max(0, target)
            if target > 0:
                order.target_value = target
                sized.append(order)

        return sized
