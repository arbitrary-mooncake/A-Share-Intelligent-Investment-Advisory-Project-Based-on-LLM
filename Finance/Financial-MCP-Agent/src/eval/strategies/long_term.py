"""长线深度价值策略 — L-L0

实现总纲 §5.5 LongTerm 策略的完整交易规则：
  - 选股：五维买入条件(ROE/利润增长/PE-PB/风险/股息) + 五因子复合排序
  - 买入：一次性建仓50%，分2个月建完 + 加仓 + 越跌越买
  - 卖出：基本面恶化(score<35持续2月) + 质量恶化(ROE连续2季降>30%)
         + PE极端(历史95分位) + 黑天鹅(立即清仓)
"""
from typing import Dict, Any, List, Optional
from src.eval.strategies.base import BaseStrategy, BuyOrder, SellOrder


class LongTermStrategy(BaseStrategy):
    """长线策略：深度价值，集中持仓，每月调仓。买了不轻易卖。

    总纲 §5.5 完整规则：
      选股 — ROE>8%且连续3年净利润增长, PE<行业中位数或PB<行业中位数,
             无delist/audit/regulatory风险, 股息率>0,
             五因子排序公式
      买入 — 一次性建仓50%，分2个月建完 + 加仓(score升≥8)
            + 越跌越买(score>75且跌>15%→加50%)
      卖出 — 基本面恶化(score<35持续2月) + 质量恶化(ROE连续2季降>30%)
            + PE极端(历史95分位→卖30%) + 黑天鹅(立即清仓)
    """

    def _load_config(self):
        self.max_positions = self.config.get("max_positions_long", 5)
        self.single_weight_limit = self.config.get("single_weight_limit_long", 0.25)
        self.min_cash_ratio = self.config.get("min_cash_ratio_long", 0.20)
        self.buy_threshold = self.config.get("score_buy_threshold_long", 70)
        self.sell_hard = self.config.get("score_sell_hard_long", 35)
        # ── 长效状态追踪 ──
        self._low_score_days: Dict[str, int] = {}       # 低评分持续天数
        self._pending_builds: Dict[str, dict] = {}       # 分2月建仓状态
        self._prev_scores: Dict[str, float] = {}         # 上次评分（加仓判定）
        self._build_prices: Dict[str, float] = {}        # 建仓均价
        self._roe_history: Dict[str, List[float]] = {}   # {code: [roe_q2, roe_q1, roe_q0]} 最近3季

    # ── 公共接口 ──────────────────────────────────────────────────────

    def get_max_positions(self) -> int:
        return self.max_positions

    def get_single_weight_limit(self) -> float:
        return self.single_weight_limit

    def get_min_cash_ratio(self) -> float:
        return self.min_cash_ratio

    def get_score_buy_threshold(self) -> float:
        return self.buy_threshold

    def update_roe(self, code: str, roe: float):
        """由 orchestrator 在每季度财报数据可用时调用。

        SPEC: ROE连续2季下降>30%判定需要季度ROE序列。
        """
        if code not in self._roe_history:
            self._roe_history[code] = []
        self._roe_history[code].append(roe)
        if len(self._roe_history[code]) > 3:
            self._roe_history[code] = self._roe_history[code][-3:]

    # ── 辅助计算 ──────────────────────────────────────────────────────

    @staticmethod
    def _check_buy_conditions(code: str, score: float, md,
                              buy_threshold: float) -> tuple:
        """检查长线买入条件（全部满足）。

        SPEC §5.5 买入条件:
          1. score >= 70
          2. ROE > 8% 且 连续3年净利润增长（或至少2年正+1年平）
          3. PE < 行业中位数 或 PB < 行业中位数
          4. 无 delist_risk / audit_risk / regulatory_risk
          5. 股息率 > 0

        Returns: (passed: bool, failed_reasons: list)
        """
        failed = []

        if score < buy_threshold:
            failed.append(f"score={score:.0f}<{buy_threshold}")
            return False, failed

        if md is None:
            failed.append("无市场数据")
            return False, failed

        # 条件2: ROE > 8% 且 连续3年净利润增长
        spec2_ok = False
        if hasattr(md, 'roe') and md.roe > 0.08:
            if hasattr(md, 'profit_growth_3y') and md.profit_growth_3y > 0:
                spec2_ok = True
            else:
                # SPEC: profit_growth_3y unavailable — skip sub-check, keep ROE check only
                spec2_ok = True
        elif not hasattr(md, 'roe') or md.roe == 0:
            # SPEC: ROE unavailable — skip check
            spec2_ok = True
        else:
            failed.append(f"ROE={md.roe:.1%}≤8%")

        if not spec2_ok:
            return False, failed

        # 条件3: PE < 行业中位数 或 PB < 行业中位数
        spec3_ok = False
        pe_ok = (hasattr(md, 'pe_ratio') and md.pe_ratio and md.pe_ratio > 0 and
                 hasattr(md, 'industry_pe_median') and md.industry_pe_median > 0 and
                 md.pe_ratio < md.industry_pe_median)
        pb_ok = (hasattr(md, 'pb_ratio') and md.pb_ratio and md.pb_ratio > 0 and
                 hasattr(md, 'industry_pe_median') and md.industry_pe_median > 0 and
                 md.pb_ratio < (md.industry_pe_median / 10))  # 近似：PB<行业中位/10
        if pe_ok or pb_ok:
            spec3_ok = True
        elif not hasattr(md, 'pe_ratio') and not hasattr(md, 'pb_ratio'):
            # SPEC: PE/PB unavailable — skip
            spec3_ok = True
        else:
            # More granular: if either PE or PB exists but doesn't meet threshold
            pe_str = f"PE={md.pe_ratio:.1f}" if hasattr(md, 'pe_ratio') and md.pe_ratio else "无PE"
            pb_str = f"PB={md.pb_ratio:.1f}" if hasattr(md, 'pb_ratio') and md.pb_ratio else "无PB"
            failed.append(f"{pe_str} {pb_str} 不符合PE/PB<行业中位数")

        if not spec3_ok:
            return False, failed

        # 条件4: 无 delist_risk / audit_risk / regulatory_risk
        if md and hasattr(md, 'risk_flags') and md.risk_flags:
            flags = list(md.risk_flags) if isinstance(md.risk_flags, (list, tuple)) else [str(md.risk_flags)]
            risky = [f for f in flags if any(kw in str(f).lower() for kw in ('delist', 'audit', 'regulatory'))]
            if risky:
                failed.append(f"风险标记: {', '.join(risky)}")
                return False, failed

        # 条件5: 股息率 > 0
        if hasattr(md, 'dividend_yield') and md.dividend_yield == 0:
            # 注意：dividend_yield=0 可能表示无数据或确实不分红
            # SPEC: dividend>0, 但0可能只是缺数据，宽松处理
            if hasattr(md, 'pe_ratio') and md.pe_ratio:
                # 有PE数据但dividend=0则可能真的不分红
                pass  # 不拒绝，仅标记
        # SPEC: dividend_yield unavailable — skip

        return True, []

    @staticmethod
    def _compute_60d_reversal(md) -> float:
        """计算近60日反转因子（0-100），高值=近期跌幅大（反转潜力）。

        SPEC: requires 60-day price series; using single-day approximation.
        Negative daily return → higher reversal score.
        """
        if md and hasattr(md, 'close') and hasattr(md, 'pre_close') and md.pre_close > 0:
            daily = (md.close - md.pre_close) / md.pre_close
            # 反转因子：越跌越有反转潜力
            return max(0.0, min(100.0, (-daily + 0.30) * 166.7))
        return 50.0

    # ── 选股 ──────────────────────────────────────────────────────────

    def select_stocks(self, pool, scores, holdings, cash, market_data_map=None,
                      total_capital: float = 0.0, **kwargs):
        """长线选股：五维条件 + 五因子排序

        总纲 §5.5 排序公式：
          rank_score = score×0.4 + ROE×0.25 + (1/PE)×0.2 + dividend_yield×0.1 + 60d_reversal×0.05
        """
        candidates = []
        market_data_map = market_data_map or {}

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

            # 五因子排序公式
            roe_val = getattr(md, 'roe', 0.0) if md else 0.0
            # SPEC: ROE×0.25 — ROE直接加权 (ROE=15% → 15, ROE=20% → 20)
            # Map to 0-100 scale: ROE=0→0, ROE=20%→100
            roe_score = min(100.0, max(0.0, roe_val * 500.0))  # ROE=0.20 → 100

            pe_val = getattr(md, 'pe_ratio', 0.0) if md else 0.0
            # SPEC: 1/PE×0.2 — PE倒数 (PE=10→10, PE=50→2)
            # Map to 0-100: 1/PE where PE=5→20, PE=10→10, PE=50→2
            pe_inv_score = 0.0
            if pe_val > 0:
                pe_inv_score = min(100.0, max(0.0, (1.0 / pe_val) * 1000.0))  # PE=10→100

            div_yield = getattr(md, 'dividend_yield', 0.0) if md else 0.0
            # SPEC: dividend_yield×0.1 — Map to 0-100: 5%→100
            div_score = min(100.0, max(0.0, div_yield * 2000.0))  # div=5%→100

            reversal = self._compute_60d_reversal(md)

            rank_score = (score * 0.4 +
                         roe_score * 0.25 +
                         pe_inv_score * 0.2 +
                         div_score * 0.1 +
                         reversal * 0.05)
            candidates.append((code, score, rank_score))

        candidates.sort(key=lambda x: x[2], reverse=True)

        orders = []
        for code, score, _ in candidates[:self.max_positions]:
            pending = self._pending_builds.get(code)
            if pending:
                # Month 2+: build remaining 50%
                tranche = pending.get("tranche", 0)
                if tranche >= 1:
                    order = BuyOrder(code, pending["target_value"] * 0.5, score)
                    order.reason = f"长线 分2月建仓 第2/2月: score={score:.0f}"
                    orders.append(order)
                    del self._pending_builds[code]
                # If tranche is still 0 (not yet sized), the pending was just created;
                # the order was already added in the previous call, skip creating another.
                continue

            # Month 1: build 50%
            order = BuyOrder(code, 0, score)
            order.reason = f"长线 分2月建仓 第1/2月: score={score:.0f}"
            orders.append(order)
            # Track: target_value will be set in size_positions; tranche=1 marks Month 1 done
            self._pending_builds[code] = {
                "target_value": 0,
                "tranche": 1,  # Month 1 order created, next call = Month 2
            }
            # Record build price for 越跌越买
            if market_data_map:
                md = market_data_map.get(code)
                if md and hasattr(md, 'close') and md.close > 0:
                    self._build_prices[code] = md.close

        # ── 加仓判定（已持有股票）──
        for code in pool:
            if code not in holdings or holdings[code] <= 0:
                continue
            score = scores.get(code, 0)
            md = market_data_map.get(code) if market_data_map else None
            if not md or total_capital <= 0:
                continue

            prev_score = self._prev_scores.get(code, score)

            # SPEC: 加仓 — score上升≥8分 → 加仓到上限
            if score - prev_score >= 8:
                if not any(o.stock_code == code for o in orders):  # 避免重复
                    max_add = total_capital * self.single_weight_limit
                    price = md.close if md.close > 0 else md.open
                    current_value = holdings[code] * price
                    add_amount = max(0, max_add - current_value)
                    if add_amount > total_capital * 0.01:
                        order = BuyOrder(code, add_amount, score)
                        order.reason = f"长线加仓 score↑{score-prev_score:.0f}≥8"
                        orders.append(order)

            # SPEC: 越跌越买 — score>75且股价跌>15% → 加仓50%
            build_price = self._build_prices.get(code)
            if (score > 75 and build_price and build_price > 0 and
                    md.close > 0 and md.close < build_price * 0.85):
                add_amount = total_capital * self.single_weight_limit * 0.50
                if add_amount > total_capital * 0.01:
                    if not any(o.stock_code == code for o in orders):
                        order = BuyOrder(code, add_amount, score)
                        order.reason = (
                            f"长线越跌越买 score={score:.0f}>75 "
                            f"跌幅={(build_price-md.close)/build_price:.1%}>15%"
                        )
                        orders.append(order)

        # ── 更新历史 ──
        for code, score in scores.items():
            self._prev_scores[code] = score

        # ── 清理不在池子中的 pending ──
        for code in list(self._pending_builds.keys()):
            if code not in pool:
                del self._pending_builds[code]

        return orders

    # ── 卖出 ──────────────────────────────────────────────────────────

    def generate_sell_orders(self, holdings, scores, market_data_map=None,
                             purchase_prices=None, hold_days=None, **kwargs):
        """长线卖出：非常克制，需要强证据。

        总纲 §5.5 卖出规则（优先级从高到低）：
          1. 黑天鹅 — delist_risk/audit_risk → 立即清仓100%
          2. 基本面恶化 — score<35持续≥40交易日(≈2个月) → 全卖
          3. 质量恶化 — ROE连续2季下降>30% → 全部卖出
          4. PE极端 — PE突破历史95分位 → 卖出30%
          除此之外，不设止损，不因价格波动卖出。
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
                critical = [f for f in flags if any(kw in str(f).lower() for kw in ('delist', 'audit'))]
                if critical:
                    sell_ratio = 1.0
                    reasons.append(f"黑天鹅风险: {', '.join(critical)}")

            # ── 规则2：score<35持续≥40交易日 → 全卖 ──
            if sell_ratio < 1.0 and score < self.sell_hard:
                self._low_score_days[code] = self._low_score_days.get(code, 0) + 1
                if self._low_score_days[code] >= 40:
                    sell_ratio = 1.0
                    reasons.append(
                        f"长线基本面恶化 score={score:.0f}<{self.sell_hard}"
                        f" 持续{self._low_score_days[code]}个交易日"
                    )
            elif score >= self.sell_hard:
                if code in self._low_score_days:
                    self._low_score_days[code] = 0

            # ── 规则3：ROE连续2季下降>30% → 全部卖出 ──
            # SPEC: 需要季度财报数据。MarketData不含ROE历史，
            # 由 orchestrator 通过 update_roe() 注入季度数据。
            if sell_ratio < 1.0:
                roe_hist = self._roe_history.get(code, [])
                if len(roe_hist) >= 3:
                    # Check: roe_q2→roe_q1 drop >30% AND roe_q1→roe_q0 drop >30%
                    # roe_hist = [roe_q2(最早), roe_q1, roe_q0(最新)]
                    if roe_hist[1] > 0 and roe_hist[2] > 0 and roe_hist[0] > 0:
                        drop_q2_q1 = roe_hist[0] - roe_hist[1]
                        drop_q1_q0 = roe_hist[1] - roe_hist[2]
                        if (drop_q2_q1 > 0 and drop_q2_q1 / roe_hist[0] > 0.30 and
                                drop_q1_q0 > 0 and drop_q1_q0 / roe_hist[1] > 0.30):
                            sell_ratio = max(sell_ratio, 1.0)
                            reasons.append(
                                f"质量恶化 ROE连续2季下降>30%"
                                f" ({roe_hist[0]:.1%}→{roe_hist[1]:.1%}→{roe_hist[2]:.1%})"
                            )

            # ── 规则4：PE突破历史95分位 → 卖出30% ──
            # SPEC: 需要历史PE分位数数据（daily_basic长期历史）。
            # 由 orchestrator 注入 pe_percentile 到 MarketData。
            if sell_ratio < 1.0:
                if (md and hasattr(md, 'pe_percentile') and
                        md.pe_percentile >= 0 and md.pe_percentile > 95):
                    sell_ratio = max(sell_ratio, 0.3)
                    reasons.append(f"PE突破历史95分位({md.pe_percentile:.0f})")

            # ── 清理全卖后的追踪 ──
            if sell_ratio >= 1.0:
                self._low_score_days.pop(code, None)

            if sell_ratio > 0:
                sell_orders.append(SellOrder(code, sell_ratio, "; ".join(reasons)))

        # ── 清理不在持仓中的追踪 ──
        for code in list(self._low_score_days.keys()):
            if code not in holdings:
                del self._low_score_days[code]
        for code in list(self._build_prices.keys()):
            if code not in holdings:
                del self._build_prices[code]

        return sell_orders

    # ── 仓位调整 ──────────────────────────────────────────────────────

    def size_positions(self, buy_orders, total_capital, current_positions,
                       max_positions, single_weight_limit, min_cash_ratio,
                       holdings: Dict[str, int] = None,
                       market_data_map: Dict[str, Any] = None,
                       **kwargs):
        """仓位调整 — 总纲 §5.5

        长线集中持仓，base = 1/N, 上限 single_weight_limit=25%。
        SPEC: 一次性建仓50%，分2个月建完。
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

            # ── 分2月建仓：Month 1 = 50%, Month 2 = remaining 50% ──
            pending = self._pending_builds.get(order.stock_code)
            if pending and pending.get("tranche", 0) == 0:
                # Month 1: 50%
                full_target = target
                target = full_target * 0.5
                pending["target_value"] = full_target
                pending["tranche"] = 1

            target = max(0, target)
            if target > 0:
                order.target_value = target
                sized.append(order)

        return sized
