"""
Loss计算引擎 — 纯代码实现，不调用LLM。
计算5维度Loss：排序质量、方向准确率、校准误差、极端判断惩罚、超额收益不足。

总纲§9：
  L_total = w_effect * L_effect + w_stability * L_stability + w_efficiency * L_efficiency
  Default: w_effect=0.75, w_stability=0.15, w_efficiency=0.10

L_effect 子维度权重(term-specific):
  Short:  w_return=0.45, w_risk=0.40, w_structure=0.15
  Medium: w_return=0.50, w_risk=0.35, w_structure=0.15
  Long:   w_return=0.55, w_risk=0.30, w_structure=0.15
"""
import math
from typing import Dict, Any, List, Optional, Tuple


def spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """计算Spearman秩相关系数，处理ties用平均秩"""
    n = len(x)
    if n < 2:
        return 0.0

    def rankify(lst: List[float]) -> List[float]:
        """Assign ranks with average for ties."""
        sorted_pairs = sorted(enumerate(lst), key=lambda p: p[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
                j += 1
            avg_rank = (i + j + 2) / 2.0  # 1-indexed average
            for k in range(i, j + 1):
                idx = sorted_pairs[k][0]
                ranks[idx] = avg_rank
            i = j + 1
        return ranks

    rx = rankify(x)
    ry = rankify(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    std_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    return cov / (std_x * std_y)


class LossEngine:
    """多维Loss计算引擎 — 纯Python，无LLM调用。

    输入: 预测序列 + 收益序列 + 组合统计量
    输出: 结构化dict，含L_total, L_return, L_risk, L_structure及所有子项
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化Loss引擎。

        Args:
            config: 可选配置字典，支持键：
                - loss_effect_weight (float): L_effect权重，默认0.75
                - loss_stability_weight (float): L_stability权重，默认0.15
                - loss_efficiency_weight (float): L_efficiency权重，默认0.10
                - return_component_weights (dict): 可覆盖L_return 5子项权重
                - risk_component_weights (dict): 可覆盖L_risk 4子项权重
                - structure_component_weights (dict): 可覆盖L_structure 4子项权重
                - term_weights (dict): 可覆盖各term的(w_return, w_risk, w_structure)
        """
        cfg = config or {}
        self.w_effect = float(cfg.get("loss_effect_weight", 0.75))
        self.w_stability = float(cfg.get("loss_stability_weight", 0.15))
        self.w_efficiency = float(cfg.get("loss_efficiency_weight", 0.10))

        # L_return 子项权重
        self.return_weights = cfg.get("return_component_weights", {
            "rank_ic": 0.35,
            "direction": 0.25,
            "calibration": 0.20,
            "extreme": 0.15,
            "excess": 0.05,
        })

        # L_risk 子项权重
        self.risk_weights = cfg.get("risk_component_weights", {
            "drawdown": 0.35,
            "volatility": 0.25,
            "downside": 0.20,
            "consecutive": 0.20,
        })

        # L_structure 子项权重
        self.structure_weights = cfg.get("structure_component_weights", {
            "concentration": 0.30,
            "turnover": 0.25,
            "cash_drag": 0.25,
            "sector": 0.20,
        })

        # 各term的L_effect子维度权重
        self.term_l_effect_weights = cfg.get("term_weights", {
            "short": (0.45, 0.40, 0.15),
            "medium": (0.50, 0.35, 0.15),
            "long": (0.55, 0.30, 0.15),
        })

        # L_stability 子项权重 (simplified vs spec §9.2)
        self.stability_weights = cfg.get("stability_component_weights", {
            "score_volatility": 0.50,
            "drawdown_consistency": 0.50,
        })

        # L_efficiency 子项权重 (simplified vs spec §9.3)
        self.efficiency_weights = cfg.get("efficiency_component_weights", {
            "turnover": 0.40,
            "cash_drag": 0.35,
            "concentration": 0.25,
        })

    def _get_term_weights(self, term: str) -> Tuple[float, float, float]:
        """获取各期限的L_effect子维度(w_return, w_risk, w_structure)权重"""
        return self.term_l_effect_weights.get(
            term, self.term_l_effect_weights["medium"]
        )

    # ========== L_return 收益端 ==========

    def compute_L_return(
        self,
        scores: List[float],
        returns: List[float],
        benchmark_returns: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """计算收益端L_return及其5子项。

        Args:
            scores: 评分列表 [0-100]
            returns: 实际收益率列表
            benchmark_returns: 基准收益率列表，默认全0

        Returns:
            dict: L_return及所有子项细节
        """
        n = len(scores)
        if n == 0:
            return {
                "L_return": 1.0,
                "L_rank_ic": 1.0,
                "L_direction": 1.0,
                "L_calibration": 1.0,
                "L_extreme": 1.0,
                "L_excess": 1.0,
                "rank_ic_raw": 0.0,
                "direction_accuracy": 0.0,
                "calibration_ece": 1.0,
                "extreme_penalty_raw": 0.0,
                "excess_return": 0.0,
                "sample_size": 0,
            }

        benchmark_returns = list(benchmark_returns) if benchmark_returns else [0.0] * n

        # ---- 1. Rank IC Normalized (0.35) ----
        if n >= 2:
            sp_r = spearman_rank_correlation(scores, returns)
            rank_ic_norm = (sp_r + 1.0) / 2.0  # [-1,1] -> [0,1]
            L_rank_ic = 1.0 - rank_ic_norm
        else:
            rank_ic_norm = 0.0
            L_rank_ic = 1.0

        # ---- 2. Direction Accuracy (0.25) ----
        correct = 0
        for s, r, br in zip(scores, returns, benchmark_returns):
            pred_dir = 1 if s >= 50 else -1
            actual_dir = 1 if r > br else -1
            if pred_dir == actual_dir:
                correct += 1
        direction_acc = correct / n if n > 0 else 0.0
        L_direction = 1.0 - direction_acc

        # ---- 3. Calibration Error / ECE (0.20) ----
        # Expected error calibration: bucket scores, compare mean pred vs mean actual
        buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
        ece = 0.0
        total_weight = 0
        for low, high in buckets:
            bucket_returns = [r for s, r in zip(scores, returns) if low <= s < high]
            if bucket_returns:
                w = len(bucket_returns) / n
                # Expected return from score bucket midpoint (normalized)
                expected = (low + high) / 200.0  # 0-100 score to ~0-1 expected return fraction
                actual = sum(bucket_returns) / len(bucket_returns)
                ece += w * abs(actual - expected)
                total_weight += w
        # If no buckets populated (unlikely), fallback
        L_calibration = min(ece, 1.0) if total_weight > 0 else 1.0

        # ---- 4. Extreme Prediction Penalty (0.15) ----
        extreme_penalty_sum = 0.0
        extreme_count = 0
        for s, r in zip(scores, returns):
            if s >= 80 and r < 0:
                extreme_penalty_sum += s / 100.0  # penalty scales with confidence
                extreme_count += 1
            elif s <= 20 and r > 0:
                extreme_penalty_sum += (100.0 - s) / 100.0
                extreme_count += 1
        L_extreme = extreme_penalty_sum / max(extreme_count, 1) if extreme_count > 0 else 0.0

        # ---- 5. Excess Return Shortfall (0.05) ----
        avg_return = sum(returns) / n if n > 0 else 0.0
        avg_benchmark = sum(benchmark_returns) / n if n > 0 else 0.0
        excess = avg_return - avg_benchmark
        if excess < 0:
            # Cap at 10% annualized shortfall -> L=1.0
            L_excess = min(abs(excess) / 0.10, 1.0)
        else:
            L_excess = 0.0

        # Aggregate
        rw = self.return_weights
        L_return = (
            rw["rank_ic"] * L_rank_ic
            + rw["direction"] * L_direction
            + rw["calibration"] * L_calibration
            + rw["extreme"] * L_extreme
            + rw["excess"] * L_excess
        )

        return {
            "L_return": L_return,
            "L_rank_ic": L_rank_ic,
            "L_direction": L_direction,
            "L_calibration": L_calibration,
            "L_extreme": L_extreme,
            "L_excess": L_excess,
            "rank_ic_raw": rank_ic_norm,
            "direction_accuracy": direction_acc,
            "calibration_ece": ece,
            "extreme_penalty_raw": extreme_penalty_sum,
            "excess_return": excess,
            "sample_size": n,
        }

    # ========== L_risk 风险端 ==========

    def compute_L_risk(self, daily_returns: List[float]) -> Dict[str, Any]:
        """计算风险端L_risk — 4子项。

        Args:
            daily_returns: 日收益率序列（百分比数值，如0.01=1%）

        Returns:
            dict: L_risk及所有子项细节
        """
        n = len(daily_returns)
        if n == 0:
            return {
                "L_risk": 1.0,
                "max_drawdown": 0.0,
                "volatility_annual": 0.0,
                "downside_deviation_annual": 0.0,
                "max_consecutive_loss_days": 0,
                "dd_penalty": 1.0,
                "vol_penalty": 1.0,
                "downside_penalty": 1.0,
                "consec_penalty": 1.0,
                "sample_size": 0,
            }

        # ---- 1. Max Drawdown (0.35) ----
        # Track portfolio value (start at 1.0), then compute MDD from running peak.
        value = 1.0
        peak = 1.0
        mdd = 0.0
        for r in daily_returns:
            value *= (1.0 + r)
            peak = max(peak, value)
            if peak > 0:
                dd = (peak - value) / peak
                mdd = max(mdd, dd)

        # Piecewise mapping
        if mdd < 0.10:
            dd_penalty = 0.0
        elif mdd < 0.20:
            dd_penalty = (mdd - 0.10) / 0.10 * 0.5
        elif mdd < 0.40:
            dd_penalty = 0.5 + (mdd - 0.20) / 0.20 * 0.5
        else:
            dd_penalty = 1.0

        # ---- 2. Annualized Volatility (0.25) ----
        mean_r = sum(daily_returns) / n
        variance = sum((r - mean_r) ** 2 for r in daily_returns) / max(n - 1, 1)
        annual_vol = math.sqrt(max(variance, 0.0) * 252)

        if annual_vol < 0.15:
            vol_penalty = 0.0
        elif annual_vol < 0.30:
            vol_penalty = (annual_vol - 0.15) / 0.15
        else:
            vol_penalty = 1.0

        # ---- 3. Downside Deviation (0.20) ----
        neg_returns = [r for r in daily_returns if r < 0]
        if neg_returns:
            neg_mean = sum(neg_returns) / len(neg_returns)
            d_var = sum((r - neg_mean) ** 2 for r in neg_returns) / max(len(neg_returns) - 1, 1)
            downside_dev = math.sqrt(max(d_var, 0.0) * 252)
            downside_penalty = min(downside_dev / 0.30, 1.0)
        else:
            downside_dev = 0.0
            downside_penalty = 0.0

        # ---- 4. Consecutive Loss Days (0.20) ----
        max_consec = 0
        curr_consec = 0
        for r in daily_returns:
            if r < 0:
                curr_consec += 1
                max_consec = max(max_consec, curr_consec)
            else:
                curr_consec = 0

        if max_consec >= 8:
            consec_penalty = 1.0
        elif max_consec >= 5:
            consec_penalty = 0.7
        elif max_consec >= 3:
            consec_penalty = 0.3
        else:
            consec_penalty = 0.0

        # Aggregate
        rkw = self.risk_weights
        L_risk = (
            rkw["drawdown"] * dd_penalty
            + rkw["volatility"] * vol_penalty
            + rkw["downside"] * downside_penalty
            + rkw["consecutive"] * consec_penalty
        )

        return {
            "L_risk": L_risk,
            "max_drawdown": mdd,
            "volatility_annual": annual_vol,
            "downside_deviation_annual": downside_dev,
            "max_consecutive_loss_days": max_consec,
            "dd_penalty": dd_penalty,
            "vol_penalty": vol_penalty,
            "downside_penalty": downside_penalty,
            "consec_penalty": consec_penalty,
            "sample_size": n,
        }

    # ========== L_structure 结构端 ==========

    def compute_L_structure(
        self,
        holdings_weights: List[float],
        turnover_rate: float,
        cash_ratio: float,
        sector_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """计算结构端L_structure — 4子项。

        Args:
            holdings_weights: 持仓个股权重列表（小数，归一到1.0）
            turnover_rate: 月换手率（小数，0.5=50%月度换手）
            cash_ratio: 现金占比（小数，0.15=15%现金）
            sector_weights: 行业权重列表（小数）

        Returns:
            dict: L_structure及所有子项细节
        """
        sector_weights = list(sector_weights) if sector_weights else []
        n_h = len(holdings_weights)

        # ---- 1. Concentration / HHI (0.30) ----
        if n_h > 0:
            hhi = sum(w ** 2 for w in holdings_weights)
            # Normalized: HHI∈[1/n, 1.0]; penalty when above 0.10
            conc_penalty = max(0.0, min((hhi - 0.10) / 0.90, 1.0))
        else:
            hhi = 0.0
            conc_penalty = 1.0  # no holdings data = worst

        # ---- 2. Turnover (0.25) ----
        # monthly turnover rate piecewise
        if turnover_rate > 2.5:
            turnover_penalty = 1.0
        elif turnover_rate > 1.5:
            turnover_penalty = (turnover_rate - 1.5) / 1.0
        elif turnover_rate > 0.5:
            turnover_penalty = (turnover_rate - 0.5) / 1.0 * 0.5
        else:
            turnover_penalty = 0.0

        # ---- 3. Cash Drag (0.25) ----
        if cash_ratio > 0.30:
            cash_penalty = min((cash_ratio - 0.30) / 0.40, 1.0)
        else:
            cash_penalty = 0.0

        # ---- 4. Sector Diversification (0.20) ----
        if sector_weights:
            max_sector = max(sector_weights)
            if max_sector > 0.80:
                sector_penalty = 1.0
            elif max_sector > 0.60:
                sector_penalty = 0.6
            elif max_sector > 0.40:
                sector_penalty = 0.3
            else:
                sector_penalty = 0.0
        else:
            max_sector = 0.0
            sector_penalty = 0.0

        # Aggregate
        sw = self.structure_weights
        L_structure = (
            sw["concentration"] * conc_penalty
            + sw["turnover"] * turnover_penalty
            + sw["cash_drag"] * cash_penalty
            + sw["sector"] * sector_penalty
        )

        return {
            "L_structure": L_structure,
            "hhi": hhi,
            "concentration_penalty": conc_penalty,
            "turnover_rate": turnover_rate,
            "turnover_penalty": turnover_penalty,
            "cash_ratio": cash_ratio,
            "cash_penalty": cash_penalty,
            "max_sector_weight": max_sector,
            "sector_penalty": sector_penalty,
            "sample_size": n_h,
        }

    # ========== L_stability 稳定性端 ==========
    # Simplified implementation per spec §9.2.
    # Full spec calls for rolling-window score correlation + multi-line drawdown
    # rank correlation. This version uses score volatility and drawdown variance
    # across periods as a practical approximation.

    def compute_L_stability(
        self,
        scores_history: Optional[List[List[float]]] = None,
        returns_history: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """计算稳定性端L_stability。

        Args:
            scores_history: 各期评分列表的列表，如 [[80,70,...], [75,65,...], ...]
            returns_history: 各期收益列表的列表，用于计算各期MDD

        Returns:
            dict: L_stability及子项细节
        """
        scores_history = scores_history or []
        returns_history = returns_history or []
        n_periods = len(scores_history)

        if n_periods < 2:
            # Insufficient historical data — default to moderate loss (0.30)
            # acknowledging stability assessment requires more data.
            return {
                "L_stability": 0.30,
                "score_volatility_penalty": 0.30,
                "drawdown_consistency_penalty": 0.30,
                "score_volatility_raw": 0.0,
                "mdds": [],
                "n_periods": n_periods,
                "insufficient_data": True,
            }

        # ---- 1. Score Volatility Penalty (0.50) ----
        # Simplified: std of mean scores across periods, normalized to [0,1]
        mean_scores = [sum(p) / max(len(p), 1) for p in scores_history]
        overall_mean = sum(mean_scores) / n_periods
        score_std = math.sqrt(
            sum((s - overall_mean) ** 2 for s in mean_scores) / max(n_periods - 1, 1)
        )
        # Normalize: std of 20 points → penalty 1.0
        score_vol_penalty = min(score_std / 20.0, 1.0)

        # ---- 2. Drawdown Consistency Penalty (0.50) ----
        # Simplified: compute MDD per period, then penalty = CV / 2.0 capped at 1.0
        # Full spec uses rank correlation of drawdown sequences across lines.
        mdds = []
        for period_returns in returns_history:
            value = 1.0
            peak = 1.0
            mdd = 0.0
            for r in period_returns:
                value *= (1.0 + float(r))
                peak = max(peak, value)
                if peak > 0:
                    dd = (peak - value) / peak
                    mdd = max(mdd, dd)
            mdds.append(mdd)

        if len(mdds) >= 2 and sum(mdds) > 0:
            mean_mdd = sum(mdds) / len(mdds)
            mdd_var = sum((m - mean_mdd) ** 2 for m in mdds) / max(len(mdds) - 1, 1)
            # Coefficient of variation; high CV = inconsistent drawdowns
            mdd_cv = math.sqrt(max(mdd_var, 0.0)) / max(mean_mdd, 0.001)
            dd_consistency_penalty = min(mdd_cv / 2.0, 1.0)
        else:
            dd_consistency_penalty = 0.30

        # Aggregate via config weights
        sw = self.stability_weights
        L_stability = (
            sw["score_volatility"] * score_vol_penalty
            + sw["drawdown_consistency"] * dd_consistency_penalty
        )

        return {
            "L_stability": L_stability,
            "score_volatility_penalty": score_vol_penalty,
            "drawdown_consistency_penalty": dd_consistency_penalty,
            "score_volatility_raw": score_std,
            "mdds": mdds,
            "n_periods": n_periods,
            "insufficient_data": False,
        }

    # ========== L_efficiency 效率端 ==========
    # Simplified implementation per spec §9.3.
    # Full spec calls for transaction cost models, multi-period turnover tracking,
    # and detailed cash-flow impact analysis. This version uses single-period
    # turnover/cash/concentration as a practical first cut.

    def compute_L_efficiency(
        self,
        turnover_rate: float = 0.0,
        cash_ratio: float = 0.0,
        holdings_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """计算效率端L_efficiency。

        Args:
            turnover_rate: 月换手率（小数）
            cash_ratio: 现金占比（小数）
            holdings_weights: 持仓个股权重列表

        Returns:
            dict: L_efficiency及子项细节
        """
        holdings_weights = list(holdings_weights) if holdings_weights else []

        # If no data at all, return default moderate loss (0.30)
        # acknowledging these metrics need more data.
        if turnover_rate == 0.0 and cash_ratio == 0.0 and not holdings_weights:
            return {
                "L_efficiency": 0.30,
                "turnover_penalty": 0.30,
                "cash_drag_penalty": 0.30,
                "concentration_penalty": 0.30,
                "hhi": 0.0,
                "turnover_rate": 0.0,
                "cash_ratio": 0.0,
                "insufficient_data": True,
            }

        # ---- 1. Turnover Penalty (0.40) ----
        # Simplified: piecewise on monthly turnover rate.
        # Tighter thresholds than L_structure (efficiency focus).
        if turnover_rate > 2.0:
            turnover_penalty = 1.0
        elif turnover_rate > 1.0:
            turnover_penalty = (turnover_rate - 1.0) / 1.0
        elif turnover_rate > 0.3:
            turnover_penalty = (turnover_rate - 0.3) / 0.7 * 0.5
        else:
            turnover_penalty = 0.0

        # ---- 2. Cash Drag Penalty (0.35) ----
        # Simplified: idle cash above 10% incurs drag.
        if cash_ratio > 0.25:
            cash_penalty = min((cash_ratio - 0.25) / 0.50, 1.0)
        elif cash_ratio > 0.10:
            cash_penalty = (cash_ratio - 0.10) / 0.15 * 0.5
        else:
            cash_penalty = 0.0

        # ---- 3. Concentration Penalty (0.25) ----
        # Simplified: HHI-based, lower threshold than L_structure (efficiency view).
        if holdings_weights:
            hhi = sum(w ** 2 for w in holdings_weights)
            conc_penalty = max(0.0, min((hhi - 0.05) / 0.95, 1.0))
        else:
            hhi = 0.0
            conc_penalty = 0.30  # neutral when no holdings data

        # Aggregate via config weights
        ew = self.efficiency_weights
        L_efficiency = (
            ew["turnover"] * turnover_penalty
            + ew["cash_drag"] * cash_penalty
            + ew["concentration"] * conc_penalty
        )

        return {
            "L_efficiency": L_efficiency,
            "turnover_penalty": turnover_penalty,
            "cash_drag_penalty": cash_penalty,
            "concentration_penalty": conc_penalty,
            "hhi": hhi,
            "turnover_rate": turnover_rate,
            "cash_ratio": cash_ratio,
            "insufficient_data": False,
        }

    # ========== Total Loss ==========

    def compute_total_loss(
        self,
        term: str,
        scores: List[float],
        returns: List[float],
        benchmark_returns: Optional[List[float]] = None,
        daily_returns: Optional[List[float]] = None,
        holdings_weights: Optional[List[float]] = None,
        turnover_rate: float = 0.0,
        cash_ratio: float = 0.0,
        sector_weights: Optional[List[float]] = None,
        scores_history: Optional[List[List[float]]] = None,
        returns_history: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """计算总Loss — 三端合一。

        L_effect = w_return * L_return + w_risk * L_risk + w_structure * L_structure
        L_total  = w_effect * L_effect + w_stability * L_stability + w_efficiency * L_efficiency

        Args:
            term: "short" / "medium" / "long"
            scores: 评分列表 [0-100]
            returns: 实际收益率列表
            benchmark_returns: 基准收益，默认全0
            daily_returns: 日收益序列（用于风险端），默认=returns（当单周期回退）
            holdings_weights: 持仓权重列表
            turnover_rate: 月换手率
            cash_ratio: 现金占比
            sector_weights: 行业权重列表
            scores_history: （可选）多期评分历史，用于L_stability计算
            returns_history: （可选）多期收益历史，用于L_stability计算

        Returns:
            dict: 完整Loss分解，含score_total=100*(1-L_total)
        """
        # Defaults
        n = len(scores)
        benchmark_returns = list(benchmark_returns) if benchmark_returns else [0.0] * max(n, 1)
        daily_returns = list(daily_returns) if daily_returns else list(returns)  # fallback
        holdings_weights = list(holdings_weights) if holdings_weights else []
        sector_weights = list(sector_weights) if sector_weights else []

        w_return, w_risk, w_structure = self._get_term_weights(term)

        # Compute three loss components
        return_result = self.compute_L_return(scores, returns, benchmark_returns)
        risk_result = self.compute_L_risk(daily_returns)
        structure_result = self.compute_L_structure(
            holdings_weights, turnover_rate, cash_ratio, sector_weights
        )

        # L_effect
        L_effect = (
            w_return * return_result["L_return"]
            + w_risk * risk_result["L_risk"]
            + w_structure * structure_result["L_structure"]
        )

        # Phase 2 implementation — stability and efficiency
        # Simplified vs full spec §9.2-9.3 (see method docstrings for details)
        stability_result = self.compute_L_stability(scores_history, returns_history)
        L_stability = stability_result["L_stability"]

        efficiency_result = self.compute_L_efficiency(turnover_rate, cash_ratio, holdings_weights)
        L_efficiency = efficiency_result["L_efficiency"]

        L_total = (
            self.w_effect * L_effect
            + self.w_stability * L_stability
            + self.w_efficiency * L_efficiency
        )

        score_total = 100.0 * (1.0 - L_total)

        return {
            "L_total": L_total,
            "L_effect": L_effect,
            "L_stability": L_stability,
            "L_efficiency": L_efficiency,
            "score_total": score_total,
            "return_detail": return_result,
            "risk_detail": risk_result,
            "structure_detail": structure_result,
            "stability_detail": stability_result,
            "efficiency_detail": efficiency_result,
            "term": term,
            "sample_size": n,
        }


    # ========== Module-Level Loss Tracking ==========
    # Simplified implementation per spec §10.
    # Computes per-module losses: stock short/medium/long lines + agent ablation.
    # Thin wrappers around existing compute_total_loss with term-specific configs.

    def compute_module_losses(
        self,
        stock_short_term_data: Optional[Dict[str, Any]] = None,
        stock_medium_term_data: Optional[Dict[str, Any]] = None,
        stock_long_term_data: Optional[Dict[str, Any]] = None,
        contribution_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """计算模块级Loss追踪。

        Args:
            stock_short_term_data: 短期线的scores/returns/daily_returns等dict
            stock_medium_term_data: 中期线的数据dict
            stock_long_term_data: 长期线的数据dict
            contribution_data: 贡献分析结果dict，含"contributions"列表

        Returns:
            dict: {
                "stock_short_term": {...},
                "stock_medium_term": {...},
                "stock_long_term": {...},
                "agent_ablation": {"deltas": {...}, "mean_delta": ...},
            }
        """
        result = {}

        if stock_short_term_data:
            result["stock_short_term"] = self.compute_total_loss(
                "short",
                stock_short_term_data.get("scores", []),
                stock_short_term_data.get("returns", []),
                stock_short_term_data.get("benchmark_returns"),
                stock_short_term_data.get("daily_returns"),
                stock_short_term_data.get("holdings_weights"),
                stock_short_term_data.get("turnover_rate", 0.0),
                stock_short_term_data.get("cash_ratio", 0.0),
                stock_short_term_data.get("sector_weights"),
            )

        if stock_medium_term_data:
            result["stock_medium_term"] = self.compute_total_loss(
                "medium",
                stock_medium_term_data.get("scores", []),
                stock_medium_term_data.get("returns", []),
                stock_medium_term_data.get("benchmark_returns"),
                stock_medium_term_data.get("daily_returns"),
                stock_medium_term_data.get("holdings_weights"),
                stock_medium_term_data.get("turnover_rate", 0.0),
                stock_medium_term_data.get("cash_ratio", 0.0),
                stock_medium_term_data.get("sector_weights"),
            )

        if stock_long_term_data:
            result["stock_long_term"] = self.compute_total_loss(
                "long",
                stock_long_term_data.get("scores", []),
                stock_long_term_data.get("returns", []),
                stock_long_term_data.get("benchmark_returns"),
                stock_long_term_data.get("daily_returns"),
                stock_long_term_data.get("holdings_weights"),
                stock_long_term_data.get("turnover_rate", 0.0),
                stock_long_term_data.get("cash_ratio", 0.0),
                stock_long_term_data.get("sector_weights"),
            )

        if contribution_data:
            contributions = contribution_data.get("contributions", [])
            ablation_deltas = {}
            for c in contributions:
                ablation_deltas[c.get("agent_name", "unknown")] = c.get("delta_L_total", 0.0)

            n_deltas = max(len(ablation_deltas), 1)
            result["agent_ablation"] = {
                "deltas": ablation_deltas,
                "mean_delta": sum(ablation_deltas.values()) / n_deltas,
                "max_positive": max(ablation_deltas.values()) if ablation_deltas else 0.0,
                "max_negative": min(ablation_deltas.values()) if ablation_deltas else 0.0,
                "n_agents": len(ablation_deltas),
            }

        return result


# ========== Utility: Agent Ablation ΔLoss ==========

def compute_agent_delta_loss(
    loss_all: float,
    loss_without_agent: float,
) -> Dict[str, Any]:
    """计算单个agent消融实验的ΔLoss。

    Args:
        loss_all: 全量agent的L_total
        loss_without_agent: 去掉某个agent后的L_total

    Returns:
        dict: delta_L_total, contribution direction
    """
    delta = loss_without_agent - loss_all
    if delta > 0.001:
        contribution = "positive"   # agent reduces loss
    elif delta < -0.001:
        contribution = "negative"   # agent increases loss
    else:
        contribution = "neutral"
    return {
        "delta_L_total": delta,
        "loss_all_agents": loss_all,
        "loss_without_agent": loss_without_agent,
        "contribution": contribution,
    }
