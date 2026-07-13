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
                - loss_effect_weight (float): L_effect权重，默认0.60
                - loss_stability_weight (float): L_stability权重，默认0.12
                - loss_efficiency_weight (float): L_efficiency权重，默认0.08
                - loss_risk_gate_weight (float): L_risk_gate权重，默认0.05
                - loss_consistency_weight (float): L_consistency权重，默认0.05
                - loss_fidelity_weight (float): L_fidelity权重，默认0.05
                - loss_eval_model_fidelity_weight (float): L_eval_model_fidelity权重，默认0.05
                - return_component_weights (dict): 可覆盖L_return 5子项权重
                - risk_component_weights (dict): 可覆盖L_risk 4子项权重
                - structure_component_weights (dict): 可覆盖L_structure 4子项权重
                - term_weights (dict): 可覆盖各term的(w_return, w_risk, w_structure)
        """
        self.config = config or {}
        cfg = self.config
        # Core weights — slightly reduced to make room for 4 new modules
        self.w_effect = float(cfg.get("loss_effect_weight", 0.60))
        self.w_stability = float(cfg.get("loss_stability_weight", 0.12))
        self.w_efficiency = float(cfg.get("loss_efficiency_weight", 0.08))

        # New module weights — monitoring metrics, not optimization targets
        self.w_risk_gate = float(cfg.get("loss_risk_gate_weight", 0.05))
        self.w_consistency = float(cfg.get("loss_consistency_weight", 0.05))
        self.w_fidelity = float(cfg.get("loss_fidelity_weight", 0.05))
        self.w_eval_model_fidelity = float(cfg.get("loss_eval_model_fidelity_weight", 0.05))

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
                # Expected return from score bucket midpoint (normalized to daily scale)
                # Score 0→-1%, 50→0%, 100→+1% daily expected return
                midpoint = (low + high) / 2.0
                expected = (midpoint - 50.0) / 5000.0  # maps [0,100] to [-0.01, +0.01]
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
    # Phase 1 (简化版): score volatility + drawdown CV approximation.
    # Phase 2 (总纲 §9.2 完整版): rolling-window score correlation +
    #   multi-line drawdown rank correlation. 需要 ≥30 期历史数据。

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
    # Phase 1 (简化版): single-period turnover / cash drag / concentration.
    # Phase 2 (总纲 §9.3 完整版): transaction cost models, multi-period
    #   turnover tracking, detailed cash-flow impact analysis. 需要 ≥12 期换手数据。

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
        risk_gate_events: Optional[List[Dict[str, Any]]] = None,
        consistency_data: Optional[Dict[str, Any]] = None,
        fidelity_result: Optional[Dict[str, Any]] = None,
        fidelity_snapshots: Optional[List[dict]] = None,
        eval_model_fidelity_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """计算总Loss — 8模块合一。

        L_effect = w_return * L_return + w_risk * L_risk + w_structure * L_structure
        L_total  = w_effect * L_effect + w_stability * L_stability + w_efficiency * L_efficiency
                 + w_risk_gate * L_risk_gate + w_consistency * L_consistency
                 + w_fidelity * L_fidelity + w_eval_model_fidelity * L_eval_model_fidelity

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
            risk_gate_events: （可选）风险门控事件列表，用于L_risk_gate计算
            consistency_data: （可选）一致性测试数据，用于L_consistency计算
            fidelity_result: （可选）FidelityEngine预计算结果
            fidelity_snapshots: （可选）快照列表，用于内部计算fidelity
            eval_model_fidelity_data: （可选）评测模型保真度数据dict

        Returns:
            dict: 完整Loss分解，含score_total=100*(1-L_total)及8模块详细分解
        """
        # Defaults
        n = len(scores)
        benchmark_returns = list(benchmark_returns) if benchmark_returns else [0.0] * max(n, 1)
        daily_returns = list(daily_returns) if daily_returns else []
        holdings_weights = list(holdings_weights) if holdings_weights else []
        sector_weights = list(sector_weights) if sector_weights else []

        w_return, w_risk, w_structure = self._get_term_weights(term)

        # Compute three loss components
        return_result = self.compute_L_return(scores, returns, benchmark_returns)
        if daily_returns:
            risk_result = self.compute_L_risk(daily_returns)
        else:
            risk_result = {
                "L_risk": 0.0,
                "max_drawdown": 0.0,
                "volatility_annual": 0.0,
                "downside_deviation_annual": 0.0,
                "max_consecutive_loss_days": 0,
                "sample_size": 0,
                "insufficient_data": True,
                "note": "No daily returns data available; risk component skipped",
            }
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

        # ---- Compute 4 new module losses ----
        risk_gate_detail = self.compute_L_risk_gate(risk_gate_events)
        consistency_detail = self.compute_L_consistency(consistency_data)
        fidelity_detail = self.compute_L_fidelity(fidelity_result, fidelity_snapshots)
        eval_fidelity_detail = self.compute_L_eval_model_fidelity(
            eval_snapshots=(eval_model_fidelity_data or {}).get("eval_snapshots"),
            production_snapshots=(eval_model_fidelity_data or {}).get("production_snapshots"),
            eval_vs_prod_score_pairs=(eval_model_fidelity_data or {}).get("eval_vs_prod_score_pairs"),
        )

        L_total += (
            self.w_risk_gate * risk_gate_detail.get("L_risk_gate", 0.0)
            + self.w_consistency * consistency_detail.get("L_consistency", 0.0)
            + self.w_fidelity * fidelity_detail.get("L_fidelity", 0.0)
            + self.w_eval_model_fidelity * eval_fidelity_detail.get("L_eval_model_fidelity", 0.0)
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
            "risk_gate_detail": risk_gate_detail,
            "consistency_detail": consistency_detail,
            "fidelity_detail": fidelity_detail,
            "eval_model_fidelity_detail": eval_fidelity_detail,
            "term": term,
            "sample_size": n,
        }


    # ========== L_risk_gate 风险门控端 ==========

    def compute_L_risk_gate(
        self,
        risk_gate_events: Optional[List[Dict[str, Any]]] = None,
        total_stocks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """计算风险门控端损失 — 衡量风险门控触发频率和评分削减幅度。

        总纲 §10: risk_gate loss 衡量风险门控规则对评分的压制程度，
        当频繁触发时说明系统在数据质量或风险检测方面存在问题。

        Args:
            risk_gate_events: 风险门控事件列表，每项包含:
                - term: 期限 (short/medium/long)
                - stock_code: 股票代码
                - original_score: 原始评分
                - capped_score: 门控后评分
                - score_cap: 触发的cap值
                - triggered_flags: 触发的风险标签列表
                - downgrade: 降级动作 (谨慎/观察/None)
            total_stocks: 评分股票总数，用于计算触发率。
                若未提供，使用事件涉及的唯一股票数作为分母。

        Returns:
            dict: L_risk_gate及所有子项细节
        """
        risk_gate_events = risk_gate_events or []

        if not risk_gate_events:
            return {
                "L_risk_gate": 0.0,
                "trigger_count": 0,
                "mean_score_reduction": 0.0,
                "max_score_reduction": 0.0,
                "trigger_rate": 0.0,
                "flag_frequency": {},
                "insufficient_data": True,
            }

        n_events = len(risk_gate_events)
        reductions = []
        flag_counter = {}
        downgrade_counter = {}

        for event in risk_gate_events:
            orig = event.get("original_score", 0)
            capped = event.get("capped_score", orig)
            reduction = orig - capped
            if reduction > 0:
                reductions.append(reduction)

            for flag in event.get("triggered_flags", []):
                flag_counter[flag] = flag_counter.get(flag, 0) + 1

            downgrade = event.get("downgrade", "")
            if downgrade:
                downgrade_counter[downgrade] = downgrade_counter.get(downgrade, 0) + 1

        mean_reduction = sum(reductions) / max(len(reductions), 1)
        max_reduction = max(reductions) if reductions else 0.0

        # Normalize: mean_reduction of 30 points → penalty 1.0
        score_reduction_penalty = min(mean_reduction / 30.0, 1.0)

        # Trigger rate penalty: >50% trigger rate → high concern
        # Use total_stocks if provided, else count unique stocks in events
        n_unique_stocks = len({e.get("stock_code", "") for e in risk_gate_events})
        denominator = total_stocks or n_unique_stocks or n_events or 1
        trigger_rate = float(n_events) / denominator
        trigger_rate_penalty = min(trigger_rate / 0.50, 1.0)  # >50% trigger rate = max concern

        L_risk_gate = 0.55 * score_reduction_penalty + 0.45 * trigger_rate_penalty

        return {
            "L_risk_gate": round(L_risk_gate, 4),
            "trigger_count": n_events,
            "mean_score_reduction": round(mean_reduction, 2),
            "max_score_reduction": round(max_reduction, 2),
            "trigger_rate": round(trigger_rate, 4),
            "flag_frequency": flag_counter,
            "downgrade_frequency": downgrade_counter,
            "insufficient_data": False,
        }

    # ========== L_consistency 一致性端 ==========

    def compute_L_consistency(
        self,
        consistency_data: Optional[Dict[str, Any]] = None,
        scores_run_pairs: Optional[List[Tuple[List[float], List[float]]]] = None,
    ) -> Dict[str, Any]:
        """计算一致性端损失 — 衡量同一输入重复运行时的评分稳定性。

        总纲 §10: consistency loss 衡量 LLM 输出的非确定性对系统的影响。
        输入可来自 ExperimentEngine.run_consistency_test() 的结果，
        或直接传入多组 (run1_scores, run2_scores) 对。

        Args:
            consistency_data: 来自 ExperimentEngine.run_consistency_test() 的结果dict。
                应包含: mean_score_diff, action_flip_rate, top_k_overlap。
            scores_run_pairs: 多组 (run1_scores, run2_scores) 的列表，
                作为备选输入路径。当 consistency_data 为空时使用。

        Returns:
            dict: L_consistency及所有子项细节
        """
        if consistency_data and consistency_data.get("experiment_type") == "consistency":
            # Direct ingestion from ExperimentEngine
            mean_diff = consistency_data.get("mean_score_diff", 0.0)
            flip_rate = consistency_data.get("action_flip_rate", 0.0)
            topk_overlap = consistency_data.get("top_k_overlap", 1.0)
            sample_size = consistency_data.get("sample_size", 0)

            # Normalize
            score_diff_penalty = min(mean_diff / 20.0, 1.0)
            flip_rate_penalty = min(flip_rate / 0.30, 1.0)
            overlap_penalty = max(0.0, (1.0 - topk_overlap) / 0.50)

            L_consistency = (
                0.35 * score_diff_penalty
                + 0.35 * flip_rate_penalty
                + 0.30 * overlap_penalty
            )

            return {
                "L_consistency": round(L_consistency, 4),
                "mean_score_diff": mean_diff,
                "action_flip_rate": flip_rate,
                "top_k_overlap": topk_overlap,
                "sample_size": sample_size,
                "insufficient_data": sample_size == 0,
            }

        if scores_run_pairs:
            all_diffs = []
            all_flips = 0
            all_pairs = 0
            all_overlaps = []
            k_default = 10

            for run1, run2 in scores_run_pairs:
                n = min(len(run1), len(run2))
                if n == 0:
                    continue

                all_pairs += n
                for i in range(n):
                    all_diffs.append(abs(run1[i] - run2[i]))
                    if (run1[i] >= 50) != (run2[i] >= 50):
                        all_flips += 1

                k = min(k_default, n)
                top1 = set(sorted(range(n), key=lambda i: run1[i], reverse=True)[:k])
                top2 = set(sorted(range(n), key=lambda i: run2[i], reverse=True)[:k])
                overlap = len(top1 & top2) / k if k > 0 else 1.0
                all_overlaps.append(overlap)

            mean_diff = sum(all_diffs) / max(len(all_diffs), 1) if all_diffs else 0.0
            flip_rate = all_flips / max(all_pairs, 1)
            avg_overlap = sum(all_overlaps) / max(len(all_overlaps), 1) if all_overlaps else 1.0

            score_diff_penalty = min(mean_diff / 20.0, 1.0)
            flip_rate_penalty = min(flip_rate / 0.30, 1.0)
            overlap_penalty = max(0.0, (1.0 - avg_overlap) / 0.50)

            L_consistency = (
                0.35 * score_diff_penalty
                + 0.35 * flip_rate_penalty
                + 0.30 * overlap_penalty
            )

            return {
                "L_consistency": round(L_consistency, 4),
                "mean_score_diff": round(mean_diff, 2),
                "action_flip_rate": round(flip_rate, 4),
                "top_k_overlap": round(avg_overlap, 4),
                "sample_size": all_pairs,
                "insufficient_data": all_pairs == 0,
            }

        # No data at all
        return {
            "L_consistency": 0.0,
            "mean_score_diff": 0.0,
            "action_flip_rate": 0.0,
            "top_k_overlap": 1.0,
            "sample_size": 0,
            "insufficient_data": True,
        }

    # ========== L_fidelity 保真度端 ==========

    def compute_L_fidelity(
        self,
        fidelity_result: Optional[Dict[str, Any]] = None,
        snapshots: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """计算保真度端损失 — 包装 FidelityEngine 的输出并映射到 0-1 损失。

        可以直接接收 FidelityEngine.compute_fidelity_loss() 的输出，
        或传入 snapshots 列表内部调用 FidelityEngine。

        Args:
            fidelity_result: FidelityEngine.compute_fidelity_loss() 的返回结果
            snapshots: 快照列表，当 fidelity_result 为空时自行计算

        Returns:
            dict: L_fidelity及所有子项细节
        """
        try:
            if fidelity_result is None and snapshots:
                from src.eval.fidelity_engine import FidelityEngine
                fe = FidelityEngine(self.config)
                fidelity_result = fe.compute_fidelity_loss(snapshots)
        except Exception as e:
            # Graceful fallback: return neutral when FidelityEngine is unavailable
            return {
                "L_fidelity": 0.0,
                "action_flip_rate": 0.0,
                "score_drift": 0.0,
                "topK_overlap": 1.0,
                "rank_drift": 0.0,
                "fidelity_loss": 0.0,
                "warnings": [f"FidelityEngine调用失败: {str(e)}"],
                "insufficient_data": True,
            }

        if fidelity_result is None:
            return {
                "L_fidelity": 0.0,
                "action_flip_rate": 0.0,
                "score_drift": 0.0,
                "topK_overlap": 1.0,
                "rank_drift": 0.0,
                "fidelity_loss": 0.0,
                "insufficient_data": True,
            }

        # Map FidelityEngine's fidelity_loss (already 0-1) to L_fidelity
        # If period_pairs_analyzed is 0, there's insufficient data
        n_pairs = fidelity_result.get("period_pairs_analyzed", 0)
        raw_fidelity_loss = fidelity_result.get("fidelity_loss", 0.0)

        L_fidelity = raw_fidelity_loss  # Already in 0-1 range from FidelityEngine

        return {
            "L_fidelity": round(L_fidelity, 4),
            "action_flip_rate": fidelity_result.get("action_flip_rate", 0.0),
            "score_drift": fidelity_result.get("score_drift", 0.0),
            "topK_overlap": fidelity_result.get("topK_overlap", 1.0),
            "rank_drift": fidelity_result.get("rank_drift", 0.0),
            "fidelity_loss_raw": raw_fidelity_loss,
            "period_pairs_analyzed": n_pairs,
            "warnings": fidelity_result.get("warnings", []),
            "insufficient_data": n_pairs == 0,
        }

    # ========== L_eval_model_fidelity 评测模型保真度端 ==========

    def compute_L_eval_model_fidelity(
        self,
        eval_snapshots: Optional[List[dict]] = None,
        production_snapshots: Optional[List[dict]] = None,
        eval_vs_prod_score_pairs: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """计算评测模型保真度损失 — 比较评测模型(M5/DeepSeek V4 Flash)与生产模型(M1/M3)的评分差异。

        总纲 §10: eval_model_fidelity 衡量评测结果在多大程度上可以代表生产环境。
        如果评测模型和生产模型的评分系统性偏离，则评测结论可能不适用于生产。

        Args:
            eval_snapshots: 评测模型(M5)的快照列表
            production_snapshots: 生产模型(M1/M3)的快照列表
            eval_vs_prod_score_pairs: (eval_score, prod_score) 对列表，作为备选输入

        Returns:
            dict: L_eval_model_fidelity及所有子项细节
        """
        try:
            if eval_snapshots and production_snapshots:
                # Build score pairs by aligning snapshots on period and symbol.
                # This measures the actual score agreement between eval (M5) and
                # production (M1/M3) models, NOT each model's self-consistency.
                eval_by_period = {}
                for s in eval_snapshots:
                    period = str(s.get("period", ""))
                    if period:
                        syms = s.get("symbols", [])
                        scores = s.get("scores", [])
                        eval_by_period[period] = dict(zip(syms, scores))

                prod_by_period = {}
                for s in production_snapshots:
                    period = str(s.get("period", ""))
                    if period:
                        syms = s.get("symbols", [])
                        scores = s.get("scores", [])
                        prod_by_period[period] = dict(zip(syms, scores))

                # Collect (eval_score, prod_score) pairs for shared period+symbol
                score_pairs = []
                for period in eval_by_period:
                    if period in prod_by_period:
                        for sym, eval_s in eval_by_period[period].items():
                            prod_s = prod_by_period[period].get(sym)
                            if prod_s is not None:
                                score_pairs.append((eval_s, prod_s))

                if score_pairs:
                    # Reuse the score-pair comparison logic below
                    eval_vs_prod_score_pairs = score_pairs
                else:
                    return {
                        "L_eval_model_fidelity": 0.0,
                        "status": "no_overlapping_periods_or_symbols",
                        "insufficient_data": True,
                    }

            if eval_vs_prod_score_pairs:
                n = len(eval_vs_prod_score_pairs)
                if n == 0:
                    return {
                        "L_eval_model_fidelity": 0.0,
                        "status": "no_production_data",
                        "insufficient_data": True,
                    }

                eval_scores = [p[0] for p in eval_vs_prod_score_pairs]
                prod_scores = [p[1] for p in eval_vs_prod_score_pairs]

                # Mean absolute difference
                mean_diff = sum(abs(e - p) for e, p in eval_vs_prod_score_pairs) / n
                # Correlation
                rho = spearman_rank_correlation(eval_scores, prod_scores)
                # Action agreement
                actions_agree = sum(
                    1 for e, p in eval_vs_prod_score_pairs
                    if (e >= 50) == (p >= 50)
                ) / n

                score_diff_penalty = min(mean_diff / 15.0, 1.0)
                rank_divergence = max(0.0, (1.0 - rho) / 2.0)  # 0-1
                action_disagreement = 1.0 - actions_agree

                L_eval_model_fidelity = (
                    0.40 * score_diff_penalty
                    + 0.35 * rank_divergence
                    + 0.25 * action_disagreement
                )

                return {
                    "L_eval_model_fidelity": round(L_eval_model_fidelity, 4),
                    "mean_score_diff": round(mean_diff, 2),
                    "spearman_rho": round(rho, 4),
                    "action_agreement_rate": round(actions_agree, 4),
                    "sample_size": n,
                    "source": "score_pairs",
                    "insufficient_data": False,
                }

            # No production data available — return neutral placeholder
            return {
                "L_eval_model_fidelity": 0.0,
                "status": "no_production_data",
                "note": "生产模型快照或评分对数据不可用，无法计算评测模型保真度",
                "insufficient_data": True,
            }

        except Exception as e:
            return {
                "L_eval_model_fidelity": 0.0,
                "status": "computation_error",
                "error": str(e),
                "insufficient_data": True,
            }

    # ========== Module-Level Loss Tracking ==========
    # Full implementation per spec §10.
    # Computes all 8 module losses: stock short/medium/long, risk_gate,
    # consistency, fidelity, agent_ablation, eval_model_fidelity.

    def compute_module_losses(
        self,
        stock_short_term_data: Optional[Dict[str, Any]] = None,
        stock_medium_term_data: Optional[Dict[str, Any]] = None,
        stock_long_term_data: Optional[Dict[str, Any]] = None,
        contribution_data: Optional[Dict[str, Any]] = None,
        risk_gate_events: Optional[List[Dict[str, Any]]] = None,
        consistency_data: Optional[Dict[str, Any]] = None,
        fidelity_result: Optional[Dict[str, Any]] = None,
        fidelity_snapshots: Optional[List[dict]] = None,
        eval_model_fidelity_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """计算模块级Loss追踪 — 全部8个模块。

        Args:
            stock_short_term_data: 短期线的scores/returns/daily_returns等dict
            stock_medium_term_data: 中期线的数据dict
            stock_long_term_data: 长期线的数据dict
            contribution_data: 贡献分析结果dict，含"contributions"列表
            risk_gate_events: 风险门控事件列表，参见 compute_L_risk_gate()
            consistency_data: 一致性测试数据，参见 compute_L_consistency()
            fidelity_result: FidelityEngine预计算结果
            fidelity_snapshots: 快照列表，用于内部计算fidelity
            eval_model_fidelity_data: dict with keys:
                - eval_snapshots: 评测模型(M5)快照列表
                - production_snapshots: 生产模型(M1/M3)快照列表
                - eval_vs_prod_score_pairs: (eval_score, prod_score) 对列表

        Returns:
            dict: 包含全部8个模块的loss结果
        """
        result = {}

        # 1-3: Stock term modules
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

        # 4: Risk gate module
        result["stock_risk_gate"] = self.compute_L_risk_gate(risk_gate_events)

        # 5: Consistency module
        result["consistency"] = self.compute_L_consistency(consistency_data)

        # 6: Fidelity module
        result["fidelity"] = self.compute_L_fidelity(fidelity_result, fidelity_snapshots)

        # 7: Agent ablation module
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
        else:
            result["agent_ablation"] = {
                "deltas": {},
                "mean_delta": 0.0,
                "max_positive": 0.0,
                "max_negative": 0.0,
                "n_agents": 0,
                "insufficient_data": True,
            }

        # 8: Eval model fidelity module
        eval_mf_data = eval_model_fidelity_data or {}
        result["evaluation_model_fidelity"] = self.compute_L_eval_model_fidelity(
            eval_snapshots=eval_mf_data.get("eval_snapshots"),
            production_snapshots=eval_mf_data.get("production_snapshots"),
            eval_vs_prod_score_pairs=eval_mf_data.get("eval_vs_prod_score_pairs"),
        )

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
