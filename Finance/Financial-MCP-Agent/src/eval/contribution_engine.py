"""
Agent贡献引擎 — 通过消融实验计算每个分析Agent的边际贡献。
ΔL = L(无该agent) - L(全agent)
ΔL > 0 → 正贡献  ΔL < 0 → 负贡献
"""
import random
import math
from typing import Dict, Any, List, Optional, Tuple
from src.eval.loss_engine import LossEngine


AGENT_NAMES = ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"]


class ContributionEngine:
    """Agent贡献分析引擎"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.loss_engine = LossEngine(config)
        self.bootstrap_iterations = self.config.get("bootstrap_iterations", 10000)
        self.significance_level = self.config.get("significance_level", 0.95)

    def compute_contributions(
        self,
        term: str,
        all_agent_results: Dict[str, List[Dict[str, Any]]],
        ablation_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ) -> Dict[str, Any]:
        """
        计算所有agent的贡献分。

        Args:
            term: short/medium/long
            all_agent_results: {"scores": [...], "returns": [...], ...} 全agent的完整结果
            ablation_results: {
                "fundamental": {"scores": [...], "returns": [...], ...},
                "technical": {...}, ...
            }

        Returns:
            {
                "term": "medium",
                "contributions": [
                    {
                        "agent_name": "fundamental",
                        "delta_L_total": 0.042,
                        "delta_L_return": 0.030,
                        "delta_L_risk": 0.008,
                        "delta_L_structure": 0.004,
                        "ci_95": [0.018, 0.086],
                        "significance": "significant_positive",
                        "stars": "★★★",
                    },
                    ...
                ]
            }
        """
        # 计算全agent的基线Loss
        baseline = self._compute_baseline_loss(term, all_agent_results)

        contributions = []
        for agent_name in AGENT_NAMES:
            if agent_name not in ablation_results:
                continue

            ablated = ablation_results[agent_name]
            if not ablated.get("scores"):
                continue

            # 计算消融后的Loss
            ablated_loss = self._compute_ablated_loss(term, ablated, all_agent_results)

            # ΔLoss
            delta_total = ablated_loss["L_total"] - baseline["L_total"]
            delta_return = ablated_loss["return_detail"]["L_return"] - baseline["return_detail"]["L_return"]
            delta_risk = ablated_loss["risk_detail"]["L_risk"] - baseline["risk_detail"]["L_risk"]
            delta_structure = ablated_loss["structure_detail"]["L_structure"] - baseline["structure_detail"]["L_structure"]

            # Bootstrap置信区间
            ci_low, ci_high = self._bootstrap_ci(
                term, all_agent_results, ablated, agent_name, delta_total
            )

            # 显著性判断
            significance, stars = self._classify_significance(delta_total, ci_low, ci_high)

            contributions.append({
                "agent_name": agent_name,
                "delta_L_total": round(delta_total, 4),
                "delta_L_return": round(delta_return, 4),
                "delta_L_risk": round(delta_risk, 4),
                "delta_L_structure": round(delta_structure, 4),
                "ci_95_lower": round(ci_low, 4),
                "ci_95_upper": round(ci_high, 4),
                "significance": significance,
                "stars": stars,
                "baseline_L_total": round(baseline["L_total"], 4),
                "ablated_L_total": round(ablated_loss["L_total"], 4),
            })

        # 按ΔL降序排列
        contributions.sort(key=lambda x: x["delta_L_total"], reverse=True)

        return {
            "term": term,
            "baseline": baseline,
            "contributions": contributions,
            "sample_size": len(all_agent_results.get("scores", [])),
        }

    def _compute_baseline_loss(self, term: str, results: Dict[str, List]) -> Dict[str, Any]:
        """计算全agent基线Loss"""
        scores = results.get("scores", [])
        returns = results.get("returns", [])
        benchmarks = results.get("benchmark_returns", [0.0] * len(returns))
        daily_rets = results.get("daily_returns", returns)
        weights = results.get("holdings_weights", [])
        turnover = results.get("turnover_rate", 0.0)
        cash = results.get("cash_ratio", 0.0)
        sectors = results.get("sector_weights", [])

        return self.loss_engine.compute_total_loss(
            term, scores, returns, benchmarks, daily_rets,
            weights, turnover, cash, sectors
        )

    def _compute_ablated_loss(self, term: str, ablated: Dict[str, List],
                               baseline_results: Dict[str, List]) -> Dict[str, Any]:
        """计算消融后的Loss（使用消融线的scores，但组合数据来自基线）"""
        scores = ablated.get("scores", [])
        returns = baseline_results.get("returns", [])
        benchmarks = baseline_results.get("benchmark_returns", [0.0] * len(returns))
        daily_rets = baseline_results.get("daily_returns", returns)
        weights = baseline_results.get("holdings_weights", [])
        turnover = baseline_results.get("turnover_rate", 0.0)
        cash = baseline_results.get("cash_ratio", 0.0)
        sectors = baseline_results.get("sector_weights", [])

        # Ensure length match
        n = min(len(scores), len(returns))
        scores = scores[:n]
        returns = returns[:n]

        return self.loss_engine.compute_total_loss(
            term, scores, returns, benchmarks[:n], daily_rets[:n],
            weights, turnover, cash, sectors
        )

    def _bootstrap_ci(self, term: str, baseline: Dict[str, List],
                      ablated: Dict[str, List], agent_name: str,
                      observed_delta: float) -> Tuple[float, float]:
        """Bootstrap置信区间 — 使用完整的L_total而非仅L_return"""
        baseline_scores = baseline.get("scores", [])
        ablated_scores = ablated.get("scores", [])
        returns = baseline.get("returns", [])

        n = min(len(baseline_scores), len(ablated_scores), len(returns))
        if n < 5:
            return observed_delta - 0.05, observed_delta + 0.05

        # Extract shared portfolio data from baseline (same for both lines)
        daily_rets = baseline.get("daily_returns", returns)
        benchmarks = baseline.get("benchmark_returns", [0.0] * n)
        weights = baseline.get("holdings_weights", [])
        turnover = baseline.get("turnover_rate", 0.0)
        cash = baseline.get("cash_ratio", 0.0)
        sectors = baseline.get("sector_weights", [])

        deltas = []
        indices = list(range(n))

        for _ in range(self.bootstrap_iterations):
            sample_idx = [random.choice(indices) for _ in range(n)]
            bs_baseline = [baseline_scores[i] for i in sample_idx]
            bs_ablated = [ablated_scores[i] for i in sample_idx]
            bs_returns = [returns[i] for i in sample_idx]

            # Resample benchmark and daily returns if available
            bs_benchmarks = (
                [benchmarks[i] for i in sample_idx]
                if len(benchmarks) >= n else [0.0] * n
            )
            bs_daily = (
                [daily_rets[i] for i in sample_idx]
                if len(daily_rets) >= n else bs_returns
            )

            # Compute full L_total for both lines (not just L_return)
            bs_base_loss = self.loss_engine.compute_total_loss(
                term, bs_baseline, bs_returns, bs_benchmarks, bs_daily,
                weights, turnover, cash, sectors
            )["L_total"]
            bs_ablate_loss = self.loss_engine.compute_total_loss(
                term, bs_ablated, bs_returns, bs_benchmarks, bs_daily,
                weights, turnover, cash, sectors
            )["L_total"]
            deltas.append(bs_ablate_loss - bs_base_loss)

        deltas.sort()
        alpha = (1 - self.significance_level) / 2
        ci_low = deltas[int(len(deltas) * alpha)]
        ci_high = deltas[int(len(deltas) * (1 - alpha))]

        return ci_low, ci_high

    def _classify_significance(self, delta: float, ci_low: float, ci_high: float) -> Tuple[str, str]:
        """根据ΔL和CI判断显著性"""
        if ci_low > 0:
            if delta > 0.03:
                return "significant_positive", "★★★"
            return "significant_positive", "★★"
        elif ci_high < 0:
            if delta < -0.03:
                return "significant_negative", "↓↓↓"
            return "significant_negative", "↓"
        else:
            return "not_significant", "☆"

    def compute_contribution_scores(self, contributions: List[Dict]) -> List[Dict]:
        """将ΔL映射到0-100贡献分（可视化用）"""
        for c in contributions:
            delta = c["delta_L_total"]
            # 线性映射: delta range [-0.1, +0.1] → contribution_score [0, 100]
            score = max(0, min(100, 50 + delta * 500))
            c["contribution_score"] = round(score, 1)

            # 分类标签
            if score >= 80:
                c["label"] = "强正贡献"
            elif score >= 60:
                c["label"] = "正贡献"
            elif score >= 40:
                c["label"] = "中性/不稳定"
            else:
                c["label"] = "负贡献/拖后腿"

        return contributions


def permutation_test(
    term: str, all_scores: List[float], ablated_scores: List[float],
    returns: List[float], n_permutations: int = 10000
) -> Dict[str, Any]:
    """Permutation Test: 随机打乱标签，测试ΔL是否显著"""
    observed = LossEngine().compute_L_return(all_scores, returns)["L_return"]
    observed_ablate = LossEngine().compute_L_return(ablated_scores, returns)["L_return"]
    observed_delta = observed_ablate - observed

    n = len(all_scores)
    permuted_deltas = []

    for _ in range(n_permutations):
        # 随机分配"全agent"和"消融"标签
        combined = list(zip(all_scores, ablated_scores))
        random.shuffle(combined)
        half = n // 2

        group_a = [c[0] for c in combined[:half]] + [c[1] for c in combined[half:]]
        group_b = [c[0] for c in combined[half:]] + [c[1] for c in combined[:half]]

        loss_a = LossEngine().compute_L_return(group_a[:n], returns[:len(group_a)])["L_return"]
        loss_b = LossEngine().compute_L_return(group_b[:n], returns[:len(group_b)])["L_return"]
        permuted_deltas.append(loss_b - loss_a)

    permuted_deltas.sort()
    p_value = sum(1 for d in permuted_deltas if abs(d) >= abs(observed_delta)) / n_permutations

    return {
        "observed_delta": observed_delta,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "n_permutations": n_permutations,
    }
