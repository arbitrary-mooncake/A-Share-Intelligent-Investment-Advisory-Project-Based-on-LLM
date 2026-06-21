"""
实验引擎 — 控制变量实验执行。
包括：agent消融、risk gate on/off、一致性测试、保真度测试、stable vs candidate。
"""
from typing import Dict, Any, List, Optional
from datetime import datetime


class ExperimentEngine:
    """控制变量实验执行器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def run_ablation_experiment(self, term: str, pool: List[str],
                                 ablation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行agent消融实验分析。

        Args:
            term: short/medium/long
            pool: 精筛池
            ablation_results: {agent_label: {scores, returns}} from backtest

        Returns:
            消融实验完整报告
        """
        from src.eval.contribution_engine import ContributionEngine

        engine = ContributionEngine(self.config)

        # 构建输入格式
        all_results = {
            "scores": ablation_results.get("full", {}).get("scores", []),
            "returns": ablation_results.get("full", {}).get("returns", []),
        }

        agent_ablation = {}
        for label, data in ablation_results.items():
            if label == "full":
                continue
            agent_name = label.lstrip("-")
            agent_ablation[agent_name] = {
                "scores": data.get("scores", []),
                "returns": data.get("returns", []),
            }

        contributions = engine.compute_contributions(term, all_results, agent_ablation)
        scored = engine.compute_contribution_scores(contributions["contributions"])

        return {
            "experiment_type": "ablation",
            "term": term,
            "pool_size": len(pool),
            "contributions": scored,
            "run_at": datetime.now().isoformat(),
        }

    def run_gate_experiment(self, term: str, scores_with_gate: List[float],
                            scores_without_gate: List[float],
                            returns: List[float]) -> Dict[str, Any]:
        """Risk gate on/off对比实验"""
        n = len(scores_with_gate)
        if n == 0:
            return {"experiment_type": "gate_experiment", "error": "no data"}

        from src.eval.loss_engine import LossEngine
        engine = LossEngine(self.config)

        loss_gate_on = engine.compute_L_return(scores_with_gate, returns)
        loss_gate_off = engine.compute_L_return(scores_without_gate, returns)

        # 错杀率：被gate截断（gate_on score < gate_off score）但实际收益>0
        false_kills = sum(1 for sg, so, r in zip(scores_with_gate, scores_without_gate, returns)
                          if sg < so and r > 0)
        total_kills = sum(1 for sg, so in zip(scores_with_gate, scores_without_gate) if sg < so)

        # 漏放率：未被gate截断但实际收益<-10%
        missed = sum(1 for sg, r in zip(scores_with_gate, returns) if sg >= 50 and r < -0.10)

        return {
            "experiment_type": "gate_experiment",
            "term": term,
            "L_gate_on": round(loss_gate_on["L_return"], 4),
            "L_gate_off": round(loss_gate_off["L_return"], 4),
            "delta_L": round(loss_gate_off["L_return"] - loss_gate_on["L_return"], 4),
            "false_kill_rate": round(false_kills / max(total_kills, 1), 4),
            "miss_rate": round(missed / max(n, 1), 4),
            "sample_size": n,
        }

    def run_consistency_test(self, term: str, scores_run1: List[float],
                              scores_run2: List[float]) -> Dict[str, Any]:
        """一致性测试：同一输入重复运行，观察输出波动"""
        n = min(len(scores_run1), len(scores_run2))
        if n == 0:
            return {"experiment_type": "consistency", "error": "no data"}

        diffs = [abs(scores_run1[i] - scores_run2[i]) for i in range(n)]
        mean_diff = sum(diffs) / n

        # Action flip rate
        flips = sum(1 for i in range(n)
                    if (scores_run1[i] >= 50) != (scores_run2[i] >= 50))

        # Rank overlap (top K)
        k = min(10, n)
        top1 = set(sorted(range(n), key=lambda i: scores_run1[i], reverse=True)[:k])
        top2 = set(sorted(range(n), key=lambda i: scores_run2[i], reverse=True)[:k])
        overlap = len(top1 & top2) / k

        return {
            "experiment_type": "consistency",
            "term": term,
            "mean_score_diff": round(mean_diff, 2),
            "action_flip_rate": round(flips / n, 4),
            "top_k_overlap": round(overlap, 4),
            "sample_size": n,
        }

    def run_fidelity_test(self, term: str, scores_fast: List[float],
                           scores_production: List[float]) -> Dict[str, Any]:
        """保真度测试：快模型 vs 生产模型"""
        return self.run_consistency_test(term, scores_fast, scores_production)
