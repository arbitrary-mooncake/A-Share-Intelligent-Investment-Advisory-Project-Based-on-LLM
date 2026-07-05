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
        """保真度测试：评测模型(M5) vs 生产模型(M1/M3)的4维度保真度分析。

        使用 FidelityEngine 计算完整的4维保真度指标：
          - action_flip_rate: 动作翻转率
          - score_drift: 评分漂移
          - topK_overlap: Top-K重叠率
          - rank_drift: 排名漂移 (1.0 - Spearman ρ)

        Args:
            term: short/medium/long
            scores_fast: 评测模型(M5)的评分列表
            scores_production: 生产模型(M1/M3)的评分列表

        Returns:
            dict: 完整的4维保真度测试结果，可被 loss_engine.compute_L_consistency() 消费
        """
        n = min(len(scores_fast), len(scores_production))
        if n == 0:
            return {
                "experiment_type": "fidelity",
                "term": term,
                "error": "no data",
                "sample_size": 0,
            }

        # Also compute basic consistency metrics (score diff, flip rate, top-K overlap)
        diffs = [abs(scores_fast[i] - scores_production[i]) for i in range(n)]
        mean_diff = sum(diffs) / n

        flips = sum(1 for i in range(n)
                    if (scores_fast[i] >= 50) != (scores_production[i] >= 50))
        action_flip_rate = flips / n

        k = min(10, n)
        top_fast = set(sorted(range(n), key=lambda i: scores_fast[i], reverse=True)[:k])
        top_prod = set(sorted(range(n), key=lambda i: scores_production[i], reverse=True)[:k])
        top_k_overlap = len(top_fast & top_prod) / k if k > 0 else 1.0

        # Compute rank drift using Spearman correlation
        from src.eval.loss_engine import spearman_rank_correlation
        rho = spearman_rank_correlation(scores_fast[:n], scores_production[:n])
        rank_drift = max(0.0, min(1.0, 1.0 - rho))

        # Build snapshots in FidelityEngine-compatible format for true 4-dim computation
        try:
            from src.eval.fidelity_engine import FidelityEngine
            fe = FidelityEngine(self.config)
            snapshots = [
                {
                    "period": "eval_model",
                    "symbols": [f"s{i}" for i in range(n)],
                    "scores": scores_fast[:n],
                    "actions": ["buy" if s >= 50 else "sell" for s in scores_fast[:n]],
                },
                {
                    "period": "production_model",
                    "symbols": [f"s{i}" for i in range(n)],
                    "scores": scores_production[:n],
                    "actions": ["buy" if s >= 50 else "sell" for s in scores_production[:n]],
                },
            ]
            fidelity_result = fe.compute_fidelity_loss(snapshots)
        except Exception:
            fidelity_result = None

        result = {
            "experiment_type": "fidelity",
            "term": term,
            "mean_score_diff": round(mean_diff, 2),
            "action_flip_rate": round(action_flip_rate, 4),
            "top_k_overlap": round(top_k_overlap, 4),
            "rank_drift": round(rank_drift, 4),
            "spearman_rho": round(rho, 4),
            "sample_size": n,
        }

        # Merge in full FidelityEngine result if available
        if fidelity_result:
            result.update({
                "fidelity_loss": fidelity_result.get("fidelity_loss", 0.0),
                "fidelity_action_flip_rate": fidelity_result.get("action_flip_rate", 0.0),
                "fidelity_score_drift": fidelity_result.get("score_drift", 0.0),
                "fidelity_topK_overlap": fidelity_result.get("topK_overlap", 1.0),
                "fidelity_rank_drift": fidelity_result.get("rank_drift", 0.0),
                "fidelity_warnings": fidelity_result.get("warnings", []),
            })

        return result
