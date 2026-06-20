"""
保真度引擎 — 检测系统评分与实际市场结果的一致性偏差。
严格按照总纲 §9.5 实现。
"""
from typing import Dict, Any, List, Optional
from datetime import datetime


class FidelityEngine:
    """保真度评估引擎"""

    def compute_fidelity_loss(
        self,
        snapshots: List[Dict[str, Any]],
        labels: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute fidelity metrics between predictions and realized outcomes.

        总纲 §9.5 四维度:
        - action_flip_rate: 连续两期间推荐动作翻转率
        - score_drift: 连续两期间评分的平均绝对差异
        - topK_overlap: Top-K推荐的Jaccard重叠率
        - rank_drift: 排名的Spearman相关系数

        Returns:
            {action_flip_rate, score_drift, topK_overlap, rank_drift,
             fidelity_loss, warnings: [...]}
        """
        result = {
            "action_flip_rate": self._compute_action_flip_rate(snapshots),
            "score_drift": self._compute_score_drift(snapshots),
            "topK_overlap": self._compute_topK_overlap(snapshots, k=10),
            "rank_drift": self._compute_rank_drift(snapshots),
            "computed_at": datetime.now().isoformat(),
        }

        # 综合保真度Loss (等权)
        result["fidelity_loss"] = (
            result["action_flip_rate"] * 0.25
            + (1.0 - result["topK_overlap"]) * 0.25
            + min(result["score_drift"] / 100.0, 1.0) * 0.25
            + (1.0 - result["rank_drift"]) * 0.25
        )

        result["warnings"] = self._check_warnings(result)
        return result

    def _compute_action_flip_rate(self, snapshots: list) -> float:
        """连续两期间推荐动作翻转率 (buy↔sell)"""
        if len(snapshots) < 2:
            return 0.0

        flips = 0
        total = 0
        for i in range(len(snapshots) - 1):
            a1 = snapshots[i].get("action", "")
            a2 = snapshots[i + 1].get("action", "")
            if a1 and a2 and a1 != a2:
                flips += 1
            total += 1

        return flips / max(total, 1)

    def _compute_score_drift(self, snapshots: list) -> float:
        """连续两期间评分的平均绝对差异"""
        if len(snapshots) < 2:
            return 0.0

        diffs = []
        for i in range(len(snapshots) - 1):
            s1 = snapshots[i].get("score", 0)
            s2 = snapshots[i + 1].get("score", 0)
            if s1 is not None and s2 is not None:
                diffs.append(abs(s1 - s2))

        return sum(diffs) / max(len(diffs), 1)

    def _compute_topK_overlap(self, snapshots: list, k: int = 10) -> float:
        """Top-K推荐的Jaccard重叠率"""
        if len(snapshots) < 2:
            return 1.0

        overlaps = []
        for i in range(len(snapshots) - 1):
            codes1 = set(self._get_top_codes(snapshots[i], k))
            codes2 = set(self._get_top_codes(snapshots[i + 1], k))
            if codes1 or codes2:
                intersection = len(codes1 & codes2)
                union = len(codes1 | codes2)
                overlaps.append(intersection / max(union, 1))

        return sum(overlaps) / max(len(overlaps), 1)

    def _compute_rank_drift(self, snapshots: list) -> float:
        """排名的Spearman相关系数 (使用scipy如果可用，否则简化为Kendall tau近似)"""
        if len(snapshots) < 2:
            return 1.0

        try:
            import scipy.stats as stats
            correlations = []
            for i in range(len(snapshots) - 1):
                codes1 = self._get_top_codes(snapshots[i], 20)
                codes2 = self._get_top_codes(snapshots[i + 1], 20)
                common = [c for c in codes1 if c in codes2]
                if len(common) >= 5:
                    r1 = [codes1.index(c) for c in common]
                    r2 = [codes2.index(c) for c in common]
                    corr, _ = stats.spearmanr(r1, r2)
                    correlations.append(max(corr, 0))
            return sum(correlations) / max(len(correlations), 1) if correlations else 0.5
        except ImportError:
            # Simplified: use overlap-based approximation
            return self._compute_topK_overlap(snapshots, k=15)

    def _get_top_codes(self, snapshot: dict, k: int) -> list:
        """从snapshot中提取Top-K股票代码"""
        if "top_codes" in snapshot:
            return snapshot["top_codes"][:k]
        if "stock_code" in snapshot:
            return [snapshot["stock_code"]]
        return []

    def _check_warnings(self, fidelity_data: dict) -> List[str]:
        """检查保真度警告阈值"""
        warnings = []
        if fidelity_data.get("action_flip_rate", 0) > 0.15:
            warnings.append(
                f"⚠ 推荐动作翻转率 {fidelity_data['action_flip_rate']:.1%} > 15% — 系统推荐不稳定"
            )
        if fidelity_data.get("topK_overlap", 1) < 0.70:
            warnings.append(
                f"⚠ Top-K重叠率 {fidelity_data['topK_overlap']:.1%} < 70% — 推荐一致性不足"
            )
        if fidelity_data.get("rank_drift", 1) < 0.50:
            warnings.append(
                f"⚠ 排名漂移 {fidelity_data['rank_drift']:.3f} < 0.50 — 排名一致性显著偏低"
            )
        return warnings
