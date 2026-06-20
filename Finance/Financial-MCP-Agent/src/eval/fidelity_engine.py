"""
保真度引擎 — 检测系统评分与实际市场结果的一致性偏差。

总纲 §9.5: 保真度 (Fidelity) 衡量系统输出的稳定性和一致性:
  - action_flip_rate:    连续周期之间推荐动作翻转的比例
  - score_drift:         连续周期之间评分的平均绝对差
  - topK_overlap:        连续周期之间Top-K推荐的重叠率 (Jaccard)
  - rank_drift:          连续周期之间排名的Spearman相关系数

同时还检测保真度告警:
  - action_flip_rate > 15%   → 推荐不稳定
  - topK_overlap < 70%       → Top选择不一致

总纲 §10.4: 短期线以周为单位Bootstrap保持周内结构 — 由 contribution_engine 实现。
"""
import math
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from src.eval.loss_engine import spearman_rank_correlation

logger = logging.getLogger(__name__)


# ──────────────────────────── Warning Thresholds ────────────────────────────

FIDELITY_WARNING_THRESHOLDS = {
    "action_flip_rate": 0.15,    # > 15% → unstable recommendations
    "score_drift": 15.0,          # > 15 points → volatile scoring
    "topK_overlap": 0.70,        # < 70% → inconsistent top picks
    "rank_drift": 0.30,           # Spearman ρ < 0.70 → significant rank change
    "fidelity_loss": 0.10,        # > 0.10 → overall fidelity concern
}


class FidelityEngine:
    """保真度引擎 — 检测系统评分与实际市场结果的一致性偏差。

    核心计算:
      - 对连续周期的快照计算4维保真度指标
      - 加权合成fidelity_loss
      - 检查是否触发告警阈值
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.k_value = self.config.get("top_k", 10)
        self.warnings = FIDELITY_WARNING_THRESHOLDS.copy()
        if "warning_thresholds" in self.config:
            self.warnings.update(self.config["warning_thresholds"])

    def compute_fidelity_loss(
        self,
        snapshots: List[dict],
        labels: List[dict] = None,
    ) -> Dict[str, Any]:
        """Compute fidelity metrics from consecutive snapshots.

        Args:
            snapshots: List of per-period snapshot dicts, each containing:
                - period: str or int (period identifier for ordering)
                - scores: List[float] — scores for each asset in this period
                - actions: List[str] — buy/sell/hold for each asset
                - symbols: List[str] — asset identifiers
                - rankings: List[int] — 1-based ranking (optional, computed from scores)
            labels: Optional list of realized labels (accepted for API compatibility).

        Returns:
            {
                "action_flip_rate": float,      # 0.0 ~ 1.0
                "score_drift": float,            # mean absolute score difference
                "topK_overlap": float,           # Jaccard overlap of top-K
                "rank_drift": float,             # 1.0 - Spearman ρ
                "fidelity_loss": float,          # weighted composite
                "period_pairs_analyzed": int,    # number of consecutive pairs
                "warnings": List[str],           # triggered warnings
                "computed_at": str,              # ISO datetime
            }
        """
        # Handle legacy format: list of individual snapshots (one per stock per date)
        if snapshots and not isinstance(snapshots[0].get("scores"), list):
            result = self._compute_fidelity_legacy(snapshots)
            return result

        if not snapshots or len(snapshots) < 2:
            return self._empty_result("至少需要2个周期的快照数据")

        sorted_snaps = sorted(snapshots, key=lambda s: str(s.get("period", "")))

        n = len(sorted_snaps)
        flip_rates = []
        score_drifts = []
        topk_overlaps = []
        rank_drifts = []

        for i in range(n - 1):
            prev = sorted_snaps[i]
            curr = sorted_snaps[i + 1]
            flip_rates.append(self._compute_action_flip_rate(prev, curr))
            score_drifts.append(self._compute_score_drift(prev, curr))
            topk_overlaps.append(self._compute_topk_overlap(prev, curr))
            rank_drifts.append(self._compute_rank_drift(prev, curr))

        avg_flip_rate = _safe_mean(flip_rates)
        avg_score_drift = _safe_mean(score_drifts)
        avg_topk_overlap = _safe_mean(topk_overlaps)
        avg_rank_drift = _safe_mean(rank_drifts)

        fidelity_loss = (
            0.40 * avg_flip_rate +
            0.15 * min(avg_score_drift / 100.0, 1.0) +
            0.25 * (1.0 - avg_topk_overlap) +
            0.20 * avg_rank_drift
        )

        warnings_list = self.check_fidelity_warnings({
            "action_flip_rate": avg_flip_rate,
            "score_drift": avg_score_drift,
            "topK_overlap": avg_topk_overlap,
            "rank_drift": avg_rank_drift,
            "fidelity_loss": fidelity_loss,
        })

        return {
            "action_flip_rate": round(avg_flip_rate, 4),
            "score_drift": round(avg_score_drift, 2),
            "topK_overlap": round(avg_topk_overlap, 4),
            "rank_drift": round(avg_rank_drift, 4),
            "fidelity_loss": round(fidelity_loss, 4),
            "period_pairs_analyzed": n - 1,
            "warnings": warnings_list,
            "computed_at": datetime.now().isoformat(),
        }

    def _compute_fidelity_legacy(self, snapshots: list) -> Dict[str, Any]:
        """Handle legacy format: list of individual snapshots (one per stock per date)."""
        if len(snapshots) < 2:
            return self._empty_result("至少需要2个快照")

        # Compute simple pairwise metrics
        flips = 0
        total = 0
        diffs = []
        codes_list = []

        for i in range(len(snapshots) - 1):
            a1 = snapshots[i].get("action", "")
            a2 = snapshots[i + 1].get("action", "")
            if a1 and a2 and a1 != a2:
                flips += 1
            total += 1

            s1 = snapshots[i].get("score", 0)
            s2 = snapshots[i + 1].get("score", 0)
            if s1 is not None and s2 is not None:
                diffs.append(abs(s1 - s2))

            code = snapshots[i].get("stock_code", snapshots[i].get("symbol", ""))
            codes_list.append(code)

        action_flip_rate = flips / max(total, 1)
        score_drift = sum(diffs) / max(len(diffs), 1) if diffs else 0.0

        # Top-K overlap from codes
        unique_codes = list(dict.fromkeys(codes_list))
        mid = len(unique_codes) // 2
        if mid > 0:
            topK_overlap = len(set(unique_codes[:mid]) & set(unique_codes[mid:])) / max(mid, 1)
        else:
            topK_overlap = 1.0

        rank_drift = 0.5  # Cannot compute rank drift from legacy format

        fidelity_loss = (
            action_flip_rate * 0.25 +
            (1.0 - topK_overlap) * 0.25 +
            min(score_drift / 100.0, 1.0) * 0.25 +
            (1.0 - rank_drift) * 0.25
        )

        fidelity_data = {
            "action_flip_rate": action_flip_rate,
            "score_drift": score_drift,
            "topK_overlap": topK_overlap,
            "rank_drift": rank_drift,
            "fidelity_loss": fidelity_loss,
        }

        return {
            "action_flip_rate": round(action_flip_rate, 4),
            "score_drift": round(score_drift, 2),
            "topK_overlap": round(topK_overlap, 4),
            "rank_drift": round(rank_drift, 4),
            "fidelity_loss": round(fidelity_loss, 4),
            "period_pairs_analyzed": max(total, 0),
            "warnings": self.check_fidelity_warnings(fidelity_data),
            "computed_at": datetime.now().isoformat(),
        }

    def _compute_action_flip_rate(self, prev: dict, curr: dict) -> float:
        """How often does recommendation flip between consecutive periods."""
        prev_actions = prev.get("actions", [])
        curr_actions = curr.get("actions", [])
        prev_symbols = prev.get("symbols", [])
        curr_symbols = curr.get("symbols", [])

        n = min(len(prev_actions), len(curr_actions))
        if n == 0:
            return 0.0

        if prev_symbols and curr_symbols:
            symbol_to_idx_prev = {s: i for i, s in enumerate(prev_symbols)}
            flips = 0
            aligned = 0
            for i, s in enumerate(curr_symbols):
                if s in symbol_to_idx_prev:
                    prev_idx = symbol_to_idx_prev[s]
                    if (prev_idx < len(prev_actions) and
                            prev_actions[prev_idx] != curr_actions[i]):
                        flips += 1
                    aligned += 1
            return flips / max(aligned, 1)

        flips = sum(1 for i in range(n) if prev_actions[i] != curr_actions[i])
        return flips / n

    def _compute_score_drift(self, prev: dict, curr: dict) -> float:
        """Mean absolute difference between consecutive scores."""
        prev_scores = prev.get("scores", [])
        curr_scores = curr.get("scores", [])
        n = min(len(prev_scores), len(curr_scores))
        if n == 0:
            return 0.0
        total_diff = sum(abs(prev_scores[i] - curr_scores[i]) for i in range(n))
        return total_diff / n

    def _compute_topk_overlap(self, prev: dict, curr: dict) -> float:
        """Jaccard overlap of top-K recommendations between periods."""
        k = self.k_value
        prev_symbols = prev.get("symbols", [])
        curr_symbols = curr.get("symbols", [])
        prev_scores = prev.get("scores", [])
        curr_scores = curr.get("scores", [])

        prev_topk = self._get_topk_symbols(prev_symbols, prev_scores, k)
        curr_topk = self._get_topk_symbols(curr_symbols, curr_scores, k)

        if not prev_topk and not curr_topk:
            return 1.0
        if not prev_topk or not curr_topk:
            return 0.0

        intersection = len(prev_topk & curr_topk)
        union = len(prev_topk | curr_topk)
        return intersection / max(union, 1)

    def _get_topk_symbols(self, symbols: List[str], scores: List[float],
                           k: int) -> set:
        """Get the set of top-K symbols by score."""
        if not symbols or not scores:
            return set()
        paired = list(zip(symbols, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        return set(s for s, _ in paired[:k])

    def _compute_rank_drift(self, prev: dict, curr: dict) -> float:
        """Drift measured as 1.0 - Spearman correlation of rankings.

        Returns 0.0 when rankings are identical, approaches 1.0 as they diverge.
        """
        prev_scores = prev.get("scores", [])
        curr_scores = curr.get("scores", [])
        n = min(len(prev_scores), len(curr_scores))
        if n < 2:
            return 0.0
        rho = spearman_rank_correlation(prev_scores[:n], curr_scores[:n])
        return max(0.0, min(1.0, 1.0 - rho))

    def check_fidelity_warnings(self, fidelity_data: dict) -> List[str]:
        """Check if fidelity metrics exceed warning thresholds.

        Args:
            fidelity_data: Output from compute_fidelity_loss(), or a dict with
                the same key structure.

        Returns:
            List of warning strings. Empty list = no warnings triggered.
        """
        warnings = []

        flip = fidelity_data.get("action_flip_rate", 0.0)
        if flip > self.warnings["action_flip_rate"]:
            warnings.append(
                f"动作翻转率 {flip:.1%} 超过阈值 {self.warnings['action_flip_rate']:.0%} — "
                f"推荐信号不稳定，建议检查短期agent的信号一致性"
            )

        drift = fidelity_data.get("score_drift", 0.0)
        if drift > self.warnings["score_drift"]:
            warnings.append(
                f"评分漂移 {drift:.1f} 超过阈值 {self.warnings['score_drift']:.1f} — "
                f"评分波动过大，建议检查评分器的校准"
            )

        overlap = fidelity_data.get("topK_overlap", 1.0)
        if overlap < self.warnings["topK_overlap"]:
            warnings.append(
                f"Top-K重叠率 {overlap:.1%} 低于阈值 {self.warnings['topK_overlap']:.0%} — "
                f"顶级选股在周期间不一致，建议检查排序稳定性"
            )

        rank_d = fidelity_data.get("rank_drift", 0.0)
        if rank_d > self.warnings["rank_drift"]:
            warnings.append(
                f"排名漂移 {rank_d:.2f} 超过阈值 {self.warnings['rank_drift']:.2f} — "
                f"资产排名在周期间大幅变动，建议检查各agent权重的稳定性"
            )

        f_loss = fidelity_data.get("fidelity_loss", 0.0)
        if f_loss > self.warnings["fidelity_loss"]:
            warnings.append(
                f"综合保真度损失 {f_loss:.3f} 超过阈值 {self.warnings['fidelity_loss']:.3f} — "
                f"系统整体输出稳定性需关注"
            )

        return warnings

    def _empty_result(self, reason: str = "") -> Dict[str, Any]:
        """Return an empty/neutral result when computation is not possible."""
        return {
            "action_flip_rate": 0.0,
            "score_drift": 0.0,
            "topK_overlap": 1.0,
            "rank_drift": 0.0,
            "fidelity_loss": 0.0,
            "period_pairs_analyzed": 0,
            "warnings": [reason] if reason else [],
            "computed_at": datetime.now().isoformat(),
        }

    def compute_fidelity_loss_from_snapshots(
        self,
        snapshots_obj_list: List[Any],
    ) -> Dict[str, Any]:
        """Compute fidelity loss from PredictionSnapshot objects.

        Converts PredictionSnapshot dataclass instances to the dict format
        expected by compute_fidelity_loss().
        """
        from collections import defaultdict
        period_groups = defaultdict(list)

        for snap in snapshots_obj_list:
            if isinstance(snap, dict):
                as_of = snap.get("as_of_date", "")
                symbol = snap.get("symbol", "")
                score = snap.get("score", 0.0)
                action = snap.get("action", "")
            else:
                as_of = getattr(snap, "as_of_date", "")
                symbol = getattr(snap, "symbol", "")
                score = getattr(snap, "score", 0.0)
                action = getattr(snap, "action", "")

            if as_of and symbol:
                period_groups[as_of].append({
                    "symbol": symbol,
                    "score": score,
                    "action": action,
                })

        snapshots = []
        for period in sorted(period_groups.keys()):
            group = period_groups[period]
            snapshots.append({
                "period": period,
                "symbols": [g["symbol"] for g in group],
                "scores": [g["score"] for g in group],
                "actions": [g["action"] for g in group],
            })

        return self.compute_fidelity_loss(snapshots)


# ──────────────────────────── Helpers ────────────────────────────

def _safe_mean(values: List[float]) -> float:
    """Compute mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)
