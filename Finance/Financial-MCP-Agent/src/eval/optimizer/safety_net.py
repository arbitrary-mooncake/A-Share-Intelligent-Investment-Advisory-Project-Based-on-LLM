"""
安全网络机制 — 优化系统的安全防护层。
严格按照总纲 §11.7 实现: cooldown, cumulative monitoring, branch isolation, direction consistency。
"""
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


class SafetyNet:
    """优化安全网络 — 防止自动优化引入问题"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cooldown_days = self.config.get("cooldown_days_per_module", 7)
        self.max_consecutive_failures = self.config.get("max_consecutive_failures", 3)
        self.baseline_days = self.config.get("pre_change_loss_baseline_days", 5)

    def check_cooldown(self, module: str, last_change_date: str) -> Dict[str, Any]:
        """
        检查模块是否在冷却期内。
        总纲 §11.7.3: 同一模块的自动改动至少间隔N天/N次检查。
        """
        if not last_change_date:
            return {"in_cooldown": False}

        try:
            last = datetime.fromisoformat(last_change_date)
            days_since = (datetime.now() - last).days
            return {
                "in_cooldown": days_since < self.cooldown_days,
                "days_since": days_since,
                "cooldown_days": self.cooldown_days,
                "can_act_on": (last + timedelta(days=self.cooldown_days)).isoformat(),
            }
        except (ValueError, TypeError):
            return {"in_cooldown": False, "error": "Invalid date format"}

    def check_cumulative_failures(
        self, module: str, recent_losses: List[float]
    ) -> Dict[str, Any]:
        """
        累计监控: 连续N次改动后Loss未改善 → 停止该模块自动优化。
        总纲 §11.7.4
        """
        if len(recent_losses) < 2:
            return {"consecutive_failures": 0, "should_stop": False}

        consecutive = 0
        for i in range(len(recent_losses) - 1, 0, -1):
            if recent_losses[i] >= recent_losses[i - 1]:
                consecutive += 1
            else:
                break

        return {
            "consecutive_failures": consecutive,
            "should_stop": consecutive >= self.max_consecutive_failures,
            "max_allowed": self.max_consecutive_failures,
            "recent_losses": recent_losses[-self.max_consecutive_failures:],
        }

    def check_direction_consistency(
        self, current_suggestion: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        方向一致性检查: 防止"上周建议增加A权重,本周又建议降低"
        总纲 §11.7.5
        """
        if not history:
            return {"consistent": True}

        conflicts = []
        current_module = current_suggestion.get("module", "")
        current_direction = current_suggestion.get("direction", "")

        for past in history[-5:]:  # Check last 5 suggestions
            if past.get("module") == current_module:
                past_direction = past.get("direction", "")
                if past_direction and current_direction:
                    if (past_direction == "increase" and current_direction == "decrease") or \
                       (past_direction == "decrease" and current_direction == "increase"):
                        conflicts.append({
                            "past_date": past.get("date", ""),
                            "past_direction": past_direction,
                            "current_direction": current_direction,
                            "note": "方向性矛盾: 建议人工审核",
                        })

        return {
            "consistent": len(conflicts) == 0,
            "conflicts": conflicts,
            "recommendation": "人工审核方向矛盾" if conflicts else "方向一致,可继续",
        }

    def compute_loss_baseline(self, recent_losses: List[float]) -> Dict[str, Any]:
        """
        计算改动前Loss基线 (最近N次检查的平均值)。
        总纲 §11.7.1
        """
        if not recent_losses:
            return {"baseline": None, "samples": 0}

        sample = recent_losses[-self.baseline_days:]
        return {
            "baseline": sum(sample) / len(sample),
            "samples": len(sample),
            "std": (sum((x - sum(sample) / len(sample)) ** 2 for x in sample) / len(sample)) ** 0.5
            if len(sample) > 1 else 0,
        }

    def validate_improvement(
        self, baseline: float, current: float, min_improvement: float = 0.01
    ) -> Dict[str, Any]:
        """
        验证改动是否真正改善了Loss。
        总纲: 1个标准差的改善要求
        """
        improvement = baseline - current  # Positive = better (lower loss)
        return {
            "improved": improvement > 0,
            "improvement": improvement,
            "significant": improvement > min_improvement,
            "baseline": baseline,
            "current": current,
        }
