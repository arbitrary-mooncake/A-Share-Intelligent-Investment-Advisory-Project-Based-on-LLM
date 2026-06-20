"""
优化路由器 — 根据问题类型路由到自动修复/半自动/全人工处理。
问题分类: PARAM_TUNE / PROMPT_PATCH / LOGIC_FIX / ARCH_CHANGE / RESEARCH
"""
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.eval.optimizer.anti_hallucination import quick_verify


class OptimizeRouter:
    """优化路由分类器"""

    # 自动修复白名单路径
    AUTO_WHITELIST_PATHS = [
        "config/eval/",
        "src/utils/model_config.py",
    ]

    # 禁止自动修复的路径模式
    AUTO_BLACKLIST_PATTERNS = [
        "agent",
        "prompt",
        "data_source",
        "pipeline",
        "main.py",
    ]

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.autofix_max_files = self.config.get("autofix_max_files", 3)

    def classify_issue(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据证据数据分类问题类型。

        Args:
            evidence: 包含loss数据、agent贡献、错误日志等

        Returns:
            {"ticket_type": "PARAM_TUNE", "route": "auto", "confidence": "HIGH", ...}
        """
        issue_type = evidence.get("type", "")
        affected_files = evidence.get("affected_files", [])
        complexity = evidence.get("complexity", "medium")

        # 默认分类
        ticket_type = "RESEARCH"
        route = "manual"

        # 分类逻辑
        if issue_type == "agent_negative_contribution":
            # Agent负贡献 → 需要研究
            ticket_type = "RESEARCH"
            route = "manual"

        elif issue_type == "parameter_suboptimal":
            # 参数次优 → 可自动调参
            ticket_type = "PARAM_TUNE"
            route = "auto"

        elif issue_type == "prompt_quality":
            # Prompt质量问题 → 半自动
            ticket_type = "PROMPT_PATCH"
            route = "semi_auto"

        elif issue_type == "logic_bug":
            # 代码逻辑问题
            if self._can_autofix(affected_files, complexity):
                ticket_type = "LOGIC_FIX"
                route = "semi_auto"
            else:
                ticket_type = "ARCH_CHANGE"
                route = "manual"

        elif issue_type == "architecture_issue":
            ticket_type = "ARCH_CHANGE"
            route = "manual"

        return {
            "ticket_type": ticket_type,
            "route": route,
            "confidence": "MEDIUM",
            "classification_reason": f"基于问题类型'{issue_type}'和复杂度'{complexity}'分类",
        }

    def _can_autofix(self, affected_files: List[str], complexity: str) -> bool:
        """判断是否可以自动修复"""
        if not affected_files:
            return False
        if len(affected_files) > self.autofix_max_files:
            return False
        if complexity == "high":
            return False

        for f in affected_files:
            # 检查白名单
            in_whitelist = any(f.startswith(w) for w in self.AUTO_WHITELIST_PATHS)
            # 检查黑名单
            in_blacklist = any(b in f for b in self.AUTO_BLACKLIST_PATTERNS)

            if not in_whitelist or in_blacklist:
                return False

        return True

    def generate_ticket(self, batch_id: str, evidence: Dict[str, Any],
                        classification: Dict[str, Any]) -> Dict[str, Any]:
        """生成优化ticket"""
        # Anti-hallucination verification (总纲 §12)
        # Only run on LLM-generated evidence (skip for metadata-only evidence)
        _llm_keys = {"diagnosis", "optimization_suggestions", "narrative_summary", "top_findings"}
        if _llm_keys & set(evidence.keys()):
            try:
                verification = quick_verify(
                    llm_output=json.dumps(evidence, ensure_ascii=False) if isinstance(evidence, dict) else str(evidence),
                    source_data=evidence if isinstance(evidence, dict) else {},
                )
                if verification.get("confidence", {}).get("level") == "LOW":
                    # Downgrade auto-fix to semi-auto when confidence is low
                    if classification.get("route") == "auto":
                        classification["route"] = "semi_auto"
                        classification["note"] = classification.get("note", "") + "; anti-hallucination: LOW confidence, downgraded from auto"
            except Exception:
                pass  # Non-critical: proceed without verification

        return {
            "batch_id": batch_id,
            "ticket_type": classification["ticket_type"],
            "route": classification["route"],
            "title": evidence.get("title", "未命名问题"),
            "summary": evidence.get("summary", ""),
            "severity": evidence.get("severity", "medium"),
            "evidence_json": json.dumps(evidence, ensure_ascii=False, default=str),
            "note": classification.get("note", ""),
            "created_at": datetime.now().isoformat(),
            "suggested_actions": self._get_suggested_actions(classification),
        }

    def _get_suggested_actions(self, classification: Dict[str, Any]) -> List[str]:
        """根据分类生成建议操作"""
        ticket_type = classification["ticket_type"]
        route = classification["route"]

        if route == "auto":
            return ["系统将自动执行参数优化", "修改后自动运行回归验证", "通过则保留，失败则回滚"]
        elif route == "semi_auto":
            return ["系统将生成修改建议供您审核", "审核通过后自动执行", "您也可以拒绝并自行修改"]
        else:
            return ["系统将生成详细分析报告", "请根据报告手动修改", "修改后可重新运行检查验证"]
