"""
人工优化包构建器 — 生成可直接用于手动修改或与Claude Code交互的结构化优化包。
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from .router import OptimizeRouter


MANUAL_PKG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "eval", "manual_packages"
)


class ManualPackageBuilder:
    """人工优化包生成器"""

    def __init__(self):
        os.makedirs(MANUAL_PKG_DIR, exist_ok=True)

    def build_package(self, ticket: Dict[str, Any], evidence: Dict[str, Any],
                      contributions: List[Dict] = None) -> Dict[str, Any]:
        """构建完整的人工优化包"""
        pkg = {
            "title": ticket.get("title", "优化建议"),
            "ticket_type": ticket.get("ticket_type", "RESEARCH"),
            "severity": ticket.get("severity", "medium"),
            "generated_at": datetime.now().isoformat(),
            "batch_id": ticket.get("batch_id", ""),
            "problem_definition": {
                "summary": ticket.get("summary", ""),
                "evidence": evidence,
            },
            "impact_assessment": self._assess_impact(ticket, contributions or []),
            "suggested_direction": self._suggest_direction(ticket),
            "affected_modules": evidence.get("affected_modules", []),
            "related_files": evidence.get("affected_files", []),
            "suggested_tests": self._suggest_tests(ticket),
            "expected_acceptance_criteria": self._define_criteria(ticket),
            "risk_warnings": self._assess_risks(ticket),
        }
        return pkg

    def save_package(self, pkg: Dict[str, Any]) -> str:
        """保存优化包到文件"""
        filename = f"manual_pkg_{pkg['ticket_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        md_path = os.path.join(MANUAL_PKG_DIR, f"{filename}.md")
        json_path = os.path.join(MANUAL_PKG_DIR, f"{filename}.json")

        # Markdown
        md = self._to_markdown(pkg)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)

        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pkg, f, ensure_ascii=False, indent=2, default=str)

        return md_path

    def _assess_impact(self, ticket: Dict, contributions: List[Dict]) -> Dict:
        return {
            "estimated_loss_improvement": "需要进一步分析",
            "affected_agents": [c.get("agent_name") for c in contributions if abs(c.get("delta_L_total", 0)) > 0.01],
            "scope": "单模块" if len(ticket.get("summary", "")) < 100 else "跨模块",
        }

    def _suggest_direction(self, ticket: Dict) -> List[str]:
        ticket_type = ticket.get("ticket_type", "")
        if ticket_type == "PARAM_TUNE":
            return ["调整相关配置参数", "在回测验证新参数", "确认改善后永久应用"]
        elif ticket_type == "PROMPT_PATCH":
            return ["修改相关agent的system prompt", "在回测环境验证prompt变更", "对比新旧prompt输出质量"]
        elif ticket_type == "LOGIC_FIX":
            return ["定位具体代码逻辑问题", "编写修复代码", "补充分单元测试", "运行回归验证"]
        else:
            return ["深入分析问题根因", "设计改进方案", "在验证环境测试", "逐步推广到生产"]

    def _suggest_tests(self, ticket: Dict) -> List[str]:
        return [
            "单元测试：验证修改后的函数行为正确",
            "回测验证：在历史数据上对比修改前后效果",
            "回归测试：确保现有功能不受影响",
        ]

    def _define_criteria(self, ticket: Dict) -> List[str]:
        return [
            "修改后Loss不高于修改前",
            "所有现有测试通过",
            "消融实验结果方向正确",
        ]

    def _assess_risks(self, ticket: Dict) -> List[str]:
        risks = ["修改可能引入新的bug", "优化方向可能在实盘中不成立"]
        if ticket.get("ticket_type") == "ARCH_CHANGE":
            risks.append("架构变更影响面大，需充分测试")
        return risks

    def _to_markdown(self, pkg: Dict) -> str:
        lines = [
            f"# 优化建议: {pkg['title']}",
            "",
            f"**类型**: {pkg['ticket_type']} | **严重度**: {pkg['severity']}",
            f"**批次**: {pkg.get('batch_id', '')} | **生成时间**: {pkg['generated_at']}",
            "",
            "## 问题诊断",
            pkg['problem_definition']['summary'],
            "",
            "## 影响评估",
            f"- 预估Loss改善: {pkg['impact_assessment']['estimated_loss_improvement']}",
            f"- 影响范围: {pkg['impact_assessment']['scope']}",
            "",
            "## 建议方向",
        ]
        for d in pkg['suggested_direction']:
            lines.append(f"- {d}")

        lines.extend([
            "",
            "## 相关文件",
        ])
        for f in pkg.get('related_files', []):
            lines.append(f"- `{f}`")

        lines.extend([
            "",
            "## 建议测试",
        ])
        for t in pkg.get('suggested_tests', []):
            lines.append(f"- {t}")

        lines.extend([
            "",
            "## 验收标准",
        ])
        for c in pkg.get('expected_acceptance_criteria', []):
            lines.append(f"- {c}")

        lines.extend([
            "",
            "## 风险提示",
        ])
        for r in pkg.get('risk_warnings', []):
            lines.append(f"- {r}")

        return "\n".join(lines)
