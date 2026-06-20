"""
LLM报告撰写Agent — DeepSeek V4 Pro + 反幻觉验证。

总纲 §11: 评测系统需生成三类自然语言报告：
  1. 批次综合报告 (write_batch_report)  — 从batch数据生成完整评测报告
  2. Agent贡献分析报告 (write_agent_contribution_report) — 分析各agent的边际贡献
  3. 优化建议报告 (write_optimization_recommendation) — 基于loss和贡献模式生成优化建议

所有LLM输出经 anti_hallucination.quick_verify() 验证后返回，LOW置信度自动降级。
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from openai import OpenAI

from src.utils.model_config import get_eval_model_config
from src.eval.optimizer.anti_hallucination import quick_verify

logger = logging.getLogger(__name__)


@dataclass
class ReportOutput:
    """报告撰写输出。"""
    text: str
    passed_verification: bool = True
    confidence: str = "HIGH"  # HIGH / MEDIUM / LOW
    verification_issues: List[str] = field(default_factory=list)
    generated_at: str = ""


REPORT_SYSTEM_PROMPT = """你是一位严谨的量化投资研究员，负责撰写A股智能投顾评测系统的分析报告。

核心原则：
1. 所有数值引用必须来自输入数据，不得编造
2. 判断性陈述必须标注为[判断]，数据事实标注为[数据]
3. 发现矛盾时必须明确指出（如：Agent A声称正贡献但ΔL为负）
4. 禁止使用模糊表述（"可能""大概""似乎"），改为"基于XX数据，推断YY"
5. 置信度必须基于数据充分性声明：数据充足→高置信，部分缺失→中等，核心缺失→低置信

报告结构：
- 摘要：3-5句核心结论
- 详细分析：分维度展开
- 风险提示：明确的数据缺口和不确定性
- 建议：基于证据的可操作建议
"""


class ReportWriterAgent:
    """LLM报告撰写Agent — DeepSeek V4 Pro + 反幻觉验证。

    负责将评测系统的结构化数据转化为自然语言报告，
    所有输出经过 anti_hallucination.quick_verify() 验证。
    """

    def __init__(self, model_profile: str = "eval_orchestrator"):
        self.model_profile = model_profile
        self.model_config = get_eval_model_config(model_profile)
        self._client = None

    @property
    def client(self) -> OpenAI:
        """Lazy init OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.model_config["api_key"],
                base_url=self.model_config["base_url"],
            )
        return self._client

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  temperature: float = 0.3, max_tokens: int = 16000) -> str:
        """Call the LLM and return text response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_config["model_name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "enabled"}},
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def _verify_and_wrap(self, report_text: str, source_data: dict) -> ReportOutput:
        """Run anti-hallucination verification and return ReportOutput."""
        result = ReportOutput(
            text=report_text,
            generated_at=datetime.now().isoformat(),
        )

        if not report_text:
            result.passed_verification = False
            result.confidence = "LOW"
            result.verification_issues = ["LLM返回空文本"]
            return result

        try:
            verification = quick_verify(report_text, source_data)
            result.confidence = verification.get("confidence", {}).get("level", "MEDIUM")
            result.passed_verification = verification.get("overall_pass", True)

            # Collect issues from verification layers
            issues = []
            for v in verification.get("verifications", []):
                if not v.get("passed", True):
                    for issue in v.get("issues", []):
                        issues.append(f"[{v.get('layer', 'unknown')}] {issue}")
            if verification.get("confidence", {}).get("reasons"):
                issues.extend(verification["confidence"]["reasons"])
            result.verification_issues = issues

            # If LOW confidence, downgrade the report with a warning prefix
            if result.confidence == "LOW":
                warning = (
                    "\n\n---\n"
                    "**⚠️ 置信度警告：** 本报告的反幻觉验证置信度为LOW。"
                    "以下结论可能存在数据支撑不足的问题，请谨慎参考，"
                    "不要基于此报告触发任何自动修改操作。\n"
                    "---\n"
                )
                result.text = warning + result.text

        except Exception as e:
            logger.warning(f"Verification failed, returning unverified text: {e}")
            result.passed_verification = False
            result.confidence = "MEDIUM"
            result.verification_issues = [f"验证异常: {str(e)}"]

        return result

    # ──────────────────────────── Batch Report ────────────────────────────

    def write_batch_report(self, batch_data: dict) -> ReportOutput:
        """Generate a natural language evaluation report from batch data.

        Args:
            batch_data: 批次评测数据，包含:
                - batch_id: 批次ID
                - status: 批次状态
                - summary_metrics: 总览指标 (L_total, L_return, L_risk, L_structure)
                - term_breakdown: 各期限维度loss
                - line_results: 各条线的详细结果
                - market_regime: 市场环境描述
                - sample_info: 样本信息

        Returns:
            ReportOutput with text, verification status, and confidence.
        """
        user_prompt = self._build_batch_report_prompt(batch_data)
        report_text = self._call_llm(REPORT_SYSTEM_PROMPT, user_prompt)
        return self._verify_and_wrap(report_text, batch_data)

    def _build_batch_report_prompt(self, batch_data: dict) -> str:
        """构建批次报告的用户prompt。"""
        parts = []

        batch_id = batch_data.get("batch_id", "unknown")
        status = batch_data.get("status", "unknown")
        parts.append(f"# 批次评测报告任务\n\n批次ID: {batch_id}\n状态: {status}")

        # Summary metrics
        metrics = batch_data.get("summary_metrics", {})
        if metrics:
            parts.append("\n## [数据] 总览指标")
            parts.append(f"- L_total (综合损失): {metrics.get('L_total', 'N/A')}")
            parts.append(f"- L_return (收益损失): {metrics.get('L_return', 'N/A')}")
            parts.append(f"- L_risk (风险损失): {metrics.get('L_risk', 'N/A')}")
            parts.append(f"- L_structure (结构损失): {metrics.get('L_structure', 'N/A')}")
            parts.append(f"- 有效样本数: {metrics.get('sample_size', 'N/A')}")

        # Term breakdown
        term_breakdown = batch_data.get("term_breakdown", {})
        if term_breakdown:
            parts.append("\n## [数据] 分期限Loss明细")
            for term, detail in term_breakdown.items():
                parts.append(f"\n### {term}")
                if isinstance(detail, dict):
                    for k, v in detail.items():
                        parts.append(f"- {k}: {v}")

        # Line results
        line_results = batch_data.get("line_results", {})
        if line_results:
            parts.append("\n## [数据] 各条线结果")
            for line_id, line_detail in list(line_results.items())[:20]:
                if isinstance(line_detail, dict):
                    parts.append(
                        f"- {line_id}: "
                        f"score={line_detail.get('score', 'N/A')}, "
                        f"L_total={line_detail.get('L_total', 'N/A')}"
                    )

        # Market regime
        regime = batch_data.get("market_regime", "")
        if regime:
            parts.append(f"\n## [数据] 市场环境\n{regime}")

        # Contribution data (if available)
        contributions = batch_data.get("contributions", [])
        if contributions:
            parts.append("\n## [数据] Agent贡献排序")
            for c in contributions[:10]:
                if isinstance(c, dict):
                    parts.append(
                        f"- {c.get('agent_name', '?')}: "
                        f"ΔL={c.get('delta_L_total', 'N/A')}, "
                        f"显著性={c.get('significance', '?')}, "
                        f"星级={c.get('stars', '?')}"
                    )

        parts.append("\n\n请基于以上数据撰写批次评测报告。要求：")
        parts.append("1. 先给出3-5句核心摘要")
        parts.append("2. 分维度展开详细分析（收益维度/风险维度/结构维度）")
        parts.append("3. 识别表现最好和最差的线/期限")
        parts.append("4. 明确指出任何数据缺口或不确定性")
        parts.append("5. 所有判断请标注[判断]，所有数据引用请标注[数据]")

        return "\n".join(parts)

    # ───────────────────── Agent Contribution Report ─────────────────────

    def write_agent_contribution_report(self, contribution_data: dict) -> ReportOutput:
        """Generate analysis of which agents help/hurt and why.

        Args:
            contribution_data: 来自ContributionEngine.compute_contributions()的输出
                包含 contributions 列表，每项包含 agent_name, delta_L_total,
                ci_95_lower, ci_95_upper, significance, stars 等。

        Returns:
            ReportOutput with natural language analysis.
        """
        user_prompt = self._build_contribution_report_prompt(contribution_data)
        report_text = self._call_llm(REPORT_SYSTEM_PROMPT, user_prompt)
        return self._verify_and_wrap(report_text, contribution_data)

    def _build_contribution_report_prompt(self, data: dict) -> str:
        """构建Agent贡献报告的用户prompt。"""
        parts = []

        term = data.get("term", "unknown")
        baseline = data.get("baseline", {})
        contributions = data.get("contributions", [])
        sample_size = data.get("sample_size", 0)

        parts.append(f"# Agent贡献分析报告\n")
        parts.append(f"期限: {term} | 样本数: {sample_size}")

        if baseline:
            parts.append(f"\n## [数据] 基线Loss（全agent）")
            parts.append(f"- L_total: {baseline.get('L_total', 'N/A')}")
            parts.append(f"- L_return: {baseline.get('return_detail', {}).get('L_return', 'N/A')}")
            parts.append(f"- L_risk: {baseline.get('risk_detail', {}).get('L_risk', 'N/A')}")
            parts.append(f"- L_structure: {baseline.get('structure_detail', {}).get('L_structure', 'N/A')}")

        if contributions:
            parts.append("\n## [数据] Agent边际贡献（按ΔL_total降序）")
            for c in contributions:
                agent = c.get("agent_name", "?")
                delta = c.get("delta_L_total", 0)
                ci_low = c.get("ci_95_lower", 0)
                ci_high = c.get("ci_95_upper", 0)
                sig = c.get("significance", "?")
                stars = c.get("stars", "?")
                parts.append(
                    f"- {agent}: ΔL_total={delta}, "
                    f"95%CI=[{ci_low}, {ci_high}], "
                    f"显著性={sig}, {stars}"
                )

        # Add detailed breakdown if available
        for c in contributions:
            agent = c.get("agent_name", "?")
            if "delta_L_return" in c:
                parts.append(
                    f"\n### {agent} 详细分解\n"
                    f"- ΔL_return: {c.get('delta_L_return', 'N/A')}\n"
                    f"- ΔL_risk: {c.get('delta_L_risk', 'N/A')}\n"
                    f"- ΔL_structure: {c.get('delta_L_structure', 'N/A')}\n"
                    f"- 贡献分: {c.get('contribution_score', 'N/A')}\n"
                    f"- 标签: {c.get('label', 'N/A')}"
                )

        parts.append("\n\n请撰写Agent贡献分析报告。要求：")
        parts.append("1. 识别显著正贡献和显著负贡献的agent")
        parts.append("2. 分析每个agent在收益/风险/结构维度的具体表现")
        parts.append("3. 检查是否存在contradiction（如某个agent在收益维度正贡献但总ΔL为负）")
        parts.append("4. 对不显著(☆)的agent给出解释（可能是样本不足或该agent确实无关）")
        parts.append("5. 所有判断标注[判断]，数据引用标注[数据]")

        return "\n".join(parts)

    # ─────────────────── Optimization Recommendation ────────────────────

    def write_optimization_recommendation(
        self, loss_data: dict, contribution_data: dict
    ) -> ReportOutput:
        """Generate optimization recommendations based on loss and contribution patterns.

        Args:
            loss_data: 来自LossEngine的损失分解数据
            contribution_data: 来自ContributionEngine的贡献分析数据

        Returns:
            ReportOutput with prioritized optimization recommendations.
        """
        source_data = {
            "loss_data": loss_data,
            "contribution_data": contribution_data,
        }

        user_prompt = self._build_optimization_prompt(loss_data, contribution_data)
        report_text = self._call_llm(REPORT_SYSTEM_PROMPT, user_prompt)
        return self._verify_and_wrap(report_text, source_data)

    def _build_optimization_prompt(self, loss_data: dict,
                                    contribution_data: dict) -> str:
        """构建优化建议报告的用户prompt。"""
        parts = ["# 优化建议分析任务\n"]

        # Loss breakdown
        parts.append("## [数据] 当前Loss分解")
        if isinstance(loss_data, dict):
            for key in ["L_total", "L_return", "L_risk", "L_structure"]:
                if key in loss_data:
                    parts.append(f"- {key}: {loss_data[key]}")
            # Sub-component breakdown if available
            return_detail = loss_data.get("return_detail", {})
            if return_detail:
                parts.append("\n### 收益维度子组件")
                for k, v in return_detail.items():
                    parts.append(f"- {k}: {v}")
            risk_detail = loss_data.get("risk_detail", {})
            if risk_detail:
                parts.append("\n### 风险维度子组件")
                for k, v in risk_detail.items():
                    parts.append(f"- {k}: {v}")
            structure_detail = loss_data.get("structure_detail", {})
            if structure_detail:
                parts.append("\n### 结构维度子组件")
                for k, v in structure_detail.items():
                    parts.append(f"- {k}: {v}")

        # Contribution patterns
        contributions = contribution_data.get("contributions", [])
        if contributions:
            parts.append("\n## [数据] Agent贡献模式")
            # Identify patterns
            positive_agents = [c for c in contributions
                               if c.get("delta_L_total", 0) > 0.005]
            negative_agents = [c for c in contributions
                               if c.get("delta_L_total", 0) < -0.005]
            neutral_agents = [c for c in contributions
                              if abs(c.get("delta_L_total", 0)) <= 0.005]

            if positive_agents:
                names = [c["agent_name"] for c in positive_agents]
                parts.append(f"正贡献Agent: {', '.join(names)}")
            if negative_agents:
                names = [c["agent_name"] for c in negative_agents]
                parts.append(f"负贡献Agent: {', '.join(names)}")
            if neutral_agents:
                names = [c["agent_name"] for c in neutral_agents]
                parts.append(f"中性Agent: {', '.join(names)}")

        parts.append("\n\n请撰写优化建议报告。要求：")
        parts.append("1. 识别loss最大的维度（收益/风险/结构），分析可能的根本原因")
        parts.append("2. 对负贡献agent提出具体的优化方向（参数调优/提示词修改/逻辑修复/架构变更）")
        parts.append("3. 对正贡献agent分析其成功模式，建议保持或增强")
        parts.append("4. 按优先级排序建议（高/中/低），每项给出预期影响估计")
        parts.append("5. 明确指出优化风险和回滚策略")
        parts.append("6. 所有判断标注[判断]，数据引用标注[数据]")

        return "\n".join(parts)

    # ──────────────────── Structured Output (JSON) ────────────────────────

    def write_structured_report(
        self, batch_data: dict, contribution_data: dict = None,
        fidelity_data: dict = None
    ) -> dict:
        """Generate a structured JSON report suitable for downstream consumption.

        Args:
            batch_data: 批次数据
            contribution_data: 可选的贡献数据
            fidelity_data: 可选的保真度数据

        Returns:
            dict with diagnosis + optimization_suggestions (matches anti_hallucination schema).
        """
        # Build comprehensive prompt
        parts = [
            "你是一位量化研究员。请基于以下数据生成结构化的诊断和优化建议JSON。",
            "",
            "输出格式必须是严格的JSON，包含以下字段：",
            "{",
            '  "diagnosis": {',
            '    "top_findings": [{"claim": "...", "confidence": "HIGH/MEDIUM/LOW", "supporting_evidence_ids": ["..."]}],',
            '    "agent_ranking": [{"agent": "...", "delta": 0.0, "contribution": 0.0}],',
            '    "market_regime_insights": {"regime": "...", "impact": "..."}',
            "  },",
            '  "optimization_suggestions": [',
            '    {"type": "PARAM_TUNE/PROMPT_PATCH/LOGIC_FIX/ARCH_CHANGE/RESEARCH",',
            '     "target_file": "src/...",',
            '     "target_param": "weight/threshold/...",',
            '     "rationale": "...",',
            '     "expected_impact": "..."}',
            "  ],",
            '  "narrative_summary": "..."',
            "}",
            "",
        ]

        # Batch metrics
        metrics = batch_data.get("summary_metrics", {})
        parts.append(f"[数据] L_total={metrics.get('L_total', 'N/A')}")
        parts.append(f"[数据] L_return={metrics.get('L_return', 'N/A')}")
        parts.append(f"[数据] L_risk={metrics.get('L_risk', 'N/A')}")
        parts.append(f"[数据] L_structure={metrics.get('L_structure', 'N/A')}")

        # Contributions
        if contribution_data:
            for c in contribution_data.get("contributions", []):
                parts.append(
                    f"[数据] {c.get('agent_name', '?')} "
                    f"Delta_L_total={c.get('delta_L_total', 0)} "
                    f"CI=[{c.get('ci_95_lower', 0)}, {c.get('ci_95_upper', 0)}] "
                    f"显著性={c.get('significance', '?')}"
                )

        # Fidelity
        if fidelity_data:
            parts.append(
                f"[数据] Fidelity: action_flip_rate={fidelity_data.get('action_flip_rate', 'N/A')}, "
                f"topK_overlap={fidelity_data.get('topK_overlap', 'N/A')}"
            )

        parts.append("\n请输出完整JSON。")

        prompt = "\n".join(parts)
        response_text = self._call_llm(REPORT_SYSTEM_PROMPT, prompt, temperature=0.1)

        # Try to parse as JSON, fall back to raw text
        try:
            # Extract JSON from response (handle markdown code blocks)
            import re
            m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
            if m:
                json_str = m.group(1).strip()
            else:
                brace_start = response_text.find('{')
                brace_end = response_text.rfind('}')
                if brace_start != -1 and brace_end > brace_start:
                    json_str = response_text[brace_start:brace_end + 1]
                else:
                    json_str = response_text

            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse structured JSON, returning raw text")
            return {
                "diagnosis": {"top_findings": [], "agent_ranking": []},
                "optimization_suggestions": [],
                "narrative_summary": response_text,
                "_parse_error": True,
            }
