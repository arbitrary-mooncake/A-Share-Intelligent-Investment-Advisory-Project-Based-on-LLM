"""
报告构建器 — 组装评测报告所需的结构化数据。
所有数值由代码计算，LLM只负责将结构化数据翻译成自然语言。
"""
import json
from datetime import datetime
from typing import Dict, Any, List, Optional


class ReportBuilder:
    """评测报告构建器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def build_batch_report_data(self, batch_id: str, orchestrator_status: Dict,
                                 loss_results: Dict = None,
                                 contribution_results: Dict = None,
                                 backtest_results: Dict = None) -> Dict[str, Any]:
        """构建完整的批次报告数据结构"""

        lines = orchestrator_status.get("lines", [])
        pools = orchestrator_status.get("pools", {})

        # 各期限汇总
        term_summaries = {}
        for term in ["short", "medium", "long"]:
            term_lines = [l for l in lines if l.get("term") == term]
            if term_lines:
                avg_return = sum(l.get("cumulative_return_pct", 0) for l in term_lines) / len(term_lines)
                avg_mdd = sum(l.get("max_drawdown_pct", 0) for l in term_lines) / len(term_lines)
                term_summaries[term] = {
                    "line_count": len(term_lines),
                    "avg_cumulative_return_pct": round(avg_return, 2),
                    "avg_max_drawdown_pct": round(avg_mdd, 2),
                }

        # 提取回测参考线数据
        reference_line_data = None
        if backtest_results and backtest_results.get("reference_line"):
            reference_line_data = backtest_results["reference_line"]

        report = {
            "batch_id": batch_id,
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {},
            "term_summaries": term_summaries,
            "line_details": lines,
            "pool_summaries": pools,
            "loss_analysis": loss_results or {},
            "agent_contributions": contribution_results or {},
            "backtest_summary": backtest_results or {},
            "reference_line": reference_line_data,
            "declarations": [
                "本报告所有数值由代码计算生成",
                "模拟盘结果仅供参考，目的是优化Agent系统",
                "评测模型使用DeepSeek V4 Flash，与生产模型(M1/M3)存在差异",
            ],
        }

        # 执行摘要
        if lines:
            all_returns = [l.get("cumulative_return_pct", 0) for l in lines if l.get("cumulative_return_pct") is not None]
            if all_returns:
                report["executive_summary"] = {
                    "best_line": max(lines, key=lambda l: l.get("cumulative_return_pct", -999)),
                    "worst_line": min(lines, key=lambda l: l.get("cumulative_return_pct", 999)),
                    "avg_return_all_lines": round(sum(all_returns) / len(all_returns), 2),
                    "total_lines": len(lines),
                }

        return report

    def build_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """将报告数据转换为Markdown格式"""
        lines = []
        lines.append(f"# 评测报告")
        lines.append(f"**批次ID**: {report_data.get('batch_id', 'N/A')[:20]}")
        lines.append(f"**生成时间**: {report_data.get('generated_at', '')[:19]}")
        lines.append("")

        # 执行摘要
        exec_summary = report_data.get("executive_summary", {})
        if exec_summary:
            lines.append("## 执行摘要")
            lines.append(f"- 总线路数: {exec_summary.get('total_lines', 0)}")
            lines.append(f"- 全线平均收益: {exec_summary.get('avg_return_all_lines', 0)}%")
            best = exec_summary.get("best_line", {})
            worst = exec_summary.get("worst_line", {})
            lines.append(f"- 最佳线路: {best.get('line_id', '?')} ({best.get('cumulative_return_pct', 0)}%)")
            lines.append(f"- 最差线路: {worst.get('line_id', '?')} ({worst.get('cumulative_return_pct', 0)}%)")
            lines.append("")

        # 各期限汇总
        term_summaries = report_data.get("term_summaries", {})
        if term_summaries:
            lines.append("## 各期限表现")
            for term, summary in term_summaries.items():
                term_cn = {"short": "短线", "medium": "中线", "long": "长线"}.get(term, term)
                lines.append(f"### {term_cn}")
                lines.append(f"- 线路数: {summary['line_count']}")
                lines.append(f"- 平均累计收益: {summary['avg_cumulative_return_pct']}%")
                lines.append(f"- 平均最大回撤: {summary['avg_max_drawdown_pct']}%")
                lines.append("")

        # 回测参考线 (SB-L6: 长持对照线)
        reference_line = report_data.get("reference_line")
        if reference_line:
            lines.append("## 长持对照线 (现实性校验)")
            lines.append("")
            lines.append(f"- **线路ID**: {reference_line.get('line_id', 'SB-L6')}")
            lines.append(f"- **类型**: 参考线（非消融，不参与ΔLoss计算）")
            lines.append(f"- **策略**: ShortLongHoldStrategy（连续持仓，成熟短线策略）")
            lines.append(f"- **说明**: {reference_line.get('description', '短线长持对照线')}")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| Anchor数 | {reference_line.get('num_anchors', 0)} |")
            lines.append(f"| 初始资金 | ¥{reference_line.get('initial_capital', 0):,.0f} |")
            lines.append(f"| 最终市值 | ¥{reference_line.get('final_value', 0):,.0f} |")
            lines.append(f"| 累计收益率 | {reference_line.get('cumulative_return_pct', 0)}% |")
            lines.append(f"| 年化收益率 | {reference_line.get('annualized_return_pct', 0)}% |")
            lines.append(f"| 最大回撤 | {reference_line.get('max_drawdown_pct', 0)}% |")
            lines.append(f"| Sharpe比率 | {reference_line.get('sharpe_ratio', 0)} |")
            lines.append(f"| 胜率 | {reference_line.get('win_rate_pct', 0)}% |")
            lines.append(f"| 最终持仓数 | {reference_line.get('final_holdings_count', 0)} |")
            lines.append("")
            lines.append("> **注**: SB-L6为参考线，使用与实盘S-L8相同的ShortLongHoldStrategy连续持仓策略。")
            lines.append("> 其表现用于交叉校验消融结论在现实连续持仓场景下是否仍然成立。")
            lines.append("> SB-L6不参与Agent消融ΔLoss计算。")
            lines.append("")

        # Agent贡献
        contributions = report_data.get("agent_contributions", {})
        if contributions and contributions.get("contributions"):
            lines.append("## Agent贡献分析")
            lines.append("| Agent | ΔL | 95% CI | 显著性 | 评价 |")
            lines.append("|-------|-----|--------|--------|------|")
            for c in contributions["contributions"][:8]:
                lines.append(f"| {c['agent_name']} | {c['delta_L_total']} | [{c['ci_95_lower']}, {c['ci_95_upper']}] | {c['stars']} | {c.get('label', '')} |")
            lines.append("")

        # 风险声明
        lines.append("## 风险声明")
        for decl in report_data.get("declarations", []):
            lines.append(f"- {decl}")
        lines.append("")
        lines.append("---")
        lines.append("*本报告由评分智能体系统自动生成*")

        return "\n".join(lines)

    def save_report(self, report_data: Dict[str, Any],
                    output_dir: str = "") -> str:
        """保存报告到文件"""
        import os
        if not output_dir:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "eval", "reports"
            )
        os.makedirs(output_dir, exist_ok=True)

        batch_id = report_data.get("batch_id", "unknown")[:16]
        filename = f"report_{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(output_dir, filename)

        md_content = self.build_markdown_report(report_data)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Also save JSON
        json_path = filepath.replace(".md", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        return filepath
