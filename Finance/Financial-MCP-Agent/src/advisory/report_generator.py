"""
收益分析报告生成器 — AdvisoryReportGenerator 类。

提供持仓收益分析报告的生成能力：
- matplotlib 收益对比图（用户 / DeepSeek 自由线 / 基准）
- LLM 报告生成提示词（含 [数据]/[判断] 防幻觉标签）
- Markdown 报告保存

Usage:
    from src.advisory.report_generator import AdvisoryReportGenerator

    g = AdvisoryReportGenerator()
    chart_path = g.generate_comparison_chart(
        user_equity=[1.0, 1.05, 1.08, ...],
        deepseek_equity=[1.0, 1.03, 1.06, ...],
        benchmark_equity=[1.0, 1.02, 1.04, ...],
    )
    prompt = g.build_report_prompt(
        "backtest",
        {"total_return_pct": 15.2, "max_drawdown_pct": 8.1, ...},
        deepseek_summary={"total_return_pct": 12.5, ...},
        chart_paths=[chart_path],
    )
    path = g.save_report("## 报告标题\\n\\n正文...", "我的报告")
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional

# ── matplotlib Agg 后端（无 GUI）───────────────────────────────────
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from matplotlib.ticker import FuncFormatter

# 注册系统中文字体，避免图表中文显示为方块
_CN_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
]
for _cn_font in _CN_FONT_CANDIDATES:
    try:
        _fm.findfont(_cn_font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_cn_font] + plt.rcParams.get("font.sans-serif", [])
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

logger = logging.getLogger(__name__)


# ── 报告提示词模板 ──────────────────────────────────────────────────

REPORT_PROMPT_TEMPLATE = """你是一位专业的A股投资报告分析师。请根据以下数据生成一份完整的收益分析报告。

## 报告类型

{report_type}

## 报告结构要求

请严格按照以下结构组织报告，每个部分都需要涵盖。

### 一、收益概览

基于以下数据，总结组合的整体表现：

**用户组合表现**：
{user_summary}
{deepseek_section}
**说明**：
- 所有收益数据使用 [数据] 标签标注（如 `[数据] 总收益率 15.2%`）
- 分析判断使用 [判断] 标签标注（如 `[判断] 该收益率在同期沪深300中排名前20%`）
- 不要混淆数据陈述与分析判断

### 二、持仓变化时间线

结合交易记录，描述持仓的主要变化节点：
1. 建仓期的主要买入标的和逻辑
2. 中间调仓的关键时间点和原因
3. 当前持仓结构与集中度
{chart_section}
### 三、DeepSeek 自由线对比

对比用户组合与DeepSeek AI自由线的表现差异：
1. 总收益差异及来源分析
2. 风险控制差异（最大回撤、波动率）
3. 交易频率和风格差异
4. 持仓行业分布对比

### 四、AI 综合评价

基于以上数据，对投资过程进行综合分析：

**亮点**：
- 列举 2-3 个表现突出的方面

**风险**：
- 指出 2-3 个需要关注的风险点

**改进建议**：
- 给出 2-3 条具体可执行的优化建议

### 五、附录

- 数据来源说明
- 分析时间段
- 免责声明

## 格式要求

1. 使用 Markdown 格式输出
2. 每个数据点必须标记 [数据] 或 [判断] 标签
3. 标签格式为 `[数据] ` 或 `[判断] ` 开头
4. 标题层级使用 ## 和 ###
5. 列表使用 - 或 1. 格式
6. 关键数字使用 **加粗**
7. 总字数控制在 1500-2500 字之间
8. 报告语言为中文

## 防幻觉规则

1. 所有直接从输入数据中读取的数字必须使用 [数据] 标签
2. 基于数据的推理、比较、评价使用 [判断] 标签
3. 禁止凭空编造数据，未提供的数据应标注为"数据未提供"
4. 当数据不完整时，明确说明分析局限性
5. 对比分析必须有数据支撑，不能泛泛而谈
"""


class AdvisoryReportGenerator:
    """收益分析报告生成器。

    职责：
    - 使用 matplotlib 生成收益对比曲线图（Agg 后端，无 GUI）。
    - 构建供 LLM 使用的报告生成提示词（含 [数据]/[判断] 防幻觉标签）。
    - 将 LLM 生成的 Markdown 报告保存到文件。
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        """初始化报告生成器。

        Args:
            output_dir: 报告保存目录。默认使用项目根目录下的
                        ``data/reports/``，自动创建。
        """
        if output_dir is None:
            root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            output_dir = os.path.join(root, "data", "reports")
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        logger.info(
            "AdvisoryReportGenerator 初始化完成，输出目录: %s", self._output_dir
        )

        # 图表输出子目录
        self._chart_dir = os.path.join(self._output_dir, "charts")
        os.makedirs(self._chart_dir, exist_ok=True)

    # ── 公共方法 ────────────────────────────────────────────────────

    def generate_comparison_chart(
        self,
        user_equity: List[float],
        deepseek_equity: Optional[List[float]] = None,
        benchmark_equity: Optional[List[float]] = None,
        title: str = "收益对比图",
    ) -> str:
        """生成收益对比曲线图。

        使用 matplotlib (Agg 后端) 绘制多条权益曲线，
        支持用户组合、DeepSeek 自由线和基准指数的对比。

        Args:
            user_equity: 用户组合的每日权益序列
                （如 [1.0, 1.01, 1.02, ...]，以 1.0 为起始净值）。
            deepseek_equity: DeepSeek 自由线的每日权益序列，可选。
            benchmark_equity: 基准指数（如沪深300）的每日权益序列，可选。
            title: 图表标题，默认 "收益对比图"。

        Returns:
            保存的图片文件绝对路径（PNG 格式）。
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # 生成 x 轴（交易日索引）
        days = list(range(len(user_equity)))

        # 绘制用户组合曲线
        ax.plot(
            days,
            user_equity,
            label="用户组合",
            color="#E63946",
            linewidth=2.0,
            zorder=5,
        )

        # 绘制 DeepSeek 自由线
        if deepseek_equity is not None and len(deepseek_equity) > 0:
            ds_len = len(deepseek_equity)
            ds_days = list(range(ds_len))
            ax.plot(
                ds_days,
                deepseek_equity,
                label="DeepSeek 自由线",
                color="#457B9D",
                linewidth=2.0,
                linestyle="--",
                zorder=4,
            )

        # 绘制基准指数
        if benchmark_equity is not None and len(benchmark_equity) > 0:
            bm_len = len(benchmark_equity)
            bm_days = list(range(bm_len))
            ax.plot(
                bm_days,
                benchmark_equity,
                label="基准指数",
                color="#2A9D8F",
                linewidth=1.5,
                linestyle=":",
                zorder=3,
            )

        # 零收益参考线
        ax.axhline(y=1.0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)

        # 格式设置
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("交易日", fontsize=11)
        ax.set_ylabel("净值", fontsize=11)
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")

        # Y 轴百分比格式（净值 → 涨跌幅）
        def _pct_format(y: float, _: Any) -> str:
            return f"{(y - 1) * 100:.1f}%"

        ax.yaxis.set_major_formatter(FuncFormatter(_pct_format))

        # 自适应 Y 轴范围
        all_values = list(user_equity)
        if deepseek_equity:
            all_values.extend(deepseek_equity)
        if benchmark_equity:
            all_values.extend(benchmark_equity)
        y_min = min(all_values)
        y_max = max(all_values)
        ax.set_ylim(min(y_min * 0.98, 0.98), max(y_max * 1.02, 1.02))

        fig.tight_layout()

        # 保存图片
        safe_title = re.sub(r"[^\w\-_]", "_", title, flags=re.UNICODE)
        hash_suffix = hashlib.md5(
            str(user_equity[:5]).encode()
        ).hexdigest()[:8]
        filename = f"comparison_{safe_title}_{hash_suffix}.png"
        filepath = os.path.join(self._chart_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)  # 释放内存

        logger.info("收益对比图已保存: %s", filepath)
        return filepath

    def build_report_prompt(
        self,
        report_type: str,
        user_summary: Dict[str, Any],
        deepseek_summary: Optional[Dict[str, Any]] = None,
        chart_paths: Optional[List[str]] = None,
    ) -> str:
        """构建 LLM 报告生成提示词。

        将用户总结数据和 DeepSeek 对比数据注入提示词模板，
        提示词内包含 [数据]/[判断] 防幻觉标签要求。

        Args:
            report_type: 报告类型描述。预定义值：
                - ``"backtest"``：历史回测分析报告
                - ``"simulation"``：模拟盘运行报告
                - ``"periodic"``：定期收益分析报告
            user_summary: 用户组合的收益总结字典，包含：
                - total_return_pct: 总收益率 (%)
                - max_drawdown_pct: 最大回撤 (%)
                - sharpe_ratio: 夏普比率
                - trade_count: 交易次数
                可选字段：annualized_return_pct, win_rate, volatility 等。
            deepseek_summary: DeepSeek 自由线的收益总结字典（可选）。
                结构与 user_summary 类似。
            chart_paths: 图表文件路径列表（可选），会在提示词中引用。

        Returns:
            注入数据后的完整报告生成提示词字符串。
        """
        # 格式化用户总结
        user_lines = self._format_summary_lines(user_summary, prefix="用户组合")
        user_summary_text = "\n".join(user_lines)

        # 格式化 DeepSeek 总结
        deepseek_section = ""
        if deepseek_summary:
            ds_lines = self._format_summary_lines(
                deepseek_summary, prefix="DeepSeek 自由线"
            )
            ds_text = "\n".join(ds_lines)
            deepseek_section = f"\n**DeepSeek 自由线表现**：\n{ds_text}\n"

        # 图表引用
        chart_section = ""
        if chart_paths:
            chart_lines = [
                f"  - 图表 {i+1}: {path}"
                for i, path in enumerate(chart_paths)
            ]
            chart_section = (
                "\n### 可视化图表\n\n"
                "以下图表已生成，请在报告中引用：\n"
                + "\n".join(chart_lines)
                + "\n"
            )

        # 报告类型标签
        type_labels = {
            "backtest": "历史回测分析报告",
            "simulation": "模拟盘运行报告",
            "periodic": "定期收益分析报告",
        }
        report_type_label = type_labels.get(
            report_type, f"收益分析报告（{report_type}）"
        )

        prompt = REPORT_PROMPT_TEMPLATE.format(
            report_type=report_type_label,
            user_summary=user_summary_text,
            deepseek_section=deepseek_section,
            chart_section=chart_section,
        )

        return prompt

    def save_report(self, content: str, title: str) -> str:
        """保存 Markdown 格式报告到文件。

        Args:
            content: Markdown 格式的报告内容。
            title: 报告标题，用于生成文件名。

        Returns:
            保存的文件绝对路径。
        """
        # 从 title 生成安全的文件名
        safe_title = re.sub(r"[^\w\-_]", "_", title, flags=re.UNICODE)
        safe_title = safe_title.strip("_")[:60]
        if not safe_title:
            safe_title = "report"

        # 用内容前缀哈希防止覆盖同名报告
        content_hash = hashlib.md5(
            content[:200].encode("utf-8")
        ).hexdigest()[:8]
        filename = f"{safe_title}_{content_hash}.md"
        filepath = os.path.join(self._output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("报告已保存: %s (%d 字符)", filepath, len(content))
        return filepath

    def save_report_html(self, content: str, title: str,
                         summary: Optional[Dict[str, Any]] = None) -> str:
        """保存 HTML 格式报告（可在浏览器中打开/打印为 PDF）。

        将 Markdown 内容包装为带样式的 HTML 报告页面，
        包含标题、元数据和响应式布局。适合打印导出为 PDF。

        Args:
            content: Markdown 格式的报告内容。
            title: 报告标题。
            summary: 可选，收益摘要字典，展示在报告头部。

        Returns:
            保存的 HTML 文件绝对路径。
        """
        safe_title = re.sub(r"[^\w\-_]", "_", title, flags=re.UNICODE)
        safe_title = safe_title.strip("_")[:60] or "report"
        content_hash = hashlib.md5(content[:200].encode("utf-8")).hexdigest()[:8]

        # 将 Markdown 内容转为简单 HTML
        html_body = self._md_to_html(content)

        # 构建摘要表
        summary_html = ""
        if summary:
            rows = []
            for key, label in [
                ("total_return_pct", "总收益率"), ("annualized_return_pct", "年化收益率"),
                ("max_drawdown_pct", "最大回撤"), ("sharpe_ratio", "夏普比率"),
                ("win_rate", "胜率"), ("trade_count", "交易次数"),
            ]:
                val = summary.get(key)
                if val is not None:
                    try:
                        rows.append(f"<tr><td>{label}</td><td>{float(val):.2f}</td></tr>")
                    except (ValueError, TypeError):
                        rows.append(f"<tr><td>{label}</td><td>{val}</td></tr>")
            if rows:
                summary_html = (
                    '<div class="summary-card"><h3>收益概览</h3>'
                    f'<table>{"".join(rows)}</table></div>'
                )

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  @media print {{ body {{ margin: 0 15mm; }} }}
  body {{ font-family: "Microsoft YaHei","SimSun",sans-serif; max-width:800px; margin:0 auto; padding:20px; color:#1a1a1a; line-height:1.8; }}
  h1 {{ border-bottom: 3px solid #2563eb; padding-bottom: 12px; }}
  h2 {{ border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; margin-top: 32px; }}
  .summary-card {{ background: #f0f7ff; border-radius: 8px; padding: 16px 24px; margin: 16px 0; }}
  .summary-card table {{ border-collapse:collapse; width:100%; }}
  .summary-card td {{ padding: 4px 16px 4px 0; }}
  .summary-card td:first-child {{ color:#64748b; }}
  .summary-card td:last-child {{ font-weight:700; }}
  .disclaimer {{ background:#fff3cd; padding:12px 16px; border-radius:6px; margin-top:32px; font-size:0.9em; color:#856404; }}
  @media print {{ .no-print {{ display:none; }} }}
</style>
</head>
<body>
<h1>{title}</h1>
{summary_html}
<div class="report-body">
{html_body}
</div>
<div class="disclaimer">
  <strong>免责声明：</strong>本报告由 AI 智能投顾系统自动生成，所有数据基于 Tushare 金融数据接口。
  报告内容仅供参考，不构成任何投资建议。市场有风险，投资需谨慎。
</div>
<p class="no-print" style="text-align:center;margin-top:24px;color:#94a3b8;">
  提示：在浏览器中按 Ctrl+P 可保存为 PDF
</p>
</body>
</html>"""

        filename = f"{safe_title}_{content_hash}.html"
        filepath = os.path.join(self._output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("HTML 报告已保存: %s (%d 字符)", filepath, len(html))
        return filepath

    def save_report_pdf(self, content: str, title: str,
                        summary: Optional[Dict[str, Any]] = None) -> str:
        """保存 PDF 格式报告（使用 fpdf2）。

        生成包含报告标题、摘要表格和正文内容的 PDF 文件。
        使用系统中文 TTF 字体（SimHei / Microsoft YaHei）渲染中文。

        Args:
            content: 报告正文内容。
            title: 报告标题。
            summary: 可选，收益摘要字典。

        Returns:
            保存的 PDF 文件绝对路径。生成失败时返回空字符串。
        """
        safe_title = re.sub(r"[^\w\-_]", "_", title, flags=re.UNICODE)
        safe_title = safe_title.strip("_")[:60] or "report"
        content_hash = hashlib.md5(content[:200].encode("utf-8")).hexdigest()[:8]

        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=20)

            # 注册中文 TTF 字体
            cn_font_family = self._register_chinese_font(pdf)
            title_font = cn_font_family or "Helvetica"
            body_font = cn_font_family or "Helvetica"

            # 标题
            pdf.set_font(title_font, "B", 18)
            pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(6)

            # 线
            pdf.set_line_width(0.2)
            y = pdf.get_y()
            pdf.line(10, y, 200, y)
            pdf.ln(6)

            # 摘要表格
            if summary:
                pdf.set_font(title_font, "B", 13)
                pdf.cell(0, 8, "收益概览", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font(body_font, "", 10)
                metrics = [
                    ("总收益率", summary.get("total_return_pct"), "%"),
                    ("年化收益率", summary.get("annualized_return_pct"), "%"),
                    ("最大回撤", summary.get("max_drawdown_pct"), "%"),
                    ("夏普比率", summary.get("sharpe_ratio"), ""),
                    ("胜率", summary.get("win_rate"), "%"),
                    ("交易次数", summary.get("trade_count"), ""),
                ]
                for label, val, unit in metrics:
                    if val is not None:
                        try:
                            display = f"{float(val):.2f}{unit}"
                        except (ValueError, TypeError):
                            display = str(val)
                        pdf.cell(95, 6, f"  {label}: {display}")
                        pdf.ln(6)
                pdf.ln(4)

            # 正文（去除 Markdown 标记后输出纯文本）
            pdf.set_font(body_font, "", 10)
            plain_text = re.sub(r"[#*`>\[\]()!_~]", "", content)
            # 按段落分割，每段作为一个多行单元格
            paragraphs = [p.strip() for p in plain_text.split("\n\n") if p.strip()]
            for para in paragraphs[:100]:  # 最多100段
                if len(para) > 500:
                    para = para[:500] + "..."
                # 清理连续空白
                para = re.sub(r"\s+", " ", para)
                try:
                    pdf.multi_cell(0, 5.5, para, align="L")
                    pdf.ln(2)
                except Exception:
                    continue

            # 免责声明
            pdf.ln(6)
            pdf.set_font(body_font, "I", 8)
            pdf.multi_cell(0, 4.5,
                "免责声明：本报告由 AI 智能投顾系统自动生成，仅供参考，不构成投资建议。市场有风险，投资需谨慎。")

            filename = f"{safe_title}_{content_hash}.pdf"
            filepath = os.path.join(self._output_dir, filename)
            pdf.output(filepath)
            logger.info("PDF 报告已保存: %s", filepath)
            return filepath

        except ImportError:
            logger.warning("fpdf2 未安装，无法生成 PDF，请安装: pip install fpdf2")
            return ""
        except Exception as e:
            logger.error("PDF 生成失败: %s", e)
            return ""

    @staticmethod
    def _register_chinese_font(pdf) -> Optional[str]:
        """为 FPDF 实例注册系统中文 TTF 字体。

        按优先级尝试 SimHei → Microsoft YaHei → SimSun。
        成功时返回字体 family 名称，失败时返回 None（调用方应回退到 Helvetica）。
        """
        candidates = [
            ("C:\\Windows\\Fonts\\simhei.ttf", "SimHei"),
            ("C:\\Windows\\Fonts\\msyh.ttc", "MSYaHei"),
            ("C:\\Windows\\Fonts\\simsun.ttc", "SimSun"),
            ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "WenQuanYi"),
            ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "WQYMicro"),
        ]
        for font_path, family_name in candidates:
            if not os.path.isfile(font_path):
                continue
            try:
                pdf.add_font(family_name, "", font_path, uni=True)
                pdf.add_font(family_name, "B", font_path, uni=True)
                pdf.add_font(family_name, "I", font_path, uni=True)
                return family_name
            except Exception as e:
                logger.debug("字体注册失败 %s: %s", font_path, e)
                continue
        logger.warning("未找到可用的中文字体，PDF 中文内容可能无法正常显示")
        return None

    @staticmethod
    def _md_to_html(md_text: str) -> str:
        """将 Markdown 文本转换为基本 HTML。

        处理标题、列表、粗体、段落等常见 Markdown 元素。
        不依赖第三方库，简单但覆盖常见报告格式。

        Args:
            md_text: Markdown 格式文本。

        Returns:
            HTML 字符串。
        """
        lines = md_text.split("\n")
        html_lines = []
        in_list = False
        in_ordered = False

        for line in lines:
            stripped = line.strip()

            # 空行 → 结束列表
            if not stripped:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                if in_ordered:
                    html_lines.append("</ol>")
                    in_ordered = False
                html_lines.append("")
                continue

            # 标题
            if stripped.startswith("### "):
                if in_list:
                    html_lines.append("</ul>"); in_list = False
                if in_ordered:
                    html_lines.append("</ol>"); in_ordered = False
                html_lines.append(f"<h3>{_escape_html(stripped[4:])}</h3>")
                continue
            if stripped.startswith("## "):
                if in_list:
                    html_lines.append("</ul>"); in_list = False
                if in_ordered:
                    html_lines.append("</ol>"); in_ordered = False
                html_lines.append(f"<h2>{_escape_html(stripped[3:])}</h2>")
                continue

            # 无序列表
            if re.match(r"^[-*]\s", stripped):
                if not in_list:
                    if in_ordered:
                        html_lines.append("</ol>"); in_ordered = False
                    html_lines.append("<ul>")
                    in_list = True
                item = re.sub(r"^[-*]\s+", "", stripped)
                html_lines.append(f"<li>{_fmt_inline(item)}</li>")
                continue

            # 有序列表
            if re.match(r"^\d+[.)]\s", stripped):
                if not in_ordered:
                    if in_list:
                        html_lines.append("</ul>"); in_list = False
                    html_lines.append("<ol>")
                    in_ordered = True
                item = re.sub(r"^\d+[.)]\s+", "", stripped)
                html_lines.append(f"<li>{_fmt_inline(item)}</li>")
                continue

            # 分隔线
            if stripped in ("---", "***", "___"):
                if in_list:
                    html_lines.append("</ul>"); in_list = False
                if in_ordered:
                    html_lines.append("</ol>"); in_ordered = False
                html_lines.append("<hr>")
                continue

            # 普通段落
            if in_list:
                html_lines.append("</ul>"); in_list = False
            if in_ordered:
                html_lines.append("</ol>"); in_ordered = False
            html_lines.append(f"<p>{_fmt_inline(stripped)}</p>")

        if in_list:
            html_lines.append("</ul>")
        if in_ordered:
            html_lines.append("</ol>")

        return "\n".join(html_lines)

    # ── 内部方法 ────────────────────────────────────────────────────

    @staticmethod
    def _format_summary_lines(
        summary: Dict[str, Any],
        prefix: str = "用户组合",
    ) -> List[str]:
        """将收益总结字典格式化为可读的文本行列表。

        Args:
            summary: 收益总结字典。
            prefix: 前缀标签，用于区分不同组合。

        Returns:
            格式化后的文本行列表。
        """
        lines: List[str] = []
        lines.append(f"  **{prefix}**：")

        # 核心指标定义（key, label, format_spec）
        key_fields: List[tuple[str, str, str]] = [
            ("total_return_pct", "总收益率", "{:.2f}%"),
            ("annualized_return_pct", "年化收益率", "{:.2f}%"),
            ("max_drawdown_pct", "最大回撤", "{:.2f}%"),
            ("sharpe_ratio", "夏普比率", "{:.2f}"),
            ("volatility", "年化波动率", "{:.2f}%"),
            ("win_rate", "胜率", "{:.2f}%"),
            ("trade_count", "交易次数", "{}"),
            ("total_value", "最终总资产", "{:.2f}"),
        ]

        for key, label, fmt in key_fields:
            val = summary.get(key)
            if val is not None:
                try:
                    formatted = fmt.format(val)
                    lines.append(f"    - {label}：{formatted}")
                except (ValueError, TypeError):
                    lines.append(f"    - {label}：{val}")

        # 补充未在核心列表中但字典中包含的字段
        core_keys = {k for k, _, _ in key_fields}
        extra_lines: List[str] = []
        for k, v in summary.items():
            if k not in core_keys and v is not None:
                extra_lines.append(f"    - {k}：{v}")
        if extra_lines:
            lines.append("")
            lines.append("    **其他指标**：")
            lines.extend(extra_lines)

        return lines


# ------------------------------------------------------------------
# HTML 辅助函数
# ------------------------------------------------------------------


def _escape_html(text: str) -> str:
    """转义 HTML 特殊字符。"""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _fmt_inline(text: str) -> str:
    """将 Markdown 行内格式转为 HTML。

    支持：**粗体**，*斜体*，`代码`，[文字](链接)。
    """
    text = _escape_html(text)
    # 粗体
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # 斜体
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # 行内代码
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # 链接
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text
