"""
基金报告产出Agent (Agent 8: Fund Report Agent)
消费fund_merge_node合并后的分析包，调用LLM生成完整的13模块基金分析报告。

读入：state.data.fund_analysis_package, fund_code, fund_name, current_time_info
写出：state.data.fund_report (str), state.metadata.fund_report_executed
模型：Model 1 (MiMo-V2.5-Pro), thinking=enabled, max_tokens=16000
"""
import os
import time
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.utils.model_config import get_model_config_for_agent, get_thinking_body
from src.utils.pdf_generator import markdown_to_pdf

load_dotenv(override=True)

logger = setup_logger(__name__)


def _build_report_prompt(
    fund_code: str,
    fund_name: str,
    fund_type: str,
    current_time_info: str,
    analysis_package: Dict[str, Any],
) -> str:
    """构建基金报告生成的系统提示词，包含13个模块要求。"""

    fund_profile = analysis_package.get("fund_profile", {})
    normalized_subscores = analysis_package.get("normalized_subscores", {})
    strengths_pool = analysis_package.get("strengths_pool", [])
    risks_pool = analysis_package.get("risks_pool", [])
    conflicts = analysis_package.get("conflicts", [])
    holding_period_hints = analysis_package.get("holding_period_hints", [])
    confidence_summary = analysis_package.get("confidence_summary", "N/A")
    missing_data_summary = analysis_package.get("missing_data_summary", [])
    frontend_ready_tags = analysis_package.get("frontend_ready_tags", [])
    raw_agent_outputs = analysis_package.get("raw_agent_outputs", {})

    # 格式化各类信息用于prompt
    fund_profile_str = "\n".join(
        f"  - **{k}**: {v}" for k, v in fund_profile.items()
    ) if fund_profile else "  暂无基金基本信息"

    subscore_str = "\n".join(
        f"  - **{k}**: {v}/100" for k, v in normalized_subscores.items()
    ) if normalized_subscores else "  暂无维度评分"

    strengths_str = "\n".join(
        f"  - {s}" for s in strengths_pool
    ) if strengths_pool else "  暂无汇总优势"

    risks_str = "\n".join(
        f"  - {r}" for r in risks_pool
    ) if risks_pool else "  暂无汇总风险"

    conflicts_str = "\n".join(
        f"  - {c}" for c in conflicts
    ) if conflicts else "  各Agent分析结论一致，无显著分歧"

    holding_str = "\n".join(
        f"  - 建议：{h.get('label', 'N/A')} | 理由：{h.get('reason', 'N/A')} | 来源：{h.get('source_agent', 'N/A')}"
        for h in holding_period_hints
    ) if holding_period_hints else "  暂无持有期建议线索"

    missing_str = "\n".join(
        f"  - {m}" for m in missing_data_summary
    ) if missing_data_summary else "  所有数据模块完整，无缺失"

    tags_str = ", ".join(frontend_ready_tags) if frontend_ready_tags else "暂无标签"

    # 格式化原始Agent输出
    raw_outputs_str = ""
    if raw_agent_outputs:
        for agent_name, output in raw_agent_outputs.items():
            if output:
                raw_outputs_str += f"\n### {agent_name}\n{output[:3000]}\n"  # 截断，保留关键信息
                if len(str(output)) > 3000:
                    raw_outputs_str += "\n...[输出过长，已截断]...\n"
    if not raw_outputs_str:
        raw_outputs_str = "  暂无原始分析输出"

    # 计算综合得分
    if normalized_subscores:
        total_score = round(sum(normalized_subscores.values()) / len(normalized_subscores), 1)
    else:
        total_score = "N/A"

    report_prompt = f"""你是一位资深基金研究分析师，拥有10年以上公募基金评价与FOF投资经验，擅长撰写面向零售投资者的基金深度研究报告。

**重要时间信息：当前实际时间是 {current_time_info}**

这是真实的当前时间，不是你的训练数据截止时间。请在生成报告时：
- 基于实际当前时间来判断数据的时效性
- 正确标注"最新"、"近期"、"历史"等时间概念
- 在报告中明确标注分析的时间基准点
- 所有时间相关的描述都要基于这个实际日期

你的任务是基于以下7维度基金分析结果，生成一份完整的基金分析报告。分析结果已经过归一化处理和多方交叉验证，请充分利用其中信息。

---

## 基金基本信息
{fund_profile_str}

## 各维度评分（满分为100）
{subscore_str}

综合得分：{total_score}/100 | 置信度：{confidence_summary}

## 汇总优势
{strengths_str}

## 汇总风险
{risks_str}

## 分析分歧
{conflicts_str}

## 持有期建议线索
{holding_str}

## 缺失数据说明
{missing_str}

## 前端标签
{tags_str}

## 原始分析详情（供深度引用）
{raw_outputs_str}

---

## 报告要求

请按以下**13个模块**生成完整的基金分析报告（Markdown格式），标题层级清晰，内容专业但不晦涩，让普通投资者能读懂。

### 模块1：报告封面信息
- 基金名称、代码、类型
- 报告生成日期（基于{current_time_info}）
- 基金管理人/托管人
- 成立日期
- 业绩比较基准
- 风险等级

### 模块2：核心结论摘要（面向时间有限的投资者）
请用5句话回答以下5个核心问题：
1. **这只基金怎么样？** —— 一句话综合评价
2. **综合得分多少？** —— 明确给出总分（如{total_score}/100）及评级（优秀/良好/一般/较差）
3. **适合什么样的投资者？** —— 风险偏好、投资期限、资金规模的匹配度
4. **建议持有多久？** —— 明确持有期限建议及简述理由
5. **最大的风险是什么？** —— 投资者最需要关注的1-2个核心风险

### 模块3：总分与分项评分
以表格形式展示7维度评分：
| 评分维度 | 得分(满分100) | 评级 | 简要评价 |
|---------|------------|------|---------|
| 产品定位与策略 | {normalized_subscores.get('product_positioning', 'N/A')} | - | - |
| 业绩与风险 | {normalized_subscores.get('performance_risk', 'N/A')} | - | - |
| 组合结构 | {normalized_subscores.get('portfolio_structure', 'N/A')} | - | - |
| 经理/团队 | {normalized_subscores.get('manager_team', 'N/A')} | - | - |
| 基准一致性与风格稳定性 | {normalized_subscores.get('benchmark_style_consistency', 'N/A')} | - | - |
| 费用与流动性 | {normalized_subscores.get('fee_liquidity', 'N/A')} | - | - |
| 事件风险 | {normalized_subscores.get('event_risk', 'N/A')} | - | - |

同时给出各维度得分的解释说明和权重逻辑（为什么该项重要）。

### 模块4：建议持有时长
- 基于7Agent分析结果的综合持有期判断
- 给出明确建议：短期（<1个月）、中短期（1-3个月）、中期（3-12个月）、中长期（1-3年）、长期（3年以上）
- 说明理由，引用分析中的具体证据
- 如果有持有期分歧（各Agent建议不同），说明分歧原因和你的最终判断
- 说明哪些情况下应调整持有期（触发条件）

### 模块5：产品定位与策略分析
- 基金类型与法律结构（ETF/LOF/主动管理/指数增强等）
- 投资目标与策略描述
- 业绩比较基准的合理性
- 投资范围与限制
- 产品创新点或差异化特征
- 在同类基金中的定位
- **策略清晰度评价**：投资策略是否容易理解、是否一致执行

### 模块6：历史业绩与风险表现
- 近期回报数据（1月/3月/YTD/1年/3年/5年）
- 最大回撤及其持续时间
- 夏普比率、信息比率、索提诺比率等风险调整后收益
- 上行捕获率与下行捕获率
- 与基准、同类的相对表现
- 不同市场环境下的表现特征（牛市/熊市/震荡市）

### 模块7：组合结构与持仓穿透
- 资产配置结构（股票/债券/现金/其他）
- 行业分布与前五大行业集中度
- 前十大持仓及集中度
- 市值风格（大盘/中盘/小盘）与价值/成长风格
- 换手率水平与趋势
- 持仓集中度的变化趋势

### 模块8：基金经理/团队/管理人
- 基金经理从业经验、历史业绩、管理规模
- 团队稳定性（是否频繁更换经理）
- 基金公司整体实力：管理规模、产品线、投研团队
- 基金经理投资哲学与风格一致性
- 激励机制是否与投资者利益对齐

### 模块9：基准一致性与风格稳定性
- 与业绩比较基准的偏离度（跟踪误差/主动份额）
- 风格漂移风险评估：是否存在风格飘移现象
- 通过不同市场周期的风格稳定性检验
- 实际持仓与基金合同约定的一致性

### 模块10：费用、流动性与持有体验
- 管理费、托管费、申购/赎回费（包括阶梯费率）
- 隐性成本：买卖价差、交易佣金、冲击成本
- 与同类基金的费用对比
- 基金规模及流动性（大额赎回冲击）
- ETF的折溢价水平（如适用）
- 分红政策与历史分红记录

### 模块11：近期重要事件与风险提示
- 近期发生的重大事件（如分红、份额折算、合同变更、高管变动、规模异动等）
- 对基金未来的潜在影响分析
- 当前面临的各项风险按优先级排列：
  1. 市场/系统性风险
  2. 行业/板块集中风险
  3. 流动性风险
  4. 管理人风险
  5. 政策/监管风险
  6. 其他特定风险
- 每个风险需说明：发生的可能性、影响程度、是否有对冲手段

### 模块12：适合人群与不适合人群
- **适合的投资者画像**：
  - 风险承受能力
  - 投资期限
  - 投资知识与经验
  - 资金规模
  - 投资目标（增值/保值/收入）
  - 典型场景举例
- **不适合的投资者画像**：
  - 不适合的投资者类型
  - 原因说明
  - 替代建议（如果投资者不符合该基金特征，可考虑什么类型的替代品）

### 模块13：免责声明/方法说明
必须包含以下固定内容：

> **免责声明**
>
> 本报告由AI投资分析系统自动生成，仅供投资者参考，不构成任何投资建议。报告中使用的数据来源于公开市场信息及第三方数据提供商，虽已尽力确保数据准确性，但不对其完整性、时效性和准确性做出任何保证。投资者应根据自身的财务状况、投资经验、投资目标和风险承受能力，独立做出投资决策，并承担相应风险。
>
> **投资有风险，入市需谨慎。过往业绩不代表未来表现。**
>
> **方法说明**
>
> 本报告基于7维度基金分析框架生成：
> 1. 产品定位与策略分析
> 2. 业绩与风险分析
> 3. 组合结构与持仓穿透
> 4. 基金经理/团队分析
> 5. 基准一致性与风格稳定性分析
> 6. 费用与流动性分析
> 7. 近期事件与风险分析
>
> 各维度由独立AI Agent完成分析后，经合并节点进行归一化评分、优势/风险提取、冲突检测和置信度评估，最终由报告产出Agent整合为完整报告。AI分析可能存在偏差或错误，投资者应结合自身判断使用本报告。
>
> **数据时效声明**
> 本报告分析基于截至 {current_time_info} 的可获取数据。部分数据可能存在更新延迟，请以基金公司官方披露为准。

---

## 输出格式要求

1. **使用Markdown格式**，标题层级清晰（# 用于大标题，## 用于模块标题，### 用于子标题）
2. **专业但不晦涩**，让普通投资者能读懂核心结论，让专业投资者能深入细节
3. **引用具体数据支撑结论**，优先使用分析包中的定量数据
4. **不确定的地方标注"基于现有数据的推断"**，不得编造未在分析数据中出现的事实或数字
5. **结尾必须包含完整的模块13免责声明/方法说明**
6. **表格在合理的地方使用**，特别是评分对比、历史业绩、费用对比等
7. **报告语言为中文**（专有名词可保留英文）
8. **不要包含任何代码块标记**（如```markdown或```），直接输出纯Markdown内容

---

## ⛔ 防幻觉输出规则

1. 报告中使用的具体数据、数字、名称必须来自提供的分析包，**绝对禁止编造**
2. 如果分析包中某个维度数据缺失，应如实说明"该维度数据暂不可用"而非臆造
3. 评分表中的分数必须与分析包中的`normalized_subscores`一致
4. 基于推断的结论需明确标注，如"【推断】"或"基于现有数据的推断"
5. 不得编造基金经理姓名、具体产品名称、或未在分析包中出现的事实

---

请严格按照以上要求，生成完整的中文基金分析报告。
"""
    return report_prompt


async def fund_report_agent(state: AgentState) -> Dict[str, Any]:
    """
    基金报告产出Agent (Agent 8)。

    从state.data中读取fund_analysis_package（由fund_merge_node合并输出），
    调用LLM生成包含13个模块的完整基金分析报告。

    Args:
        state: LangGraph AgentState，包含data和metadata

    Returns:
        包含更新后的data和metadata的字典
    """
    logger.info(f"{WAIT_ICON} FundReportAgent: Starting fund report generation.")

    execution_logger = get_execution_logger()
    agent_name = "fund_report_agent"

    current_data = state.get("data", {})
    messages = state.get("messages", [])
    metadata = state.get("metadata", {})

    # ── 读取输入 ──
    analysis_package = current_data.get("fund_analysis_package", {})
    fund_code = current_data.get("fund_code", "Unknown Fund")
    fund_name = current_data.get("fund_name", "Unknown Fund")
    current_time_info = current_data.get("current_time_info", "未知时间")

    fund_type = analysis_package.get("fund_profile", {}).get("fund_type", "未知类型")

    execution_logger.log_agent_start(agent_name, {
        "fund_code": fund_code,
        "fund_name": fund_name,
        "fund_type": fund_type,
        "has_analysis_package": bool(analysis_package),
        "subscores_count": len(analysis_package.get("normalized_subscores", {})),
        "strengths_count": len(analysis_package.get("strengths_pool", [])),
        "risks_count": len(analysis_package.get("risks_pool", [])),
        "confidence_summary": analysis_package.get("confidence_summary", "N/A"),
    })

    agent_start_time = time.time()

    # 如果分析包为空，生成错误报告
    if not analysis_package:
        logger.error(f"{ERROR_ICON} FundReportAgent: fund_analysis_package is empty.")
        error_report = _build_empty_package_report(fund_code, fund_name, current_time_info)
        current_data["fund_report"] = error_report

        execution_logger.log_agent_complete(
            agent_name, {"fund_report": error_report, "error": "empty_package"},
            time.time() - agent_start_time, False, "fund_analysis_package is empty"
        )

        return {
            "data": current_data,
            "messages": messages,
            "metadata": {**metadata, "fund_report_executed": True}
        }

    try:
        # ── 构建prompt ──
        logger.info(f"{WAIT_ICON} FundReportAgent: Building 13-section report prompt...")
        system_prompt = _build_report_prompt(
            fund_code=fund_code,
            fund_name=fund_name,
            fund_type=fund_type,
            current_time_info=current_time_info,
            analysis_package=analysis_package,
        )

        user_prompt = f"""请基于上述分析数据，为 **{fund_name}**（{fund_code}）生成完整的13模块基金分析报告。

基金代码：{fund_code}
基金名称：{fund_name}
基金类型：{fund_type}
分析基准时间：{current_time_info}

请确保：
1. 严格按照13个模块的结构生成报告
2. 每个模块都有实质性内容，不空泛
3. 基于分析包中的定量数据给出具体结论
4. 输出纯Markdown格式，不包含代码块标记
5. 完整包含模块13的免责声明
"""

        # ── 获取模型配置 ──
        model_cfg = get_model_config_for_agent("fund_report_agent", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]

        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} FundReportAgent: Missing OpenAI environment variables.")
            current_data["fund_report_error"] = "Missing OpenAI environment variables."

            execution_logger.log_agent_complete(
                agent_name, current_data, time.time() - agent_start_time,
                False, "Missing OpenAI environment variables"
            )

            return {
                "data": current_data,
                "messages": messages,
                "metadata": {**metadata, "fund_report_executed": True}
            }

        # ── 记录模型配置 ──
        model_config = {
            "model": model_name,
            "temperature": 0.6,
            "max_tokens": 16000,
            "thinking": "enabled",
            "api_base": base_url
        }
        logger.info(
            f"{WAIT_ICON} FundReportAgent: Using model={model_name}, "
            f"max_tokens=16000, thinking=enabled"
        )

        # ── 创建LLM实例 ──
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.6,
            request_timeout=720,
            max_tokens=16000,
            extra_body=get_thinking_body(base_url, True)
        )

        # ── 调用LLM ──
        llm_start_time = time.time()
        llm_message = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        llm_execution_time = time.time() - llm_start_time

        fund_report = llm_message.content

        # ── 记录LLM交互 ──
        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="fund_report_generation",
            input_messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            output_content=fund_report,
            model_config=model_config,
            execution_time=llm_execution_time
        )

        # ── 清理输出 ──
        fund_report = fund_report.replace("```markdown", "").replace("```", "").strip()

        logger.info(
            f"{SUCCESS_ICON} FundReportAgent: Fund report generated for "
            f"{fund_name} ({fund_code}), length={len(fund_report)} chars."
        )

        # ── 保存报告到文件 ──
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_fund_name = fund_name.replace(" ", "_").replace(".", "").replace("/", "_")
        clean_fund_code = fund_code.replace("sh.", "").replace("sz.", "").replace("of.", "")
        report_filename = f"fund_report_{safe_fund_name}_{clean_fund_code}_{timestamp}.md"

        reports_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "reports"
        )
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(fund_report)
        logger.info(f"{SUCCESS_ICON} FundReportAgent: Report saved to {report_path}")

        # ── 生成PDF版本 ──
        pdf_path = None
        pdf_filename = report_filename.replace(".md", ".pdf")
        pdf_full_path = os.path.join(reports_dir, pdf_filename)
        try:
            markdown_to_pdf(
                fund_report, pdf_full_path,
                company_name=f"{fund_name} ({fund_code})",
                stock_code=fund_code
            )
            pdf_path = pdf_full_path
            logger.info(f"{SUCCESS_ICON} FundReportAgent: PDF report saved to {pdf_path}")
        except Exception as pdf_err:
            logger.warning(f"Failed to generate PDF report: {pdf_err}. Markdown is available.")

        # ── 写入状态 ──
        current_data["fund_report"] = fund_report
        current_data["fund_report_path"] = report_path
        if pdf_path:
            current_data["fund_report_pdf_path"] = pdf_path

        total_execution_time = time.time() - agent_start_time
        execution_logger.log_agent_complete(agent_name, {
            "fund_report_length": len(fund_report),
            "fund_report_path": report_path,
            "fund_report_pdf_path": pdf_path,
            "llm_execution_time": llm_execution_time,
            "total_execution_time": total_execution_time,
        }, total_execution_time, True)

        return {
            "data": current_data,
            "messages": messages,
            "metadata": {**metadata, "fund_report_executed": True}
        }

    except Exception as e:
        logger.error(
            f"{ERROR_ICON} FundReportAgent: Error generating fund report: {e}",
            exc_info=True
        )

        # 生成错误报告
        error_report = _build_error_report(
            fund_code, fund_name, fund_type, current_time_info, str(e), analysis_package
        )
        current_data["fund_report"] = error_report
        current_data["fund_report_error"] = f"Error generating fund report: {e}"

        # 保存错误报告
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_fund_name = fund_name.replace(" ", "_").replace(".", "").replace("/", "_")
        clean_fund_code = fund_code.replace("sh.", "").replace("sz.", "").replace("of.", "")
        report_filename = f"fund_error_report_{safe_fund_name}_{clean_fund_code}_{timestamp}.md"
        reports_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "reports"
        )
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(error_report)
        current_data["fund_report_path"] = report_path

        execution_logger.log_agent_complete(
            agent_name, current_data, time.time() - agent_start_time, False, str(e)
        )

        return {
            "data": current_data,
            "messages": messages,
            "metadata": {**metadata, "fund_report_executed": True}
        }


def _build_empty_package_report(
    fund_code: str,
    fund_name: str,
    current_time_info: str,
) -> str:
    """当fund_analysis_package为空时，生成说明报告。"""
    return f"""# {fund_name}（{fund_code}）基金分析报告

## 报告封面信息
- 报告生成日期：{current_time_info}
- 基金：{fund_name}（{fund_code}）
- 状态：**数据不可用**

## 核心结论摘要
很抱歉，由于分析数据包为空，无法生成完整的基金分析报告。

**可能的原因**：
1. 上游基金分析Agent未能成功获取数据
2. 数据合并节点（fund_merge_node）未正常执行
3. 该基金代码可能无效或暂不支持

**建议**：
- 请检查基金代码是否正确
- 请确认上游分析Agent是否正常执行
- 可通过系统日志查看具体错误信息

## 免责声明
本报告由AI投资分析系统自动生成。由于数据缺失，本次分析未能完成。建议通过基金公司官网或第三方基金研究平台获取该基金的详细信息。

**投资有风险，入市需谨慎。**
"""


def _build_error_report(
    fund_code: str,
    fund_name: str,
    fund_type: str,
    current_time_info: str,
    error_msg: str,
    analysis_package: Dict[str, Any],
) -> str:
    """当LLM调用失败时，基于已有分析包生成降级报告。"""

    normalized_subscores = analysis_package.get("normalized_subscores", {})
    strengths_pool = analysis_package.get("strengths_pool", [])
    risks_pool = analysis_package.get("risks_pool", [])
    fund_profile = analysis_package.get("fund_profile", {})

    # 计算综合得分
    if normalized_subscores:
        total_score = round(sum(normalized_subscores.values()) / len(normalized_subscores), 1)
    else:
        total_score = "N/A"

    subscore_table = "\n".join(
        f"| {k} | {v} | - | - |"
        for k, v in normalized_subscores.items()
    ) if normalized_subscores else "| - | - | - | - |"

    strengths_list = "\n".join(f"- {s}" for s in strengths_pool) if strengths_pool else "- 暂无"
    risks_list = "\n".join(f"- {r}" for r in risks_pool) if risks_pool else "- 暂无"

    profile_items = "\n".join(
        f"- **{k}**: {v}" for k, v in fund_profile.items()
    ) if fund_profile else "- 暂无"

    return f"""# {fund_name}（{fund_code}）基金分析报告（降级版）

> ⚠️ 报告生成过程中LLM调用失败，以下为基于原始分析数据的降级报告，缺少AI综合分析和解读。

## 报告封面信息
- 报告生成日期：{current_time_info}
- 基金名称：{fund_name}
- 基金代码：{fund_code}
- 基金类型：{fund_type}

{profile_items}

## 综合得分
综合得分：{total_score}/100

## 各维度评分
| 评分维度 | 得分 | 评级 | 简要评价 |
|---------|------|------|---------|
{subscore_table}

## 汇总优势
{strengths_list}

## 汇总风险
{risks_list}

## 错误详情
生成完整报告时发生错误：{error_msg}

## 免责声明
本报告为降级版本，仅展示原始分析数据。AI综合分析因技术错误未能生成。请通过标准流程重新尝试分析。

**投资有风险，入市需谨慎。**
"""


# ── 本地测试函数 ──
async def test_fund_report_agent():
    """基金报告Agent的测试函数"""
    from src.utils.state_definition import AgentState

    test_package = {
        "fund_profile": {
            "fund_code": "sh.510050",
            "fund_name": "华夏上证50ETF",
            "fund_type": "ETF",
            "benchmark": "上证50指数",
            "risk_level": "R4",
            "inception_date": "2004-12-30",
            "management_company": "华夏基金管理有限公司",
        },
        "normalized_subscores": {
            "product_positioning": 85,
            "performance_risk": 78,
            "portfolio_structure": 84,
            "manager_team": 80,
            "benchmark_style_consistency": 79,
            "fee_liquidity": 76,
            "event_risk": 83,
        },
        "strengths_pool": [
            "产品定位清晰，紧密跟踪上证50指数",
            "管理团队稳定，华夏基金投研实力强",
            "流动性极佳，日均成交额超10亿",
            "费率在同类中处于较低水平",
        ],
        "risks_pool": [
            "行业集中度偏高，金融行业占比超30%",
            "大盘价值风格敞口大，中小盘牛市可能跑输",
            "跟踪误差在市场剧烈波动时有放大趋势",
        ],
        "conflicts": [
            "产品定位Agent评级较高(85)，费用Agent认为费率仍有优化空间(76)"
        ],
        "holding_period_hints": [
            {"label": "建议1年以上", "reason": "指数基金长期持有胜率高", "source_agent": "fund_perf_risk"},
            {"label": "适合定投", "reason": "波动适中，定投可平滑成本", "source_agent": "fund_fee"},
        ],
        "confidence_summary": 0.81,
        "missing_data_summary": [
            "基金经理详细从业经历数据部分缺失",
        ],
        "frontend_ready_tags": ["中高波动", "被动指数", "1年以上", "大盘价值"],
        "raw_agent_outputs": {
            "fund_product_doc": "该基金为被动指数型ETF，跟踪上证50指数。上证50指数由沪市A股中规模大、流动性好的50只股票组成，综合反映上海证券市场最具影响力的一批龙头企业的整体状况。基金采用完全复制法跟踪标的指数，年化跟踪误差目标控制在2%以内。",
            "fund_perf_risk": "近1年收益12.3%，近3年年化收益5.8%，近5年年化收益3.2%。最大回撤-28.5%（发生在2021-2022年）。夏普比率0.42，信息比率0.15。上行捕获率95%，下行捕获率97%。",
            "fund_holdings": "前十大持仓占比42%，行业集中在金融(32%)、食品饮料(18%)、医药(8%)。前五大重仓：贵州茅台、招商银行、中国平安、兴业银行、长江电力。",
            "fund_manager": "基金经理从业年限10年+，管理规模超800亿。华夏基金为头部公募，总管理规模超1.5万亿。基金经理团队稳定，近3年无变更。",
            "fund_benchmark": "跟踪误差年化1.2%，信息比率0.15。风格与上证50指数高度一致，无显著风格漂移。大盘价值风格稳定。",
            "fund_fee": "管理费0.5%/年，托管费0.1%/年。ETF交易佣金按券商标准收取。总费用率0.6%，在同类ETF中处于较低水平。日均成交额超10亿，流动性充裕。",
            "fund_event": "近期无重大负面事件。注意：2025年分红方案已公告，每份分红0.15元。基金经理未发生变更。",
        },
    }

    test_state = AgentState(
        messages=[],
        data={
            "fund_code": "sh.510050",
            "fund_name": "华夏上证50ETF",
            "fund_analysis_package": test_package,
            "current_time_info": "2026年6月7日 14:30:00",
        },
        metadata={}
    )

    result = await fund_report_agent(test_state)
    report = result.get("data", {}).get("fund_report", "No report generated")
    print("=" * 80)
    print("FUND REPORT:")
    print("=" * 80)
    print(report[:2000])
    if len(report) > 2000:
        print(f"\n... [总长度: {len(report)} 字符] ...")
    print("=" * 80)
    print(f"Report saved to: {result.get('data', {}).get('fund_report_path', 'Not saved')}")
    print(f"Metadata fund_report_executed: {result.get('metadata', {}).get('fund_report_executed')}")

    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fund_report_agent())
