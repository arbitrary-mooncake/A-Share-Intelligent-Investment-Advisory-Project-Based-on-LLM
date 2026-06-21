"""
Summary Agent: Consolidates analyses from other agents into a final report.
汇总 Agent：将其他 Agent的分析结果整合成最终报告
"""
import os
import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI
import re

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.utils.model_config import get_model_config_for_agent, get_thinking_body
from src.utils.pdf_generator import markdown_to_pdf
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv(override=True)

logger = setup_logger(__name__)


def truncate_report_at_baseline_time(report_content: str, current_time_info: str) -> str:
    """
    使用正则表达式截断报告，在"分析基准时间"那一行之后停止

    Args:
        report_content: 完整的报告内容
        current_time_info: 当前时间信息

    Returns:
        截断后的报告内容
    """
    # 构建多种可能的"分析基准时间"模式
    baseline_patterns = [
        rf'分析基准时间[：:]\s*{re.escape(current_time_info)}',
        rf'分析基准时间[：:]\s*{re.escape(current_time_info)}\s*$',
        rf'基准时间[：:]\s*{re.escape(current_time_info)}',
        rf'时间基准[：:]\s*{re.escape(current_time_info)}',
        rf'分析时间[：:]\s*{re.escape(current_time_info)}',
        rf'报告时间[：:]\s*{re.escape(current_time_info)}',
        rf'生成时间[：:]\s*{re.escape(current_time_info)}',
        rf'更新时间[：:]\s*{re.escape(current_time_info)}',
        rf'数据时间[：:]\s*{re.escape(current_time_info)}',
        rf'分析基准[：:]\s*{re.escape(current_time_info)}'
    ]

    # 尝试匹配各种模式
    for pattern in baseline_patterns:
        match = re.search(pattern, report_content, re.MULTILINE | re.IGNORECASE)
        if match:
            # 找到"分析基准时间"行，检查它是否在报告开头（标题之后）
            # 如果是，说明这是正常的报告结构，返回完整内容
            line_end = report_content.find('\n', match.end())
            if line_end == -1:
                line_end = match.end()
            # 如果基准时间行在报告前500字符内，说明是正常的报告头部，
            # 保留完整内容（标题+基准时间+正文），只在末尾做LLM续写截断
            if match.start() < 500:
                logger.info(f"分析基准时间在报告头部(位置{match.start()})，保留完整内容")
                return report_content.strip()
            # 如果基准时间出现在报告中部（重复标题），截断到此处
            truncated_content = report_content[:line_end].strip()
            logger.info(f"截断报告在重复'分析基准时间'行，截断位置: {match.end()}")
            return truncated_content

    # 如果没有找到匹配的模式，尝试查找包含时间信息的行
    time_patterns = [
        rf'.*{re.escape(current_time_info)}.*',
        rf'.*{re.escape(current_time_info.split()[0])}.*',  # 只匹配日期部分
        rf'.*{re.escape(current_time_info.split()[1])}.*'   # 只匹配时间部分
    ]

    for pattern in time_patterns:
        match = re.search(pattern, report_content, re.MULTILINE | re.IGNORECASE)
        if match:
            end_pos = match.end()
            line_end = report_content.find('\n', end_pos)
            if line_end == -1:
                truncated_content = report_content[:end_pos].strip()
            else:
                truncated_content = report_content[:line_end].strip()

            logger.info(f"截断报告在时间信息行之后，截断位置: {end_pos}")
            return truncated_content

    # 如果都没有找到，返回原始内容
    logger.warning("未找到'分析基准时间'模式，返回原始报告内容")
    return report_content


async def summary_agent(state: AgentState) -> Dict[str, Any]:
    """
    整合基本面、技术面、估值和新闻分析的结果
    使用LLM生成最终的综合性报告
    """
    logger.info(f"{WAIT_ICON} SummaryAgent: Starting to consolidate analyses.")

    # 获取执行日志记录器，用于记录 Agent的执行过程
    execution_logger = get_execution_logger()
    agent_name = "summary_agent"

    # 从状态中提取当前数据、消息和用户查询
    current_data = state.get("data", {})
    messages = state.get("messages", [])
    user_query = current_data.get("query", "")

    # 记录 Agent开始执行，包含可用的分析类型
    execution_logger.log_agent_start(agent_name, {
        "user_query": user_query,
        "available_analyses": {
            "fundamental": "fundamental_analysis" in current_data,
            "technical": "technical_analysis" in current_data,
            "value": "value_analysis" in current_data,
            "news": "news_analysis" in current_data,
            "event": "event_analysis" in current_data,
            "quality_risk": "quality_risk_analysis" in current_data,
            "moneyflow": "moneyflow_analysis" in current_data,
        },
        "input_data_keys": list(current_data.keys())
    })

    # 记录 Agent开始时间，用于计算执行时长
    agent_start_time = time.time()

    # 获取之前 Agent的分析结果（mimo 大上下文，无需截断）
    current_date = current_data.get("current_date", "未知日期")

    def _sanitize(text: str) -> str:
        """Clean raw analysis text: replace tool-level data-unavailable phrases
        that mislead the summary LLM into thinking entire modules are missing."""
        import re as _re
        if not text or text == "Not available":
            return text
        t = text
        t = _re.sub(r'tushare_\w+\s*数据不可用', '[该子项工具未返回数据]', t)
        t = _re.sub(r'\[[^\]]+\]\s*数据不可用[（(](?:超时|返回过短|调用异常)[)）]', '[该子项工具未返回数据]', t)
        t = _re.sub(r'数据不可用[（(](?:超时|返回过短)[)）]', '[该子项工具未返回数据]', t)
        return t

    fundamental_analysis = _sanitize(current_data.get("fundamental_analysis", "Not available"))
    technical_analysis = _sanitize(current_data.get("technical_analysis", "Not available"))
    value_analysis = _sanitize(current_data.get("value_analysis", "Not available"))
    news_analysis = _sanitize(current_data.get("news_analysis", "Not available"))
    event_analysis = _sanitize(current_data.get("event_analysis", "Not available"))
    quality_risk_analysis = _sanitize(current_data.get("quality_risk_analysis", "Not available"))
    moneyflow_analysis = _sanitize(current_data.get("moneyflow_analysis", "Not available"))

    from src.utils.analysis_package_builder import build_analysis_package
    pkg = build_analysis_package(current_data, current_date)

    # 处理各个分析的错误信息
    errors = []
    if "fundamental_analysis_error" in current_data:
        errors.append(
            f"Fundamental Analysis Error: {current_data['fundamental_analysis_error']}")
    if "technical_analysis_error" in current_data:
        errors.append(
            f"Technical Analysis Error: {current_data['technical_analysis_error']}")
    if "value_analysis_error" in current_data:
        errors.append(
            f"Value Analysis Error: {current_data['value_analysis_error']}")
    if "news_analysis_error" in current_data:
        errors.append(
            f"News Analysis Error: {current_data['news_analysis_error']}")
    if "event_analysis_error" in current_data:
        errors.append(
            f"Event Analysis Error: {current_data['event_analysis_error']}")
    if "quality_risk_analysis_error" in current_data:
        errors.append(
            f"Quality Risk Analysis Error: {current_data['quality_risk_analysis_error']}")
    if "moneyflow_analysis_error" in current_data:
        errors.append(
            f"Moneyflow Analysis Error: {current_data['moneyflow_analysis_error']}")

    # 基本股票标识信息
    stock_code = current_data.get("stock_code", "Unknown Stock")
    company_name = current_data.get("company_name", "Unknown Company")

    try:
        # 获取当前时间信息，用于报告中的时间标注
        current_time_info = current_data.get("current_time_info", "未知时间")
        current_date = current_data.get("current_date", "未知日期")

        # 准备汇总的系统提示词
        system_prompt = f"""
        你是一位资深A股证券分析师，拥有10年以上卖方研究经验。

        **重要时间信息：当前实际时间是 {current_time_info}**
        **分析基准日期：{current_date}**

        你的任务是综合7种分析结果，创建一份结构清晰的专业研究报告。

        ## 报告结构（必须严格遵循此9段格式）

        ### 1. 核心结论
        [3-5句话：最核心的投资判断、关键理由、总体评级]

        ### 2. 多维信号总览
        [表格或列表形式：列出每个分析维度的方向(看多/中性/看空)和置信度]
        [必须包含: 基本面、技术面、估值、新闻舆情、事件驱动、质量风险、资金面]

        ### 3. 关键利多因素
        [列出最重要的看多证据，每条标注来源agent和数据基础]
        [区分: [数据] 事实 vs [判断] 推断 vs [建议] 建议]

        ### 4. 关键利空与反证
        [必须写！列出最重要的看空证据和风险点]
        [当不同agent结论冲突时，必须明确写出冲突，不要只保留一种声音]

        ### 5. 事件与催化剂时间线
        [近期已发生和即将发生的关键事件，标注日期/时效/影响方向/影响期限]

        ### 6. 短线 / 中线 / 长线判断
        [分别给出三个期限的专业判断，明确区分不同期限的逻辑和风险]
        [短线: 量价/资金/催化剂驱动；中线: 基本面/估值/事件；长线: 护城河/行业/治理]

        ### 7. 主要风险与需要继续核验的数据
        [列出剩余不确定性、数据缺口、需要后续跟踪的指标]
        [数据不足时必须承认不确定性]

        ### 8. 结论的置信度与适用边界
        [当前结论的置信度评估(高/中/低)，在什么条件下结论会改变]
        [说明该结论适用的投资者类型和市场环境]

        ### 9. 风险提示
        [市场有风险，投资需谨慎]
        [具体到该股票的风险提示，不使用泛泛套话]

        ## 写作原则
        1. 明确区分[数据]事实、[判断]推断、[建议]建议
        2. 当不同分析维度冲突时，必须写出冲突而非只保留一种声音
        3. 必须有反证部分
        4. 数据不足时承认不确定性
        5. 使用简洁专业语言，避免空泛套话

        ⛔ 防幻觉规则:
        - 所有陈述必须标注 [数据] 或 [判断]
        - 禁止编造数值、新闻、或未在输入中出现的事实
        - 数据可用性判断规则：
          1. 只有在"结构化分析摘要"中某模块显示为"未执行agent"时，才写"该模块数据不可用"
          2. 如果"结构化分析摘要"中某模块有 bias 和置信度（即使置信度较低），说明该模块已产出分析结论，必须正常展示其核心观点
          3. 某模块原始分析中提到"部分数据缺失"（如"未获取到商誉数据"）不等于整个模块缺失——应在第7节"数据缺口"中注明，而不是在第2节总览中将该模块标记为不可用
          4. 第2节总览表格中，每个模块的方向和置信度应以"结构化分析摘要"为准，不要自行降级

        输出为纯Markdown，不含代码块标记。
        """

        # 准备汇总提示词
        user_prompt = f"""
请为 {company_name} ({stock_code}) 创建综合分析报告。

原始用户查询: {user_query}

## 各维度原始分析（参考用，注意：原始文本中提到的部分数据缺口不代表整个模块缺失）

FUNDAMENTAL ANALYSIS:
{fundamental_analysis}

TECHNICAL ANALYSIS:
{technical_analysis}

VALUE ANALYSIS:
{value_analysis}

NEWS ANALYSIS:
{news_analysis}

EVENT ANALYSIS:
{event_analysis}

QUALITY & RISK ANALYSIS:
{quality_risk_analysis}

MONEYFLOW ANALYSIS:
{moneyflow_analysis}

{"ANALYSIS ISSUES:" if errors else ""}
{". ".join(errors) if errors else ""}

## 结构化分析摘要（⚠️ 最终权威 — 以此为准判断各模块是否有数据）
{pkg.compact_prompt_context}

⚠️ 重要：上方结构化摘要中列出的每个agent，只要显示有bias和置信度，就代表该模块已成功产出分析结论。即使上方原始分析文本中提到"部分数据缺失"或"未获取到XX"，也不代表整个模块不可用——那只是该模块中某些子项的数据缺口，应在报告第7节中提及，而不是在第2节总览中标记为不可用。

IMPORTANT: Output in valid Markdown with proper headings. No code block markers.
"""

        # 使用OpenAI API生成报告
        logger.info(f"{WAIT_ICON} SummaryAgent: Using OpenAI API...")

        # 模型配置：优先 state 覆盖（快筛），否则使用 agent 分配的模型 (MiMo-V2.5-Pro)
        model_cfg = get_model_config_for_agent("summary_agent", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]

        # 验证必要的环境变量是否存在
        if not all([api_key, base_url, model_name]):
            logger.error(
                f"{ERROR_ICON} SummaryAgent: Missing OpenAI environment variables.")
            current_data["summary_error"] = "Missing OpenAI environment variables."

            execution_logger.log_agent_complete(agent_name, current_data, time.time(
            ) - agent_start_time, False, "Missing OpenAI environment variables")

            return {"data": current_data, "messages": messages}

        # 记录模型配置信息
        model_config = {
            "model": model_name,
            "temperature": 0.6,
            "max_tokens": 32000,
            "thinking": "enabled",
            "api_base": base_url
        }

        # 准备汇总提示词消息列表
        summary_prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 使用ChatOpenAI模型
        logger.info(f"{WAIT_ICON} SummaryAgent: Creating ChatOpenAI with model {model_name}")
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.6,
            request_timeout=720,
            max_tokens=32000,
            extra_body=get_thinking_body(base_url, True)
        )

        # 记录LLM交互开始时间
        llm_start_time = time.time()

        # 调用LLM生成最终报告
        llm_message = await llm.ainvoke(summary_prompt_messages)
        final_report = llm_message.content

        # 记录LLM交互执行时间
        llm_execution_time = time.time() - llm_start_time

        # 记录LLM交互详情，用于后续分析和优化
        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="summary_generation",
            input_messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            output_content=final_report,
            model_config=model_config,
            execution_time=llm_execution_time
        )

        # 移除任何可能出现的markdown代码块标记
        final_report = final_report.replace(
            "```markdown", "").replace("```", "").strip()

        # 保留完整报告内容（LLM可能产生少量后记，但不影响正文阅读）
        # truncate_report_at_baseline_time 存在逻辑缺陷已被禁用

        logger.info(
            f"{SUCCESS_ICON} SummaryAgent: Final report generated for {company_name} ({stock_code}).")
        logger.debug(f"Final report preview: {final_report[:300]}...")

        # 将报告保存到文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 处理公司名称和股票代码，确保文件名有意义
        if stock_code == "Unknown Stock" or stock_code == "Extracted from analysis":
            query_based_name = user_query.replace(
                " ", "_").replace("分析", "").strip()
            if not query_based_name:
                query_based_name = "financial_analysis"
            safe_file_prefix = f"report_{query_based_name}"
        else:
            safe_company_name = company_name.replace(" ", "_").replace(".", "")
            if safe_company_name == "Unknown_Company" or safe_company_name == "Extracted_from_analysis":
                safe_company_name = user_query.replace(
                    " ", "_").replace("分析", "").strip()
                if not safe_company_name:
                    safe_company_name = "company"

            clean_stock_code = stock_code.replace("sh.", "").replace("sz.", "")
            safe_file_prefix = f"report_{safe_company_name}_{clean_stock_code}"

        report_filename = f"{safe_file_prefix}_{timestamp}.md"
        pdf_filename = f"{safe_file_prefix}_{timestamp}.pdf"

        # 确保reports目录存在
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, report_filename)
        pdf_path = os.path.join(reports_dir, pdf_filename)

        # 将报告写入 Markdown 文件（保留作为备份）
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)

        logger.info(
            f"{SUCCESS_ICON} SummaryAgent: Report saved to {report_path}")

        # 生成 PDF 版本
        try:
            markdown_to_pdf(
                final_report, pdf_path,
                company_name=company_name,
                stock_code=stock_code
            )
            logger.info(
                f"{SUCCESS_ICON} SummaryAgent: PDF report saved to {pdf_path}")
        except Exception as pdf_err:
            logger.warning(
                f"Failed to generate PDF report: {pdf_err}. "
                f"Markdown report is still available at {report_path}"
            )
            pdf_path = None

        # 返回更新后的状态，包含最终报告
        current_data["final_report"] = final_report
        current_data["report_path"] = report_path
        current_data["report_pdf_path"] = pdf_path

        # 记录 Agent执行成功
        total_execution_time = time.time() - agent_start_time
        execution_logger.log_agent_complete(agent_name, {
            "final_report_length": len(final_report),
            "report_path": report_path,
            "report_pdf_path": pdf_path,
            "report_preview": final_report,
            "llm_execution_time": llm_execution_time,
            "total_execution_time": total_execution_time
        }, total_execution_time, True)

        return {"data": current_data, "messages": messages}

    except Exception as e:
        logger.error(
            f"{ERROR_ICON} SummaryAgent: Error generating final report: {e}", exc_info=True)
        current_data["summary_error"] = f"Error generating final report: {e}"

        # 即使出现错误也创建最小化的报告
        error_report = f"""
        # Analysis Report for {company_name} ({stock_code})

        **Error encountered during report generation**: {e}

        ## Available Analysis Fragments:

        - Fundamental Analysis: {"Available" if fundamental_analysis != "Not available" else "Not available"}
        - Technical Analysis: {"Available" if technical_analysis != "Not available" else "Not available"}
        - Value Analysis: {"Available" if value_analysis != "Not available" else "Not available"}
        - News Analysis: {"Available" if news_analysis != "Not available" else "Not available"}

        Please review the individual analyses directly for more information.
        """
        current_data["final_report"] = error_report

        # 也将错误报告保存到文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if stock_code == "Unknown Stock" or stock_code == "Extracted from analysis":
            query_based_name = user_query.replace(
                " ", "_").replace("分析", "").strip()
            if not query_based_name:
                query_based_name = "financial_analysis"
            safe_file_prefix = f"error_report_{query_based_name}"
        else:
            safe_company_name = company_name.replace(" ", "_").replace(".", "")
            if safe_company_name == "Unknown_Company" or safe_company_name == "Extracted_from_analysis":
                safe_company_name = user_query.replace(
                    " ", "_").replace("分析", "").strip()
                if not safe_company_name:
                    safe_company_name = "company"

            clean_stock_code = stock_code.replace("sh.", "").replace("sz.", "")
            safe_file_prefix = f"error_report_{safe_company_name}_{clean_stock_code}"

        report_filename = f"{safe_file_prefix}_{timestamp}.md"
        pdf_filename = f"{safe_file_prefix}_{timestamp}.pdf"

        reports_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, report_filename)
        pdf_path = os.path.join(reports_dir, pdf_filename)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(error_report)

        logger.info(
            f"{ERROR_ICON} SummaryAgent: Error report saved to {report_path}")

        # 尝试生成 PDF 版本
        try:
            markdown_to_pdf(
                error_report, pdf_path,
                company_name=company_name,
                stock_code=stock_code
            )
        except Exception as pdf_err:
            logger.warning(f"Failed to generate error PDF: {pdf_err}")
            pdf_path = None

        current_data["report_path"] = report_path
        current_data["report_pdf_path"] = pdf_path

        execution_logger.log_agent_complete(
            agent_name, current_data, time.time() - agent_start_time, False, str(e))

        return {"data": current_data, "messages": messages}


# 本地测试函数
async def test_summary_agent():
    """汇总 Agent的测试函数"""
    from src.utils.state_definition import AgentState

    # 用于测试的示例状态，包含模拟分析结果
    test_state = AgentState(
        messages=[],
        data={
            "query": "分析嘉友国际",
            "stock_code": "603871",
            "company_name": "嘉友国际",
            "fundamental_analysis": "嘉友国际基本面分析：公司主营业务为跨境物流、供应链贸易以及供应链增值服务。财务状况良好，负债率较低，现金流充裕。近年来业绩稳步增长，毛利率保持在行业较高水平。",
            "technical_analysis": "嘉友国际技术分析：短期内股价处于上升通道，突破了200日均线。RSI指标显示股票尚未达到超买区域。MACD指标呈现多头形态，成交量有所放大，支持价格继续上行。",
            "value_analysis": "嘉友国际估值分析：当前市盈率为15倍，低于行业平均水平。市净率为1.8倍，处于合理区间。与同行业公司相比，嘉友国际的估值较为合理，具有一定的投资价值。",
            "news_analysis": "嘉友国际新闻分析：近期公司发布了2023年业绩预告，预计净利润同比增长15-25%，超出市场预期。同时，公司宣布与多家国际物流巨头达成战略合作，市场反应积极。分析师普遍上调了目标价，市场情绪偏向乐观。"
        },
        metadata={}
    )

    # 运行 Agent并输出结果
    result = await summary_agent(test_state)
    print("Summary Report:")
    print(result.get("data", {}).get("final_report", "No report generated"))
    print(
        f"Report saved to: {result.get('data', {}).get('report_path', 'Not saved')}")

    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_summary_agent())
