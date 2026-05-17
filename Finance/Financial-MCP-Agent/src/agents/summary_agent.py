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
from src.utils.model_config import get_model_config_for_agent
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
            "news": "news_analysis" in current_data
        },
        "input_data_keys": list(current_data.keys())
    })

    # 记录 Agent开始时间，用于计算执行时长
    agent_start_time = time.time()

    # 获取之前 Agent的分析结果（mimo 大上下文，无需截断）
    fundamental_analysis = current_data.get(
        "fundamental_analysis", "Not available")
    technical_analysis = current_data.get(
        "technical_analysis", "Not available")
    value_analysis = current_data.get("value_analysis", "Not available")
    news_analysis = current_data.get("news_analysis", "Not available")

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

    # 基本股票标识信息
    stock_code = current_data.get("stock_code", "Unknown Stock")
    company_name = current_data.get("company_name", "Unknown Company")

    try:
        # 获取当前时间信息，用于报告中的时间标注
        current_time_info = current_data.get("current_time_info", "未知时间")
        current_date = current_data.get("current_date", "未知日期")

        # 准备汇总的系统提示词
        system_prompt = f"""
        你是一位资深A股证券分析师，拥有10年以上卖方研究经验，擅长撰写券商级个股深度研究报告。

        **重要时间信息：当前实际时间是 {current_time_info}**
        **分析基准日期：{current_date}**

        这是真实的当前时间，不是你的训练数据截止时间。请在生成报告时：
        - 基于实际当前时间来判断数据的时效性
        - 正确标注"最新"、"近期"、"历史"等时间概念
        - 在报告中明确标注分析的时间基准点为：{current_date}
        - 所有时间相关的描述都要基于这个实际日期

        你的任务是综合四种不同的分析结果：
        1. 基本面分析 - 关注财务报表、商业模式、杜邦分析(ROE拆解)、现金流质量、资产负债结构。⚠️ 基本面分析中已包含ST风险警示数据，必须提取并在ST风险专区中使用
        2. 技术分析 - 关注价格趋势、均线系统、支撑位和阻力位、MACD/RSI等技术指标、成交量配合
        3. 估值分析 - 关注PE/PB/PS/EV-EBITDA等估值指标、相对行业估值溢价、历史估值分位、股息收益率
        4. 新闻分析 - 关注市场情绪、重要新闻事件、资金流向、分析师评级变化、媒体报道对股价的影响。⚠️ 新闻分析中可能包含ST风险筛查结果，注意提取使用

        特别说明：基本面分析Agent已调用Tushare/AkShare接口查询ST状态，新闻分析Agent也进行了ST风险筛查。
        你必须在ST风险警示专区中整合这些数据，不得遗漏或编造ST相关数据。

        请创建一份结构清晰、内容连贯的券商级研究报告，整合所有四种分析的见解。
        即使某些分析数据不完整或缺失，也请基于可用信息提供最佳的综合分析。

        **严格遵循以下报告格式和结构（参照券商研报"3×9×3"框架）：**

        # [公司名称]([股票代码]) 综合分析报告

        ## 执行摘要
        [提供简明扼要的总体分析和投资评级（买入/增持/中性/减持，相对沪深300基准），包括核心投资逻辑(3-4个要点)、风险等级和预期回报]

        投资评级：[买入/增持/中性/减持]
        风险等级：[低/中/高]
        综合评分：[0-100分的具体数字]

        ## 公司概况
        [简要介绍公司的业务模式、行业地位、主要产品或服务、股权结构、核心竞争力（护城河分析）]

        ## 基本面分析
        [
          - 盈利能力：ROE（杜邦分析：净利率×资产周转率×权益乘数）、毛利率、净利率趋势
          - 成长性：营收增速、利润增速、与行业对比
          - 运营效率：应收账款周转率、存货周转率、总资产周转率
          - 偿债能力：资产负债率、流动比率、利息保障倍数
          - 现金流质量：经营性现金流/净利润比率、自由现金流状况
        ]
        基本面评分：[0-100分的具体数字]

        ## 技术分析
        [
          - 价格趋势：短期/中期趋势判断、关键均线(20/60/200日)位置
          - 技术指标：MACD金叉/死叉、RSI超买/超卖区域
          - 关键价位：支撑位和阻力位、成交量配合情况
        ]
        技术面评分：[0-100分的具体数字]

        ## 估值分析
        [
          - 相对估值：PE/PB/PS与行业均值对比、历史估值分位(过去3年/5年)
          - 绝对估值：如有DCF/FCFF结果请引用
          - 估值合理性：当前估值是否充分反映增长预期
        ]
        估值评分：[0-100分的具体数字]

        ## 新闻分析
        [
          - 近期重要新闻事件及其对公司的影响
          - 市场情绪分析：投资者关注度、资金流向、分析师评级变化
          - 每条新闻的风险评估(1-5分)和情感评分(1-5分)的综合解读
        ]
        舆情评分：[0-100分的具体数字]

        ## 综合评估
        [
          - 四种分析方法的一致性分析：哪些结论相互印证？哪些存在分歧？
          - 对于分歧点，给出你的专业判断和权衡依据
          - 核心催化剂：未来6-12个月可能驱动股价的关键事件
        ]

        ## 风险因素
        [
          - 系统性风险：宏观经济下行、货币政策收紧、地缘政治风险
          - 行业风险：行业竞争加剧、政策变化、技术替代
          - 公司特定风险：客户集中度高、管理层变动、诉讼纠纷、大股东减持
          - 必须声明："市场有风险，投资需谨慎"
        ]

        ## ST风险警示专区 ⚠️
        [
          - **当前ST状态**：[数据] 基于实际查询结果：正常/ST/*ST
          - **ST类型**：[数据] 退市风险警示（*ST）/ 其他风险警示（ST）/ 无
          - **风险起始日期**：[数据] 进入风险警示板的具体日期，或标注"不适用"
          - **触发原因分析**：[判断] 基于财务数据的综合分析：
            - 连续亏损风险：近两年净利润是否持续为负
            - 净资产风险：最近一期净资产是否已为负值
            - 审计风险：审计报告是否存在否定意见或无法表示意见
            - 重大违规：是否涉及重大信息披露违法违规或财务造假
          - **退市风险等级**：[判断] 基于ST状态+财务指标+持续时间的综合评估（低/中/高/极高）
          - **交易限制提醒**：[数据] 风险警示板交易规则：
            - 涨跌幅限制±5%（2026新规，此前风险警示板为±5%）
            - 投资者需签署风险揭示书方可买入
            - 单日买入数量限制50万股
          - **整改可能性评估**：[判断] 基于行业地位、财务改善潜力、股东背景的综合判断
          - **对投资决策的影响**：[判断] ST状态对估值、流动性、投资策略的具体影响分析

          ⚠️ 如果各Agent的ST查询结果均为"数据不可用"，必须声明："本次分析中ST风险数据暂不可用，建议投资者通过交易所官网自行查询最新ST状态。"
        ]

## 投资建议
        [
          - 明确的投资评级（买入/增持/中性/减持）
          - 目标价格及估值方法
          - 建议投资时间范围（短线/中线/长线）
          - 适合的投资者类型（稳健型/成长型/风险偏好型）
          - 若股票处于ST/*ST状态，投资建议应极度谨慎，并明确提示退市风险
        ]

        ## 数据可靠性声明
        [必须包含以下内容：
        - 列出各分析模块的数据完整度（如：基本面数据完整/技术面数据部分缺失）
        - 标注哪些结论有直接数据支撑，哪些是分析师推断
        - 如果某个分析模块的数据完全缺失，必须在此明确声明
        - 声明格式："数据可靠性评估：[高/中/低]，具体说明..."]

        ## 附录：数据来源与限制
        [说明数据来源(MCP工具、Tushare、新浪等)，以及分析过程中遇到的任何数据限制或缺失]

        ⛔ 防幻觉输出规则（必须严格遵守）：
        1. 报告中使用以下标记区分信息来源：
           - **[数据]** 开头 = 该陈述有API返回的具体数据支持
           - **[判断]** 开头 = 该陈述是分析师的推断或观点
        2. 每个关键结论前必须标注 [数据] 或 [判断]
        3. 如果某个分析模块数据缺失，必须写"该模块数据不可用"而不是编造内容
        4. 评分章节中的分数是判断，但必须引用 [数据] 段落中的具体数值作为依据
        5. 绝对禁止编造任何数值、新闻、公司名、或未在分析输入中出现的事实

        输出必须是有效的Markdown格式，使用适当的标题、项目符号和格式。
        不要包含任何代码块标记，如```markdown或```，直接输出纯Markdown内容。

        使用专业的金融语言和券商研报写作风格，报告应该全面且深入，包含足够的细节和数据支持，
        同时聚焦于最重要的见解，帮助投资者做出决策。

        **重要提醒：**
        - 请在报告末尾明确标注分析基准时间：{current_time_info}
        - 基于这个实际时间来判断所有数据的时效性
        - 避免使用模糊的时间概念，要基于实际当前时间进行判断
        - 严格按照上述格式和结构生成报告，确保每个章节都有实质性内容
        - 分析要有数据支撑，避免空洞的定性描述，尽量引用具体数字
        - **评分必须为0-100之间的具体整数**，不得省略
        - **所有陈述必须标注 [数据] 或 [判断]，不可遗漏**

        如果某些分析数据不完整或有错误，请在报告中明确说明，并尽可能基于可用信息提供有价值的分析。
        """

        # 准备汇总提示词
        user_prompt = f"""
        Please create a comprehensive analysis report for {company_name} ({stock_code}) based on the following analyses.

        Original user query: {user_query}

        FUNDAMENTAL ANALYSIS:
        {fundamental_analysis}

        TECHNICAL ANALYSIS:
        {technical_analysis}

        VALUE ANALYSIS:
        {value_analysis}

        NEWS ANALYSIS:
        {news_analysis}

        {"ANALYSIS ISSUES:" if errors else ""}
        {". ".join(errors) if errors else ""}

        IMPORTANT: Your output MUST be in valid Markdown format with proper headings, bullet points,
        and formatting. Include a clear recommendation section at the end.

        DO NOT include any code block markers like ```markdown or ``` in your output.
        Just write pure Markdown content directly.
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
            "temperature": 1.0,
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
            temperature=1.0,
            request_timeout=720,
            max_tokens=32000,
            extra_body={"thinking": {"type": "enabled"}}  # 大上下文+深度思考，需要充足时间
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
