"""
FundamentalAnalysis Agent: Performs fundamental analysis of a stock using ReAct Agent framework.
基本面分析 Agent：使用ReAct Agent框架对股票进行基本面分析
"""
import os
import json
from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langgraph.prebuilt import create_react_agent
import time

from src.utils.state_definition import AgentState
from src.tools.mcp_client import get_mcp_tools
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv(override=True)

logger = setup_logger(__name__)


async def fundamental_agent(state: AgentState) -> AgentState:
    """
    使用ReAct框架进行基本面分析，直接集成MCP工具
    
    Args:
        state: 包含用户查询的当前 Agent状态

    Returns:
        更新后的AgentState，包含基本面分析结果
    """
    logger.info(
        f"{WAIT_ICON} FundamentalAgent: Starting fundamental analysis using ReAct framework.")

    # 获取执行日志记录器，用于记录 Agent的执行过程
    execution_logger = get_execution_logger()
    agent_name = "fundamental_agent"

    # 从状态中提取当前数据、消息和元数据
    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})
    user_query = current_data.get("query")

    # 记录 Agent开始执行，包含关键信息
    execution_logger.log_agent_start(agent_name, {
        "user_query": user_query,
        "stock_code": current_data.get("stock_code"),
        "company_name": current_data.get("company_name"),
        "input_data_keys": list(current_data.keys())
    })

    # 验证用户查询是否存在
    if not user_query:
        logger.error(
            f"{ERROR_ICON} FundamentalAgent: User query is missing in state data.")
        current_data["fundamental_analysis_error"] = "User query is missing."

        # 记录 Agent执行失败
        execution_logger.log_agent_complete(
            agent_name, current_data, 0, False, "User query is missing")

        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    # 记录 Agent开始时间，用于计算执行时长
    agent_start_time = time.time()

    try:
        # 使用API调用
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

        # 验证必要的环境变量是否存在
        if not all([api_key, base_url, model_name]):
            logger.error(
                f"{ERROR_ICON} FundamentalAgent: Missing OpenAI environment variables.")
            current_data["fundamental_analysis_error"] = "Missing OpenAI environment variables."

            # 记录 Agent执行失败
            execution_logger.log_agent_complete(agent_name, current_data, time.time(
            ) - agent_start_time, False, "Missing OpenAI environment variables")

            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        logger.info(
            f"{WAIT_ICON} FundamentalAgent: Creating ChatOpenAI with model {model_name}")
        # 创建LLM实例，设置合适的参数
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.6,  # kimi-k2.5 instant 模式要求值
            max_tokens=8000,  # 基本面分析需要更多输出空间
            extra_body={"thinking": {"type": "disabled"}}  # 工具调用时必须关闭思考模式
        )

        # 2. 获取MCP工具集
        logger.info(f"{WAIT_ICON} FundamentalAgent: Fetching MCP tools...")
        try:
            mcp_tools = await get_mcp_tools()
            if not mcp_tools:
                logger.error(
                    f"{ERROR_ICON} FundamentalAgent: No MCP tools available.")
                current_data["fundamental_analysis_error"] = "No MCP tools available."

                # 记录 Agent执行失败
                execution_logger.log_agent_complete(agent_name, current_data, time.time(
                ) - agent_start_time, False, "No MCP tools available")

                return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

            logger.info(
                f"{SUCCESS_ICON} FundamentalAgent: Successfully loaded {len(mcp_tools)} tools.")

            # 打印可用工具列表，便于调试
            tool_names = [tool.name for tool in mcp_tools]
            logger.info(f"Available tools: {tool_names}")

            # 3. 创建ReAct Agent - 只传入LLM和工具
            logger.info(
                f"{WAIT_ICON} FundamentalAgent: Creating ReAct agent...")
            agent = create_react_agent(llm, mcp_tools)

            # 4. 准备输入数据，构建详细的分析请求
            stock_code = current_data.get('stock_code', 'Unknown')
            company_name = current_data.get('company_name', 'Unknown')
            current_time_info = current_data.get('current_time_info', '未知时间')
            current_date = current_data.get('current_date', '未知日期')

            # 构建详细的基本面分析请求，包含多个分析维度
            agent_input = f"""请以券商分析师的标准，对{company_name}（股票代码：{stock_code}）进行基本面分析。

当前时间：{current_time_info}
当前日期：{current_date}

请进行以下基本面分析（每个维度都需要基于实际数据，引用具体数字）：

1. 公司概况与行业地位
   - 主营业务、行业分类、市场地位
   - 核心竞争力（护城河）：技术壁垒、品牌、规模效应、客户粘性等

2. 财务报表分析（基于最新可用财报数据）
   - 盈利能力：毛利率、净利率、ROE（请进行杜邦分析：净利率×资产周转率×权益乘数拆解）
   - 成长性：营业收入增长率、净利润增长率、与行业增速对比
   - 运营效率：应收账款周转天数、存货周转天数、总资产周转率
   - 偿债能力：资产负债率、流动比率、速动比率、利息保障倍数
   - 现金流质量：经营活动现金流净额/净利润比率、自由现金流

3. 资产负债结构
   - 主要资产构成（流动资产vs非流动资产）
   - 主要负债构成（有息负债vs无息负债）
   - 应收账款和存货是否存在减值风险

4. 分红与股东回报
   - 历史分红记录、股息率
   - 股东变化趋势（机构/散户持仓比例）

5. 行业对比分析
   - 核心财务指标与同行业可比公司对比
   - 公司在行业中的相对优势和劣势

6. 综合评估
   - 基本面优势总结
   - 基本面风险提示

重要限制：
- 请专注于财务数据和基本面指标分析，不要使用crawl_news工具获取新闻信息
- 分析必须有数据支撑，引用具体的财务数字，避免空洞的定性描述
- 如果某些数据无法获取，请说明原因并基于可用数据提供分析

请使用可用的工具获取实际数据进行分析，而不是基于假设。"""

            logger.info(f"Agent input: {agent_input}")

            # 5. 调用ReAct Agent - 使用正确的messages格式
            logger.info(
                f"{WAIT_ICON} FundamentalAgent: Calling ReAct agent...")
            start_time = time.time()

            # LangGraph ReAct Agent需要messages格式的输入
            input_data = {
                "messages": [HumanMessage(content=agent_input)]
            }

            # 调用 Agent执行分析
            response = await agent.ainvoke(input_data)

            end_time = time.time()
            execution_time = end_time - start_time

            logger.info(
                f"ReAct agent execution completed in {execution_time:.2f} seconds")

            # 6. 提取分析结果
            final_output = "No analysis generated."

            if "messages" in response and isinstance(response["messages"], list):
                messages = response["messages"]
                # 查找最后一条AI消息，这通常包含最终的分析结果
                ai_messages = [
                    msg for msg in messages if isinstance(msg, AIMessage)]
                if ai_messages:
                    last_ai_message = ai_messages[-1]
                    final_output = last_ai_message.content
                    logger.info(
                        f"Successfully extracted analysis from AI message.")
                else:
                    logger.warning("No AI messages found in response")
                    # 如果没有AI消息，尝试获取所有消息的内容
                    all_content = []
                    for msg in messages:
                        if hasattr(msg, 'content') and msg.content:
                            all_content.append(str(msg.content))
                    if all_content:
                        final_output = "\n".join(all_content)
            else:
                logger.error(f"Unexpected response format: {type(response)}")
                logger.error(
                    f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")

            logger.info(
                f"Final extracted analysis length: {len(final_output)} characters")
            print(f"FUNDAMENTALAGENT: {final_output}")
            # 7. 记录LLM交互，用于后续分析和优化
            model_config = {
                "model": model_name,
                "temperature": 0.6,
                "max_tokens": 8000,
                "thinking": "disabled",
                "api_base": base_url
            }
            
            execution_logger.log_llm_interaction(
                agent_name=agent_name,
                interaction_type="react_agent",
                input_messages=[{"role": "user", "content": agent_input}],
                output_content=final_output,
                model_config=model_config,
                execution_time=execution_time
            )

            logger.info(
                f"{SUCCESS_ICON} FundamentalAgent: Successfully completed fundamental analysis.")
            
            # 8. 更新状态，保存分析结果和元数据
            current_data["fundamental_analysis"] = final_output
            current_metadata["fundamental_agent_executed"] = True
            current_metadata["fundamental_agent_timestamp"] = str(time.time())
            current_metadata["fundamental_agent_execution_time"] = f"{execution_time:.2f} seconds"

            # 9. 添加消息记录，保持对话历史
            new_message = {"role": "assistant", "content": "基本面分析已完成"}
            updated_messages = current_messages + [new_message]

            # 记录 Agent执行成功
            total_execution_time = time.time() - agent_start_time
            execution_logger.log_agent_complete(agent_name, {
                "fundamental_analysis_length": len(final_output),
                "analysis_preview": final_output[:500] if len(final_output) > 500 else final_output,
                "llm_execution_time": execution_time,
                "total_execution_time": total_execution_time
            }, total_execution_time, True)

            return {
                "data": current_data,
                "messages": updated_messages,
                "metadata": current_metadata
            }

        except Exception as e:
            logger.error(
                f"{ERROR_ICON} FundamentalAgent: Error in MCP or agent execution: {e}", exc_info=True)
            current_data[
                "fundamental_analysis_error"] = f"Error in MCP or agent execution: {e}"
            current_data["fundamental_analysis"] = f"基本面分析过程中出现错误: {str(e)}"
            current_metadata["fundamental_agent_error"] = str(e)

            # 记录 Agent执行失败
            execution_logger.log_agent_complete(
                agent_name, current_data, time.time() - agent_start_time, False, str(e))

            return {
                "data": current_data,
                "messages": current_messages,
                "metadata": current_metadata
            }

    except Exception as e:
        logger.error(
            f"{ERROR_ICON} FundamentalAgent: Error during execution: {e}", exc_info=True)
        current_data["fundamental_analysis_error"] = f"Error during execution: {e}"
        current_metadata["fundamental_agent_error"] = str(e)

        # 记录 Agent执行失败
        execution_logger.log_agent_complete(
            agent_name, current_data, time.time() - agent_start_time, False, str(e))

        return {
            "data": current_data,
            "messages": current_messages,
            "metadata": current_metadata
        }


# 本地测试函数
async def test_fundamental_agent():
    """基本面分析 Agent的测试函数"""
    from src.utils.state_definition import AgentState
    from datetime import datetime

    # 准备测试数据，包含当前时间信息
    current_datetime = datetime.now()
    current_date_cn = current_datetime.strftime("%Y年%m月%d日")
    current_date_en = current_datetime.strftime("%Y-%m-%d")
    current_weekday_cn = ["星期一", "星期二", "星期三", "星期四",
                          "星期五", "星期六", "星期日"][current_datetime.weekday()]
    current_time = current_datetime.strftime("%H:%M:%S")
    current_time_info = f"{current_date_cn} ({current_date_en}) {current_weekday_cn} {current_time}"

    # 创建测试状态，模拟真实的用户查询
    test_state = AgentState(
        messages=[],
        data={
            "query": "分析嘉友国际的财务状况",
            "stock_code": "sh.603871",
            "company_name": "嘉友国际",
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat()
        },
        metadata={}
    )

    # 运行 Agent并输出结果
    result = await fundamental_agent(test_state)
    print("Fundamental Analysis Result:")
    print(result)
    print(result.get("data", {}).get("fundamental_analysis", "No analysis found"))

    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fundamental_agent())
