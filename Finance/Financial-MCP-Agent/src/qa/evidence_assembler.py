"""
证据装配器 — 并行MCP工具调用 + 指标预计算 + 证据包组装

快路径(L1/L2): asyncio.gather 并行拉取所有工具数据
ReAct路径(L4): 使用 LangGraph ReAct Agent（Phase 2 完善）
"""
import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from src.tools.mcp_client import get_mcp_tools
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)

TOOL_TIMEOUT = 30  # 单个工具超时（秒）


@dataclass
class EvidencePackage:
    """证据包 — 分析友好型数据结构"""
    subject: str = ""                 # 分析主体
    stock_code: str = ""
    company_name: str = ""
    data_time: str = ""               # 数据截至时间
    domains_queried: List[str] = field(default_factory=list)
    facts: List[Dict[str, str]] = field(default_factory=list)
    raw_text: str = ""                # 原始数据文本（供LLM参考）
    missing: List[str] = field(default_factory=list)
    tool_call_summary: str = ""
    elapsed_seconds: float = 0.0


async def _call_tool_safe(tool, kwargs: dict, timeout: float, label: str) -> str:
    """安全调用单个MCP工具，带超时和异常保护"""
    try:
        result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=timeout)
        text = str(result).strip()
        if len(text) > 25:
            logger.info(f"{SUCCESS_ICON} QA Evidence: {label} 成功 ({len(text)} 字符)")
            return text
        else:
            logger.warning(f"QA Evidence: {label} 返回过短 ({len(text)} 字符)")
            return ""
    except asyncio.TimeoutError:
        logger.warning(f"QA Evidence: {label} 超时({timeout}s)")
        return ""
    except Exception as e:
        logger.warning(f"QA Evidence: {label} 失败: {e}")
        return ""


async def assemble_evidence_fast(
    stock_code: str,
    company_name: str,
    tools: List[str],
    question: str,
    current_date: str,
) -> EvidencePackage:
    """
    快路径：并行拉取所有所需工具的数据，组装为证据包。

    类比快筛股票池的做法 — 确定数据域后，asyncio.gather 一次性拉取。
    """
    start_time = time.time()
    evidence = EvidencePackage(
        subject=company_name or stock_code or question,
        stock_code=stock_code or "",
        company_name=company_name or "",
        domains_queried=[],
    )

    # 获取 MCP 工具
    try:
        all_mcp_tools = await get_mcp_tools(tool_filter=tools)
    except Exception as e:
        logger.error(f"{ERROR_ICON} QA Evidence: 获取MCP工具失败: {e}")
        evidence.missing.append(f"MCP工具不可用: {e}")
        return evidence

    if not all_mcp_tools:
        evidence.missing.append("无可用MCP工具")
        return evidence

    logger.info(f"{WAIT_ICON} QA Evidence: 并行调用 {len(all_mcp_tools)} 个工具...")

    # 并行调用所有工具
    tasks = []
    labels = []
    for tool in all_mcp_tools:
        kwargs = _build_tool_kwargs(tool.name, stock_code, company_name, question)
        tasks.append(_call_tool_safe(tool, kwargs, TOOL_TIMEOUT, tool.name))
        labels.append(tool.name)

    results = await asyncio.gather(*tasks)

    # 组装原始文本
    raw_parts = []
    success_count = 0
    for label, text in zip(labels, results):
        if text:
            raw_parts.append(f"### [{label}]\n{text}")
            success_count += 1
        else:
            evidence.missing.append(label)

    evidence.raw_text = "\n\n".join(raw_parts) if raw_parts else "(无数据)"
    evidence.tool_call_summary = f"{success_count}/{len(labels)} 工具成功"
    evidence.elapsed_seconds = time.time() - start_time

    logger.info(
        f"{SUCCESS_ICON} QA Evidence: 装配完成 "
        f"({evidence.elapsed_seconds:.1f}s, {success_count}/{len(labels)} 成功)"
    )

    return evidence


async def assemble_evidence_react(
    stock_code: str,
    company_name: str,
    question: str,
    current_date: str,
    current_time_info: str,
) -> EvidencePackage:
    """
    ReAct 路径：使用 LangGraph ReAct Agent 迭代调工具。
    Phase 2 完善。Phase 1 提供基本实现。
    """
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage, AIMessage

    evidence = EvidencePackage(
        subject=company_name or stock_code or question,
        stock_code=stock_code or "",
        company_name=company_name or "",
    )
    start_time = time.time()

    try:
        all_mcp_tools = await get_mcp_tools()
    except Exception as e:
        evidence.missing.append(f"ReAct MCP工具不可用: {e}")
        return evidence

    if not all_mcp_tools:
        evidence.missing.append("无可用MCP工具")
        return evidence

    from src.utils.model_config import get_model_config_for_agent
    model_cfg = get_model_config_for_agent("qa_engine")

    llm = ChatOpenAI(
        model=model_cfg["model_name"],
        api_key=model_cfg["api_key"],
        base_url=model_cfg["base_url"],
        temperature=0.6,
        request_timeout=180,
        max_tokens=8000,
        extra_body={"thinking": {"type": "disabled"}},
    )

    agent = create_react_agent(llm, all_mcp_tools)

    agent_input = (
        f"请获取以下问题的相关数据：{question}\n\n"
        f"当前股票：{company_name}（{stock_code}）\n"
        f"当前日期：{current_date}\n\n"
        f"请使用可用的工具获取实际数据，基于数据回答，不要编造。"
    )

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=agent_input)]},
        config={"recursion_limit": 20}
    )

    if "messages" in response and isinstance(response["messages"], list):
        ai_msgs = [m for m in response["messages"] if isinstance(m, AIMessage)]
        if ai_msgs:
            evidence.raw_text = ai_msgs[-1].content
        else:
            evidence.raw_text = str(response["messages"][-1])

    evidence.elapsed_seconds = time.time() - start_time
    evidence.tool_call_summary = f"ReAct完成 ({evidence.elapsed_seconds:.0f}s)"
    return evidence


def _build_tool_kwargs(tool_name: str, stock_code: str, company_name: str,
                        question: str) -> dict:
    """根据工具名构建合适的参数"""
    clean_code = stock_code.replace("sh.", "").replace("sz.", "") if stock_code else ""
    code = stock_code or ""

    tool_kwargs_map = {
        "get_stock_basic_info": {"code": code},
        "tushare_stock_info": {"code": code},
        "get_stock_industry": {"code": code},
        "get_historical_k_data": {"code": code, "days": 120},
        "tushare_kline": {"code": code, "days": 120},
        "tushare_daily_basic": {"code": code},
        "tushare_pe_percentile": {"code": code},
        "get_profit_data": {"code": code},
        "get_balance_data": {"code": code},
        "get_cash_flow_data": {"code": code},
        "get_growth_data": {"code": code},
        "get_operation_data": {"code": code},
        "get_dupont_data": {"code": code},
        "tushare_fina_indicator": {"code": code},
        "tushare_moneyflow": {"code": code},
        "tushare_dividend": {"code": code},
        "get_dividend_data": {"code": code},
        "tushare_ev_ebitda": {"code": code},
        "get_st_risk_data": {"code": code},
        "tushare_st_status": {"code": code},
        "get_latest_trading_date": {},
        "get_market_analysis_timeframe": {},
    }

    if tool_name in tool_kwargs_map:
        return tool_kwargs_map[tool_name]
    elif tool_name == "crawl_news":
        return {"query": company_name or clean_code or question, "top_k": 10}
    else:
        return {"query": question}
