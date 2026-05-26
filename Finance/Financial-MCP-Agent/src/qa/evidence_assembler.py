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
from src.qa.session_manager import (
    get_cached_evidence, set_cached_evidence,
    get_global_cached_evidence, set_global_cached_evidence,
)

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


async def _call_tool_safe(tool, kwargs: dict, timeout: float, label: str,
                         session_id: str = "", max_retries: int = 2) -> str:
    """安全调用单个MCP工具，带超时、per-session缓存+全局缓存+空数据重试"""
    # 先查 per-session 缓存
    if session_id:
        cached = get_cached_evidence(session_id, label, kwargs)
        if cached:
            logger.info(f"{SUCCESS_ICON} QA Evidence: {label} 命中会话缓存 ({len(cached)} 字符)")
            return cached

    # 再查跨会话全局缓存
    cached = get_global_cached_evidence(label, kwargs)
    if cached:
        logger.info(f"{SUCCESS_ICON} QA Evidence: {label} 命中全局缓存 ({len(cached)} 字符)")
        return cached

    last_error = ""
    for attempt in range(1 + max_retries):
        try:
            result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=timeout)
            text = str(result).strip()
            if len(text) > 25:
                logger.info(f"{SUCCESS_ICON} QA Evidence: {label} 成功 ({len(text)} 字符)"
                           + (f" (第{attempt+1}次)" if attempt > 0 else ""))
                if session_id:
                    set_cached_evidence(session_id, label, kwargs, text)
                set_global_cached_evidence(label, kwargs, text)
                return text
            else:
                last_error = f"返回过短 ({len(text)} 字符)"
        except asyncio.TimeoutError:
            last_error = f"超时({timeout}s)"
        except Exception as e:
            last_error = str(e)

        if attempt < max_retries:
            wait = 1.5 * (attempt + 1)  # 递增等待: 1.5s, 3s
            logger.warning(f"QA Evidence: {label} {last_error}，{wait:.1f}s后重试({attempt+1}/{max_retries})...")
            await asyncio.sleep(wait)

    logger.warning(f"QA Evidence: {label} 最终失败: {last_error}（已重试{max_retries}次）")
    return ""


async def assemble_evidence_fast(
    stock_code: str,
    company_name: str,
    tools: List[str],
    question: str,
    current_date: str,
    session_id: str = "",
    topic_name: str = "",
    representative_stocks: list = None,
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

    # 无股票代码时：仅当请求的全部是宏观/市场类工具（无需代码）时才继续
    _NO_CODE_TOOLS = {
        "get_deposit_rate_data", "get_loan_rate_data",
        "get_required_reserve_ratio_data", "get_money_supply_data_month",
        "get_money_supply_data_year", "get_latest_trading_date",
        "get_market_analysis_timeframe", "get_trade_dates", "get_all_stock",
        "get_sz50_stocks", "get_hs300_stocks", "get_zz500_stocks",
        "tushare_concept_list", "tushare_ths_index", "tushare_dc_index",
        "tushare_search_stock",
        "tushare_cn_cpi", "tushare_cn_gdp", "tushare_cn_pmi",
        "tushare_cn_ppi", "tushare_cn_m", "tushare_shibor",
        "tushare_fx_daily", "tushare_eco_cal",
    }
    if not stock_code and not company_name:
        if tools and all(t in _NO_CODE_TOOLS for t in tools):
            pass  # 宏观/市场类查询，无需股票代码，继续执行
        else:
            evidence.raw_text = (
                "当前问题未指定具体的A股股票或公司，且A股数据工具主要覆盖个股、行业和板块数据。"
                "对于黄金、商品期货、宏观经济等非A股标的的问题，请基于现有知识和行业理解作答，"
                "明确说明数据来源限制，不编造任何数据。"
            )
            evidence.tool_call_summary = "无股票代码，跳过数据获取"
            evidence.missing.append("未指定A股标的")
            return evidence

    # 获取 MCP 工具（含重试）
    all_mcp_tools = None
    for attempt in range(3):
        try:
            all_mcp_tools = await get_mcp_tools(tool_filter=tools)
            break
        except Exception as e:
            logger.warning(f"QA Evidence: MCP连接尝试 {attempt+1}/3 失败: {e}")
            if attempt < 2:
                await asyncio.sleep(2)
            else:
                logger.error(f"{ERROR_ICON} QA Evidence: MCP连接3次均失败")
                evidence.raw_text = (
                    "数据服务暂时不可用（MCP连接失败）。请稍后重试，"
                    "或基于现有知识和行业理解作答，明确标注数据来源限制。"
                )
                evidence.tool_call_summary = "MCP连接失败（3次重试）"
                evidence.missing.append("MCP数据服务不可用")
                return evidence

    if not all_mcp_tools:
        evidence.raw_text = (
            "当前数据工具未返回有效结果。请基于现有知识和行业理解作答，"
            "明确标注数据来源限制。"
        )
        evidence.tool_call_summary = "无可用工具"
        evidence.missing.append("无可用MCP工具")
        return evidence

    logger.info(f"{WAIT_ICON} QA Evidence: 并行调用 {len(all_mcp_tools)} 个工具...")

    # 并行调用所有工具
    tasks = []
    labels = []
    for tool in all_mcp_tools:
        kwargs = _build_tool_kwargs(tool.name, stock_code, company_name, question, current_date, topic_name)
        tasks.append(_call_tool_safe(tool, kwargs, TOOL_TIMEOUT, tool.name, session_id))
        labels.append(tool.name)

    results = await asyncio.gather(*tasks)

    # 板块/主题查询：并行拉取代表性个股的估值+基本信息+近K线
    if representative_stocks:
        _REP_TOOLS = ["get_stock_basic_info", "tushare_daily_basic", "tushare_kline"]
        rep_tool_map = {t.name: t for t in all_mcp_tools}
        rep_tasks = []
        rep_labels = []
        for rep_code, rep_name in representative_stocks:
            for tname in _REP_TOOLS:
                tool = rep_tool_map.get(tname)
                if tool:
                    rep_kwargs = _build_tool_kwargs(tname, rep_code, rep_name, question, current_date, topic_name)
                    rep_tasks.append(_call_tool_safe(tool, rep_kwargs, TOOL_TIMEOUT, f"{tname}({rep_name})", session_id))
                    rep_labels.append(f"{tname}({rep_name})")
        if rep_tasks:
            rep_results = await asyncio.gather(*rep_tasks)
            labels.extend(rep_labels)
            results = list(results) + list(rep_results)

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

    # ReAct路径只加载核心分析工具（避免全量工具导致prompt过大）
    _REACT_TOOL_FILTER = [
        "get_stock_basic_info", "get_stock_industry",
        "get_historical_k_data", "tushare_kline",
        "tushare_daily_basic", "tushare_pe_percentile",
        "tushare_fina_indicator", "tushare_moneyflow",
        "tushare_dividend", "tushare_ev_ebitda",
        "get_profit_data", "get_balance_data", "get_cash_flow_data",
        "get_growth_data", "get_dupont_data",
        "crawl_news", "tushare_st_status", "get_st_risk_data",
        "get_latest_trading_date", "get_market_analysis_timeframe",
        "tushare_search_stock", "tushare_stock_info",
        "tushare_concept_list", "tushare_hsgt_flow",
        "tushare_top10_holders", "tushare_news",
    ]
    try:
        all_mcp_tools = await get_mcp_tools(tool_filter=_REACT_TOOL_FILTER)
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
                        question: str, current_date: str = "",
                        topic_name: str = "") -> dict:
    """根据工具名构建合适的参数（含日期计算）"""
    clean_code = stock_code.replace("sh.", "").replace("sz.", "") if stock_code else ""
    code = stock_code or ""

    # 计算常用日期
    from datetime import datetime, timedelta
    try:
        base_date = datetime.strptime(current_date, "%Y-%m-%d") if current_date else datetime.now()
    except ValueError:
        base_date = datetime.now()
    this_year = str(base_date.year)
    this_quarter = (base_date.month - 1) // 3 + 1  # int, 财务工具期望整数
    start_date = (base_date - timedelta(days=180)).strftime("%Y-%m-%d")
    end_date = base_date.strftime("%Y-%m-%d")

    tool_kwargs_map = {
        # 行情类
        "get_stock_basic_info": {"code": code},
        "get_historical_k_data": {"code": code, "start_date": start_date, "end_date": end_date},
        "tushare_kline": {"code": code, "days": 250},
        "tushare_daily_basic": {"code": code, "days": 500},
        "get_latest_trading_date": {},
        "get_market_analysis_timeframe": {},
        # 宏观类（Tushare，无需股票代码）
        "tushare_cn_cpi": {"start_date": start_date, "end_date": end_date},
        "tushare_cn_gdp": {"start_date": start_date, "end_date": end_date},
        "tushare_cn_pmi": {"start_date": start_date, "end_date": end_date},
        "tushare_cn_ppi": {"start_date": start_date, "end_date": end_date},
        "tushare_cn_m": {"start_date": start_date, "end_date": end_date},
        "tushare_shibor": {"date": end_date},
        "tushare_fx_daily": {"start_date": start_date, "end_date": end_date},
        "tushare_eco_cal": {"date": end_date},
        # baostock 宏观工具
        "get_deposit_rate_data": {"start_date": start_date, "end_date": end_date},
        "get_loan_rate_data": {"start_date": start_date, "end_date": end_date},
        "get_required_reserve_ratio_data": {"start_date": start_date, "end_date": end_date, "year_type": "0"},
        "get_money_supply_data_month": {"start_date": start_date, "end_date": end_date},
        "get_money_supply_data_year": {"start_date": start_date, "end_date": end_date},
        # 估值类
        "tushare_pe_percentile": {"code": code},
        "tushare_ev_ebitda": {"code": code},
        "tushare_dividend": {"code": code},
        "get_dividend_data": {"code": code, "year": this_year, "year_type": "report"},
        # 财务类 — 需要 year + quarter
        "get_profit_data": {"code": code, "year": this_year, "quarter": this_quarter},
        "get_balance_data": {"code": code, "year": this_year, "quarter": this_quarter},
        "get_cash_flow_data": {"code": code, "year": this_year, "quarter": this_quarter},
        "get_growth_data": {"code": code, "year": this_year, "quarter": this_quarter},
        "get_operation_data": {"code": code, "year": this_year, "quarter": this_quarter},
        "get_dupont_data": {"code": code, "year": this_year, "quarter": this_quarter},
        "tushare_fina_indicator": {"code": code, "years": 3},
        "tushare_stock_info": {"code": code},
        # 资金类
        "tushare_moneyflow": {"code": code, "days": 60},
        "tushare_hsgt_flow": {"code": code},
        # 行业类
        "get_stock_industry": {"code": code},
        # 新闻/风险类
        # 主题类问题直接用问题原文搜新闻（含关键词），股票类用公司名
        "crawl_news": {
            "query": question if ("主题" in (company_name or "")) else (company_name or clean_code or question),
            "top_k": 10,
        },
        "get_st_risk_data": {"code": code},
        "tushare_st_status": {"code": code},
        "tushare_news": {"code": code},
        # 板块/概念类（快路径只用搜索工具，明细由ReAct二次调用）
        "tushare_concept_list": {"keyword": topic_name or company_name or question},
        "tushare_ths_index": {"keyword": topic_name or company_name or question},
        "tushare_dc_index": {"keyword": topic_name or company_name or question},
        # 股票名称搜索
        "tushare_search_stock": {"keyword": company_name or clean_code or question},
    }

    if tool_name in tool_kwargs_map:
        return tool_kwargs_map[tool_name]
    # 未知工具：尝试传 code（多数工具都接受）
    return {"code": code} if code else {}
