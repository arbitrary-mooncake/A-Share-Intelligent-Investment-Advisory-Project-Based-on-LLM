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
MAX_CONCURRENT_TOOLS = 4  # 最大并发工具调用数（防止 Windows 子进程竞争）
REACT_TOTAL_TIMEOUT = 120  # ReAct路径总超时（秒），防止LLM迭代过久导致前端断连

# L1 快速通道常量
L1_TOOL_TIMEOUT = 12   # L1 单工具超时（秒），简单查询应快速返回
L1_MAX_RETRIES = 1     # L1 单工具最多重试 1 次


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
    complexity_level: str = "",
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
        "tushare_concept_list", "tushare_ths_index", "tushare_dc_index",
        "tushare_search_stock",
        "tushare_cn_cpi", "tushare_cn_gdp", "tushare_cn_pmi",
        "tushare_cn_ppi", "tushare_cn_m", "tushare_shibor",
        "tushare_fx_daily", "tushare_eco_cal",
        "tushare_latest_trading_date", "tushare_top_list",
        "tushare_major_news",  # keyword-based news search, no code needed
        # Phase 1: Web Search (no code needed)
        "web_search", "web_fetch",
        # Phase 1: AKShare international (no code needed)
        "get_us_cpi", "get_us_pmi", "get_us_non_farm",
        "get_us_unemployment", "get_us_gdp", "get_us_retail_sales",
        "get_comex_inventory", "get_spot_gold_sge",
        "get_global_futures_spot", "get_sge_spot_prices", "get_fx_rates",
        # Phase 2: Yahoo Finance (no code needed)
        "get_commodity_price", "get_us_treasury_yield",
        "get_dollar_index", "get_gold_etf",
    }
    if not stock_code and not company_name:
        # 过滤：只保留不需要股票代码的工具（宏观数据、Web Search、国际数据等）
        _no_code_tools = [t for t in tools if t in _NO_CODE_TOOLS]
        if not _no_code_tools:
            evidence.raw_text = (
                "当前问题未指定具体的A股股票或公司，且A股数据工具主要覆盖个股、行业和板块数据。"
                "对于黄金、商品期货、宏观经济等非A股标的的问题，请基于现有知识和行业理解作答，"
                "明确说明数据来源限制，不编造任何数据。"
            )
            evidence.tool_call_summary = "无股票代码，跳过数据获取"
            evidence.missing.append("未指定A股标的")
            return evidence
        logger.info(
            f"QA Evidence: 无股票代码，从{len(tools)}个工具中筛选"
            f"{len(_no_code_tools)}个代码无关工具继续执行"
        )
        tools = _no_code_tools

    # 安全网：有公司名但无代码（名称反查失败）→ 不用空code调个股工具，避免全部失败
    if not stock_code and company_name:
        needs_code = any(t not in _NO_CODE_TOOLS for t in tools) if tools else True
        if needs_code:
            logger.warning(
                f"QA Evidence: 有公司名'{company_name}'但无代码，且名称反查失败，"
                f"避免用空code调用个股工具导致全部失败"
            )
            evidence.raw_text = (
                f"已识别到查询对象「{company_name}」，但当前无法解析其对应的A股股票代码。"
                f"请尝试提供股票代码（6位数字，如600519）或使用更完整的公司全称。"
            )
            evidence.tool_call_summary = "无股票代码（名称反查失败），跳过数据获取"
            evidence.missing.append(f"未解析到股票代码: {company_name}")
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

    # Tushare→AkShare 托底映射（当 Tushare 工具全部失败时回退到旧 mcp_server.py 工具）
    _FALLBACK_TOOL_MAP = {
        "tushare_kline": "get_historical_k_data",
        "tushare_daily_basic": None,  # AkShare 无直接等价工具
        "tushare_stock_info": "get_stock_basic_info",
        "tushare_latest_trading_date": "get_latest_trading_date",
        "tushare_adj_factor": "get_adjust_factor_data",
        "tushare_fina_indicator": None,  # AkShare 有 get_profit_data 等但参数不兼容
    }

    is_l1 = (complexity_level == "L1")
    tool_timeout = L1_TOOL_TIMEOUT if is_l1 else TOOL_TIMEOUT
    tool_retries = L1_MAX_RETRIES if is_l1 else 2

    logger.info(
        f"{WAIT_ICON} QA Evidence: 并行调用 {len(all_mcp_tools)} 个工具 "
        f"(并发上限={MAX_CONCURRENT_TOOLS}, 复杂度={complexity_level or 'L2+'}, "
        f"超时={tool_timeout}s, 重试={tool_retries}), stock_code={stock_code}..."
    )

    # 使用信号量限制并发，防止 Windows 上同时启动过多 MCP 子进程
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TOOLS)

    async def _sem_tool_call(tool, kwargs, label):
        async with semaphore:
            return await _call_tool_safe(
                tool, kwargs, tool_timeout, label, session_id,
                max_retries=tool_retries,
            )

    # 并行调用所有工具
    tasks = []
    labels = []
    for tool in all_mcp_tools:
        kwargs = _build_tool_kwargs(tool.name, stock_code, company_name, question, current_date, topic_name)
        tasks.append(_sem_tool_call(tool, kwargs, tool.name))
        labels.append(tool.name)

    results = await asyncio.gather(*tasks)

    # 板块/主题查询：L2+ 才拉取代表性个股数据（L1 精简路径跳过）
    if representative_stocks and not is_l1:
        _REP_TOOLS = ["tushare_stock_info", "tushare_daily_basic", "tushare_kline"]
        rep_tool_map = {t.name: t for t in all_mcp_tools}
        rep_tasks = []
        rep_labels = []
        for rep_code, rep_name in representative_stocks:
            for tname in _REP_TOOLS:
                tool = rep_tool_map.get(tname)
                if tool:
                    rep_kwargs = _build_tool_kwargs(tname, rep_code, rep_name, question, current_date, topic_name)
                    rep_tasks.append(_sem_tool_call(tool, rep_kwargs, f"{tname}({rep_name})"))
                    rep_labels.append(f"{tname}({rep_name})")
        if rep_tasks:
            rep_results = await asyncio.gather(*rep_tasks)
            labels.extend(rep_labels)
            results = list(results) + list(rep_results)

    # ── 托底：Tushare 工具成功率低于 50% 时，回退到 AkShare（L2+ 专属，L1 跳过）──
    total_called = len(labels)
    initial_success = sum(1 for t in results if t)
    if not is_l1 and total_called > 0 and initial_success / total_called < 0.5 and stock_code:
        fallback_tools_to_try = []
        fallback_tool_objs = {}
        # 先加载全部旧工具供回退
        try:
            _all_fb_tools = await get_mcp_tools()
            for t in _all_fb_tools:
                fallback_tool_objs[t.name] = t
        except Exception:
            pass

        for orig_name in labels:
            fb_name = _FALLBACK_TOOL_MAP.get(orig_name)
            if fb_name and fb_name in fallback_tool_objs:
                fb_kwargs = _build_tool_kwargs(fb_name, stock_code, company_name,
                                                question, current_date, topic_name)
                fallback_tools_to_try.append(
                    _sem_tool_call(fallback_tool_objs[fb_name], fb_kwargs, f"{fb_name}(托底)")
                )
        if fallback_tools_to_try:
            logger.warning(f"{WAIT_ICON} QA Evidence: Tushare工具全部失败，尝试 {len(fallback_tools_to_try)} 个AkShare托底工具...")
            fb_results = await asyncio.gather(*fallback_tools_to_try)
            fb_labels = [f"{_FALLBACK_TOOL_MAP.get(l, l)}(托底)" for l in labels if _FALLBACK_TOOL_MAP.get(l) in fallback_tool_objs]
            labels = list(labels) + fb_labels
            results = list(results) + list(fb_results)

    # ── ETF 专属托底（L2+ 专属，L1 跳过）──
    # 即使MCP工具"成功"返回了数据，也是stock类API的无效数据（ETF无ROE/PE等个股指标）
    # 因此ETF查询一律补充fund_basic/fund_daily/fund_adj直连数据
    _is_etf = stock_code and stock_code.replace("sh.", "").replace("sz.", "").startswith(("51", "58", "15", "16", "18"))
    if _is_etf and stock_code and not is_l1:
        logger.info(f"{WAIT_ICON} QA Evidence: ETF标的，补充fund_daily/fund_basic直连数据...")
        try:
            from src.utils.tushare_client import get_fund_basic, get_fund_daily, get_fund_adj
            ts_code = stock_code.replace("sh.", "").replace("sz.", "")
            _is_bse = (ts_code.startswith(("430", "431", "920")) or
                       (len(ts_code) >= 3 and ts_code[:3] in
                        ("830", "831", "832", "833", "834", "835", "836", "837", "838", "839",
                         "870", "871", "872", "873")))
            if _is_bse:
                ts_code = f"{ts_code}.BJ"
            elif ts_code.startswith(("6", "688", "5")):
                ts_code = f"{ts_code}.SH"
            else:
                ts_code = f"{ts_code}.SZ"

            loop = asyncio.get_running_loop()

            # ETF基础信息
            fund_info = await loop.run_in_executor(None, get_fund_basic, ts_code)
            if fund_info:
                info_text = f"名称: {fund_info.get('name', '')}, 管理公司: {fund_info.get('management', '')}, " \
                           f"类型: {fund_info.get('fund_type', '')}, 成立日: {fund_info.get('setup_date', '') or fund_info.get('found_date', '')}, " \
                           f"跟踪指数: {fund_info.get('index_name', '') or fund_info.get('index_code', '')}"
                if fund_info.get("m_fee"):
                    info_text += f", 管理费{fund_info['m_fee']}%, 托管费{fund_info.get('c_fee', '')}%"
                labels.append("etf_fund_basic(ETF直连)")
                results.append(info_text)

            # ETF日K线
            fund_kline = await loop.run_in_executor(None, get_fund_daily, ts_code, 120)
            if fund_kline and len(fund_kline) >= 5:
                lines = ["trade_date | open | high | low | close | pct_chg | vol | amount"]
                lines.append("--- | --- | --- | --- | --- | --- | --- | ---")
                for d in fund_kline[:30]:
                    lines.append(f"{d.get('trade_date','')} | {d.get('open','')} | {d.get('high','')} | "
                                f"{d.get('low','')} | {d.get('close','')} | {d.get('pct_chg','')} | "
                                f"{d.get('vol','')} | {d.get('amount','')}")
                kline_text = "\n".join(lines)
                chg_5d = sum(float(d.get("pct_chg", 0) or 0) for d in fund_kline[:5])
                chg_20d = sum(float(d.get("pct_chg", 0) or 0) for d in fund_kline[:20]) if len(fund_kline) >= 20 else None
                chg_60d = sum(float(d.get("pct_chg", 0) or 0) for d in fund_kline[:60]) if len(fund_kline) >= 60 else None
                trend = f"近5日累计{chg_5d:+.2f}%"
                if chg_20d is not None:
                    trend += f", 近20日累计{chg_20d:+.2f}%"
                if chg_60d is not None:
                    trend += f", 近60日累计{chg_60d:+.2f}%"
                kline_text = f"趋势: {trend}\n\n" + kline_text
                labels.append("etf_fund_daily(ETF直连)")
                results.append(kline_text)

            # ETF复权收益（用 fund_daily 500天 + fund_adj 500天，按日期对齐）
            fund_kline_long = await loop.run_in_executor(None, get_fund_daily, ts_code, 500)
            fund_adj_long = await loop.run_in_executor(None, get_fund_adj, ts_code, 500)
            if fund_kline_long and len(fund_kline_long) >= 60 and fund_adj_long and len(fund_adj_long) >= 60:
                adj_map = {}
                for d in fund_adj_long:
                    dt = d.get("trade_date", "")
                    if dt:
                        adj_map[dt] = float(d.get("adj_factor", 1) or 1)
                latest_adj_val = adj_map.get(fund_kline_long[0].get("trade_date", ""), 1)

                def _qfq_price(day_data):
                    raw_close = float(day_data.get("close", 0) or 0)
                    day_adj = adj_map.get(day_data.get("trade_date", ""), latest_adj_val)
                    return raw_close * day_adj / latest_adj_val if latest_adj_val > 0 else raw_close

                qfq_now = _qfq_price(fund_kline_long[0])
                ret_parts = []
                for days, label in [(60, "60日"), (120, "120日"), (250, "250日")]:
                    idx = min(days, len(fund_kline_long) - 1)
                    qfq_then = _qfq_price(fund_kline_long[idx])
                    if qfq_then > 0:
                        ret_pct = (qfq_now / qfq_then - 1) * 100
                        ret_parts.append(f"{label}{ret_pct:+.1f}%")
                adj_text = f"前复权收益 — {', '.join(ret_parts)}"
                labels.append("etf_adj_return(ETF直连)")
                results.append(adj_text)

            logger.info(f"{SUCCESS_ICON} QA Evidence: ETF直连补充数据完成")
        except Exception as e:
            logger.warning(f"QA Evidence: ETF直连数据获取失败: {e}")

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
        "tushare_kline", "tushare_daily_basic", "tushare_stock_info",
        "tushare_pe_percentile", "tushare_fina_indicator",
        "tushare_moneyflow", "tushare_dividend", "tushare_ev_ebitda",
        "tushare_news", "tushare_st_status",
        "tushare_search_stock", "tushare_concept_list",
        "tushare_hsgt_flow", "tushare_top10_holders",
        "tushare_adj_factor", "tushare_latest_trading_date",
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

    try:
        response = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [HumanMessage(content=agent_input)]},
                config={"recursion_limit": 20}
            ),
            timeout=REACT_TOTAL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.warning(
            f"QA Evidence: ReAct路径超时({REACT_TOTAL_TIMEOUT}s)，返回部分证据"
        )
        evidence.raw_text = (
            f"（ReAct数据获取在{REACT_TOTAL_TIMEOUT}秒内未完成，已超时终止。"
            f"以下基于已有信息和行业知识分析。）"
        )
        evidence.tool_call_summary = f"ReAct超时 ({elapsed:.0f}s)"
        evidence.missing.append(f"ReAct超时({REACT_TOTAL_TIMEOUT}s)")
        evidence.elapsed_seconds = elapsed
        return evidence

    if "messages" in response and isinstance(response["messages"], list):
        ai_msgs = [m for m in response["messages"] if isinstance(m, AIMessage)]
        if ai_msgs:
            evidence.raw_text = ai_msgs[-1].content
        else:
            evidence.raw_text = str(response["messages"][-1])

    evidence.elapsed_seconds = time.time() - start_time
    evidence.tool_call_summary = f"ReAct完成 ({evidence.elapsed_seconds:.0f}s)"
    return evidence


def _build_search_query(question: str, topic_name: str) -> str:
    """从用户问题中提取核心关键词构建搜索查询。
    去口语化 + 注入英文金融术语提高搜索质量。"""
    import re
    if topic_name:
        cleaned = re.sub(r'(深度|详细|全面|系统)?(分析一下|分析|帮我)?(最近|当前|未来)?(的)?', '', question)
        cleaned = cleaned.strip()
        if len(cleaned) > 60:
            return f"{topic_name} {cleaned[:40]}"
        return f"{topic_name} {cleaned}" if cleaned else topic_name

    # 无主题：去口语化 + 补英文关键词
    q = question
    # 去掉口语前缀和疑问语气词
    q = re.sub(r'^(最近|当前|目前|帮我|请|麻烦|能否|怎么|如何|为什么|什么是|是多少)', '', q)
    q = re.sub(r'(分析一下|分析|看看|查询|查一下|了解|说一下|介绍|讲讲)?(的)?(各自)?(是)?(多少|什么|怎么样|如何|吗)?[？?！!。，,、\s]*$', '', q)
    q = re.sub(r'[？?！!。，,、]+', ' ', q).strip()

    # 注入英文关键词提高搜索质量
    _EN_TERMS = {
        '通货膨胀': 'CPI inflation rate', '通胀': 'CPI inflation',
        'GDP': 'GDP', 'PMI': 'PMI', 'PPI': 'PPI',
        '失业': 'unemployment rate', '就业': 'employment data',
        '非农': 'nonfarm payroll', '利率': 'interest rate',
        '汇率': 'exchange rate', '零售': 'retail sales',
        'M2': 'M2 money supply', 'M1': 'M1 money supply',
        '贸易': 'trade data', '关税': 'tariff',
    }
    en_terms = []
    for cn_term, en_term in _EN_TERMS.items():
        if cn_term in question:
            en_terms.append(en_term)
    if en_terms:
        q = f"{q} {' '.join(en_terms)}"

    return q[:150]


def _topic_to_commodity_symbol(topic_name: str) -> str:
    """将投资主题映射为 Yahoo Finance 商品期货代码。
    未映射的主题返回空字符串，调用方应跳过 get_commodity_price。"""
    if not topic_name:
        return ""
    mapping = {
        "黄金": "GC=F",
        "白银": "SI=F",
        "原油": "CL=F",
    }
    return mapping.get(topic_name, "")


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
        # 行情类（tushare）
        "tushare_kline": {"code": code, "days": 250},
        "tushare_daily_basic": {"code": code, "days": 500},
        "tushare_stock_info": {"code": code},
        "tushare_adj_factor": {"code": code, "days": 500},
        "tushare_latest_trading_date": {},
        # 宏观类（Tushare，无需股票代码）
        "tushare_cn_cpi": {"start_date": start_date, "end_date": end_date},
        "tushare_cn_gdp": {"start_date": start_date, "end_date": end_date},
        "tushare_cn_pmi": {"start_date": start_date, "end_date": end_date},
        "tushare_cn_ppi": {"start_date": start_date, "end_date": end_date},
        "tushare_cn_m": {"start_date": start_date, "end_date": end_date},
        "tushare_shibor": {"date": end_date},
        "tushare_fx_daily": {"start_date": start_date, "end_date": end_date},
        "tushare_eco_cal": {"date": end_date},
        # 估值类
        "tushare_pe_percentile": {"code": code},
        "tushare_ev_ebitda": {"code": code},
        "tushare_dividend": {"code": code},
        # 财务类
        "tushare_fina_indicator": {"code": code, "years": 3},
        "tushare_income": {"code": code},
        "tushare_balancesheet": {"code": code},
        "tushare_cashflow": {"code": code},
        # 资金类
        "tushare_moneyflow": {"code": code, "days": 60},
        "tushare_hsgt_flow": {"code": code},
        "tushare_top10_holders": {"code": code},
        "tushare_holder_num": {"code": code},
        # 新闻/风险类
        "tushare_st_status": {"code": code},
        "tushare_news": {"code": code},
        "tushare_major_news": {"keyword": _build_search_query(question, topic_name), "days": 30},
        # 板块/概念类
        "tushare_concept_list": {"keyword": topic_name or company_name or question},
        "tushare_concept_detail": {"concept_code": topic_name or company_name or question},
        "tushare_ths_index": {"keyword": topic_name or company_name or question},
        "tushare_dc_index": {"keyword": topic_name or company_name or question},
        # 股票名称搜索
        "tushare_search_stock": {"keyword": company_name or clean_code or question},
        # 龙虎榜
        "tushare_top_list": {"date": end_date.replace("-", "")},
        # 国际宏观（AKShare — 无需参数）
        "get_us_cpi": {},
        "get_us_pmi": {},
        "get_us_non_farm": {},
        "get_us_unemployment": {},
        "get_us_gdp": {},
        "get_us_retail_sales": {},
        "get_comex_inventory": {},
        "get_spot_gold_sge": {},
        "get_global_futures_spot": {},
        "get_sge_spot_prices": {},
        "get_fx_rates": {},
        # Web Search（使用问题文本作为查询）
        "web_search": {"query": _build_search_query(question, topic_name)},
        # web_fetch 是动态工具（URL由后续调用决定），不在预规划kwargs中
        # Yahoo Finance（国际商品/利率 — 按主题选择默认symbol）
        "get_commodity_price": {"symbol": _topic_to_commodity_symbol(topic_name)},
        "get_us_treasury_yield": {"tenor": "10y"},
        "get_dollar_index": {},
        "get_gold_etf": {"symbol": "GLD"},
    }

    if tool_name in tool_kwargs_map:
        return tool_kwargs_map[tool_name]
    # 未知工具：尝试传 code（多数工具都接受）
    return {"code": code} if code else {}
