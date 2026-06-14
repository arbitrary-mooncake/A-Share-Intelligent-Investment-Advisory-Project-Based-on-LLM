"""
ValueAnalysis Agent: 两阶段架构 — 并行数据预取 + 单次 LLM 深度分析。
Phase 1: asyncio.gather 并行获取白名单全部 16 个工具的数据
Phase 2: 将所有原始数据喂给 LLM 一次性完成估值分析（thinking 开启）
"""
import asyncio
import os
import time
from datetime import datetime
from typing import Dict, Any, List

from openai import AsyncOpenAI
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from src.utils.state_definition import AgentState
from src.tools.mcp_client import get_mcp_tools
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.utils.cache_utils import read_cache, write_cache
from src.utils.model_config import get_model_config_for_agent, get_thinking_body
from src.utils.fetch_utils import retry_failed_fetches, is_empty_result

load_dotenv(override=True)

logger = setup_logger(__name__)

TOOL_TIMEOUT = 30
LLM_TIMEOUT = 300

# 估值分析白名单（2026-05 重构：全部切换为 Tushare 工具，移除 Baostock 依赖）
VALUE_TOOL_NAMES = [
    # 基本信息与行业
    "tushare_stock_info", "get_stock_industry",
    # 三大报表 + 财务指标
    "tushare_income", "tushare_balancesheet", "tushare_cashflow",
    "tushare_fina_indicator",
    # 估值核心
    "tushare_daily_basic", "tushare_pe_percentile", "tushare_ev_ebitda",
    # 分红与股东
    "tushare_dividend", "tushare_top10_holders", "tushare_holder_num",
    # 保留的 AkShare 工具（无 Tushare 平替）
    "get_dupont_data", "get_operation_data", "get_growth_data",
]


def _extract_code(stock_code: str) -> str:
    return stock_code.replace("sh.", "").replace("sz.", "").replace("bj.", "").replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()


def _build_etf_fallback(company_name: str, stock_code: str,
                         etf_info: str, etf_price: str) -> str:
    """ETF估值数据不足时的降级输出"""
    parts = []
    parts.append(f"## ETF估值分析\n")
    if etf_info:
        parts.append(f"### 基本信息\n{etf_info}\n")
        parts.append(f"\n**[数据]** 以上为{company_name}({stock_code})的ETF基本信息。\n")
    if etf_price:
        parts.append(f"### 行情数据\n{etf_price}\n")
        parts.append(f"\n**[数据]** 以上为近期行情数据（含市价与累计净值acc_close）。\n")
        parts.append(f"\n**[判断]** 请关注市价(close)与累计净值(acc_close)的差值以计算折溢价率：")
        parts.append(f"折溢价率 = (市价 - 累计净值) / 累计净值 × 100%。正值为溢价，负值为折价。\n")
    if not etf_info and not etf_price:
        parts.append(f"\n**[数据]** {company_name}({stock_code})的ETF估值数据暂时不可用。\n")
        parts.append(f"\n**[判断]** ETF估值应关注：基金净值(NAV)、市价折溢价率、跟踪指数的历史PE/PB分位、管理费率与跟踪误差。\n")
    parts.append(f"\n**估值评分：不适用（数据不足）**\n")
    return "".join(parts)


def _get_recent_quarters(date_str: str, count: int = 2) -> List[Dict[str, Any]]:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        dt = datetime.now()
    quarters = []
    current_q = (dt.month - 1) // 3 + 1
    current_year = dt.year
    q = current_q - 1
    y = current_year
    if q <= 0:
        q = 4
        y -= 1
    for _ in range(count):
        quarters.append({"year": str(y), "quarter": q})
        q -= 1
        if q <= 0:
            q = 4
            y -= 1
    return quarters


async def _noop_result(text: str) -> str:
    """返回固定文本的占位协程"""
    return text


async def _call_tool_safe(tool, kwargs: dict, label: str) -> str:
    try:
        result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=TOOL_TIMEOUT)
        text = str(result).strip()
        if len(text) > 20:
            logger.info(f"{SUCCESS_ICON} ValueAgent: {label} 获取成功 ({len(text)} 字符)")
            return text
        logger.warning(f"ValueAgent: {label} 返回过短 ({len(text)} 字符)")
        return f"[{label}] 数据不可用（返回过短）"
    except asyncio.TimeoutError:
        logger.warning(f"ValueAgent: {label} 超时({TOOL_TIMEOUT}s)")
        return f"[{label}] 数据不可用（超时）"
    except Exception as e:
        logger.warning(f"ValueAgent: {label} 调用失败: {e}")
        return f"[{label}] 数据不可用（调用失败: {str(e)[:80]}）"


async def value_agent(state: AgentState) -> AgentState:
    """
    两阶段估值分析：
    Phase 1: 并行获取全部 16 个工具的数据
    Phase 2: 单次 LLM 深度分析（Kimi K2.6, thinking=enabled）
    """
    logger.info(f"{WAIT_ICON} ValueAgent: Starting two-phase valuation analysis.")

    execution_logger = get_execution_logger()
    agent_name = "value_agent"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})
    user_query = current_data.get("query")

    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    cache_code = current_data.get("stock_code", "")

    # 缓存检查（TTL=7天）
    if not skip_cache and cache_date and cache_code:
        cached = read_cache("value_analysis", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} ValueAgent: 命中缓存，跳过分析 ({cache_code})")
            current_data["value_analysis"] = cached
            current_metadata["value_agent_executed"] = True
            current_metadata["value_agent_cached"] = True
            return {"data": current_data,
                    "messages": current_messages + [{"role": "assistant", "content": "估值分析已完成（缓存）"}],
                    "metadata": current_metadata}

    if current_data.get("is_etf", False):
        logger.info(f"{SUCCESS_ICON} ValueAgent: ETF标的，执行ETF专用估值分析")
        try:
            # 加载 ETF 估值相关工具
            etf_tool_names = ["tushare_stock_info", "tushare_daily_basic", "tushare_kline"]
            etf_tools = await get_mcp_tools(tool_filter=etf_tool_names)
            tool_map = {t.name: t for t in etf_tools}

            stock_code = current_data.get("stock_code", "Unknown")
            company_name = current_data.get("company_name", "Unknown")
            current_time_info = current_data.get("current_time_info", "未知时间")
            current_date = current_data.get("current_date", "未知日期")
            clean_code = _extract_code(stock_code)

            # ETF 估值数据预取
            etf_info_text = ""
            etf_price_text = ""

            if "tushare_stock_info" in tool_map:
                try:
                    r = await asyncio.wait_for(
                        tool_map["tushare_stock_info"].ainvoke({"code": clean_code}), timeout=15)
                    etf_info_text = str(r).strip()
                except Exception as e:
                    logger.warning(f"ETF信息获取失败: {e}")

            if "tushare_daily_basic" in tool_map:
                try:
                    r = await asyncio.wait_for(
                        tool_map["tushare_daily_basic"].ainvoke({"code": clean_code, "days": 120}), timeout=15)
                    etf_price_text = str(r).strip()
                except Exception as e:
                    logger.warning(f"ETF行情数据获取失败: {e}")

            # 使用 LLM 生成 ETF 估值分析
            model_cfg = get_model_config_for_agent("value_agent", current_data)
            api_key = model_cfg["api_key"]
            base_url = model_cfg["base_url"]
            model_name = model_cfg["model_name"]

            if all([api_key, base_url, model_name, etf_price_text or etf_info_text]):
                _client = AsyncOpenAI(
                    api_key=api_key, base_url=base_url,
                    timeout=httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=10.0),
                    max_retries=1,
                )
                etf_prompt = f"""你是一位资深ETF分析师，请对以下ETF产品进行估值分析。

ETF名称：{company_name}（代码：{stock_code}）
当前时间：{current_time_info}
当前日期：{current_date}

ETF基本信息：
{etf_info_text if etf_info_text else "数据不可用"}

近期行情数据（来自fund_daily，close=市价，acc_close=累计净值）：
{etf_price_text if etf_price_text else "数据不可用"}

请严格按以下格式输出 ## ETF估值分析 内容：

## 📊 数据事实区
1. ETF基本信息：名称、代码、管理公司、成立时间、跟踪指数
2. 近期市价与净值数据：最新收盘价、最新累计净值(acc_close)、5日/20日净值趋势
3. 折溢价分析：计算最新折溢价率 = (市价 - 累计净值) / 累计净值 × 100
4. 成交量与流动性：近期日均成交量、换手率
5. 跟踪指数信息：所跟踪指数的名称及近期表现

## 🔍 分析判断区
1. 折溢价状态评估：当前折溢价是否处于合理范围（ETF正常±2%以内）
2. 净值趋势分析：近期净值走势的技术性判断
3. 流动性评估：该ETF的日均成交量和流动性状况
4. 估值综合判断：基于以上数据的ETF估值综合评分(0-100分)
5. 投资建议：针对当前折溢价水平和净值趋势的配置建议

评分规则：
- 折溢价在±1%以内 +30分，±1-2% +20分，±2-3% +10分，超过±3% +0分
- 净值处于20日均线上方 +20分，处于60日均线上方 +15分，处于120日均线上方 +10分
- 日均成交额>1亿 +20分，>5000万 +10分，否则+0分
- 跟踪指数走势健康 +20分
- 管理费率合理(<0.5%) +10分

最终评分和投资建议请明确给出，不要含糊。"""

                try:
                    response = await asyncio.wait_for(
                        _client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "你是专业的ETF分析师，所有分析必须基于提供的实际数据。使用[数据]标记数据事实，[判断]标记分析推断。评分必须为0-100的整数。"},
                                {"role": "user", "content": etf_prompt},
                            ],
                            temperature=1.0,
                            max_tokens=16000,
                        ),
                        timeout=130.0,
                    )
                    etf_analysis = response.choices[0].message.content if response.choices else ""
                    logger.info(f"ETF估值分析完成 ({len(etf_analysis)} 字符)")
                except Exception as llm_err:
                    logger.warning(f"ETF估值 LLM 失败: {llm_err}")
                    etf_analysis = _build_etf_fallback(company_name, stock_code, etf_info_text, etf_price_text)
            else:
                etf_analysis = _build_etf_fallback(company_name, stock_code, etf_info_text, etf_price_text)

            current_data["value_analysis"] = etf_analysis
            current_metadata["value_agent_executed"] = True
            if not skip_cache and cache_date and cache_code:
                write_cache("value_analysis", cache_code, cache_date, etf_analysis)
            return {"data": current_data,
                    "messages": current_messages + [{"role": "assistant", "content": "ETF估值分析已完成"}],
                    "metadata": current_metadata}
        except Exception as e:
            logger.error(f"{ERROR_ICON} ValueAgent: ETF估值分析失败: {e}", exc_info=True)
            current_data["value_analysis_error"] = f"ETF估值分析失败: {e}"
            current_data["value_analysis"] = _build_etf_fallback(
                current_data.get("company_name", ""),
                current_data.get("stock_code", ""), "", "")
            current_metadata["value_agent_executed"] = True
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    execution_logger.log_agent_start(agent_name, {
        "user_query": user_query,
        "stock_code": cache_code,
        "company_name": current_data.get("company_name"),
        "input_data_keys": list(current_data.keys()),
        "architecture": "two-phase",
    })

    if not user_query:
        logger.error(f"{ERROR_ICON} ValueAgent: User query is missing.")
        current_data["value_analysis_error"] = "User query is missing."
        execution_logger.log_agent_complete(agent_name, current_data, 0, False, "User query is missing")
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    agent_start_time = time.time()

    try:
        model_cfg = get_model_config_for_agent("value_agent", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]

        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} ValueAgent: Missing OpenAI environment variables.")
            current_data["value_analysis_error"] = "Missing OpenAI environment variables."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "Missing env vars")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        stock_code = current_data.get("stock_code", "Unknown")
        company_name = current_data.get("company_name", "Unknown")
        current_time_info = current_data.get("current_time_info", "未知时间")
        current_date = current_data.get("current_date", "未知日期")
        clean_code = _extract_code(stock_code) if stock_code else ""

        # ── Phase 1: 并行数据预取 ──────────────────────────────
        logger.info(f"{WAIT_ICON} ValueAgent: Phase 1 — 并行获取 {len(VALUE_TOOL_NAMES)} 个工具数据...")
        phase1_start = time.time()

        try:
            all_tools = await get_mcp_tools(tool_filter=VALUE_TOOL_NAMES)
        except Exception as e:
            logger.error(f"{ERROR_ICON} ValueAgent: 获取 MCP 工具失败: {e}")
            all_tools = []

        if not all_tools:
            logger.error(f"{ERROR_ICON} ValueAgent: No MCP tools available.")
            current_data["value_analysis_error"] = "No MCP tools available."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "No tools")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        tool_map = {t.name: t for t in all_tools}
        logger.info(f"{SUCCESS_ICON} ValueAgent: 已加载 {len(all_tools)}/{len(VALUE_TOOL_NAMES)} 个工具")

        quarters = _get_recent_quarters(current_date, count=2)
        q_latest = quarters[0]
        q_prior = quarters[1] if len(quarters) > 1 else q_latest

        tasks = []
        labels = []
        tool_infos = []  # (tool, kwargs) 用于空数据重试

        def _add(task, label, ti=None):
            tasks.append(task)
            labels.append(label)
            tool_infos.append(ti)

        def _placeholder(label, msg):
            labels.append(label); tasks.append(_noop_result(msg)); tool_infos.append(None)

        # --- 基本信息与行业 ---
        code_tools = [
            ("tushare_stock_info", "Tushare基本信息"),
            ("get_stock_industry", "行业分类"),
        ]
        for tname, label in code_tools:
            if tname in tool_map:
                kwargs = {"code": clean_code}
                if tname == "get_stock_industry":
                    kwargs["date"] = current_date
                _add(_call_tool_safe(tool_map[tname], kwargs, label), label, (tool_map[tname], kwargs))
            else:
                _placeholder(label, f"[{tname}] 工具不可用")

        # --- 三大报表（Tushare） ---
        for tname, label in [("tushare_income", "Tushare利润表"), ("tushare_balancesheet", "Tushare资产负债表"), ("tushare_cashflow", "Tushare现金流量表")]:
            if tname in tool_map:
                _add(_call_tool_safe(tool_map[tname], {"code": clean_code}, label), label, (tool_map[tname], {"code": clean_code}))
            else: _placeholder(label, f"[{tname}] 工具不可用")

        # --- 财务指标与估值 ---
        valuation_tools = [
            ("tushare_fina_indicator", "Tushare财务指标"),
            ("tushare_daily_basic", "Tushare日线基础"),
            ("tushare_pe_percentile", "Tushare PE分位"),
            ("tushare_ev_ebitda", "Tushare EV/EBITDA"),
        ]
        for tname, label in valuation_tools:
            if tname in tool_map:
                _add(_call_tool_safe(tool_map[tname], {"code": clean_code}, label), label, (tool_map[tname], {"code": clean_code}))
            else:
                _placeholder(label, f"[{tname}] 工具不可用")

        # --- 分红与股东 ---
        shareholder_tools = [
            ("tushare_dividend", "Tushare分红"),
            ("tushare_top10_holders", "Tushare十大股东"),
            ("tushare_holder_num", "Tushare股东人数"),
        ]
        for tname, label in shareholder_tools:
            if tname in tool_map:
                _add(_call_tool_safe(tool_map[tname], {"code": clean_code}, label), label, (tool_map[tname], {"code": clean_code}))
            else:
                _placeholder(label, f"[{tname}] 工具不可用")

        # --- 保留的 AkShare 工具（无 Tushare 平替：杜邦/运营/成长） ---
        legacy_fin_tools = [
            ("get_dupont_data", "杜邦分析"),
            ("get_operation_data", "运营数据"),
            ("get_growth_data", "成长数据"),
        ]
        for tool_name, label_base in legacy_fin_tools:
            if tool_name in tool_map:
                t = tool_map[tool_name]
                kw_latest = {"code": clean_code, "year": q_latest["year"], "quarter": q_latest["quarter"]}
                kw_prior = {"code": clean_code, "year": q_prior["year"], "quarter": q_prior["quarter"]}
                _add(_call_tool_safe(t, kw_latest, f"{label_base}({q_latest['year']}Q{q_latest['quarter']})"),
                     f"{label_base}(最新)", (t, kw_latest))
                _add(_call_tool_safe(t, kw_prior, f"{label_base}({q_prior['year']}Q{q_prior['quarter']})"),
                     f"{label_base}(上期)", (t, kw_prior))
            else:
                _placeholder(f"{label_base}(最新)", f"[{tool_name}] 工具不可用")
                _placeholder(f"{label_base}(上期)", f"[{tool_name}] 工具不可用")

        # 并行执行（第一轮）
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as gather_err:
            logger.error(f"{ERROR_ICON} ValueAgent: Phase 1 并行调用异常: {gather_err}")
            results = [f"并行调用异常: {gather_err}"] * len(tasks)

        # 空数据重试（最多额外3轮（并发递减8→4→2），覆盖率100%则提前跳出）
        results = await retry_failed_fetches(
            results, tool_infos, labels, _call_tool_safe,
            agent_label="ValueAgent",
        )

        safe_results = []
        for r in results:
            if isinstance(r, Exception):
                safe_results.append(f"[工具调用异常: {str(r)[:100]}]")
            else:
                safe_results.append(str(r) if r else "[空返回]")

        phase1_elapsed = time.time() - phase1_start
        success_count = sum(1 for r in safe_results if not is_empty_result(str(r)))
        total_real = len([ti for ti in tool_infos if ti is not None])
        logger.info(f"{SUCCESS_ICON} ValueAgent: Phase 1 完成 ({phase1_elapsed:.1f}s, {success_count}/{total_real} 个工具有效数据)")

        data_sections = []
        for label, result in zip(labels, safe_results):
            data_sections.append(f"### [{label}]\n{result}")
        raw_data_text = "\n\n".join(data_sections)

        # ── Phase 2: LLM 深度分析 ──────────────────────────────
        logger.info(f"{WAIT_ICON} ValueAgent: Phase 2 — LLM 深度分析 (model={model_name}, thinking=enabled)...")
        phase2_start = time.time()

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=1.0,
            request_timeout=LLM_TIMEOUT,
            max_tokens=16000,
            extra_body=get_thinking_body(base_url, True),
        )

        analysis_prompt = f"""请以券商分析师的标准，对{company_name}（股票代码：{stock_code}）进行估值分析。

当前时间：{current_time_info}
当前日期：{current_date}

## 原始数据

以下是通过工具获取的全部原始数据，请基于这些数据进行深度分析：

{raw_data_text}

## 分析要求

请进行以下估值分析（每个维度都需要基于上述原始数据，引用具体数字）：

1. 当前估值指标
   - 市盈率（PE TTM）：当前值，计算方式（股价/EPS）
   - 市净率（PB MRQ）：当前值，计算方式（股价/每股净资产）
   - 市销率（PS TTM）：当前值
   - 市现率（PCF TTM）：当前值
   - 如可用：EV/EBITDA

2. 行业对比估值
   - 将上述各估值指标与行业平均水平对比（高估/低估/合理）
   - 与同行业3-5家可比公司进行横向对比
   - 解释估值差异的原因（增长预期、盈利能力、风险水平等）

3. 历史估值水平
   - 当前PE/PB在过去3年中的分位水平（如：处于历史30%分位，说明比70%的时间便宜）
   - 历史估值趋势（估值扩张/收缩）
   - 估值与股价走势的关系

4. 股息收益率分析
   - 当前股息率
   - 历史分红记录与分红稳定性
   - 股息率与无风险利率（存款利率/国债收益率）的对比

5. 内在价值评估
   - 基于各估值指标给出合理价值区间
   - 当前股价相对于合理价值区间的偏离程度
   - 安全边际分析

6. 综合估值结论
   - 当前估值判断（低估/合理/高估）
   - 估值风险提示
   - 简要投资建议

重要限制：
- 请专注于估值指标和财务数据分析，不要编造新闻信息
- 分析必须有数据支撑，引用具体的估值数字
- 如果某些数据无法获取，请说明原因并基于可用数据提供分析

⛔ 输出格式要求（防幻觉机制）：
请将分析输出严格分为两个区域：

## 📊 数据事实区
列出上述原始数据中的关键客观数据，每条标注数据来源标签：
- [标签] 具体数值（如：PE_TTM=15.2，PB_MRQ=1.8）
- [标签] 具体数值
- ...
如果某项数据在上述原始数据中标注为「数据不可用」，必须在此如实声明，不得推测。

## 🔍 分析判断区
基于上述数据事实进行分析和推断。每个判断必须：
1. 引用数据事实区的具体数值
2. 使用「【基于数据的推断】」或「【行业知识补充】」标注推断性质
3. 如果某个结论无法从数据中直接得出，必须声明「此为分析师推断」
4. 不得在任何地方编造数据事实区没有的数值"""

        messages = [
            {"role": "system", "content": "你是一位资深券商估值分析师，专注于A股公司的估值分析和投资价值判断。"},
            {"role": "user", "content": analysis_prompt},
        ]

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=float(LLM_TIMEOUT)
            )
            final_output = response.content.strip() if hasattr(response, 'content') else str(response)
            phase2_elapsed = time.time() - phase2_start
            logger.info(f"ValueAgent: Phase 2 完成 ({phase2_elapsed:.1f}s, {len(final_output)} 字符)")
            llm_success = True
        except asyncio.TimeoutError:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} ValueAgent: LLM 超时 ({phase2_elapsed:.0f}s)")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析判断区\n\n⚠️ LLM分析超时，以上为原始数据，请人工分析。\n"
            llm_success = False
        except Exception as llm_err:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} ValueAgent: LLM 失败: {llm_err}")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析判断区\n\n⚠️ LLM分析失败({str(llm_err)[:100]})，以上为原始数据，请人工分析。\n"
            llm_success = False

        model_config_log = {
            "model": model_name,
            "temperature": 1.0,
            "max_tokens": 16000,
            "thinking": "enabled",
            "api_base": base_url,
            "architecture": "two-phase",
        }
        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="two_phase_analysis",
            input_messages=[{"role": "user", "content": analysis_prompt[:5000]}],
            output_content=final_output,
            model_config=model_config_log,
            execution_time=phase2_elapsed,
        )

        total_time = time.time() - agent_start_time
        logger.info(f"{SUCCESS_ICON if llm_success else WAIT_ICON} ValueAgent: 分析完成 (总耗时 {total_time:.1f}s, Phase1={phase1_elapsed:.1f}s, Phase2={phase2_elapsed:.1f}s)")

        current_data["value_analysis"] = final_output
        if not skip_cache and cache_date and cache_code:
            write_cache("value_analysis", cache_code, cache_date, final_output)
        current_metadata["value_agent_executed"] = True
        current_metadata["value_agent_timestamp"] = str(time.time())
        current_metadata["value_agent_execution_time"] = f"{total_time:.2f} seconds"

        execution_logger.log_agent_complete(agent_name, {
            "value_analysis_length": len(final_output),
            "analysis_preview": final_output[:500],
            "phase1_time": phase1_elapsed,
            "phase2_time": phase2_elapsed,
            "total_time": total_time,
            "tools_queried": len(labels),
            "tools_with_data": success_count,
            "llm_success": llm_success,
        }, total_time, True)

        return {
            "data": current_data,
            "messages": current_messages + [{"role": "assistant", "content": "估值分析已完成"}],
            "metadata": current_metadata,
        }

    except Exception as e:
        logger.error(f"{ERROR_ICON} ValueAgent: Error: {e}", exc_info=True)
        current_data["value_analysis_error"] = f"Error: {e}"
        current_data["value_analysis"] = f"估值分析过程中出现错误: {str(e)}"
        current_metadata["value_agent_error"] = str(e)
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}


# 本地测试函数
async def test_value_agent():
    """估值分析 Agent的测试函数"""
    from src.utils.state_definition import AgentState

    current_datetime = datetime.now()
    current_date_cn = current_datetime.strftime("%Y年%m月%d日")
    current_date_en = current_datetime.strftime("%Y-%m-%d")
    current_weekday_cn = ["星期一", "星期二", "星期三", "星期四",
                          "星期五", "星期六", "星期日"][current_datetime.weekday()]
    current_time = current_datetime.strftime("%H:%M:%S")
    current_time_info = f"{current_date_cn} ({current_date_en}) {current_weekday_cn} {current_time}"

    test_state = AgentState(
        messages=[],
        data={
            "query": "分析嘉友国际的估值",
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

    result = await value_agent(test_state)
    print("Valuation Analysis Result:")
    print(result.get("data", {}).get("value_analysis", "No analysis found"))
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_value_agent())
