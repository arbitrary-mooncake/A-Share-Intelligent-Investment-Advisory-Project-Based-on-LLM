"""
FundHoldingsAgent: 两阶段架构 — 基金数据预取 + 持股穿透 + 单次LLM深度分析。
Phase 1a: 并行获取基金基本信息与持仓数据（fund_basic + fund_portfolio）
Phase 1b: 从持仓中解析重仓股代码，为每只重仓股并行获取个股数据
Phase 2: 将所有原始数据喂给 LLM 一次性完成持仓穿透分析（thinking 开启）

核心分析维度：
  1. 资产配置结构  2. 行业暴露  3. 持仓集中度
  4. 重仓股质量  5. 与基准偏离度  6. 风格暴露
  7. 风格漂移检测  8. 数据时滞标注  9. 名义vs实际
"""
import asyncio
import os
import re
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

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

# 单个工具超时（秒）
TOOL_TIMEOUT = 30
# LLM 整体超时（秒）
LLM_TIMEOUT = 300

# 基金持仓分析白名单
FUND_HOLDINGS_TOOLS = [
    "tushare_fund_portfolio",   # 前十大持仓（权重/行业）
    "tushare_fund_basic",        # 基金类型/基准指数
    "tushare_stock_info",        # 持仓股行业分类
    "tushare_daily_basic",       # 持仓股估值（PE/PB）
    "tushare_fina_indicator",    # 持仓股财务指标（ROE）
]


def _extract_code(code: str) -> str:
    """提取纯6位数字代码（兼容 sh.xxx / sz.xxx / xxx.SH / xxx.SZ / 纯数字）"""
    code = code.strip()
    # 处理 sh.xxx / sz.xxx 格式
    m = re.search(r"[sS][hHzZ]\.(\d{6})", code)
    if m:
        return m.group(1)
    # 处理 xxx.SH / xxx.SZ 格式
    m = re.search(r"(\d{6})\.[sS][hHzZ]", code)
    if m:
        return m.group(1)
    # 纯6位数字
    m = re.search(r"(\d{6})", code)
    if m:
        return m.group(1)
    return code


def _extract_stock_codes_from_portfolio(portfolio_text: str, max_holdings: int = 10) -> List[str]:
    """
    从基金持仓数据文本中提取重仓股代码列表。
    会去重并保持顺序，限制最多 max_holdings 只。
    """
    if not portfolio_text:
        return []

    codes = set()
    ordered = []

    # 匹配各种格式的6位数字码（排除9位基金代码）
    # 格式: 000858.SZ, 000858.SH, sh.000858, sz.000858, 或纯6位数字
    patterns = [
        r'(?:^|\s|[\["\',;:=])(\d{6})(?:\.(?:SH|SZ|sh|sz))?(?:$|\s|[\]"\',;:\s])',  # 6位数字（可能有后缀）
    ]

    for pattern in patterns:
        for m in re.finditer(pattern, portfolio_text):
            candidate = m.group(1)
            # 排除明显的非股票代码（如基金代码9开头且后面跟着更多位数字）
            # 跳过出现在较长数字串中的6位数字（可能不是股票代码）
            if candidate not in codes and len(ordered) < max_holdings:
                codes.add(candidate)
                ordered.append(candidate)

    logger.info(f"FundHoldingsAgent: 从持仓数据中提取到 {len(ordered)} 只重仓股代码: {ordered[:5]}...")
    return ordered


def _get_report_freshness_penalty(current_date: str) -> Dict[str, Any]:
    """
    计算季报数据时效性信息。
    年报/季报截止日：Q1=3/31, Q2=6/30, Q3=9/30, Q4=12/31
    季报披露存在1-2个月滞后期（一季报4/30前，中报8/31前，三季报10/31前，年报次年4/30前）
    返回最新完成报告期及距今月数，用于prompt中的数据时滞标注。
    """
    try:
        dt = datetime.strptime(current_date, "%Y-%m-%d")
    except ValueError:
        dt = datetime.now()

    current_year = dt.year
    current_month = dt.month

    # 根据当前月份确定最新已完成的季度
    # 例如：当前5月 → 最新完成Q1（3/31结束），Q1报告在4/30前披露完
    #       当前8月 → 最新完成Q2（6/30结束），Q2报告在8/31前披露完
    #       当前11月 → 最新完成Q3（9/30结束），Q3报告在10/31前披露完
    #       当前1-3月 → 最新完成上年Q4（12/31结束），年报在次年4/30前披露完
    if current_month <= 3:
        prev_q, prev_y = 4, current_year - 1  # 上年Q4
    elif current_month <= 6:
        prev_q, prev_y = 1, current_year  # Q1
    elif current_month <= 9:
        prev_q, prev_y = 2, current_year  # Q2
    else:
        prev_q, prev_y = 3, current_year  # Q3

    # 计算上一完成季度
    report_end_months = {4: 12, 1: 3, 2: 6, 3: 9}
    report_end_month = report_end_months[prev_q]

    # 计算距今月数（以报告期末月为基准）
    months_ago = max(0, (dt.year - prev_y) * 12 + (dt.month - report_end_month))

    # 披露月份（报告期结束后1-2个月）
    disclose_start_month = report_end_month + 1 if report_end_month < 12 else 1
    disclose_end_month = report_end_month + 2 if report_end_month < 11 else (report_end_month + 2 - 12)
    disclose_year = prev_y
    if report_end_month >= 11:
        disclose_year = prev_y + 1

    return {
        "previous_report_period": f"{prev_y}Q{prev_q}",
        "months_since_period_end": months_ago,
        "staleness_warning": (
            f"持仓数据最新完整报告期为{prev_y}Q{prev_q}（截止{prev_y}年{report_end_month}月底），"
            f"距今约{months_ago}个月。"
            f"季报披露存在1-2个月滞后（{prev_y}Q{prev_q}报告约在{disclose_year}年{disclose_start_month}-{disclose_end_month}月披露），"
            f"当前日期距该报告期末已{months_ago}个月，持仓可能已发生较大变化。"
        ),
        "months_ago": months_ago,
    }


async def _noop_result(text: str) -> str:
    """返回固定文本的占位协程，用于工具不可用时的占位"""
    return text


async def _call_tool_safe(tool, kwargs: dict, label: str) -> str:
    """调用单个 MCP 工具，带超时和异常保护"""
    from src.utils.tool_cache import get_cached_tool_result, set_cached_tool_result
    tool_name = getattr(tool, 'name', 'unknown')
    cached = await get_cached_tool_result(tool_name, kwargs)
    if cached:
        return cached
    try:
        result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=TOOL_TIMEOUT)
        text = str(result).strip()
        if len(text) > 20:
            logger.info(f"{SUCCESS_ICON} FundHoldingsAgent: {label} 获取成功 ({len(text)} 字符)")
            await set_cached_tool_result(tool_name, kwargs, text)
            return text
        logger.warning(f"FundHoldingsAgent: {label} 返回过短 ({len(text)} 字符)")
        return f"[{label}] 数据不可用（返回过短）"
    except asyncio.TimeoutError:
        logger.warning(f"FundHoldingsAgent: {label} 超时({TOOL_TIMEOUT}s)")
        return f"[{label}] 数据不可用（超时）"
    except Exception as e:
        logger.warning(f"FundHoldingsAgent: {label} 调用失败: {e}")
        return f"[{label}] 数据不可用（调用失败: {str(e)[:80]}）"


async def fund_holdings_analysis(state: AgentState) -> AgentState:
    """
    两阶段基金持仓穿透分析：
    Phase 1a: 并行获取基金基本信息与持仓数据
    Phase 1b: 解析持仓中的重仓股代码，为每只重仓股并行获取个股数据
    Phase 2: 单次 LLM 深度分析（MiMo-V2.5-Pro, thinking=enabled）
    """
    logger.info(f"{WAIT_ICON} FundHoldingsAgent: Starting two-phase fund holdings analysis.")

    execution_logger = get_execution_logger()
    agent_name = "fund_holdings_agent"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})

    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    # 使用 fund_code 作为缓存 key（如果是基金分析场景）
    cache_code = current_data.get("fund_code", "") or current_data.get("stock_code", "")

    # 缓存检查（TTL=15天，由 cache_utils 控制）
    if not skip_cache and cache_date and cache_code:
        cached = read_cache("fund_holdings", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} FundHoldingsAgent: 命中缓存，跳过分析 ({cache_code})")
            current_data["fund_holdings"] = cached
            from src.utils.cache_utils import read_signal_pack_cache
            cached_sp = read_signal_pack_cache("fund_holdings", cache_code, cache_date)
            if cached_sp:
                current_data["fund_holdings_signal_pack"] = cached_sp
            current_metadata["fund_holdings_executed"] = True
            current_metadata["fund_holdings_cached"] = True
            return {"data": current_data,
                    "messages": current_messages + [{"role": "assistant", "content": "基金持仓穿透分析已完成（缓存）"}],
                    "metadata": current_metadata}

    execution_logger.log_agent_start(agent_name, {
        "user_query": current_data.get("query"),
        "fund_code": current_data.get("fund_code"),
        "fund_name": current_data.get("fund_name"),
        "input_data_keys": list(current_data.keys()),
        "architecture": "two-phase-with-holdings-penetration",
    })

    agent_start_time = time.time()

    try:
        # 模型配置：MiMo-V2.5-Pro
        model_cfg = get_model_config_for_agent("fund_holdings_agent", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]

        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} FundHoldingsAgent: Missing OpenAI environment variables.")
            current_data["fund_holdings_error"] = "Missing OpenAI environment variables."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "Missing env vars")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        fund_code_raw = current_data.get("fund_code", "") or current_data.get("stock_code", "Unknown")
        fund_name = current_data.get("fund_name", "") or current_data.get("company_name", "Unknown")
        current_time_info = current_data.get("current_time_info", "未知时间")
        current_date = current_data.get("current_date", "未知日期")
        clean_fund_code = _extract_code(fund_code_raw) if fund_code_raw else ""

        # 计算季报时效性
        freshness = _get_report_freshness_penalty(current_date)

        # ── Phase 1a: 基金数据预取（fund_basic + fund_portfolio） ──
        logger.info(f"{WAIT_ICON} FundHoldingsAgent: Phase 1a — 获取基金基本信息与持仓数据...")
        phase1a_start = time.time()

        try:
            fund_tool_names = ["tushare_fund_basic", "tushare_fund_portfolio"]
            fund_tools = await get_mcp_tools(tool_filter=fund_tool_names)
        except Exception as e:
            logger.error(f"{ERROR_ICON} FundHoldingsAgent: 获取基金 MCP 工具失败: {e}")
            fund_tools = []

        if not fund_tools:
            logger.error(f"{ERROR_ICON} FundHoldingsAgent: No fund MCP tools available.")
            current_data["fund_holdings_error"] = "No fund MCP tools available."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "No tools")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        fund_tool_map = {t.name: t for t in fund_tools}

        # Phase 1a tasks
        fund_tasks = []
        fund_labels = []
        fund_tool_infos = []

        def _add_fund(task, label, ti=None):
            fund_tasks.append(task)
            fund_labels.append(label)
            fund_tool_infos.append(ti)

        def _placeholder_fund(label, msg):
            fund_labels.append(label)
            fund_tasks.append(_noop_result(msg))
            fund_tool_infos.append(None)

        # fund_basic
        if "tushare_fund_basic" in fund_tool_map:
            _add_fund(
                _call_tool_safe(fund_tool_map["tushare_fund_basic"], {"code": clean_fund_code}, "基金基本信息"),
                "基金基本信息",
                (fund_tool_map["tushare_fund_basic"], {"code": clean_fund_code})
            )
        else:
            _placeholder_fund("基金基本信息", "[tushare_fund_basic] 工具不可用")

        # fund_portfolio
        if "tushare_fund_portfolio" in fund_tool_map:
            _add_fund(
                _call_tool_safe(fund_tool_map["tushare_fund_portfolio"], {"code": clean_fund_code}, "基金持仓"),
                "基金持仓",
                (fund_tool_map["tushare_fund_portfolio"], {"code": clean_fund_code})
            )
        else:
            _placeholder_fund("基金持仓", "[tushare_fund_portfolio] 工具不可用")

        # 并行执行 Phase 1a
        try:
            fund_results = await asyncio.gather(*fund_tasks, return_exceptions=True)
        except Exception as gather_err:
            logger.error(f"{ERROR_ICON} FundHoldingsAgent: Phase 1a 并行调用异常: {gather_err}")
            fund_results = [f"并行调用异常: {gather_err}"] * len(fund_tasks)

        # 空数据重试
        fund_results = await retry_failed_fetches(
            fund_results, fund_tool_infos, fund_labels, _call_tool_safe,
            agent_label="FundHoldingsAgent",
        )

        # 处理异常结果
        fund_safe_results = []
        for r in fund_results:
            if isinstance(r, Exception):
                fund_safe_results.append(f"[工具调用异常: {str(r)[:100]}]")
            else:
                fund_safe_results.append(str(r) if r else "[空返回]")

        phase1a_elapsed = time.time() - phase1a_start
        logger.info(f"{SUCCESS_ICON} FundHoldingsAgent: Phase 1a 完成 ({phase1a_elapsed:.1f}s)")

        # ── Phase 1b: 解析持仓 → 为每只重仓股获取个股数据 ──
        logger.info(f"{WAIT_ICON} FundHoldingsAgent: Phase 1b — 解析持仓并获取重仓股个股数据...")
        phase1b_start = time.time()

        # 从基金持仓结果中提取重仓股代码
        portfolio_text = ""
        for label, result in zip(fund_labels, fund_safe_results):
            if "持仓" in label and "数据不可用" not in str(result) and "工具不可用" not in str(result):
                portfolio_text = str(result)
                break

        holding_codes = _extract_stock_codes_from_portfolio(portfolio_text, max_holdings=10)

        # 加载个股相关工具
        stock_tool_names = ["tushare_stock_info", "tushare_daily_basic", "tushare_fina_indicator"]
        try:
            stock_tools = await get_mcp_tools(tool_filter=stock_tool_names)
        except Exception as e:
            logger.error(f"{ERROR_ICON} FundHoldingsAgent: 获取个股 MCP 工具失败: {e}")
            stock_tools = []

        stock_tool_map = {t.name: t for t in stock_tools}
        logger.info(f"{SUCCESS_ICON} FundHoldingsAgent: 已加载 {len(stock_tools)}/{len(stock_tool_names)} 个个股工具")

        # 为每只重仓股构建并行任务
        stock_tasks = []
        stock_labels = []
        stock_tool_infos = []

        def _add_stock(task, label, ti=None):
            stock_tasks.append(task)
            stock_labels.append(label)
            stock_tool_infos.append(ti)

        def _placeholder_stock(label, msg):
            stock_labels.append(label)
            stock_tasks.append(_noop_result(msg))
            stock_tool_infos.append(None)

        for hcode in holding_codes:
            hcode_clean = _extract_code(hcode)

            # tushare_stock_info
            if "tushare_stock_info" in stock_tool_map:
                _add_stock(
                    _call_tool_safe(stock_tool_map["tushare_stock_info"], {"code": hcode_clean}, f"持仓股{hcode}基本信息"),
                    f"持仓股{hcode}基本信息",
                    (stock_tool_map["tushare_stock_info"], {"code": hcode_clean})
                )
            else:
                _placeholder_stock(f"持仓股{hcode}基本信息", f"[tushare_stock_info] 工具不可用")

            # tushare_daily_basic
            if "tushare_daily_basic" in stock_tool_map:
                _add_stock(
                    _call_tool_safe(stock_tool_map["tushare_daily_basic"], {"code": hcode_clean, "days": 60}, f"持仓股{hcode}估值数据"),
                    f"持仓股{hcode}估值数据",
                    (stock_tool_map["tushare_daily_basic"], {"code": hcode_clean, "days": 60})
                )
            else:
                _placeholder_stock(f"持仓股{hcode}估值数据", f"[tushare_daily_basic] 工具不可用")

            # tushare_fina_indicator
            if "tushare_fina_indicator" in stock_tool_map:
                _add_stock(
                    _call_tool_safe(stock_tool_map["tushare_fina_indicator"], {"code": hcode_clean, "years": 4}, f"持仓股{hcode}财务指标"),
                    f"持仓股{hcode}财务指标",
                    (stock_tool_map["tushare_fina_indicator"], {"code": hcode_clean, "years": 4})
                )
            else:
                _placeholder_stock(f"持仓股{hcode}财务指标", f"[tushare_fina_indicator] 工具不可用")

        # 如果没有提取到任何持仓代码，记录警告
        if not holding_codes:
            logger.warning(f"{WAIT_ICON} FundHoldingsAgent: 未能从持仓数据中提取到重仓股代码，跳过个股数据获取")
            _placeholder_stock("持仓股信息", "[无持仓股代码] 未能从基金持仓数据中解析出重仓股代码，可能该基金为债基/货基/FoF")

        # 并行执行 Phase 1b
        stock_safe_results = []
        if stock_tasks:
            try:
                stock_results = await asyncio.gather(*stock_tasks, return_exceptions=True)
            except Exception as gather_err:
                logger.error(f"{ERROR_ICON} FundHoldingsAgent: Phase 1b 并行调用异常: {gather_err}")
                stock_results = [f"并行调用异常: {gather_err}"] * len(stock_tasks)

            # 空数据重试（对持仓股数据也应用重试机制）
            stock_results = await retry_failed_fetches(
                stock_results, stock_tool_infos, stock_labels, _call_tool_safe,
                agent_label="FundHoldingsAgent-Holdings",
            )

            for r in stock_results:
                if isinstance(r, Exception):
                    stock_safe_results.append(f"[工具调用异常: {str(r)[:100]}]")
                else:
                    stock_safe_results.append(str(r) if r else "[空返回]")

        phase1b_elapsed = time.time() - phase1b_start
        stock_success = sum(1 for r in stock_safe_results if not is_empty_result(str(r)))
        stock_total = len([ti for ti in stock_tool_infos if ti is not None])
        logger.info(f"{SUCCESS_ICON} FundHoldingsAgent: Phase 1b 完成 ({phase1b_elapsed:.1f}s, {stock_success}/{stock_total} 个持仓工具有效数据)")

        # ── 聚合全部数据 ──
        data_sections = []

        # 基金数据
        for label, result in zip(fund_labels, fund_safe_results):
            data_sections.append(f"### [{label}]\n{result}")

        # 持仓股数据
        for label, result in zip(stock_labels, stock_safe_results):
            data_sections.append(f"### [{label}]\n{result}")

        raw_data_text = "\n\n".join(data_sections)

        phase1_total = phase1a_elapsed + phase1b_elapsed
        total_success = sum(1 for r in fund_safe_results + stock_safe_results if not is_empty_result(str(r)))
        total_real = len([ti for ti in fund_tool_infos + stock_tool_infos if ti is not None])
        logger.info(f"{SUCCESS_ICON} FundHoldingsAgent: Phase 1 全部完成 ({phase1_total:.1f}s, {total_success}/{total_real} 个工具有效数据)")

        # ── Phase 2: LLM 深度分析 ──
        logger.info(f"{WAIT_ICON} FundHoldingsAgent: Phase 2 — LLM 深度分析 (model={model_name}, thinking=enabled)...")
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

        # 构建持有股列表（用于 prompt 引用）
        holdings_list = "\n".join([f"  - 注意：上文中 ### [持仓股{code}基本信息] 等标签对应的是此持仓股" for code in holding_codes]) if holding_codes else "  - 未能解析到具体持仓股代码"

        analysis_prompt = f"""请以基金分析师的视角，对{fund_name}（基金代码：{fund_code_raw}）进行**持仓穿透与组合结构分析**。

当前时间：{current_time_info}
当前日期：{current_date}

## 数据时效性警告

⚠️ **季报数据时滞提示**：
- 基金持仓数据来源于公开季报/年报披露，最新报告期为 {freshness['previous_report_period']}，距今约 {freshness['months_ago']} 个月
- {freshness['staleness_warning']}
- 请在分析中明确标注数据的截止日期，对可能已变化的持仓保持审慎态度
- 对于距今超过2个月的持仓数据，分析结论应附加「数据时滞风险」标注

## 原始数据

以下是通过工具获取的全部原始数据，请基于这些数据进行深度分析：

{raw_data_text}

## 重仓股列表

以下是从持仓数据中解析出的重仓股代码（共 {len(holding_codes)} 只）：
{', '.join(holding_codes) if holding_codes else '未能解析到具体持仓股代码'}

## 分析要求

请进行以下基金持仓穿透分析（每个维度都需要基于上述原始数据，引用具体数字）：

### 1. 资产配置结构
- 股票/债券/现金/其他资产的大致占比（基于持仓数据推算）
- 实际权益仓位 vs 名义仓位（是否悄然加仓或减仓）
- 如有行业ETF/主题基金重仓，需标注其底层资产性质

### 2. 行业暴露分析
- 通过重仓股的行业分类，汇总该基金的行业配置分布
- 前三大行业的集中度
- 是否存在行业押注（单一行业占比>30%即为行业押注）
- 是否存在行业漂移（当前行业配置 vs 基金招募说明书中的投资范围）

### 3. 持仓集中度
- 前十大重仓股合计占比
- 前三大重仓股合计占比
- 第一大重仓股占比及个股权重风险
- 集中度评价：<30%分散、30-50%适度集中、>50%高度集中

### 4. 重仓股质量评估
- 对前十大重仓股逐一进行简要评估（基于各自的PE、ROE、行业地位数据）
- 核心关注：高ROE+合理PE的组合占比、是否存在"低质量+高仓位"的异常高配
- 重仓股的整体质量评分（优/良/中/差）

### 5. 与基准偏离度
- 根据基金基本信息中的基准指数，分析持仓行业分布 vs 基准指数行业分布的偏离度
- 是否存在显著的行业超配/低配
- 主动偏离的合理性与风险

### 6. 风格暴露
- 大盘/中盘/小盘股的比例分布（基于重仓股市值推断）
- 成长/价值/平衡风格倾向
- 周期/防御/进攻型配置比例

### 7. 风格漂移检测
- 对比基金招募说明书中的投资目标与实际持仓风格是否一致
- 是否存在「名义上是A，实际上像B」的风格漂移问题（例如名称是"中小盘"但重仓大盘股）
- 前期持仓变化趋势（如果数据允许，推测基金经理的操作方向）

### 8. 名义vs实际分析（⚠️ 重点关注）
- 基金名称暗示的投资方向 vs 实际操作
- 是否存在"名为XX主题基金、实为抱团取暖"的情况
- 是否存在追逐热点、风格摇摆的迹象

### 9. 综合评估
- 组合结构的核心优势
- 组合结构的核心风险（含数据时滞风险）
- 组合结构评分（0-100分）
- 对投资者的配置建议

重要限制：
- 请专注于基金持仓数据和组合结构分析，不要编造个股新闻信息
- 分析必须有数据支撑，引用上述原始数据中的具体数字
- 如果某些数据无法获取，请说明原因并基于可用数据提供分析
- 如果基金为债基/货基/FoF/LoF，请明确标注并调整分析框架

⛔ 输出格式要求（防幻觉机制）：
请将分析输出严格分为两个区域：

## 📊 数据事实区
列出上述原始数据中的关键客观数据，每条标注数据来源标签：
- [基金基本信息] 具体数值（如：基金类型=偏股混合型，基准=沪深300指数）
- [基金持仓] 具体数值（如：前十大占比=45.2%，第一大=贵州茅台/8.5%）
- [持仓股000858基本信息] 具体数值（如：行业=白酒，市值=xxx亿）
- [持仓股000858估值数据] 具体数值（如：PE_TTM=25.3，PB=5.2）
- [持仓股000858财务指标] 具体数值（如：ROE=28.5%）
- ...
如果某项数据在上述原始数据中标注为「数据不可用」，必须在此如实声明，不得推测。

## 🔍 分析判断区
基于上述数据事实进行分析和推断。每个判断必须：
1. 引用数据事实区的具体数值
2. 使用「【基于数据的推断】」或「【行业知识补充】」标注推断性质
3. 如果某个结论无法从数据中直接得出，必须声明「此为分析师推断」
4. 不得在任何地方编造数据事实区没有的数值
5. 不得编造原始数据中不存在的新闻或事件
6. ⚠️ 对基于季报数据的分析结论，必须标注「【数据时滞风险：基于{freshness['previous_report_period']}报告期数据，距今{freshness['months_ago']}个月】」
7. ⚠️ 风格漂移判断必须在存在明确的名称与实际持仓矛盾时才能下结论，否则标注「数据不足以判断风格漂移」

## 🏷️ 综合评分
给出组合结构最终评分（0-100分），并简要说明评分理由。评分维度权重：
- 持仓质量（重仓股ROE/PE质量）: 30%
- 组合分散度（行业/个股集中度）: 25%
- 风格一致性（不漂移/不抱团）: 20%
- 资产配置合理性: 15%
- 数据时效性（越新越高）: 10%

⛔ 结构化输出要求：
在完成上述分析后，请额外输出一个 JSON block：

<SIGNAL_PACK>
{{
    "bias": "bullish"|"neutral"|"bearish",
    "confidence": 0.0-1.0 (基金分析置信度),
    "key_points": ["关键结论1", "关键结论2", ...] (最多5条,每条<80字),
    "signals": [
        {{
            "factor": "因子名",
            "direction": 1(利多)|-1(利空)|0(中性),
            "strength": 0-100,
            "time_horizon": ["medium","long"],
            "source_level": "structured"|"derived",
            "note": "一句话说明"
        }}
    ] (最多4条),
    "risk_flags": [],
    "missing_data": ["缺失项"],
    "source_summary": "数据来源简述"
}}
</SIGNAL_PACK>"""

        messages = [
            {"role": "system", "content": "你是一位资深基金分析师，专注于公募基金的持仓穿透、组合结构分析和风格漂移检测。你的分析以数据为驱动，对季报时滞风险保持高度警惕。"},
            {"role": "user", "content": analysis_prompt},
        ]

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=float(LLM_TIMEOUT)
            )
            final_output = response.content.strip() if hasattr(response, 'content') else str(response)
            # Extract signal_pack from LLM output
            import json as _json, re as _re
            sp = None
            tag_match = _re.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', final_output)
            if tag_match:
                try:
                    sp = _json.loads(tag_match.group(1))
                    sp["agent_name"] = "fund_holdings"
                    sp["as_of_date"] = current_date
                except Exception:
                    pass
            if sp is None:
                from src.utils.analysis_package_builder import text_to_signal_pack
                sp = text_to_signal_pack(final_output, "fund_holdings", current_date)
            current_data["fund_holdings_signal_pack"] = sp
            phase2_elapsed = time.time() - phase2_start
            logger.info(f"FundHoldingsAgent: Phase 2 完成 ({phase2_elapsed:.1f}s, {len(final_output)} 字符)")
            llm_success = True
        except asyncio.TimeoutError:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundHoldingsAgent: LLM 超时 ({phase2_elapsed:.0f}s)")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析判断区\n\n⚠️ LLM分析超时，以上为原始数据，请人工分析。\n"
            llm_success = False
        except Exception as llm_err:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundHoldingsAgent: LLM 失败: {llm_err}")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析判断区\n\n⚠️ LLM分析失败({str(llm_err)[:100]})，以上为原始数据，请人工分析。\n"
            llm_success = False

        # 记录 LLM 交互
        model_config_log = {
            "model": model_name,
            "temperature": 1.0,
            "max_tokens": 16000,
            "thinking": "enabled",
            "api_base": base_url,
            "architecture": "two-phase-with-holdings-penetration",
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
        logger.info(f"{SUCCESS_ICON if llm_success else WAIT_ICON} FundHoldingsAgent: 分析完成 (总耗时 {total_time:.1f}s, Phase1a={phase1a_elapsed:.1f}s, Phase1b={phase1b_elapsed:.1f}s, Phase2={phase2_elapsed:.1f}s)")

        # 更新状态
        current_data["fund_holdings"] = final_output
        if not skip_cache and cache_date and cache_code:
            write_cache("fund_holdings", cache_code, cache_date, final_output)
            if "fund_holdings_signal_pack" in current_data:
                from src.utils.cache_utils import write_signal_pack_cache
                write_signal_pack_cache("fund_holdings", cache_code, cache_date, current_data["fund_holdings_signal_pack"])
        current_metadata["fund_holdings_executed"] = True
        current_metadata["fund_holdings_timestamp"] = str(time.time())
        current_metadata["fund_holdings_execution_time"] = f"{total_time:.2f} seconds"

        execution_logger.log_agent_complete(agent_name, {
            "fund_holdings_length": len(final_output),
            "analysis_preview": final_output[:500],
            "phase1a_time": phase1a_elapsed,
            "phase1b_time": phase1b_elapsed,
            "phase2_time": phase2_elapsed,
            "total_time": total_time,
            "holdings_count": len(holding_codes),
            "fund_tools_queried": len(fund_labels),
            "stock_tools_queried": len(stock_labels),
            "total_tools_with_data": total_success,
            "llm_success": llm_success,
        }, total_time, True)

        return {
            "data": current_data,
            "messages": current_messages + [{"role": "assistant", "content": "基金持仓穿透分析已完成"}],
            "metadata": current_metadata,
        }

    except Exception as e:
        logger.error(f"{ERROR_ICON} FundHoldingsAgent: Error: {e}", exc_info=True)
        current_data["fund_holdings_error"] = f"Error: {e}"
        current_data["fund_holdings"] = f"基金持仓穿透分析过程中出现错误: {str(e)}"
        current_metadata["fund_holdings_error"] = str(e)
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}


# 本地测试函数
async def test_fund_holdings_agent():
    """基金持仓穿透分析 Agent的测试函数"""
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
            "query": "分析黄金ETF华安的持仓穿透与组合结构",
            "fund_code": "sh.518880",
            "fund_name": "黄金ETF华安",
            "stock_code": "sh.518880",
            "company_name": "黄金ETF华安",
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat()
        },
        metadata={}
    )

    result = await fund_holdings_analysis(test_state)
    print("Fund Holdings Analysis Result:")
    print(result.get("data", {}).get("fund_holdings", "No analysis found"))
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fund_holdings_agent())
