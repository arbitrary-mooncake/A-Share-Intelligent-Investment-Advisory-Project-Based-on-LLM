"""
FundamentalAnalysis Agent: 两阶段架构 — 并行数据预取 + 单次 LLM 深度分析。
Phase 1: asyncio.gather 并行获取白名单全部 18 个工具的数据
Phase 2: 将所有原始数据喂给 LLM 一次性完成深度分析（thinking 开启）
"""
import asyncio
import os
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

# 基本面分析白名单（2026-06 全量迁移至 Tushare，移除 AkShare 依赖）
FUNDAMENTAL_TOOL_NAMES = [
    # 基本信息与行业
    "tushare_stock_info",       # 行业分类（替代 get_stock_industry）
    # 财务报表（Tushare 三大报表 + 财务指标）
    "tushare_income", "tushare_balancesheet", "tushare_cashflow",
    "tushare_fina_indicator",    # 财务指标（替代 get_dupont_data/get_operation_data/get_growth_data）
    # 估值与分红
    "tushare_daily_basic", "tushare_dividend",
    "tushare_ev_ebitda", "tushare_adj_factor",
    # 股东与ST
    "tushare_top10_holders", "tushare_st_status",  # ST状态（替代 get_st_risk_data）
]


def _extract_code(stock_code: str) -> str:
    """提取纯数字股票代码"""
    return stock_code.replace("sh.", "").replace("sz.", "").replace("bj.", "").replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()


def _get_recent_quarters(date_str: str, count: int = 2) -> List[Dict[str, Any]]:
    """根据当前日期计算最近 N 个已完成的季度"""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        dt = datetime.now()
    quarters = []
    # 当前季度
    current_q = (dt.month - 1) // 3 + 1
    current_year = dt.year
    # 最新完成的季度是上一个季度
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


def _extract_signal_pack_from_llm(llm_output: str, agent_name: str, as_of_date: str) -> dict:
    """
    从LLM输出中提取signal_pack JSON。
    三层fallback: JSON解析 → 正则提取 → 文本推断
    """
    import json as _json
    import re as _re

    # 第一层: <SIGNAL_PACK> 标签
    tag_match = _re.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', llm_output)
    if tag_match:
        try:
            sp = _json.loads(tag_match.group(1))
            sp["agent_name"] = agent_name
            sp["as_of_date"] = as_of_date
            sp.setdefault("analysis_text", llm_output[:500])
            sp.setdefault("data_quality_score", 0.7)
            return sp
        except (_json.JSONDecodeError, ValueError):
            pass

    # 第二层: 从文本中找包含bias和signals的JSON
    json_match = _re.search(r'\{[\s\S]*"bias"[\s\S]*"signals"[\s\S]*\}', llm_output)
    if json_match:
        try:
            sp = _json.loads(json_match.group(0))
            sp["agent_name"] = agent_name
            sp["as_of_date"] = as_of_date
            sp.setdefault("analysis_text", llm_output[:500])
            sp.setdefault("data_quality_score", 0.5)
            return sp
        except (_json.JSONDecodeError, ValueError):
            pass

    # 第三层: 纯文本推断
    from src.utils.analysis_package_builder import text_to_signal_pack
    return text_to_signal_pack(llm_output, agent_name, as_of_date)


async def _noop_result(text: str) -> str:
    """返回固定文本的占位协程，用于工具不可用时的占位"""
    return text


async def _call_tool_safe(tool, kwargs: dict, label: str) -> str:
    """调用单个 MCP 工具，带超时、异常保护、和工具级缓存"""
    # Check tool cache first
    from src.utils.tool_cache import get_cached_tool_result, set_cached_tool_result
    tool_name = getattr(tool, 'name', 'unknown')
    cached = await get_cached_tool_result(tool_name, kwargs)
    if cached:
        logger.info(f"{SUCCESS_ICON} FundamentalAgent: {label} [工具缓存命中]")
        return cached

    try:
        result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=TOOL_TIMEOUT)
        text = str(result).strip()
        if len(text) > 20:
            logger.info(f"{SUCCESS_ICON} FundamentalAgent: {label} 获取成功 ({len(text)} 字符)")
            await set_cached_tool_result(tool_name, kwargs, text)
            return text
        logger.warning(f"FundamentalAgent: {label} 返回过短 ({len(text)} 字符)")
        return f"[{label}] 数据不可用（返回过短）"
    except asyncio.TimeoutError:
        logger.warning(f"FundamentalAgent: {label} 超时({TOOL_TIMEOUT}s)")
        return f"[{label}] 数据不可用（超时）"
    except Exception as e:
        logger.warning(f"FundamentalAgent: {label} 调用失败: {e}")
        return f"[{label}] 数据不可用（调用失败: {str(e)[:80]}）"


async def fundamental_agent(state: AgentState) -> AgentState:
    """
    两阶段基本面分析：
    Phase 1: 并行获取全部 18 个工具的数据
    Phase 2: 单次 LLM 深度分析（Kimi K2.6, thinking=enabled）
    """
    logger.info(f"{WAIT_ICON} FundamentalAgent: Starting two-phase fundamental analysis.")

    execution_logger = get_execution_logger()
    agent_name = "fundamental_agent"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})
    user_query = current_data.get("query")

    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    cache_code = current_data.get("stock_code", "")

    # 缓存检查（TTL=15天，由 cache_utils 控制）
    if not skip_cache and cache_date and cache_code:
        cached = read_cache("fundamental_analysis", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} FundamentalAgent: 命中缓存，跳过分析 ({cache_code})")
            current_data["fundamental_analysis"] = cached
            from src.utils.cache_utils import read_signal_pack_cache
            cached_sp = read_signal_pack_cache("fundamental_analysis", cache_code, cache_date)
            if cached_sp:
                current_data["fundamental_signal_pack"] = cached_sp
            else:
                current_data["fundamental_signal_pack"] = _extract_signal_pack_from_llm(cached, "fundamental", cache_date)
            current_metadata["fundamental_agent_executed"] = True
            current_metadata["fundamental_agent_cached"] = True
            return {"data": current_data,
                    "messages": current_messages + [{"role": "assistant", "content": "基本面分析已完成（缓存）"}],
                    "metadata": current_metadata}

    if current_data.get("is_etf", False):
        logger.info(f"{SUCCESS_ICON} FundamentalAgent: ETF标的，跳过基本面分析")
        current_data["fundamental_analysis"] = "该标的为ETF产品，不适用基本面分析（无ROE、毛利率、财报等财务指标）。"
        current_metadata["fundamental_agent_executed"] = True
        return {"data": current_data,
                "messages": current_messages + [{"role": "assistant", "content": "ETF标的，已跳过基本面分析"}],
                "metadata": current_metadata}

    execution_logger.log_agent_start(agent_name, {
        "user_query": user_query,
        "stock_code": cache_code,
        "company_name": current_data.get("company_name"),
        "input_data_keys": list(current_data.keys()),
        "architecture": "two-phase",
    })

    if not user_query:
        logger.error(f"{ERROR_ICON} FundamentalAgent: User query is missing.")
        current_data["fundamental_analysis_error"] = "User query is missing."
        execution_logger.log_agent_complete(agent_name, current_data, 0, False, "User query is missing")
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    agent_start_time = time.time()

    try:
        # 模型配置：Kimi K2.6
        model_cfg = get_model_config_for_agent("fundamental_agent", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]

        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} FundamentalAgent: Missing OpenAI environment variables.")
            current_data["fundamental_analysis_error"] = "Missing OpenAI environment variables."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "Missing env vars")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        stock_code = current_data.get("stock_code", "Unknown")
        company_name = current_data.get("company_name", "Unknown")
        current_time_info = current_data.get("current_time_info", "未知时间")
        current_date = current_data.get("current_date", "未知日期")
        clean_code = _extract_code(stock_code) if stock_code else ""

        # ── Phase 1: 并行数据预取 ──────────────────────────────
        logger.info(f"{WAIT_ICON} FundamentalAgent: Phase 1 — 并行获取 {len(FUNDAMENTAL_TOOL_NAMES)} 个工具数据...")
        phase1_start = time.time()

        try:
            all_tools = await get_mcp_tools(tool_filter=FUNDAMENTAL_TOOL_NAMES)
        except Exception as e:
            logger.error(f"{ERROR_ICON} FundamentalAgent: 获取 MCP 工具失败: {e}")
            all_tools = []

        if not all_tools:
            logger.error(f"{ERROR_ICON} FundamentalAgent: No MCP tools available.")
            current_data["fundamental_analysis_error"] = "No MCP tools available."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "No tools")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        tool_map = {t.name: t for t in all_tools}
        logger.info(f"{SUCCESS_ICON} FundamentalAgent: 已加载 {len(all_tools)}/{len(FUNDAMENTAL_TOOL_NAMES)} 个工具")

        # 计算最近季度
        quarters = _get_recent_quarters(current_date, count=2)
        q_latest = quarters[0]
        q_prior = quarters[1] if len(quarters) > 1 else q_latest

        # 构建并行任务列表（覆盖白名单全部工具，不遗漏任何一个）
        tasks = []
        labels = []
        tool_infos = []  # (tool, kwargs) 用于空数据重试，占位任务为 None

        def _add(task, label, ti=None):
            tasks.append(task)
            labels.append(label)
            tool_infos.append(ti)

        # Helper: 注册占位任务（工具不可用）
        def _placeholder(label, msg):
            labels.append(label); tasks.append(_noop_result(msg)); tool_infos.append(None)

        # --- 基本信息与行业 ---
        if "tushare_stock_info" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_stock_info"], {"code": clean_code}, "Tushare基本信息"), "Tushare基本信息",
                 (tool_map["tushare_stock_info"], {"code": clean_code}))
        else: _placeholder("Tushare基本信息", "[tushare_stock_info] 工具不可用")


        # --- 三大报表（Tushare，无需 quarter 参数） ---
        for tname, label in [("tushare_income", "利润表"), ("tushare_balancesheet", "资产负债表"), ("tushare_cashflow", "现金流量表")]:
            if tname in tool_map:
                _add(_call_tool_safe(tool_map[tname], {"code": clean_code}, f"Tushare{label}"), f"Tushare{label}",
                     (tool_map[tname], {"code": clean_code}))
            else: _placeholder(f"Tushare{label}", f"[{tname}] 工具不可用")

        # --- 财务指标 ---
        if "tushare_fina_indicator" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fina_indicator"], {"code": clean_code, "years": 4}, "Tushare财务指标"), "Tushare财务指标",
                 (tool_map["tushare_fina_indicator"], {"code": clean_code, "years": 4}))
        else: _placeholder("Tushare财务指标", "[tushare_fina_indicator] 工具不可用")

        # --- 估值数据 ---
        if "tushare_daily_basic" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_daily_basic"], {"code": clean_code, "days": 250}, "Tushare日线基础"), "Tushare日线基础",
                 (tool_map["tushare_daily_basic"], {"code": clean_code, "days": 250}))
        else: _placeholder("Tushare日线基础", "[tushare_daily_basic] 工具不可用")

        if "tushare_ev_ebitda" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_ev_ebitda"], {"code": clean_code}, "Tushare EV/EBITDA"), "Tushare EV/EBITDA",
                 (tool_map["tushare_ev_ebitda"], {"code": clean_code}))
        else: _placeholder("Tushare EV/EBITDA", "[tushare_ev_ebitda] 工具不可用")

        # --- 分红 ---
        if "tushare_dividend" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_dividend"], {"code": clean_code}, "Tushare分红"), "Tushare分红",
                 (tool_map["tushare_dividend"], {"code": clean_code}))
        else: _placeholder("Tushare分红", "[tushare_dividend] 工具不可用")

        # --- 复权因子 ---
        if "tushare_adj_factor" in tool_map:
            kw_adj = {"code": clean_code, "days": 500}
            _add(_call_tool_safe(tool_map["tushare_adj_factor"], kw_adj, "Tushare复权因子"),
                 "Tushare复权因子", (tool_map["tushare_adj_factor"], kw_adj))
        else: _placeholder("Tushare复权因子", "[tushare_adj_factor] 工具不可用")

        # --- 股东与ST ---
        if "tushare_top10_holders" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_top10_holders"], {"code": clean_code}, "Tushare十大股东"), "Tushare十大股东",
                 (tool_map["tushare_top10_holders"], {"code": clean_code}))
        else: _placeholder("Tushare十大股东", "[tushare_top10_holders] 工具不可用")

        if "tushare_st_status" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_st_status"], {"code": clean_code}, "Tushare ST状态"), "Tushare ST状态",
                 (tool_map["tushare_st_status"], {"code": clean_code}))
        else: _placeholder("Tushare ST状态", "[tushare_st_status] 工具不可用")


        # 并行执行所有任务（第一轮）
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as gather_err:
            logger.error(f"{ERROR_ICON} FundamentalAgent: Phase 1 并行调用异常: {gather_err}")
            results = [f"并行调用异常: {gather_err}"] * len(tasks)

        # 空数据重试（最多额外3轮（并发递减8→4→2），覆盖率100%则提前跳出）
        results = await retry_failed_fetches(
            results, tool_infos, labels, _call_tool_safe,
            agent_label="FundamentalAgent",
        )

        # 处理异常结果
        safe_results = []
        for r in results:
            if isinstance(r, Exception):
                safe_results.append(f"[工具调用异常: {str(r)[:100]}]")
            else:
                safe_results.append(str(r) if r else "[空返回]")

        phase1_elapsed = time.time() - phase1_start
        success_count = sum(1 for r in safe_results if not is_empty_result(str(r)))
        total_real = len([ti for ti in tool_infos if ti is not None])
        logger.info(f"{SUCCESS_ICON} FundamentalAgent: Phase 1 完成 ({phase1_elapsed:.1f}s, {success_count}/{total_real} 个工具有效数据)")

        # 聚合数据
        data_sections = []
        for label, result in zip(labels, safe_results):
            data_sections.append(f"### [{label}]\n{result}")
        raw_data_text = "\n\n".join(data_sections)

        # ── Phase 2: LLM 深度分析 ──────────────────────────────
        logger.info(f"{WAIT_ICON} FundamentalAgent: Phase 2 — LLM 深度分析 (model={model_name}, thinking=enabled)...")
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

        analysis_prompt = f"""请以券商分析师的标准，对{company_name}（股票代码：{stock_code}）进行基本面分析。

当前时间：{current_time_info}
当前日期：{current_date}

## 原始数据

以下是通过工具获取的全部原始数据，请基于这些数据进行深度分析：

{raw_data_text}

## 分析要求

请进行以下基本面分析（每个维度都需要基于上述原始数据，引用具体数字）：

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

5. ST风险警示分析（⚠️ 必查项）
   - 检查上述原始数据中的 tushare_st_status 数据
   - 分析当前ST状态：是否为ST/*ST、进入风险警示板的具体日期
   - ST类型判断：退市风险警示（*ST）还是其他风险警示（ST）
   - 触发原因分析（结合财务数据）：是否连续两年净利润为负、净资产是否为负等
   - 退市风险等级评估
   - 如果ST相关数据均不可用，必须明确标注"ST数据不可用，无法完成ST风险评估"

6. 行业对比分析
   - 核心财务指标与同行业可比公司对比
   - 公司在行业中的相对优势和劣势

7. 综合评估
   - 基本面优势总结
   - 基本面风险提示

重要限制：
- 请专注于财务数据和基本面指标分析，不要分析新闻信息
- 分析必须有数据支撑，引用上述原始数据中的具体数字，避免空洞的定性描述
- 如果某些数据无法获取，请说明原因并基于可用数据提供分析
- 不要使用模型训练数据中的知识来补充数据事实

⛔ 输出格式要求（防幻觉机制）：
请将分析输出严格分为两个区域：

## 📊 数据事实区
列出上述原始数据中的关键客观数据，每条标注数据来源标签（如 [利润表(最新)]）：
- [标签] 具体数值（如：ROE=15.2%）
- [标签] 具体数值
- ...
如果某项数据在上述原始数据中标注为「数据不可用」，必须在此如实声明，不得推测。

## 🔍 分析判断区
基于上述数据事实进行分析和推断。每个判断必须：
1. 引用数据事实区的具体数值
2. 使用「【基于数据的推断】」或「【行业知识补充】」标注推断性质
3. 如果某个结论无法从数据中直接得出，必须声明「此为分析师推断」
4. 不得在任何地方编造数据事实区没有的数值
5. 不得编造原始数据中不存在的新闻或事件

⛔ 结构化输出要求：
在完成上述分析的「🔍 分析判断区」之后，请额外输出一个 JSON block：

<SIGNAL_PACK>
{{
    "bias": "bullish"|"neutral"|"bearish",
    "confidence": 0.0-1.0,
    "key_points": ["关键结论1", "关键结论2"] (最多6条,每条<80字),
    "signals": [
        {{
            "factor": "因子名(如:ROE持续性/现金流质量/负债率)",
            "direction": 1|-1|0,
            "strength": 0-100,
            "time_horizon": ["medium","long"],
            "source_level": "official_like"|"structured"|"derived",
            "risk_flags": [],
            "freshness": "quarterly",
            "note": "一句话说明"
        }}
    ] (最多6条),
    "risk_flags": ["cashflow_mismatch", "earnings_quality_concern"],
    "missing_data": ["未获取到质押数据"],
    "source_summary": "Tushare财报+AkShare杜邦/运营/成长数据"
}}
</SIGNAL_PACK>

请确保SIGNAL_PACK内的JSON完全有效。"""

        messages = [
            {"role": "system", "content": "你是一位资深券商基本面分析师，专注于A股公司的财务分析和基本面评估。"},
            {"role": "user", "content": analysis_prompt},
        ]

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=float(LLM_TIMEOUT)
            )
            final_output = response.content.strip() if hasattr(response, 'content') else str(response)
            # 提取 signal_pack
            fundamental_signal_pack = _extract_signal_pack_from_llm(final_output, "fundamental", current_date)
            current_data["fundamental_signal_pack"] = fundamental_signal_pack
            phase2_elapsed = time.time() - phase2_start
            logger.info(f"FundamentalAgent: Phase 2 完成 ({phase2_elapsed:.1f}s, {len(final_output)} 字符)")
            llm_success = True
        except asyncio.TimeoutError:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundamentalAgent: LLM 超时 ({phase2_elapsed:.0f}s)")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析判断区\n\n⚠️ LLM分析超时，以上为原始数据，请人工分析。\n"
            from src.utils.analysis_package_builder import text_to_signal_pack
            fundamental_signal_pack = text_to_signal_pack(final_output, "fundamental", current_date)
            current_data["fundamental_signal_pack"] = fundamental_signal_pack
            llm_success = False
        except Exception as llm_err:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundamentalAgent: LLM 失败: {llm_err}")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析判断区\n\n⚠️ LLM分析失败({str(llm_err)[:100]})，以上为原始数据，请人工分析。\n"
            from src.utils.analysis_package_builder import text_to_signal_pack
            fundamental_signal_pack = text_to_signal_pack(final_output, "fundamental", current_date)
            current_data["fundamental_signal_pack"] = fundamental_signal_pack
            llm_success = False

        # 记录 LLM 交互
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
        logger.info(f"{SUCCESS_ICON if llm_success else WAIT_ICON} FundamentalAgent: 分析完成 (总耗时 {total_time:.1f}s, Phase1={phase1_elapsed:.1f}s, Phase2={phase2_elapsed:.1f}s)")

        # 更新状态
        current_data["fundamental_analysis"] = final_output
        if not skip_cache and cache_date and cache_code:
            write_cache("fundamental_analysis", cache_code, cache_date, final_output)
            if "fundamental_signal_pack" in current_data:
                from src.utils.cache_utils import write_signal_pack_cache
                write_signal_pack_cache("fundamental_analysis", cache_code, cache_date, current_data["fundamental_signal_pack"])
        current_metadata["fundamental_agent_executed"] = True
        current_metadata["fundamental_agent_timestamp"] = str(time.time())
        current_metadata["fundamental_agent_execution_time"] = f"{total_time:.2f} seconds"

        execution_logger.log_agent_complete(agent_name, {
            "fundamental_analysis_length": len(final_output),
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
            "messages": current_messages + [{"role": "assistant", "content": "基本面分析已完成"}],
            "metadata": current_metadata,
        }

    except Exception as e:
        logger.error(f"{ERROR_ICON} FundamentalAgent: Error: {e}", exc_info=True)
        current_data["fundamental_analysis_error"] = f"Error: {e}"
        current_data["fundamental_analysis"] = f"基本面分析过程中出现错误: {str(e)}"
        current_metadata["fundamental_agent_error"] = str(e)
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}


# 本地测试函数
async def test_fundamental_agent():
    """基本面分析 Agent的测试函数"""
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

    result = await fundamental_agent(test_state)
    print("Fundamental Analysis Result:")
    print(result.get("data", {}).get("fundamental_analysis", "No analysis found"))
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fundamental_agent())
