"""
EventAnalyst Agent: 事件驱动分析
职责: 识别重大事件/催化剂、标记时效、输出事件方向(利多/利空/中性)
模型: M3 (Qwen3.7-Plus), thinking=enabled
"""
import asyncio
import json
import re
import time
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.utils.state_definition import AgentState
from src.tools.mcp_client import get_mcp_tools
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.utils.cache_utils import read_cache, write_cache
from src.utils.model_config import get_model_config_for_agent, get_thinking_body
from src.utils.fetch_utils import retry_failed_fetches, is_empty_result
from src.utils.analysis_package_builder import text_to_signal_pack

load_dotenv(override=True)
logger = setup_logger(__name__)

TOOL_TIMEOUT = 30
LLM_TIMEOUT = 120

EVENT_TOOL_NAMES = [
    "tushare_anns_d",
    "tushare_new_share",
    "tushare_pledge_stat",
    "tushare_repurchase",
    "tushare_share_float",
    "tushare_top10_holders",
    "tushare_stk_holdertrade",
    "tushare_dividend",
    "tushare_namechange",
    "tushare_suspend",
    "crawl_news",
    "tushare_st_status",
]


def _extract_code(stock_code: str) -> str:
    return stock_code.replace("sh.", "").replace("sz.", "").replace("bj.", "").replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()


async def _call_tool_safe(tool, kwargs: dict, label: str) -> str:
    """调用单个 MCP 工具，带超时、异常保护、和工具级缓存"""
    from src.utils.tool_cache import get_cached_tool_result, set_cached_tool_result
    tool_name = getattr(tool, 'name', 'unknown')
    cached = await get_cached_tool_result(tool_name, kwargs)
    if cached:
        logger.info(f"{SUCCESS_ICON} EventAnalyst: {label} [工具缓存命中]")
        return cached

    try:
        result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=TOOL_TIMEOUT)
        text = str(result).strip()
        if len(text) > 20:
            logger.info(f"{SUCCESS_ICON} EventAnalyst: {label} 获取成功 ({len(text)} 字符)")
            await set_cached_tool_result(tool_name, kwargs, text)
            return text
        return f"[{label}] 数据不可用（返回过短）"
    except asyncio.TimeoutError:
        return f"[{label}] 数据不可用（超时）"
    except Exception as e:
        return f"[{label}] 数据不可用: {str(e)[:80]}"


def _extract_signal_pack(llm_output: str, agent_name: str, as_of_date: str) -> Dict[str, Any]:
    tag_match = re.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', llm_output)
    if tag_match:
        try:
            sp = json.loads(tag_match.group(1))
            sp["agent_name"] = agent_name
            sp["as_of_date"] = as_of_date
            sp.setdefault("analysis_text", llm_output[:500])
            sp.setdefault("data_quality_score", 0.6)
            return sp
        except (json.JSONDecodeError, ValueError):
            pass
    return text_to_signal_pack(llm_output, agent_name, as_of_date)


async def event_analyst_agent(state: AgentState) -> AgentState:
    logger.info(f"{WAIT_ICON} EventAnalyst: 开始事件驱动分析")
    execution_logger = get_execution_logger()
    agent_name = "event_analyst"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})

    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    cache_code = current_data.get("stock_code", "")

    if not skip_cache and cache_date and cache_code:
        cached = read_cache("event_analysis", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} EventAnalyst: 命中缓存")
            current_data["event_analysis"] = cached
            current_metadata["event_agent_executed"] = True
            current_metadata["event_agent_cached"] = True
            from src.utils.cache_utils import read_signal_pack_cache
            cached_sp = read_signal_pack_cache("event_analysis", cache_code, cache_date)
            if cached_sp:
                current_data["event_signal_pack"] = cached_sp
            else:
                current_data["event_signal_pack"] = _extract_signal_pack(cached, "event", cache_date)
            return {"data": current_data, "messages": current_messages + [{"role": "assistant", "content": "事件分析已完成（缓存）"}], "metadata": current_metadata}

    stock_code = current_data.get("stock_code", "")
    company_name = current_data.get("company_name", "")
    current_time_info = current_data.get("current_time_info", "")
    current_date = current_data.get("current_date", "")
    clean_code = _extract_code(stock_code) if stock_code else ""

    agent_start_time = time.time()
    execution_logger.log_agent_start(agent_name, {"stock_code": stock_code, "company_name": company_name})

    try:
        model_cfg = get_model_config_for_agent("event_analyst", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]
        if not all([api_key, base_url, model_name]):
            raise ValueError("缺少环境变量")

        # Phase 1: 并行数据预取
        try:
            all_tools = await get_mcp_tools(tool_filter=EVENT_TOOL_NAMES)
        except Exception:
            all_tools = []

        tool_map = {t.name: t for t in all_tools} if all_tools else {}
        tasks = []
        labels = []
        tool_infos = []  # (tool, kwargs) for retry

        for tname in EVENT_TOOL_NAMES:
            if tname in tool_map:
                kwargs = {"code": clean_code}
                if tname == "tushare_anns_d":
                    kwargs["days"] = 90
                tasks.append(_call_tool_safe(tool_map[tname], kwargs, tname))
                labels.append(tname)
                tool_infos.append((tool_map[tname], kwargs))

        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []

        # 空数据重试（最多额外3轮，并发递减8→4→2）
        results = await retry_failed_fetches(
            results, tool_infos, labels, _call_tool_safe,
            agent_label="EventAnalyst",
        )

        safe_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                safe_results.append(f"[{labels[i]}] 调用异常")
            else:
                safe_results.append(str(r) if r else f"[{labels[i]}] 空返回")

        data_sections = [f"### [{label}]\n{result}" for label, result in zip(labels, safe_results)]
        raw_data_text = "\n\n".join(data_sections) if data_sections else "无可用数据源"

        # Phase 2: LLM 分析
        llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url,
                        temperature=1.0, request_timeout=LLM_TIMEOUT, max_tokens=12000,
                        extra_body=get_thinking_body(base_url, True))

        system_prompt = f"""你是一位A股事件驱动分析师。
当前时间: {current_time_info}

职责:
1. 识别重大事件/催化剂: 业绩预告/快报、回购、增减持、重大合同、并购重组、诉讼仲裁、处罚/问询、股权质押、异常停复牌
2. 标记事件时效: 事件日期、新近程度(freshness)、影响期限
3. 输出事件方向: 利多/利空/中性; 一次性/持续性; 是否已被市场交易过
4. source_level优先级: official_like(正式公告) > structured(数值工具) > news(媒体) > derived(推断)

⛔ 先输出「📊 数据事实区」「🔍 分析判断区」的自然语言分析。
末尾输出: <SIGNAL_PACK>{{JSON}}</SIGNAL_PACK>
JSON含: bias, confidence, key_points(≤5条), signals(≤8条,含factor/direction/strength/confidence/time_horizon/source_level/freshness/note), risk_flags, missing_data, source_summary
"""

        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请分析{company_name}({stock_code})的重大事件。\n\n## 原始数据\n{raw_data_text}"}
        ])
        final_output = response.content.strip() if hasattr(response, 'content') else str(response)

        signal_pack = _extract_signal_pack(final_output, "event", current_date)
        current_data["event_signal_pack"] = signal_pack
        current_data["event_analysis"] = final_output

        if not skip_cache and cache_date and cache_code:
            write_cache("event_analysis", cache_code, cache_date, final_output)
            if "event_signal_pack" in current_data:
                from src.utils.cache_utils import write_signal_pack_cache
                write_signal_pack_cache("event_analysis", cache_code, cache_date, current_data["event_signal_pack"])
        current_metadata["event_agent_executed"] = True

        return {"data": current_data, "messages": current_messages + [{"role": "assistant", "content": "事件分析已完成"}], "metadata": current_metadata}

    except Exception as e:
        logger.error(f"{ERROR_ICON} EventAnalyst 失败: {e}", exc_info=True)
        current_data["event_analysis"] = f"事件分析失败: {str(e)}"
        current_data["event_analysis_error"] = str(e)
        current_data["event_signal_pack"] = text_to_signal_pack(current_data.get("event_analysis", ""), "event", current_date)
        current_metadata["event_agent_error"] = str(e)
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}
