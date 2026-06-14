"""
QualityRiskAnalyst Agent: 财务质量与治理风险分析
职责: 利润现金含量、应收/存货/商誉/减值风险、质押/减持/监管/退市风险
模型: M4 (Kimi K2.6), thinking=enabled
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
from src.utils.analysis_package_builder import text_to_signal_pack

load_dotenv(override=True)
logger = setup_logger(__name__)

TOOL_TIMEOUT = 30
LLM_TIMEOUT = 300

QUALITY_RISK_TOOL_NAMES = [
    "tushare_income",
    "tushare_balancesheet",
    "tushare_cashflow",
    "tushare_fina_indicator",
    "tushare_pledge_stat",
    "tushare_top10_holders",
    "tushare_stk_holdertrade",
    "tushare_st_status",
    "get_st_risk_data",
    "get_dupont_data",
]


def _extract_code(stock_code: str) -> str:
    return stock_code.replace("sh.", "").replace("sz.", "").replace("bj.", "").replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()


async def _call_tool_safe(tool, kwargs: dict, label: str) -> str:
    try:
        result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=TOOL_TIMEOUT)
        text = str(result).strip()
        if len(text) > 20:
            logger.info(f"{SUCCESS_ICON} QualityRiskAnalyst: {label} 获取成功 ({len(text)} 字符)")
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


async def quality_risk_analyst_agent(state: AgentState) -> AgentState:
    logger.info(f"{WAIT_ICON} QualityRiskAnalyst: 开始财务质量与治理风险分析")
    execution_logger = get_execution_logger()
    agent_name = "quality_risk_analyst"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})

    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    cache_code = current_data.get("stock_code", "")

    if not skip_cache and cache_date and cache_code:
        cached = read_cache("quality_risk_analysis", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} QualityRiskAnalyst: 命中缓存")
            current_data["quality_risk_analysis"] = cached
            current_metadata["quality_risk_agent_executed"] = True
            current_metadata["quality_risk_agent_cached"] = True
            current_data["quality_risk_signal_pack"] = _extract_signal_pack(cached, "quality_risk", cache_date)
            return {"data": current_data, "messages": current_messages + [{"role": "assistant", "content": "质量风险分析已完成（缓存）"}], "metadata": current_metadata}

    stock_code = current_data.get("stock_code", "")
    company_name = current_data.get("company_name", "")
    current_time_info = current_data.get("current_time_info", "")
    current_date = current_data.get("current_date", "")
    clean_code = _extract_code(stock_code) if stock_code else ""

    agent_start_time = time.time()
    execution_logger.log_agent_start(agent_name, {"stock_code": stock_code, "company_name": company_name})

    try:
        model_cfg = get_model_config_for_agent("quality_risk_analyst", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]
        if not all([api_key, base_url, model_name]):
            raise ValueError("缺少环境变量")

        # Phase 1: 并行数据预取
        try:
            all_tools = await get_mcp_tools(tool_filter=QUALITY_RISK_TOOL_NAMES)
        except Exception:
            all_tools = []

        tool_map = {t.name: t for t in all_tools} if all_tools else {}
        tasks = []
        labels = []

        for tname in QUALITY_RISK_TOOL_NAMES:
            if tname in tool_map:
                kwargs = {"code": clean_code}
                tasks.append(_call_tool_safe(tool_map[tname], kwargs, tname))
                labels.append(tname)

        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
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
                        temperature=1.0, request_timeout=LLM_TIMEOUT, max_tokens=16000,
                        extra_body=get_thinking_body(base_url, True))

        system_prompt = f"""你是一位A股财务质量与治理风险分析师。
当前时间: {current_time_info}

职责:
1. 财务质量: 利润现金含量(经营现金流/净利润)、应收/存货/商誉/减值风险、非经常性损益依赖
2. 治理与股东风险: 质押比例、冻结、控制权变化、关联交易、大股东减持
3. 风险标签输出: cashflow_mismatch, high_pledge_risk, regulatory_risk, audit_risk, impairment_risk, earnings_quality_concern, goodwill_risk, debt_risk, delist_risk
4. 不能确认时显式写「未获取到」

⛔ 先输出「📊 数据事实区」「🔍 分析判断区」的自然语言分析。
末尾输出: <SIGNAL_PACK>{{JSON}}</SIGNAL_PACK>
JSON含: bias, confidence, key_points(≤5条), signals(≤8条,含factor/direction/strength/confidence/time_horizon/source_level/freshness/note), risk_flags(优先输出上述风险标签), missing_data, source_summary
"""

        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请分析{company_name}({stock_code})的财务质量与治理风险。\n\n## 原始数据\n{raw_data_text}"}
        ])
        final_output = response.content.strip() if hasattr(response, 'content') else str(response)

        signal_pack = _extract_signal_pack(final_output, "quality_risk", current_date)
        current_data["quality_risk_signal_pack"] = signal_pack
        current_data["quality_risk_analysis"] = final_output

        if not skip_cache and cache_date and cache_code:
            write_cache("quality_risk_analysis", cache_code, cache_date, final_output)
        current_metadata["quality_risk_agent_executed"] = True

        return {"data": current_data, "messages": current_messages + [{"role": "assistant", "content": "质量风险分析已完成"}], "metadata": current_metadata}

    except Exception as e:
        logger.error(f"{ERROR_ICON} QualityRiskAnalyst 失败: {e}", exc_info=True)
        current_data["quality_risk_analysis"] = f"质量风险分析失败: {str(e)}"
        current_data["quality_risk_signal_pack"] = text_to_signal_pack(current_data.get("quality_risk_analysis", ""), "quality_risk", current_date)
        current_metadata["quality_risk_agent_error"] = str(e)
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}
