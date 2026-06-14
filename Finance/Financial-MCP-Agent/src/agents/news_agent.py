"""
NewsAnalysis Agent: 新闻分析 Agent — 并行多源采集 + 深度分析

架构（≤140s预算）：
  Phase 1 (≤70s): asyncio.gather 并行调用 2 路新闻源
    - crawl_news(clean_stock_code) → 主要数据源（akshare东方财富）
    - crawl_news(company_name)    → 名称备选查询
  Phase 2 (<1s): 数据聚合
  Phase 3 (≤60s): LLM 深度分析（Qwen3.7-Plus, thinking=enabled, 直接调用 openai）

模型: Qwen3.7-Plus (M3) — 2026-06 从 Kimi K2.6 迁移。开启 thinking 提升情感分析深度，关闭温度降低幻觉。
"""
import asyncio
import os
from typing import Dict, Any
from openai import AsyncOpenAI
import httpx
import time

from src.utils.state_definition import AgentState
from src.tools.mcp_client import get_mcp_tools
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.utils.cache_utils import read_cache, write_cache
from src.utils.model_config import get_model_config_for_agent, get_thinking_body
from src.utils.fetch_utils import retry_failed_fetches, is_empty_result
from dotenv import load_dotenv

load_dotenv(override=True)

logger = setup_logger(__name__)

# 工具超时（秒）—— 并行调用，总工具阶段 ≤ TOOL_TIMEOUT
TOOL_TIMEOUT = 30
# LLM 阶段超时预算（Qwen3.7-Plus + thinking 通常在 30-50s，90s 留足余量）
LLM_TIMEOUT = 90


def _extract_code(stock_code: str) -> str:
    """提取纯数字股票代码，去除交易所前缀"""
    return stock_code.replace("sh.", "").replace("sz.", "").replace("bj.", "").replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()


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


def _deduplicate_news(news_items: list) -> list:
    """按标题去重，保留首次出现的条目"""
    seen = set()
    result = []
    for item in news_items:
        title = item.get("title", "")
        key = title.strip()[:60]
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


async def _call_tool_with_timeout(tool, kwargs: dict, timeout: float, label: str) -> str:
    """调用单个MCP工具，带超时保护"""
    try:
        result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=timeout)
        text = str(result).strip()
        if len(text) > 25:
            logger.info(f"{SUCCESS_ICON} NewsAgent: {label} 获取成功 ({len(text)} 字符)")
            return text
        else:
            logger.warning(f"NewsAgent: {label} 返回过短 ({len(text)} 字符)")
            return ""
    except asyncio.TimeoutError:
        logger.warning(f"NewsAgent: {label} 超时({timeout}s)，跳过")
        return ""
    except Exception as e:
        logger.warning(f"NewsAgent: {label} 调用失败: {e}")
        return ""


def _build_fallback_output(news_text: str, news_parts: list, company_name: str,
                           stock_code: str, current_date: str) -> str:
    """当LLM调用失败时，用原始新闻数据构建降级输出（仅陈述数据，不做分析判断）"""
    if not news_text:
        return (
            f"## 📊 数据事实区\n"
            f"新闻数据有限。经过多路新闻源并行查询，当前时段均未获取到与{company_name}相关的新闻数据。\n\n"
            f"## 🔍 分析判断区\n"
            f"由于新闻数据不可用，无法进行分析。请注意：本声明基于实际工具查询结果，非推断。\n"
        )
    return (
        f"## 📊 数据事实区\n"
        f"以下是通过东方财富新闻接口获取的 {company_name}（{stock_code}）原始新闻数据"
        f"（采集日期：{current_date}）。因LLM分析服务暂时不可用，仅列出数据事实，不做分析判断。\n\n"
        f"{news_text}\n\n"
        f"## 🔍 分析判断区\n"
        f"⚠️ LLM分析服务暂时不可用，以上为原始新闻数据，未经过AI分析处理。"
        f"请人工查阅上述新闻进行判断。\n"
    )


async def news_agent(state: AgentState) -> AgentState:
    """
    新闻分析 Agent：并行多源采集 + LLM深度分析

    严格防幻觉：LLM 仅分析工具返回的真实新闻数据，
    无数据时如实声明，不使用训练数据编造信息。
    """
    logger.info(f"{WAIT_ICON} NewsAgent: Starting parallel multi-source news analysis.")

    execution_logger = get_execution_logger()
    agent_name = "news_agent"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})

    # 缓存检查
    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    cache_code = current_data.get("stock_code", "")
    if not skip_cache and cache_date and cache_code:
        cached = read_cache("news_analysis", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} NewsAgent: 命中缓存，跳过分析 ({cache_code})")
            current_data["news_analysis"] = cached
            current_metadata["news_agent_executed"] = True
            current_metadata["news_agent_cached"] = True
            from src.utils.cache_utils import read_signal_pack_cache
            cached_sp = read_signal_pack_cache("news_analysis", cache_code, cache_date)
            if cached_sp:
                current_data["news_signal_pack"] = cached_sp
            else:
                from src.utils.analysis_package_builder import text_to_signal_pack
                import re, json
                # Try to re-extract SIGNAL_PACK from cached LLM output
                sp = None
                tag_match = re.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', cached)
                if tag_match:
                    try:
                        sp = json.loads(tag_match.group(1))
                        sp["agent_name"] = "news"
                        sp["as_of_date"] = cache_date
                    except Exception:
                        pass
                if sp is None:
                    sp = text_to_signal_pack(cached, "news", cache_date)
                current_data["news_signal_pack"] = sp
            return {"data": current_data,
                    "messages": current_messages + [{"role": "assistant", "content": "新闻分析已完成（缓存）"}],
                    "metadata": current_metadata}

    execution_logger.log_agent_start(agent_name, {
        "stock_code": current_data.get("stock_code"),
        "company_name": current_data.get("company_name"),
        "input_data_keys": list(current_data.keys())
    })

    agent_start_time = time.time()

    # 提取基本信息
    stock_code = current_data.get("stock_code", "")
    company_name = current_data.get("company_name", "")
    current_time_info = current_data.get("current_time_info", "")
    current_date = current_data.get("current_date", "")

    if not company_name:
        logger.error(f"{ERROR_ICON} NewsAgent: 缺少公司名称")
        current_data["news_analysis_error"] = "缺少公司名称"
        current_data["news_analysis"] = "新闻分析失败：缺少公司名称。"
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    clean_code = _extract_code(stock_code) if stock_code else ""

    # ── Phase 1: 并行采集 ──────────────────────────────────
    logger.info(f"{WAIT_ICON} NewsAgent: Phase 1 — 并行获取新闻工具...")
    phase1_start = time.time()

    try:
        news_tools = await get_mcp_tools(tool_filter=["crawl_news", "get_st_risk_data", "tushare_st_status"])
    except Exception as e:
        logger.error(f"{ERROR_ICON} NewsAgent: 获取MCP工具失败: {e}")
        news_tools = []

    if not news_tools:
        logger.error(f"{ERROR_ICON} NewsAgent: 无可用新闻工具")
        current_data["news_analysis"] = "新闻分析失败：新闻工具不可用。"
        current_metadata["news_agent_error"] = "No news tools available"
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    # 构建并行任务
    tasks = []
    task_labels = []
    news_tool_infos = []  # (tool, kwargs) 用于空数据重试

    for tool in news_tools:
        if tool.name == "crawl_news":
            # 主查询：股票代码
            if clean_code:
                kw1 = {"query": clean_code, "top_k": 10}
                tasks.append(_call_tool_with_timeout(tool, kw1, TOOL_TIMEOUT, f"crawl_news(code={clean_code})"))
                task_labels.append(f"crawl_news(code)")
                news_tool_infos.append((tool, kw1))
            # 备选查询：公司名称
            kw2 = {"query": company_name, "top_k": 10}
            tasks.append(_call_tool_with_timeout(tool, kw2, TOOL_TIMEOUT, f"crawl_news(name={company_name})"))
            task_labels.append(f"crawl_news(name)")
            news_tool_infos.append((tool, kw2))

        elif tool.name == "get_st_risk_data":
            if clean_code:
                _is_bse = (clean_code.startswith(("430", "431", "920")) or
                           (len(clean_code) >= 3 and clean_code[:3] in
                            ("830", "831", "832", "833", "834", "835", "836", "837", "838", "839",
                             "870", "871", "872", "873")))
                if _is_bse:
                    sh_code = f"bj.{clean_code}"
                elif clean_code.startswith(("6", "688", "5")):
                    sh_code = f"sh.{clean_code}"
                else:
                    sh_code = f"sz.{clean_code}"
                kw = {"code": sh_code}
                tasks.append(_call_tool_with_timeout(tool, kw, 5.0, f"st_risk(code={sh_code})"))
                task_labels.append(f"st_risk")
                news_tool_infos.append((tool, kw))

    # 并行执行（第一轮）
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
    except Exception as gather_err:
        logger.error(f"{ERROR_ICON} NewsAgent: 并行工具调用异常: {gather_err}")
        results = []

    # 空数据重试包装（news_agent 使用 _call_tool_with_timeout）
    async def _news_retry_call(tool, kwargs, label):
        return await _call_tool_with_timeout(tool, kwargs, TOOL_TIMEOUT, label)

    results = await retry_failed_fetches(
        results, news_tool_infos, task_labels, _news_retry_call,
        agent_label="NewsAgent",
    )

    phase1_elapsed = time.time() - phase1_start

    # 聚合结果
    news_parts = []
    success_count = 0
    for label, text in zip(task_labels, results):
        if text and not isinstance(text, Exception):
            text_str = str(text)
            # 过滤空结果和无效返回
            if is_empty_result(text_str):
                logger.info(f"NewsAgent: {label} 返回空结果")
                continue
            # 额外过滤"未找到"等短回复
            if "未找到" in text_str[:50] and len(text_str) < 100:
                logger.info(f"NewsAgent: {label} 未找到新闻")
                continue
            news_parts.append(f"### {label}\n{text_str}")
            success_count += 1
        else:
            logger.info(f"NewsAgent: {label} 返回空结果")

    news_text = "\n\n".join(news_parts) if news_parts else ""
    logger.info(
        f"{SUCCESS_ICON if success_count else WAIT_ICON} NewsAgent: Phase 1 完成 "
        f"({phase1_elapsed:.1f}s, {success_count}/{len(task_labels)} 路有数据)"
    )

    # ── Phase 2: LLM 深度分析 ──────────────────────────────
    logger.info(f"{WAIT_ICON} NewsAgent: Phase 2 — LLM 深度分析...")

    # 模型配置
    model_cfg = get_model_config_for_agent("news_agent", current_data)
    api_key = model_cfg["api_key"]
    base_url = model_cfg["base_url"]
    model_name = model_cfg["model_name"]

    if not all([api_key, base_url, model_name]):
        logger.error(f"{ERROR_ICON} NewsAgent: Missing OpenAI environment variables.")
        current_data["news_analysis_error"] = "Missing OpenAI environment variables."
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    # 直接使用 openai 客户端（绕开 langchain），确保 httpx 超时可控
    _client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(connect=15.0, read=float(LLM_TIMEOUT), write=30.0, pool=10.0),
        max_retries=1,
    )
    _messages = [
        {"role": "system", "content": (
            "你是一位资深的A股新闻舆情分析师。\n\n"
            "你的职责范围（只做这些）：\n"
            "1. 媒体新闻情绪判断\n"
            "2. 行业/政策舆情分析\n"
            "3. 题材热度与叙事强度评估\n"
            "4. 市场关注点是否集中、是否形成一致预期\n\n"
            "你不再负责（这些交给event_analyst）：\n"
            "- 公司正式公告的事实判断\n"
            "- 监管事件是否成立\n"
            "- 重大事项的事实核实\n\n"
            "换言之：你分析的是'市场如何看、如何传'，而不是'真实发生了什么'。\n\n"
            "⛔ 输出格式：先输出「📊 数据事实区」「🔍 分析判断区」，"
            "然后在末尾输出: <SIGNAL_PACK>{JSON}</SIGNAL_PACK>\n"
            "其中JSON包含: bias, confidence, key_points(≤5条), signals(≤5条, source_level=\"news\"), "
            "risk_flags, source_summary\n"
        )}
    ]
    if news_text:
        _messages.append({"role": "user", "content": (
            f"请对{company_name}（股票代码：{stock_code}）进行新闻分析。\n\n"
            f"当前时间：{current_time_info}\n"
            f"当前日期：{current_date}\n\n"
            f"## 新闻原始数据\n{news_text}\n\n"
            f"请基于以上新闻数据逐条分析，输出格式请严格遵循系统指令中的要求。"
        )})
    else:
        _messages.append({"role": "user", "content": (
            f"请对{company_name}（股票代码：{stock_code}）进行新闻分析。\n\n"
            f"当前时间：{current_time_info}\n"
            f"当前日期：{current_date}\n\n"
            f"## 新闻原始数据\n"
            f"经过多路新闻源并行查询，当前时段均未获取到与{company_name}相关的新闻数据。\n\n"
            f"请如实声明「新闻数据有限」，不得编造任何新闻内容。"
        )})

    logger.info(f"{WAIT_ICON} NewsAgent: Calling LLM (model={model_name}, {len(news_text)} 字符输入)...")
    phase2_start = time.time()

    try:
        response = await asyncio.wait_for(
            _client.chat.completions.create(
                model=model_name,
                messages=_messages,
                temperature=0.6,
                max_tokens=16000,
                extra_body=get_thinking_body(base_url, True),
            ),
            timeout=float(LLM_TIMEOUT + 10)  # Qwen3.7-Plus thinking 通常在 20-30s，60s 余量充足
        )
        phase2_elapsed = time.time() - phase2_start
        final_output = response.choices[0].message.content if response.choices else "No analysis generated."
        # 提取 signal_pack
        news_signal_pack = _extract_signal_pack_from_llm(final_output, "news", current_date)
        current_data["news_signal_pack"] = news_signal_pack
        logger.info(f"NewsAgent: Phase 2 完成 ({phase2_elapsed:.1f}s, {len(final_output)} 字符)")
        llm_success = True

    except asyncio.TimeoutError:
        phase2_elapsed = time.time() - phase2_start
        logger.error(f"{ERROR_ICON} NewsAgent: LLM 超时 ({phase2_elapsed:.0f}s)，降级返回原始数据")
        final_output = _build_fallback_output(news_text, news_parts, company_name, stock_code, current_date)
        from src.utils.analysis_package_builder import text_to_signal_pack
        news_signal_pack = text_to_signal_pack(final_output, "news", current_date)
        current_data["news_signal_pack"] = news_signal_pack
        llm_success = False

    except Exception as llm_err:
        phase2_elapsed = time.time() - phase2_start
        err_msg = str(llm_err) or type(llm_err).__name__
        logger.error(f"{ERROR_ICON} NewsAgent: LLM 失败 ({err_msg})，降级返回原始数据")
        final_output = _build_fallback_output(news_text, news_parts, company_name, stock_code, current_date)
        from src.utils.analysis_package_builder import text_to_signal_pack
        news_signal_pack = text_to_signal_pack(final_output, "news", current_date)
        current_data["news_signal_pack"] = news_signal_pack
        llm_success = False

    # 记录LLM交互
    model_config_log = {
        "model": model_name,
        "temperature": 0.6,
        "max_tokens": 16000,
        "thinking": "enabled",
        "api_base": base_url
    }
    try:
        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="direct_llm_parallel_sources",
            input_messages=_messages,
            output_content=final_output,
            model_config=model_config_log,
            execution_time=phase2_elapsed
        )
    except Exception:
        pass  # 日志记录失败不影响主流程

    logger.info(f"{SUCCESS_ICON if llm_success else WAIT_ICON} NewsAgent: 分析完成 (总耗时 {time.time() - agent_start_time:.1f}s)")

    # 更新状态
    current_data["news_analysis"] = final_output
    if not skip_cache and cache_date and cache_code:
        write_cache("news_analysis", cache_code, cache_date, final_output)
        if "news_signal_pack" in current_data:
            from src.utils.cache_utils import write_signal_pack_cache
            write_signal_pack_cache("news_analysis", cache_code, cache_date, current_data["news_signal_pack"])
    current_metadata["news_agent_executed"] = True
    current_metadata["news_agent_timestamp"] = str(time.time())

    total_time = time.time() - agent_start_time
    current_metadata["news_agent_execution_time"] = f"{total_time:.2f} seconds"

    execution_logger.log_agent_complete(agent_name, {
        "news_analysis_length": len(final_output),
        "analysis_preview": final_output[:500],
        "sources_queried": len(task_labels),
        "sources_with_data": success_count,
        "phase1_time": phase1_elapsed,
        "phase2_time": phase2_elapsed,
        "llm_success": llm_success,
        "total_time": total_time,
    }, total_time, True)

    return {
        "data": current_data,
        "messages": current_messages + [{"role": "assistant", "content": "新闻分析已完成"}],
        "metadata": current_metadata
    }
