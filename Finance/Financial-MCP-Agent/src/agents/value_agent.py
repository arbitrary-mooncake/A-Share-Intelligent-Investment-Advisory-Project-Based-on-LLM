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

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from src.utils.state_definition import AgentState
from src.tools.mcp_client import get_mcp_tools
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.utils.cache_utils import read_cache, write_cache
from src.utils.model_config import get_model_config_for_agent
from src.utils.fetch_utils import retry_failed_fetches, is_empty_result

load_dotenv(override=True)

logger = setup_logger(__name__)

TOOL_TIMEOUT = 30
LLM_TIMEOUT = 300

# 估值分析白名单（与旧 ReAct 白名单完全一致，16 个工具）
VALUE_TOOL_NAMES = [
    "get_stock_basic_info", "get_stock_industry",
    "get_profit_data", "get_balance_data", "get_cash_flow_data",
    "get_growth_data", "get_operation_data", "get_dividend_data",
    "tushare_stock_info", "tushare_fina_indicator",
    "tushare_dividend", "tushare_ev_ebitda",
    "tushare_daily_basic", "tushare_pe_percentile",
    "tushare_top10_holders", "tushare_holder_num",
]


def _extract_code(stock_code: str) -> str:
    return stock_code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").strip()


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
        logger.info(f"{SUCCESS_ICON} ValueAgent: ETF标的，跳过估值分析")
        current_data["value_analysis"] = "该标的为ETF产品，估值分析参考基金净值和折溢价率，不适用个股估值框架。"
        current_metadata["value_agent_executed"] = True
        return {"data": current_data,
                "messages": current_messages + [{"role": "assistant", "content": "ETF标的，已跳过估值分析"}],
                "metadata": current_metadata}

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

        # --- 纯代码类工具 ---
        code_tools = [
            ("get_stock_basic_info", "基本信息"),
            ("get_stock_industry", "行业分类"),
            ("tushare_stock_info", "Tushare基本信息"),
            ("tushare_fina_indicator", "Tushare财务指标"),
            ("tushare_daily_basic", "Tushare日线基础"),
            ("tushare_pe_percentile", "Tushare PE分位"),
            ("tushare_top10_holders", "Tushare十大股东"),
            ("tushare_holder_num", "Tushare股东人数"),
            ("tushare_ev_ebitda", "Tushare EV/EBITDA"),
            ("tushare_dividend", "Tushare分红"),
        ]
        for tname, label in code_tools:
            if tname in tool_map:
                kwargs = {"code": clean_code}
                if tname == "get_stock_industry":
                    kwargs["date"] = current_date
                _add(_call_tool_safe(tool_map[tname], kwargs, label), label, (tool_map[tname], kwargs))
            else:
                _placeholder(label, f"[{tname}] 工具不可用")

        # --- 财务报表类工具 ---
        fin_tools_params = [
            ("get_profit_data", "利润表"),
            ("get_balance_data", "资产负债表"),
            ("get_cash_flow_data", "现金流量表"),
            ("get_growth_data", "成长数据"),
            ("get_operation_data", "运营数据"),
        ]
        for tool_name, label_base in fin_tools_params:
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

        # --- 分红数据 ---
        if "get_dividend_data" in tool_map:
            kw_div = {"code": clean_code, "year": q_latest["year"], "year_type": "report"}
            _add(_call_tool_safe(tool_map["get_dividend_data"], kw_div, f"分红数据({q_latest['year']})"),
                 "分红数据", (tool_map["get_dividend_data"], kw_div))
        else:
            _placeholder("分红数据", "[get_dividend_data] 工具不可用")

        # 并行执行（第一轮）
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as gather_err:
            logger.error(f"{ERROR_ICON} ValueAgent: Phase 1 并行调用异常: {gather_err}")
            results = [f"并行调用异常: {gather_err}"] * len(tasks)

        # 空数据重试（最多额外2轮，覆盖率100%则提前跳出）
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
            temperature=0.6,
            request_timeout=LLM_TIMEOUT,
            max_tokens=8000,
            extra_body={"thinking": {"type": "enabled"}},
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
            "temperature": 0.6,
            "max_tokens": 8000,
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
