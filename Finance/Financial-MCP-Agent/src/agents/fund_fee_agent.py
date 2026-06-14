"""
FundFee Agent: 两阶段架构 — 并行数据预取 + 单次 LLM 费用/流动性/持有期分析。
Phase 1: asyncio.gather 并行获取白名单 3 个基金工具的数据
Phase 2: 将所有原始数据喂给 LLM 一次性完成费用持有期分析（thinking 关闭，max_tokens=8000 精简输出）
"""
import asyncio
import os
import time
from datetime import datetime
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

load_dotenv(override=True)

logger = setup_logger(__name__)

# 单个工具超时（秒）
TOOL_TIMEOUT = 30
# LLM 整体超时（秒）
LLM_TIMEOUT = 300

# 费用/流动性/持有期分析白名单
FUND_FEE_TOOLS = [
    "tushare_fund_basic",       # m_fee, c_fee, min_amount, purc/redm dates, 基金类型
    "tushare_fund_share",       # 份额规模变化趋势（流动性代理指标）
    "tushare_fund_nav",         # NAV 波动率（影响持有期判断）
]


def _extract_code(code: str) -> str:
    """提取纯数字基金代码（去掉 sh./sz./of./.SH/.SZ/.OF 等前缀后缀）"""
    return code.replace("sh.", "").replace("sz.", "").replace("of.", "").replace(
        ".SH", "").replace(".SZ", "").replace(".OF", "").strip()


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
            logger.info(f"{SUCCESS_ICON} FeeAgent: {label} 获取成功 ({len(text)} 字符)")
            await set_cached_tool_result(tool_name, kwargs, text)
            return text
        logger.warning(f"FeeAgent: {label} 返回过短 ({len(text)} 字符)")
        return f"[{label}] 数据不可用（返回过短）"
    except asyncio.TimeoutError:
        logger.warning(f"FeeAgent: {label} 超时({TOOL_TIMEOUT}s)")
        return f"[{label}] 数据不可用（超时）"
    except Exception as e:
        logger.warning(f"FeeAgent: {label} 调用失败: {e}")
        return f"[{label}] 数据不可用（调用失败: {str(e)[:80]}）"


async def fund_fee_agent(state: AgentState) -> AgentState:
    """
    两阶段费用/流动性/持有期分析：
    Phase 1: 并行获取 3 个基金工具的数据
    Phase 2: 单次 LLM 分析（Model 3: Qwen3.7-Plus, thinking=disabled）
    """
    logger.info(f"{WAIT_ICON} FeeAgent: Starting two-phase fee & holding period analysis.")

    execution_logger = get_execution_logger()
    agent_name = "fund_fee_agent"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})

    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    cache_code = current_data.get("fund_code", "") or current_data.get("stock_code", "")

    # 缓存检查
    if not skip_cache and cache_date and cache_code:
        cached = read_cache("fund_fee", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} FeeAgent: 命中缓存，跳过分析 ({cache_code})")
            current_data["fund_fee"] = cached
            from src.utils.cache_utils import read_signal_pack_cache
            cached_sp = read_signal_pack_cache("fund_fee", cache_code, cache_date)
            if cached_sp:
                current_data["fund_fee_signal_pack"] = cached_sp
            else:
                # Fallback: re-extract from cached LLM output text
                import json as _json, re as _re
                sp = None
                tag_match = _re.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', cached)
                if tag_match:
                    try:
                        sp = _json.loads(tag_match.group(1))
                        sp["agent_name"] = "fund_fee"
                        sp["as_of_date"] = cache_date
                    except Exception:
                        pass
                if sp is None:
                    from src.utils.analysis_package_builder import text_to_signal_pack
                    sp = text_to_signal_pack(cached, "fund_fee", cache_date)
                current_data["fund_fee_signal_pack"] = sp
            current_metadata["fund_fee_executed"] = True
            current_metadata["fund_fee_cached"] = True
            return {"data": current_data,
                    "messages": current_messages + [{"role": "assistant", "content": "费用与持有期分析已完成（缓存）"}],
                    "metadata": current_metadata}

    execution_logger.log_agent_start(agent_name, {
        "fund_code": cache_code,
        "fund_name": current_data.get("fund_name", ""),
        "input_data_keys": list(current_data.keys()),
        "architecture": "two-phase",
    })

    agent_start_time = time.time()

    try:
        # 模型配置：Model 3 (Qwen3.7-Plus)
        model_cfg = get_model_config_for_agent("fund_fee_agent", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]

        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} FeeAgent: Missing OpenAI environment variables.")
            current_data["fund_fee_error"] = "Missing OpenAI environment variables."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "Missing env vars")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        fund_code = current_data.get("fund_code", "") or current_data.get("stock_code", "Unknown")
        fund_name = current_data.get("fund_name", "") or current_data.get("company_name", "Unknown")
        current_time_info = current_data.get("current_time_info", "未知时间")
        current_date = current_data.get("current_date", "未知日期")
        clean_code = _extract_code(fund_code) if fund_code else ""

        # ── Phase 1: 并行数据预取 ──────────────────────────────
        logger.info(f"{WAIT_ICON} FeeAgent: Phase 1 — 并行获取 {len(FUND_FEE_TOOLS)} 个工具数据...")
        phase1_start = time.time()

        try:
            all_tools = await get_mcp_tools(tool_filter=FUND_FEE_TOOLS)
        except Exception as e:
            logger.error(f"{ERROR_ICON} FeeAgent: 获取 MCP 工具失败: {e}")
            all_tools = []

        if not all_tools:
            logger.error(f"{ERROR_ICON} FeeAgent: No MCP tools available.")
            current_data["fund_fee_error"] = "No MCP tools available."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "No tools")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        tool_map = {t.name: t for t in all_tools}
        logger.info(f"{SUCCESS_ICON} FeeAgent: 已加载 {len(all_tools)}/{len(FUND_FEE_TOOLS)} 个工具")

        tasks = []
        labels = []
        tool_infos = []  # (tool, kwargs) 用于空数据重试

        def _add(task, label, ti=None):
            tasks.append(task)
            labels.append(label)
            tool_infos.append(ti)

        def _placeholder(label, msg):
            labels.append(label); tasks.append(_noop_result(msg)); tool_infos.append(None)

        # --- 基金基本信息（费率、申购赎回、持有期限制） ---
        if "tushare_fund_basic" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_basic"], {"code": clean_code}, "基金基本信息"),
                 "基金基本信息", (tool_map["tushare_fund_basic"], {"code": clean_code}))
        else:
            _placeholder("基金基本信息", "[tushare_fund_basic] 工具不可用")

        # --- 基金份额变动（流动性代理指标） ---
        if "tushare_fund_share" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_share"], {"code": clean_code}, "基金份额变动"),
                 "基金份额变动", (tool_map["tushare_fund_share"], {"code": clean_code}))
        else:
            _placeholder("基金份额变动", "[tushare_fund_share] 工具不可用")

        # --- 基金净值（NAV波动率） ---
        if "tushare_fund_nav" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_nav"], {"code": clean_code}, "基金净值"),
                 "基金净值", (tool_map["tushare_fund_nav"], {"code": clean_code}))
        else:
            _placeholder("基金净值", "[tushare_fund_nav] 工具不可用")

        # 并行执行所有任务（第一轮）
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as gather_err:
            logger.error(f"{ERROR_ICON} FeeAgent: Phase 1 并行调用异常: {gather_err}")
            results = [f"并行调用异常: {gather_err}"] * len(tasks)

        # 空数据重试
        results = await retry_failed_fetches(
            results, tool_infos, labels, _call_tool_safe,
            agent_label="FeeAgent",
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
        logger.info(f"{SUCCESS_ICON} FeeAgent: Phase 1 完成 ({phase1_elapsed:.1f}s, {success_count}/{total_real} 个工具有效数据)")

        # 聚合数据
        data_sections = []
        for label, result in zip(labels, safe_results):
            data_sections.append(f"### [{label}]\n{result}")
        raw_data_text = "\n\n".join(data_sections)

        # ── Phase 2: LLM 分析 ──────────────────────────────
        logger.info(f"{WAIT_ICON} FeeAgent: Phase 2 — LLM 分析 (model={model_name}, thinking=disabled)...")
        phase2_start = time.time()

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=1.0,
            request_timeout=LLM_TIMEOUT,
            max_tokens=8000,
            extra_body=get_thinking_body(base_url, False),
        )

        analysis_prompt = f"""请以基金研究员的专业标准，对{fund_name}（基金代码：{fund_code}）进行费率结构、交易便利性与持有期分析。

当前时间：{current_time_info}
当前日期：{current_date}

## 原始数据

以下是通过工具获取的全部原始数据，请基于这些数据进行深度分析：

{raw_data_text}

## 分析要求

请完成以下七个维度的分析（每个维度都必须基于上述原始数据中的具体数字）：

### 1. 费率结构评价
- 管理费率（m_fee）：当前水平，与同类基金平均费率对比
- 托管费率（c_fee）：当前水平，与同类基金平均费率对比
- 综合费率（管理费+托管费）：年化总成本
- 是否有销售服务费（s_fee）：如存在，列明年费率
- 费率水平判定：低费率（<同类30分位）/ 中费率（30-70分位）/ 高费率（>70分位）

### 2. 申购赎回成本
- 申购费率：前端收费还是后端收费，具体费率水平
- 赎回费率：是否有阶梯式赎回费率（如持有<7天收1.5%惩罚性赎回费）
- 最低申购金额（min_amount）：对小额投资者是否友好
- 是否有申购/赎回限制期（如封闭期、定期开放）

### 3. 最短持有期与赎回限制
- 从 fund_basic 数据中的 duration_year、purc_startdate（可申购起始日）、redm_startdate（可赎回起始日）判断
- 是否设置了最短持有期（如持有满X天/月/年方可赎回）
- 是否有封闭期：封闭期内无法赎回
- 判断依据必须引用上述原始数据中的具体日期或年限数值

### 4. 交易便利性
- 基金类型：ETF/LOF（场内可交易）/ 普通开放式（仅场外申赎）/ 封闭式 / 定期开放
- ETF/LOF场内交易便利性：T+0还是T+1，流动性如何
- 场外申赎便利性：申购确认日(T+N)、赎回到账日(T+N)
- 基金规模（从fund_share数据判断）：规模是否过小（<5000万有清盘风险）

### 5. 流动性提示
- 基金份额变化趋势（从fund_share数据中提取）：近3/6/12个月份额变动方向
- 是否出现持续净赎回（份额持续缩减）
- 持续净赎回的含义：可能反映持有人信心不足，或市场风格切换导致资金流出
- 规模预警：如规模<5000万或连续多季度净赎回>20%

### 6. 持有期建议（五档判定）
请根据上述分析，从以下5个持有期建议中选择一个，并给出明确理由：

**五档持有期建议：**
| 建议 | 适用条件 |
|------|---------|
| 不建议持有 | 证据冲突严重、费率极高、流动性枯竭、或基金底层逻辑不匹配 |
| 试持有观察1-3个月 | 数据尚不充分、或费率偏高但产品有潜力、或处于建仓观察期 |
| 建议持有6-12个月 | 费率合理、流动性尚可、数据趋势中性偏正面 |
| 建议持有1年以上 | 费率较低、流动性好、无封闭期限制、趋势正面 |
| 适合长期配置3年以上 | 低费率（管理费<0.5%）、规模健康、流动性充裕、产品定位清晰且与投资者长期目标匹配 |

### 7. 持有期推理
结合以下因素综合推理持有期建议：
- 产品定位：宽基指数ETF vs 行业主题ETF vs 主动管理基金 vs 货币/债券基金
- 费率结构：长期持有的复利损耗（管理费1.5%持5年=累计侵蚀约7.2%收益），低费率产品的长期复利优势
- 锁定期约束：封闭期长短、赎回费率阶梯
- 净值波动特征（从fund_nav数据判断）：高波动适合短期交易，低波动适合长期配置
- **2026年证监会基金费率改革政策提示**：根据2026年CSRC基金费率新规，持有期超过1年的基金产品免征销售服务费，鼓励长期持有；对设置较长持有期的基金给予更灵活的资产配置权限

## ⛔ 输出格式要求（防幻觉机制）

请将分析输出严格分为两个区域：

## 📊 数据事实区
列出上述原始数据中的关键客观数据，每条标注数据来源标签：
- [基金基本信息] m_fee=XXX, c_fee=XXX, min_amount=XXX, 类型=XXX
- [基金基本信息] duration_year=XXX, purc_startdate=XXX, redm_startdate=XXX
- [基金份额变动] 最新日期份额=XXX, 3个月前份额=XXX, 变动率=±XX%
- [基金净值] NAV最新=XXX, 近期波动率=±XX%
- ...

如果某项数据在上述原始数据中标注为「数据不可用」，必须在此如实声明，不得推测。

## 🔍 分析判断区
基于上述数据事实进行分析和推断。必须严格按以下顺序输出：

### 费率结构评价
...
### 申购赎回成本
...
### 最短持有期与赎回限制
...
### 交易便利性
...
### 流动性提示
...
### 持有期建议
**建议：[五选一：不建议持有 / 试持有观察1-3个月 / 建议持有6-12个月 / 建议持有1年以上 / 适合长期配置3年以上]**
### 持有期推理
...

每个判断必须：
1. 引用数据事实区的具体数值
2. 使用「【基于数据的推断】」或「【行业知识补充】」标注推断性质
3. 如果某个结论无法从数据中直接得出，必须声明「此为分析师推断」
4. 不得在任何地方编造数据事实区没有的数值
5. 持有期建议必须是上述5个选项之一，不得自定义其他建议

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
            {"role": "system", "content": "你是一位资深基金研究员，专注于基金费用结构、流动性评估和持有期策略分析。你严格基于数据事实进行分析，不编造任何数据。"},
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
                    sp["agent_name"] = "fund_fee"
                    sp["as_of_date"] = current_date
                except Exception:
                    pass
            if sp is None:
                from src.utils.analysis_package_builder import text_to_signal_pack
                sp = text_to_signal_pack(final_output, "fund_fee", current_date)
            current_data["fund_fee_signal_pack"] = sp
            phase2_elapsed = time.time() - phase2_start
            logger.info(f"FeeAgent: Phase 2 完成 ({phase2_elapsed:.1f}s, {len(final_output)} 字符)")
            llm_success = True
        except asyncio.TimeoutError:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FeeAgent: LLM 超时 ({phase2_elapsed:.0f}s)")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析判断区\n\n⚠️ LLM分析超时，以上为原始数据，请人工分析。\n"
            llm_success = False
        except Exception as llm_err:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FeeAgent: LLM 失败: {llm_err}")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析判断区\n\n⚠️ LLM分析失败({str(llm_err)[:100]})，以上为原始数据，请人工分析。\n"
            llm_success = False

        # 记录 LLM 交互
        model_config_log = {
            "model": model_name,
            "temperature": 1.0,
            "max_tokens": 8000,
            "thinking": "disabled",
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
        logger.info(f"{SUCCESS_ICON if llm_success else WAIT_ICON} FeeAgent: 分析完成 (总耗时 {total_time:.1f}s, Phase1={phase1_elapsed:.1f}s, Phase2={phase2_elapsed:.1f}s)")

        # 更新状态
        current_data["fund_fee"] = final_output
        if not skip_cache and cache_date and cache_code:
            write_cache("fund_fee", cache_code, cache_date, final_output)
            if "fund_fee_signal_pack" in current_data:
                from src.utils.cache_utils import write_signal_pack_cache
                write_signal_pack_cache("fund_fee", cache_code, cache_date, current_data["fund_fee_signal_pack"])
        current_metadata["fund_fee_executed"] = True
        current_metadata["fund_fee_timestamp"] = str(time.time())
        current_metadata["fund_fee_execution_time"] = f"{total_time:.2f} seconds"

        execution_logger.log_agent_complete(agent_name, {
            "fund_fee_length": len(final_output),
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
            "messages": current_messages + [{"role": "assistant", "content": "费用与持有期分析已完成"}],
            "metadata": current_metadata,
        }

    except Exception as e:
        logger.error(f"{ERROR_ICON} FeeAgent: Error: {e}", exc_info=True)
        current_data["fund_fee_error"] = f"Error: {e}"
        current_data["fund_fee"] = f"费用与持有期分析过程中出现错误: {str(e)}"
        current_metadata["fund_fee_agent_error"] = str(e)
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}


# 本地测试函数
async def test_fund_fee_agent():
    """费用/持有期分析 Agent 的测试函数"""
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
            "query": "分析黄金ETF华安的费用和持有期",
            "fund_code": "sh.518880",
            "fund_name": "黄金ETF华安",
            "company_name": "黄金ETF华安",
            "stock_code": "sh.518880",
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat()
        },
        metadata={}
    )

    result = await fund_fee_agent(test_state)
    print("Fund Fee & Holding Period Analysis Result:")
    print(result.get("data", {}).get("fund_fee", "No analysis found"))
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fund_fee_agent())
