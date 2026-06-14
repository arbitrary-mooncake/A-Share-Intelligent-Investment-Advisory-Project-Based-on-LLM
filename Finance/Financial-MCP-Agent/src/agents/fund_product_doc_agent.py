"""
基金产品文件解析 Agent (Agent 1/7): 两阶段架构 — 并行数据预取 + 单次 LLM 分析。

Phase 1: asyncio.gather 并行获取白名单全部 3 个基金数据工具
Phase 2: 将所有原始数据喂给 LLM 一次性完成基金产品文档解析（thinking 关闭）
"""
import asyncio
import os
import time
from typing import Dict, Any, List

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

# 基金产品文档解析白名单
FUND_PRODUCT_DOC_TOOLS = [
    "tushare_fund_basic",
    "tushare_fund_company",
    "tushare_fund_search",
]


def _extract_fund_code(fund_code: str) -> str:
    """提取纯数字基金代码"""
    return fund_code.replace("sh.", "").replace("sz.", "").replace("of.", "").replace(".SH", "").replace(".SZ", "").replace(".OF", "").strip()


async def _noop_result(text: str) -> str:
    """返回固定文本的占位协程，用于工具不可用时的占位"""
    return text


async def _call_tool_safe(tool, kwargs: dict, label: str) -> str:
    """调用单个 MCP 工具，带超时和异常保护"""
    try:
        result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=TOOL_TIMEOUT)
        text = str(result).strip()
        if len(text) > 20:
            logger.info(f"{SUCCESS_ICON} FundProductDocAgent: {label} 获取成功 ({len(text)} 字符)")
            return text
        logger.warning(f"FundProductDocAgent: {label} 返回过短 ({len(text)} 字符)")
        return f"[{label}] 数据不可用（返回过短）"
    except asyncio.TimeoutError:
        logger.warning(f"FundProductDocAgent: {label} 超时({TOOL_TIMEOUT}s)")
        return f"[{label}] 数据不可用（超时）"
    except Exception as e:
        logger.warning(f"FundProductDocAgent: {label} 调用失败: {e}")
        return f"[{label}] 数据不可用（调用失败: {str(e)[:80]}）"


async def fund_product_doc_agent(state: AgentState) -> AgentState:
    """
    两阶段基金产品文档解析：
    Phase 1: 并行获取全部 3 个基金数据工具
    Phase 2: 单次 LLM 分析（Qwen3.7-Plus, thinking=disabled）
    """
    logger.info(f"{WAIT_ICON} FundProductDocAgent: Starting two-phase fund product doc analysis.")

    execution_logger = get_execution_logger()
    agent_name = "fund_product_doc_agent"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})
    user_query = current_data.get("query")

    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    cache_code = current_data.get("fund_code", "") or current_data.get("stock_code", "")

    # 缓存检查（TTL=7天，由 cache_utils 控制）
    if not skip_cache and cache_date and cache_code:
        cached = read_cache("fund_product_doc", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} FundProductDocAgent: 命中缓存，跳过分析 ({cache_code})")
            current_data["fund_product_doc"] = cached
            current_metadata["fund_product_doc_executed"] = True
            current_metadata["fund_product_doc_cached"] = True
            return {"data": current_data,
                    "messages": current_messages + [{"role": "assistant", "content": "基金产品文件解析已完成（缓存）"}],
                    "metadata": current_metadata}

    execution_logger.log_agent_start(agent_name, {
        "user_query": user_query,
        "fund_code": cache_code,
        "fund_name": current_data.get("fund_name"),
        "input_data_keys": list(current_data.keys()),
        "architecture": "two-phase",
    })

    if not user_query:
        logger.error(f"{ERROR_ICON} FundProductDocAgent: User query is missing.")
        current_data["fund_product_doc_error"] = "User query is missing."
        execution_logger.log_agent_complete(agent_name, current_data, 0, False, "User query is missing")
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    agent_start_time = time.time()

    try:
        # 模型配置：Model 3 (Qwen3.7-Plus)
        model_cfg = get_model_config_for_agent("fund_product_doc_agent", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]

        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} FundProductDocAgent: Missing OpenAI environment variables.")
            current_data["fund_product_doc_error"] = "Missing OpenAI environment variables."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "Missing env vars")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        fund_code = current_data.get("fund_code", "") or current_data.get("stock_code", "Unknown")
        fund_name = current_data.get("fund_name", "") or current_data.get("company_name", "Unknown")
        current_time_info = current_data.get("current_time_info", "未知时间")
        current_date = current_data.get("current_date", "未知日期")
        clean_code = _extract_fund_code(fund_code) if fund_code else ""

        # ── Phase 1: 并行数据预取 ──────────────────────────────
        logger.info(f"{WAIT_ICON} FundProductDocAgent: Phase 1 — 并行获取 {len(FUND_PRODUCT_DOC_TOOLS)} 个工具数据...")
        phase1_start = time.time()

        try:
            all_tools = await get_mcp_tools(tool_filter=FUND_PRODUCT_DOC_TOOLS)
        except Exception as e:
            logger.error(f"{ERROR_ICON} FundProductDocAgent: 获取 MCP 工具失败: {e}")
            all_tools = []

        if not all_tools:
            logger.error(f"{ERROR_ICON} FundProductDocAgent: No MCP tools available.")
            current_data["fund_product_doc_error"] = "No MCP tools available."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "No tools")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        tool_map = {t.name: t for t in all_tools}
        logger.info(f"{SUCCESS_ICON} FundProductDocAgent: 已加载 {len(all_tools)}/{len(FUND_PRODUCT_DOC_TOOLS)} 个工具")

        # 构建并行任务列表
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

        # --- 基金基本信息 ---
        if "tushare_fund_basic" in tool_map:
            _add(
                _call_tool_safe(tool_map["tushare_fund_basic"], {"code": clean_code}, "基金基本信息"),
                "基金基本信息",
                (tool_map["tushare_fund_basic"], {"code": clean_code}),
            )
        else:
            _placeholder("基金基本信息", "[tushare_fund_basic] 工具不可用")

        # --- 基金公司信息 ---
        if "tushare_fund_company" in tool_map:
            _add(
                _call_tool_safe(tool_map["tushare_fund_company"], {"code": clean_code}, "基金公司信息"),
                "基金公司信息",
                (tool_map["tushare_fund_company"], {"code": clean_code}),
            )
        else:
            _placeholder("基金公司信息", "[tushare_fund_company] 工具不可用")

        # --- 基金搜索（用于交叉验证和同类对比） ---
        if "tushare_fund_search" in tool_map:
            _add(
                _call_tool_safe(tool_map["tushare_fund_search"], {"keyword": fund_name}, "基金搜索结果"),
                "基金搜索结果",
                (tool_map["tushare_fund_search"], {"keyword": fund_name}),
            )
        else:
            _placeholder("基金搜索结果", "[tushare_fund_search] 工具不可用")

        # 并行执行所有任务（第一轮）
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as gather_err:
            logger.error(f"{ERROR_ICON} FundProductDocAgent: Phase 1 并行调用异常: {gather_err}")
            results = [f"并行调用异常: {gather_err}"] * len(tasks)

        # 空数据重试（最多额外3轮，并发递减8→4→2）
        results = await retry_failed_fetches(
            results, tool_infos, labels, _call_tool_safe,
            agent_label="FundProductDocAgent",
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
        logger.info(f"{SUCCESS_ICON} FundProductDocAgent: Phase 1 完成 ({phase1_elapsed:.1f}s, {success_count}/{total_real} 个工具有效数据)")

        # 聚合数据
        data_sections = []
        for label, result in zip(labels, safe_results):
            data_sections.append(f"### [{label}]\n{result}")
        raw_data_text = "\n\n".join(data_sections)

        # ── Phase 2: LLM 基金产品文件解析 ─────────────────────
        logger.info(f"{WAIT_ICON} FundProductDocAgent: Phase 2 — LLM 基金产品文档解析 (model={model_name}, thinking=disabled)...")
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

        analysis_prompt = f"""请以基金产品分析师的标准，对{fund_name}（基金代码：{fund_code}）进行基金产品文档解析。

当前时间：{current_time_info}
当前日期：{current_date}

## 原始数据

以下是通过工具获取的全部基金产品数据，请基于这些数据进行全面解析：

{raw_data_text}

## 分析要求

请进行以下维度的基金产品文档解析（每个维度都需要基于上述原始数据，引用具体字段和数值）：

1. 基金类型判定
   - 基金大类：ETF/LOF/主动权益/债券/货币/QDII/FOF/养老目标/REITs/其他
   - 具体子类型：股票型ETF/债券型LOF/混合型FOF等
   - 运作方式：开放式/封闭式/定期开放
   - 交易场所：场内（上交所/深交所）、场外、跨市场

2. 投资目标与策略
   - 基金投资目标和投资范围（从基金名称+基金类型+业绩基准综合判断）
   - 是否跟踪特定指数？若是，指数全称和编制方？
   - 主动还是被动管理？
   - 投资策略特点（如Smart Beta、因子策略、行业轮动等）

3. 业绩基准分析
   - 基准指数的构成（如沪深300指数收益率×95%+银行活期存款利率×5%）
   - 基准代表的市场/资产类别
   - 实际跟踪标的（如果是ETF/指数基金）

4. 风险等级评估（R1-R5）
   - 基于基金类型判断风险等级：
     - R1（低风险）：货币基金
     - R2（中低风险）：短债基金、纯债基金
     - R3（中风险）：混合债基、偏债混合、FOF
     - R4（中高风险）：偏股混合、主动权益、宽基ETF
     - R5（高风险）：行业/主题ETF、QDII、商品ETF、杠杆基金
   - 结合业绩基准的波动特征说明风险来源

5. 费率结构
   - 管理费率（%/年）
   - 托管费率（%/年）
   - 申购/认购费率（最高/最低）
   - 赎回费率和阶梯规则
   - 是否有销售服务费
   - 综合持有成本估算（以持有1年为例）

6. 申购赎回规则
   - 申购/认购起始日期
   - 赎回起始日期
   - 最低申购金额
   - 最低赎回份额
   - 是否支持定投
   - 分红方式（现金分红/红利再投资）

7. 产品结构特征
   - 是否ETF：若是，是否支持T+0交易、是否支持套利
   - 是否LOF：若是，场内场外价差风险提示
   - 是否QDII：若是，汇率风险提示
   - 是否FOF：若是，双重收费风险提示
   - 是否养老目标：若是，个税递延政策说明
   - 是否联接基金：若是，说明联接的主基金

8. 持有期约束
   - 基金存续期限（duration_year，如适用）
   - 是否有最短持有期
   - 是否有封闭期
   - 到期日/到期处理规则

9. 综合评估
   - 产品适配投资者画像（风险承受能力、投资期限、投资目标）
   - 同类产品对比优势（如有搜索结果数据）
   - 信心程度评级（高/中/低，标注原因）

重要限制：
- 请专注于基金产品文档信息分析，不要分析基金的短期业绩表现或市场行情
- 分析必须有数据支撑，引用上述原始数据中的具体字段值，避免空洞的定性描述
- 如果某些数据无法获取，请说明原因并基于可用数据提供分析
- 不要使用模型训练数据中的知识来补充数据事实

⛔ 输出格式要求（防幻觉机制）：
请将分析输出严格分为两个区域：

## 📊 数据事实区
列出上述原始数据中的关键客观数据，每条标注数据来源标签（如 [基金基本信息]）：
- [标签] 字段名=具体值（如：基金类型=ETF、管理费率=0.50%）
- [标签] 字段名=具体值
- ...
如果某项数据在上述原始数据中标注为「数据不可用」，必须在此如实声明，不得推测。

## 🔍 分析判断区
基于上述数据事实进行分析和推断。每个判断必须：
1. 引用数据事实区的具体数值
2. 使用「【基于数据的推断】」或「【行业知识补充】」标注推断性质
3. 如果某个结论无法从数据中直接得出，必须声明「此为分析师推断」
4. 不得在任何地方编造数据事实区没有的数值
5. 不得编造原始数据中不存在的基金名称、日期或事件"""

        messages = [
            {"role": "system", "content": "你是一位资深基金产品分析师，专注于公募基金的产品要素解析和投资者适配评估。"},
            {"role": "user", "content": analysis_prompt},
        ]

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=float(LLM_TIMEOUT)
            )
            final_output = response.content.strip() if hasattr(response, 'content') else str(response)
            phase2_elapsed = time.time() - phase2_start
            logger.info(f"FundProductDocAgent: Phase 2 完成 ({phase2_elapsed:.1f}s, {len(final_output)} 字符)")
            llm_success = True
        except asyncio.TimeoutError:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundProductDocAgent: LLM 超时 ({phase2_elapsed:.0f}s)")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析判断区\n\n⚠️ LLM分析超时，以上为原始数据，请人工分析。\n"
            llm_success = False
        except Exception as llm_err:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundProductDocAgent: LLM 失败: {llm_err}")
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
        logger.info(f"{SUCCESS_ICON if llm_success else WAIT_ICON} FundProductDocAgent: 分析完成 (总耗时 {total_time:.1f}s, Phase1={phase1_elapsed:.1f}s, Phase2={phase2_elapsed:.1f}s)")

        # 更新状态
        current_data["fund_product_doc"] = final_output
        if not skip_cache and cache_date and cache_code:
            write_cache("fund_product_doc", cache_code, cache_date, final_output)
        current_metadata["fund_product_doc_executed"] = True
        current_metadata["fund_product_doc_timestamp"] = str(time.time())
        current_metadata["fund_product_doc_execution_time"] = f"{total_time:.2f} seconds"

        execution_logger.log_agent_complete(agent_name, {
            "fund_product_doc_length": len(final_output),
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
            "messages": current_messages + [{"role": "assistant", "content": "基金产品文件解析已完成"}],
            "metadata": current_metadata,
        }

    except Exception as e:
        logger.error(f"{ERROR_ICON} FundProductDocAgent: Error: {e}", exc_info=True)
        current_data["fund_product_doc_error"] = f"Error: {e}"
        current_data["fund_product_doc"] = f"基金产品文件解析过程中出现错误: {str(e)}"
        current_metadata["fund_product_doc_error"] = str(e)
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}


# 本地测试函数
async def test_fund_product_doc_agent():
    """基金产品文件解析 Agent的测试函数"""
    from datetime import datetime
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
            "query": "解析基金产品华夏上证50ETF",
            "fund_code": "sh.510050",
            "fund_name": "华夏上证50ETF",
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
        },
        metadata={}
    )

    result = await fund_product_doc_agent(test_state)
    print("Fund Product Doc Analysis Result:")
    print(result.get("data", {}).get("fund_product_doc", "No analysis found"))
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fund_product_doc_agent())
