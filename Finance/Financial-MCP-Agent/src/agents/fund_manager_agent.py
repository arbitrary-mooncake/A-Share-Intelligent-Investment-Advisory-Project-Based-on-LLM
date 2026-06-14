"""
FundManager Analysis Agent: 两阶段架构 — 并行数据预取 + 单次 LLM 评估。
Phase 1: asyncio.gather 并行获取基金经理/管理人/基金基本信息
Phase 2: 将所有原始数据喂给 LLM 一次性完成经理质量评估（thinking 关闭）
"""
import asyncio
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

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

# 基金经理分析白名单
FUND_MANAGER_TOOLS = [
    "tushare_fund_manager",     # 基金经理信息：姓名、学历、任职期限、简历
    "tushare_fund_company",     # 基金管理人/公司信息：规模、员工数、注册资本
    "tushare_fund_basic",       # 基金基本信息：类型、管理人、费率、成立日期
]


def _clean_code(fund_code: str) -> str:
    """清理基金代码，移除交易所前后缀"""
    return fund_code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").replace(".OF", "").strip()


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
            logger.info(f"{SUCCESS_ICON} FundManagerAgent: {label} 获取成功 ({len(text)} 字符)")
            await set_cached_tool_result(tool_name, kwargs, text)
            return text
        logger.warning(f"FundManagerAgent: {label} 返回过短 ({len(text)} 字符)")
        return f"[{label}] 数据不可用（返回过短）"
    except asyncio.TimeoutError:
        logger.warning(f"FundManagerAgent: {label} 超时({TOOL_TIMEOUT}s)")
        return f"[{label}] 数据不可用（超时）"
    except Exception as e:
        logger.warning(f"FundManagerAgent: {label} 调用失败: {e}")
        return f"[{label}] 数据不可用（调用失败: {str(e)[:80]}）"


async def fund_manager_agent(state: AgentState) -> AgentState:
    """
    两阶段基金经理/团队/管理人评估：
    Phase 1: 并行获取基金经理信息、管理人平台数据、基金基本信息
    Phase 2: 单次 LLM 结构化评估（Qwen3.7-Plus, thinking=disabled）
    """
    logger.info(f"{WAIT_ICON} FundManagerAgent: Starting two-phase fund manager analysis.")

    execution_logger = get_execution_logger()
    agent_name = "fund_manager"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})

    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    # 优先使用 fund_code，回退到 stock_code（ETF 场景）
    fund_code = current_data.get("fund_code", "") or current_data.get("stock_code", "")

    # 缓存检查
    if not skip_cache and cache_date and fund_code:
        cached = read_cache("fund_manager", fund_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} FundManagerAgent: 命中缓存，跳过分析 ({fund_code})")
            current_data["fund_manager"] = cached
            from src.utils.cache_utils import read_signal_pack_cache
            cached_sp = read_signal_pack_cache("fund_manager", fund_code, cache_date)
            if cached_sp:
                current_data["fund_manager_signal_pack"] = cached_sp
            current_metadata["fund_manager_executed"] = True
            current_metadata["fund_manager_cached"] = True
            return {"data": current_data,
                    "messages": current_messages + [{"role": "assistant", "content": "基金经理评估已完成（缓存）"}],
                    "metadata": current_metadata}

    if not fund_code:
        logger.warning(f"{WAIT_ICON} FundManagerAgent: 未提供基金代码，跳过分析")
        current_data["fund_manager"] = "未提供基金代码，无法进行基金经理评估。"
        current_metadata["fund_manager_executed"] = True
        return {"data": current_data,
                "messages": current_messages + [{"role": "assistant", "content": "无基金代码，已跳过基金经理评估"}],
                "metadata": current_metadata}

    fund_name = current_data.get("fund_name", "") or current_data.get("company_name", "")
    current_date = current_data.get("current_date", "未知日期")
    current_time_info = current_data.get("current_time_info", "未知时间")
    clean_code = _clean_code(fund_code)

    execution_logger.log_agent_start(agent_name, {
        "fund_code": fund_code,
        "fund_name": fund_name,
        "current_date": current_date,
        "input_data_keys": list(current_data.keys()),
        "architecture": "two-phase",
    })

    agent_start_time = time.time()

    try:
        # 模型配置：Model 3 (Qwen3.7-Plus), thinking=disabled
        model_cfg = get_model_config_for_agent("fund_manager_agent", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]

        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} FundManagerAgent: Missing OpenAI environment variables.")
            current_data["fund_manager"] = "基金经理评估失败：缺少模型配置。"
            current_metadata["fund_manager_error"] = "Missing OpenAI environment variables."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "Missing env vars")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        # ── Phase 1: 并行数据预取 ──────────────────────────────
        logger.info(f"{WAIT_ICON} FundManagerAgent: Phase 1 — 并行获取 {len(FUND_MANAGER_TOOLS)} 个工具数据...")
        phase1_start = time.time()

        try:
            all_tools = await get_mcp_tools(tool_filter=FUND_MANAGER_TOOLS)
        except Exception as e:
            logger.error(f"{ERROR_ICON} FundManagerAgent: 获取 MCP 工具失败: {e}")
            all_tools = []

        if not all_tools:
            logger.error(f"{ERROR_ICON} FundManagerAgent: No MCP tools available.")
            current_data["fund_manager"] = "基金经理评估失败：MCP 工具不可用。"
            current_metadata["fund_manager_error"] = "No MCP tools available."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "No tools")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        tool_map = {t.name: t for t in all_tools}
        logger.info(f"{SUCCESS_ICON} FundManagerAgent: 已加载 {len(all_tools)}/{len(FUND_MANAGER_TOOLS)} 个工具")

        # 构建并行任务列表
        tasks = []
        labels = []
        tool_infos = []

        def _add(task, label, ti=None):
            tasks.append(task)
            labels.append(label)
            tool_infos.append(ti)

        def _placeholder(label, msg):
            labels.append(label)
            tasks.append(_noop_result(msg))
            tool_infos.append(None)

        # --- 基金基本信息 ---
        if "tushare_fund_basic" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_basic"], {"code": clean_code}, "基金基本信息"),
                 "基金基本信息", (tool_map["tushare_fund_basic"], {"code": clean_code}))
        else:
            _placeholder("基金基本信息", "[tushare_fund_basic] 工具不可用")

        # --- 基金经理信息（核心） ---
        if "tushare_fund_manager" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_manager"], {"code": clean_code}, "基金经理信息"),
                 "基金经理信息", (tool_map["tushare_fund_manager"], {"code": clean_code}))
        else:
            _placeholder("基金经理信息", "[tushare_fund_manager] 工具不可用")

        # --- 基金管理人/公司信息 ---
        if "tushare_fund_company" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_company"], {"code": clean_code}, "基金公司信息"),
                 "基金公司信息", (tool_map["tushare_fund_company"], {"code": clean_code}))
        else:
            _placeholder("基金公司信息", "[tushare_fund_company] 工具不可用")

        # 并行执行所有任务（第一轮）
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as gather_err:
            logger.error(f"{ERROR_ICON} FundManagerAgent: Phase 1 并行调用异常: {gather_err}")
            results = [f"并行调用异常: {gather_err}"] * len(tasks)

        # 空数据重试
        results = await retry_failed_fetches(
            results, tool_infos, labels, _call_tool_safe,
            agent_label="FundManagerAgent",
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
        logger.info(f"{SUCCESS_ICON} FundManagerAgent: Phase 1 完成 ({phase1_elapsed:.1f}s, {success_count}/{total_real} 个工具有效数据)")

        # 聚合数据
        data_sections = []
        for label, result in zip(labels, safe_results):
            data_sections.append(f"### [{label}]\n{result}")
        raw_data_text = "\n\n".join(data_sections)

        # ── Phase 2: LLM 结构化评估 ──────────────────────────────
        logger.info(f"{WAIT_ICON} FundManagerAgent: Phase 2 — LLM 经理评估 (model={model_name}, thinking=disabled)...")
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

        analysis_prompt = f"""请以基金评级机构分析师的标准，对基金"{fund_name}"（代码：{fund_code}）的基金经理和团队进行系统性评估。

当前时间：{current_time_info}
当前日期：{current_date}

## 原始数据

以下是通过工具获取的全部原始数据，请基于这些数据进行深度分析：

{raw_data_text}

## 分析要求

请对以下 9 个维度逐一评估（每个维度必须引用原始数据中的具体信息）：

### 1. 基金经理任职年限
- 总从业年限：从简历(resume)中推断基金经理何时入行
- 管理本基金年限：从 begin_date 计算至今的时长
- 评分参考：<2年为短，2-5年为中，5-10年为长，>10年为资深

### 2. 教育背景与专业资质
- 学历（edu 字段）：本科/硕士/博士
- 专业背景：从简历中提取（金融/经济/理工/复合背景）
- 是否具备 CFA/CPA/FRM 等高含金量证书（从简历中提取）

### 3. 在管产品风格一致性
- 从简历中评估该经理的投资风格是否一贯（价值/成长/均衡/量化等）
- 分析其从业经历是否出现风格漂移

### 4. 历史回撤管理能力
- 简历中是否体现风控经验和回撤管理理念
- 是否经历过完整牛熊周期
- 是否有危机应对的表述

### 5. 团队稳定性（关键指标）
- 检查基金经理的 end_date 字段：
  - 若 end_date 为 None 或空：表示现任经理仍在任
  - 若有明确的 end_date：表示已离任，需记录离任日期
- 从基金经理列表中判断是否存在频繁更换基金经理的情况：
  - 数一数该基金历史上出现过几位经理
  - 计算平均每位经理的任职时长
  - 与行业平均水平（约3年）对比

### 6. 是否频繁更换基金经理
- 对比现任经理的 begin_date 与基金的 found_date（成立日期）
- 如果基金成立以来更换超过3任经理，标注为"频繁更换"
- 如果现任经理管理时间超过3年且基金历史经理数不超过2人，标注为"管理稳定"

### 7. 管理人平台能力
- 基金公司名称、规模（从 tushare_fund_company 数据中提取）
- 员工人数（employees）、注册资本（reg_capital）
- 成立日期（setup_date）：评估平台历史积淀
- 平台综合实力评级：头部/大型/中型/小型

### 8. 同类产品线是否拥挤
- 从基金基本信息中识别该基金的类型（fund_type, invest_type）
- 结合基金公司信息，评估该公司旗下同类型产品的数量
- 判断是否存在"同类产品过度发行、资源分散"的风险
- 如果数据不足以判断，请如实声明

### 9. 综合评估：是否因"人"而加分
- 综合以上 8 个维度的分析，给出该基金经理/团队的总评价
- 明确指出哪些维度是加分项（如：资深经理+名校背景+团队稳定）
- 明确指出哪些维度是减分项（如：频繁更换经理+平台实力弱）
- 最终判断：该基金的经理/团队质量是其投资价值的加分因素还是减分因素

## 输出格式

请按以下结构输出：

## 📊 数据事实区
列出上述原始数据中的关键客观数据，每条标注数据来源标签：
- [基金经理信息] 姓名=XXX, 学历=XXX, 任职起始日=XXX, end_date=XXX
- [基金公司信息] 公司名称=XXX, 员工数=XXX, 注册资本=XXX
- [基金基本信息] 基金类型=XXX, 成立日期=XXX, 管理费率=XXX
- ...
如果某项数据标注为「数据不可用」，必须在此如实声明，不得推测。

## 🔍 经理评估区
逐项对上述 9 个维度进行分析判断。每个判断必须：
1. 引用数据事实区的具体信息
2. 如果某个结论无法从数据中直接得出，必须声明「此为分析师推断」
3. 不得在任何地方编造数据事实区没有的数值
4. 不得编造原始数据中不存在的新闻或事件

## 📋 综合评分
| 维度 | 评价 | 得分(满10) |
|------|------|-----------|
| 任职年限 | ... | X/10 |
| 教育背景 | ... | X/10 |
| 团队稳定性 | ... | X/10 |
| 平台实力 | ... | X/10 |
| **综合** | **加分/中性/减分** | **X/10** |

重要限制：
- 请专注于基金经理和团队质量评估，不要分析基金的投资价值或业绩表现
- 分析必须有数据支撑，引用上述原始数据中的具体信息
- 如果某些数据无法获取，请说明原因并基于可用数据提供分析
- 不要使用模型训练数据中的知识来补充数据事实

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
            {"role": "system", "content": "你是一位资深基金评级分析师，专注于公募基金的基金经理和团队质量评估。你擅长从基金经理简历中提取关键信息，能客观评价管理人的专业能力和稳定性。"},
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
                    sp["agent_name"] = "fund_manager"
                    sp["as_of_date"] = current_date
                except Exception:
                    pass
            if sp is None:
                from src.utils.analysis_package_builder import text_to_signal_pack
                sp = text_to_signal_pack(final_output, "fund_manager", current_date)
            current_data["fund_manager_signal_pack"] = sp
            phase2_elapsed = time.time() - phase2_start
            logger.info(f"FundManagerAgent: Phase 2 完成 ({phase2_elapsed:.1f}s, {len(final_output)} 字符)")
            llm_success = True
        except asyncio.TimeoutError:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundManagerAgent: LLM 超时 ({phase2_elapsed:.0f}s)")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 经理评估区\n\n⚠️ LLM分析超时，以上为原始数据，请人工分析。\n"
            llm_success = False
        except Exception as llm_err:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundManagerAgent: LLM 失败: {llm_err}")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 经理评估区\n\n⚠️ LLM分析失败({str(llm_err)[:100]})，以上为原始数据，请人工分析。\n"
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
            interaction_type="fund_manager_analysis",
            input_messages=[{"role": "user", "content": analysis_prompt[:5000]}],
            output_content=final_output,
            model_config=model_config_log,
            execution_time=phase2_elapsed,
        )

        total_time = time.time() - agent_start_time
        logger.info(f"{SUCCESS_ICON if llm_success else WAIT_ICON} FundManagerAgent: 分析完成 (总耗时 {total_time:.1f}s, Phase1={phase1_elapsed:.1f}s, Phase2={phase2_elapsed:.1f}s)")

        # 更新状态
        current_data["fund_manager"] = final_output
        if not skip_cache and cache_date and fund_code:
            write_cache("fund_manager", fund_code, cache_date, final_output)
            if "fund_manager_signal_pack" in current_data:
                from src.utils.cache_utils import write_signal_pack_cache
                write_signal_pack_cache("fund_manager", fund_code, cache_date, current_data["fund_manager_signal_pack"])
        current_metadata["fund_manager_executed"] = True
        current_metadata["fund_manager_timestamp"] = str(time.time())
        current_metadata["fund_manager_execution_time"] = f"{total_time:.2f} seconds"

        execution_logger.log_agent_complete(agent_name, {
            "fund_manager_length": len(final_output),
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
            "messages": current_messages + [{"role": "assistant", "content": "基金经理评估已完成"}],
            "metadata": current_metadata,
        }

    except Exception as e:
        logger.error(f"{ERROR_ICON} FundManagerAgent: Error: {e}", exc_info=True)
        current_data["fund_manager"] = f"基金经理评估过程中出现错误: {str(e)}"
        current_metadata["fund_manager_error"] = str(e)
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}


# 本地测试函数
async def test_fund_manager_agent():
    """基金经理评估 Agent 的测试函数"""
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
            "query": "评估黄金ETF(512480)华安基金的基金经理和团队质量",
            "fund_code": "512480",
            "fund_name": "黄金ETF华安",
            "stock_code": "sh.512480",
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

    result = await fund_manager_agent(test_state)
    print("Fund Manager Analysis Result:")
    print(result.get("data", {}).get("fund_manager", "No analysis found"))
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fund_manager_agent())
