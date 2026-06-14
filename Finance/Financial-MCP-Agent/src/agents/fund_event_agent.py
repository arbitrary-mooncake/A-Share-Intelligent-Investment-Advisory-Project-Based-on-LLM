"""
Fund Event Agent (Agent 7: 基金事件/变更/风险公告): 两阶段架构 — 并行数据预取 + 单次 LLM 事件分类。
Phase 1: asyncio.gather 并行获取白名单 5 个基金事件相关工具的数据
Phase 2: 将所有原始数据喂给 LLM 一次性完成事件分类（thinking 关闭，事件分类不需要深度推理）

数据覆盖率：~60%（基金合同修订和招募说明书更新无法通过 API 获取）
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
# LLM 整体超时（秒，thinking 关闭下更快）
LLM_TIMEOUT = 180

# 基金事件分析白名单（6 个 Tushare 工具）
FUND_EVENT_TOOLS = [
    "tushare_fund_manager",      # 基金经理变更检测
    "tushare_fund_div",          # 分红事件
    "tushare_fund_share",        # 份额规模变化（赎回压力检测）
    "tushare_fund_basic",        # 基金基本信息（状态：上市/退市/暂停申赎）
    "tushare_news",              # 基金个股新闻（季报/年报等）
    "tushare_major_news",        # 主流财经新闻（基金公司/经理相关）
]


def _extract_fund_code(fund_code: str) -> str:
    """提取纯数字基金代码，去掉 sh./sz./of./.SH/.SZ/.OF 等前缀后缀"""
    code = str(fund_code).strip()
    for prefix in ["sh.", "sz.", "of.", "SH.", "SZ.", "OF.", "bj."]:
        if code.lower().startswith(prefix.lower()):
            code = code[len(prefix):]
    for suffix in [".OF", ".SH", ".SZ", ".BJ", ".of", ".sh", ".sz", ".bj"]:
        if code.lower().endswith(suffix.lower()):
            code = code[:-len(suffix)]
    return code.strip()


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
            logger.info(f"{SUCCESS_ICON} FundEventAgent: {label} 获取成功 ({len(text)} 字符)")
            await set_cached_tool_result(tool_name, kwargs, text)
            return text
        logger.warning(f"FundEventAgent: {label} 返回过短 ({len(text)} 字符)")
        return f"[{label}] 数据不可用（返回过短）"
    except asyncio.TimeoutError:
        logger.warning(f"FundEventAgent: {label} 超时({TOOL_TIMEOUT}s)")
        return f"[{label}] 数据不可用（超时）"
    except Exception as e:
        logger.warning(f"FundEventAgent: {label} 调用失败: {e}")
        return f"[{label}] 数据不可用（调用失败: {str(e)[:80]}）"


async def fund_event_agent(state: AgentState) -> AgentState:
    """
    两阶段基金事件分析：
    Phase 1: 并行获取 5 个基金事件相关工具的数据
    Phase 2: 单次 LLM 事件分类（Qwen3.7-Plus, thinking=disabled）

    检测维度：基金经理变更、分红、申赎状态、清盘预警、合同修订、
             招募说明书更新、基准调整、侧袋机制/流动性风险、合规事件、
             近期变化对投资结论的影响
    """
    logger.info(f"{WAIT_ICON} FundEventAgent: Starting two-phase fund event analysis.")

    execution_logger = get_execution_logger()
    agent_name = "fund_event_agent"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})

    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    cache_code = current_data.get("fund_code", "") or current_data.get("stock_code", "")

    # 缓存检查（TTL=1天，由 cache_utils 控制）
    if not skip_cache and cache_date and cache_code:
        cached = read_cache("fund_event", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} FundEventAgent: 命中缓存，跳过分析 ({cache_code})")
            current_data["fund_event"] = cached
            from src.utils.cache_utils import read_signal_pack_cache
            cached_sp = read_signal_pack_cache("fund_event", cache_code, cache_date)
            if cached_sp:
                current_data["fund_event_signal_pack"] = cached_sp
            current_metadata["fund_event_executed"] = True
            current_metadata["fund_event_cached"] = True
            return {"data": current_data,
                    "messages": current_messages + [{"role": "assistant", "content": "基金事件分析已完成（缓存）"}],
                    "metadata": current_metadata}

    execution_logger.log_agent_start(agent_name, {
        "fund_code": cache_code,
        "fund_name": current_data.get("fund_name"),
        "input_data_keys": list(current_data.keys()),
        "architecture": "two-phase",
        "data_coverage_warning": "~60%",
    })

    agent_start_time = time.time()

    try:
        # 模型配置：Model 3 (Qwen3.7-Plus)，thinking 关闭
        model_cfg = get_model_config_for_agent("fund_event_agent", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]

        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} FundEventAgent: Missing OpenAI environment variables.")
            current_data["fund_event_error"] = "Missing OpenAI environment variables."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "Missing env vars")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        fund_code = current_data.get("fund_code", "") or current_data.get("stock_code", "Unknown")
        fund_name = current_data.get("fund_name", "") or current_data.get("company_name", "Unknown")
        current_time_info = current_data.get("current_time_info", "未知时间")
        current_date = current_data.get("current_date", "未知日期")
        clean_code = _extract_fund_code(fund_code) if fund_code else ""

        # ── Phase 1: 并行数据预取 ──────────────────────────────
        logger.info(f"{WAIT_ICON} FundEventAgent: Phase 1 — 并行获取 {len(FUND_EVENT_TOOLS)} 个工具数据...")
        phase1_start = time.time()

        try:
            all_tools = await get_mcp_tools(tool_filter=FUND_EVENT_TOOLS)
        except Exception as e:
            logger.error(f"{ERROR_ICON} FundEventAgent: 获取 MCP 工具失败: {e}")
            all_tools = []

        if not all_tools:
            logger.error(f"{ERROR_ICON} FundEventAgent: No MCP tools available.")
            current_data["fund_event_error"] = "No MCP tools available."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "No tools")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        tool_map = {t.name: t for t in all_tools}
        logger.info(f"{SUCCESS_ICON} FundEventAgent: 已加载 {len(all_tools)}/{len(FUND_EVENT_TOOLS)} 个工具")

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
            labels.append(label)
            tasks.append(_noop_result(msg))
            tool_infos.append(None)

        # --- 基金经理变更 ---
        if "tushare_fund_manager" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_manager"], {"code": clean_code}, "基金经理变更"),
                 "基金经理变更", (tool_map["tushare_fund_manager"], {"code": clean_code}))
        else:
            _placeholder("基金经理变更", "[tushare_fund_manager] 工具不可用")

        # --- 分红事件 ---
        if "tushare_fund_div" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_div"], {"code": clean_code}, "分红事件"),
                 "分红事件", (tool_map["tushare_fund_div"], {"code": clean_code}))
        else:
            _placeholder("分红事件", "[tushare_fund_div] 工具不可用")

        # --- 份额规模变化 ---
        if "tushare_fund_share" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_share"], {"code": clean_code}, "份额规模变化"),
                 "份额规模变化", (tool_map["tushare_fund_share"], {"code": clean_code}))
        else:
            _placeholder("份额规模变化", "[tushare_fund_share] 工具不可用")

        # --- 基金基本信息（状态：上市/退市/暂停申赎）---
        if "tushare_fund_basic" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_basic"], {"code": clean_code}, "基金基本信息"),
                 "基金基本信息", (tool_map["tushare_fund_basic"], {"code": clean_code}))
        else:
            _placeholder("基金基本信息", "[tushare_fund_basic] 工具不可用")

        # --- 新闻舆情 ---
        if "tushare_news" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_news"], {"code": clean_code}, "新闻舆情"),
                 "新闻舆情", (tool_map["tushare_news"], {"code": clean_code}))
        else:
            _placeholder("新闻舆情", "[tushare_news] 工具不可用")

        # --- 主流财经新闻（用基金公司/经理名称搜索） ---
        # 优先用管理公司名称搜索（如"华商基金"），命中率更高；
        # 若首轮失败，重试用基金全称（如"华商优势行业混合"）
        _search_name = fund_name or clean_code
        _company_name = ""
        try:
            from src.utils.tushare_client import get_fund_basic as _direct_get_fund_basic
            loop = asyncio.get_running_loop()
            _fb = await loop.run_in_executor(None, _direct_get_fund_basic, clean_code)
            if _fb:
                _company_name = _fb.get("management", "") or ""
        except Exception as _e:
            logger.debug(f"FundEventAgent: 获取管理公司名称失败: {_e}")

        if _company_name:
            _primary_kw = _company_name
            _fallback_kw = fund_name or clean_code
            logger.info(f"FundEventAgent: 主流财经新闻首选关键词='{_primary_kw}', 备选='{_fallback_kw}'")
        else:
            _primary_kw = _search_name
            _fallback_kw = None

        if "tushare_major_news" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_major_news"], {"keyword": _primary_kw, "days": 90}, "主流财经新闻"),
                 "主流财经新闻", (tool_map["tushare_major_news"], {"keyword": _primary_kw, "days": 90}))
        else:
            _placeholder("主流财经新闻", "[tushare_major_news] 工具不可用")

        # 并行执行所有任务（第一轮）
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as gather_err:
            logger.error(f"{ERROR_ICON} FundEventAgent: Phase 1 并行调用异常: {gather_err}")
            results = [f"并行调用异常: {gather_err}"] * len(tasks)

        # 空数据重试（最多额外3轮，并发递减8→4→2）
        # 为 major_news 构建备选关键词（首轮重试用基金全称替代公司名称）
        _alt_kwargs = [None] * len(tool_infos)
        if _fallback_kw and _fallback_kw != _primary_kw:
            for _i, (_lbl, _ti) in enumerate(zip(labels, tool_infos)):
                if _lbl == "主流财经新闻" and _ti is not None:
                    _alt_kwargs[_i] = {"keyword": _fallback_kw, "days": 90}
                    break
        results = await retry_failed_fetches(
            results, tool_infos, labels, _call_tool_safe,
            agent_label="FundEventAgent",
            alt_kwargs_list=_alt_kwargs,
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
        logger.info(f"{SUCCESS_ICON} FundEventAgent: Phase 1 完成 ({phase1_elapsed:.1f}s, {success_count}/{total_real} 个工具有效数据)")

        # 聚合数据
        data_sections = []
        for label, result in zip(labels, safe_results):
            data_sections.append(f"### [{label}]\n{result}")
        raw_data_text = "\n\n".join(data_sections)

        # ── Phase 2: LLM 事件分类（thinking 关闭）──────────────
        logger.info(f"{WAIT_ICON} FundEventAgent: Phase 2 — LLM 事件分类 (model={model_name}, thinking=disabled)...")
        phase2_start = time.time()

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=1.0,
            request_timeout=LLM_TIMEOUT,
            max_tokens=8000,
            extra_body=get_thinking_body(base_url, False),  # thinking 关闭
        )

        analysis_prompt = f"""请以基金研究分析师的标准，对{fund_name}（基金代码：{fund_code}）进行基金事件/变更/风险公告分析。

当前时间：{current_time_info}
当前日期：{current_date}

## ⚠️ 重要：数据覆盖率声明

本代理的数据覆盖率约为 **60%**，以下数据**无法通过当前API直接获取**：
- 基金合同全文及修订内容
- 招募说明书更新全文
- 基金公司内部合规审查文件
- 侧袋机制具体持仓明细
- 实时申赎限制变更

分析中如遇上述数据缺口，请如实标注「数据不可用」，不得推测或编造。

## 原始数据

以下是通过工具获取的全部原始数据，请基于这些数据进行事件分类和风险评估：

{raw_data_text}

## 分析要求

请逐一检查以下 10 个事件维度（每个维度都需要引用原始数据中的具体信息）：

### 1. 基金经理变更检测
- 检查 tushare_fund_manager 数据中的 ann_date（公告日期）和 end_date（离职/变更日期）
- 判断是否有近期基金经理离职（离当前日期 3 个月内）
- 判断是否有即将发生的变更（end_date 在未来 1 个月内）
- 评估新任基金经理的历史业绩和管理规模（如果数据包含）
- 基金经理变更频繁程度（历史变更次数）
- **结论**：变更风险等级（低/中/高），是否构成实质性影响

### 2. 分红事件
- 检查 tushare_fund_div 数据中的分红公告日期、除息日、分红金额
- 判断最近一次分红是否在近期（3 个月内）
- 分红频率和金额趋势（与历史对比）
- **结论**：分红对持有收益的影响评估

### 3. 暂停申购/赎回检测
- 检查 tushare_fund_basic 数据中的 fund_status 字段
- 是否存在申购限制（暂停申购、限额申购）
- 是否存在赎回限制（暂停赎回、巨额赎回延迟）
- 检测 fund_share 数据中份额急剧下降（可能预示大额赎回）
- **结论**：申赎风险等级，流动性风险提示

### 4. 清盘预警
- 检查 tushare_fund_share 数据中的基金规模趋势
- 基金规模是否持续缩小（连续 3 期份额下降）
- 当前规模是否接近清盘线（通常 5000 万元以下）或 2 亿元（发起式基金）
- 从 fund_basic 检查基金运作方式（是否有到期日、是否发起式）
- **结论**：清盘风险等级（无/低/中/高）

### 5. 基金合同修订
- ⚠️ **数据限制提醒**：基金合同全文及修订公告无法通过当前 API 直接获取
- 尝试从 tushare_news 中检测是否有合同修订相关新闻
- 若有合同修订新闻，分析修订要点
- **结论**：如新闻中无合同修订信息，如实标注「合同修订数据不可用」

### 6. 招募说明书更新
- ⚠️ **数据限制提醒**：招募说明书更新全文无法通过当前 API 直接获取
- 尝试从 tushare_news 中检测是否有招募说明书更新相关公告
- 若有相关新闻，分析更新要点
- **结论**：如新闻中无招募说明书更新信息，如实标注「招募说明书更新数据不可用」

### 7. 基准调整检测
- 检查 tushare_fund_basic 数据中的 benchmark 字段
- 对比历史 fund_basic 数据（如果可用），判断基准是否发生变更
- 基准变更对基金投资策略的潜在影响
- **结论**：如无法进行历史对比，如实标注「基准历史对比数据不可用」

### 8. 侧袋机制/流动性风险评估
- ⚠️ **数据限制提醒**：侧袋机制具体持仓明细无法通过当前 API 获取
- 根据 tushare_fund_basic 中的基金类型和投资标的特征，评估流动性风险
- 如果基金持有停牌股票或低流动性资产比例较高，流动性风险上升
- 检查 tushare_news 中是否有侧袋机制相关公告
- **结论**：流动性风险等级评估，如实标注数据限制

### 9. 管理人重大合规事件
- 检查 tushare_news 数据中是否涉及基金管理人的负面新闻
- 检查内容包括但不限于：监管处罚、违规调查、高管涉案、重大投诉
- **结论**：合规风险等级（无/低/中/高），如有负面事件请详细说明

### 10. 近期变化综合评估：是否改变投资结论
- 综合以上 9 个维度的分析结果
- 判断近期事件是否对基金的投资价值产生了实质性改变
- 若有重大负面事件（如基金经理变更+规模大幅下降），给出明确的"改变结论"
- 若近期无明显负面事件，给出"维持原结论"
- 标注各维度的数据可用性状态

## 输出格式要求

请将分析输出严格分为两个区域：

## 📊 数据事实区
列出上述原始数据中的关键客观数据，每条标注数据来源标签：
- [基金经理变更] 具体信息（如：2026-05-20 张某离任，2026-06-01 李某接任）
- [分红事件] 具体信息（如：2026-04-15 每份分红 0.05 元，除息日 2026-04-20）
- [份额规模变化] 具体数值（如：2026Q1 份额 12.3 亿→2026Q2 份额 8.7 亿，下降 29%）
- [基金基本信息] fund_status、benchmark、基金类型等
- [新闻舆情] 相关新闻标题和日期
如果某项数据标注为「数据不可用」，必须在此如实声明，不得推测。

## 🔍 分析与风险评估区
基于上述数据事实，逐项完成 10 个维度的分析。每个判断必须：
1. 引用数据事实区的具体数值
2. 使用「【基于数据的判断】」或「【行业知识补充】」标注推断性质
3. 如果某个结论无法从数据中直接得出，必须声明「此为分析师推断」
4. 不得在任何地方编造数据事实区没有的数值
5. 不得编造原始数据中不存在的事件或新闻
6. 数据不可用时，使用「⚠️ 数据不可用：<具体说明缺失内容>」标注

## ⚠️ 数据限制最终声明
在分析末尾，请以清单形式列出本次分析中所有「数据不可用」的维度和具体缺失内容，确保阅读者了解分析的局限性。

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
            {"role": "system", "content": "你是一位资深基金研究分析师，专注于中国公募基金的事件监测、风险预警和信息披露分析。你严格遵守数据限制，绝不编造信息，对无法获取的数据如实标注。"},
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
                    sp["agent_name"] = "fund_event"
                    sp["as_of_date"] = current_date
                except Exception:
                    pass
            if sp is None:
                from src.utils.analysis_package_builder import text_to_signal_pack
                sp = text_to_signal_pack(final_output, "fund_event", current_date)
            current_data["fund_event_signal_pack"] = sp
            phase2_elapsed = time.time() - phase2_start
            logger.info(f"FundEventAgent: Phase 2 完成 ({phase2_elapsed:.1f}s, {len(final_output)} 字符)")
            llm_success = True
        except asyncio.TimeoutError:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundEventAgent: LLM 超时 ({phase2_elapsed:.0f}s)")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析与风险评估区\n\n⚠️ LLM分析超时，以上为原始数据，请人工分析。\n\n## ⚠️ 数据限制最终声明\n\n- 因 LLM 超时，本次分析未能完成，以上为原始数据汇总。"
            llm_success = False
        except Exception as llm_err:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundEventAgent: LLM 失败: {llm_err}")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析与风险评估区\n\n⚠️ LLM分析失败({str(llm_err)[:100]})，以上为原始数据，请人工分析。\n\n## ⚠️ 数据限制最终声明\n\n- 因 LLM 分析失败，本次分析未能完成，以上为原始数据汇总。"
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
            interaction_type="two_phase_fund_event_classification",
            input_messages=[{"role": "user", "content": analysis_prompt[:5000]}],
            output_content=final_output,
            model_config=model_config_log,
            execution_time=phase2_elapsed,
        )

        total_time = time.time() - agent_start_time
        logger.info(f"{SUCCESS_ICON if llm_success else WAIT_ICON} FundEventAgent: 分析完成 (总耗时 {total_time:.1f}s, Phase1={phase1_elapsed:.1f}s, Phase2={phase2_elapsed:.1f}s)")

        # 更新状态
        current_data["fund_event"] = final_output
        if not skip_cache and cache_date and cache_code:
            write_cache("fund_event", cache_code, cache_date, final_output)
            if "fund_event_signal_pack" in current_data:
                from src.utils.cache_utils import write_signal_pack_cache
                write_signal_pack_cache("fund_event", cache_code, cache_date, current_data["fund_event_signal_pack"])
        current_metadata["fund_event_executed"] = True
        current_metadata["fund_event_timestamp"] = str(time.time())
        current_metadata["fund_event_execution_time"] = f"{total_time:.2f} seconds"

        execution_logger.log_agent_complete(agent_name, {
            "fund_event_length": len(final_output),
            "fund_event_preview": final_output[:500],
            "phase1_time": phase1_elapsed,
            "phase2_time": phase2_elapsed,
            "total_time": total_time,
            "tools_queried": len(labels),
            "tools_with_data": success_count,
            "llm_success": llm_success,
        }, total_time, True)

        return {
            "data": current_data,
            "messages": current_messages + [{"role": "assistant", "content": "基金事件分析已完成"}],
            "metadata": current_metadata,
        }

    except Exception as e:
        logger.error(f"{ERROR_ICON} FundEventAgent: Error: {e}", exc_info=True)
        current_data["fund_event_error"] = f"Error: {e}"
        current_data["fund_event"] = f"基金事件分析过程中出现错误: {str(e)}"
        current_metadata["fund_event_agent_error"] = str(e)
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}


# 本地测试函数
async def test_fund_event_agent():
    """基金事件分析 Agent 的测试函数"""
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
            "query": "分析黄金ETF华安的基金事件和风险",
            "fund_code": "of.518880",
            "fund_name": "黄金ETF华安",
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat()
        },
        metadata={}
    )

    result = await fund_event_agent(test_state)
    print("Fund Event Analysis Result:")
    print(result.get("data", {}).get("fund_event", "No analysis found"))
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fund_event_agent())
