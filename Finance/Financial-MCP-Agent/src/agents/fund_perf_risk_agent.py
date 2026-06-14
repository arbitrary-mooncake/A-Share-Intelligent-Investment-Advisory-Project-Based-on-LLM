"""
Fund Performance & Risk Agent: 两阶段架构 — 并行数据预取 + 单次 LLM 深度分析。
Phase 1: asyncio.gather 并行获取基金净值/K线/基本信息数据
Phase 2: 将所有原始数据喂给 LLM 一次性完成业绩与风险评价（thinking 开启）
"""
import asyncio
import os
import time
from datetime import datetime
from typing import Dict, Any, List

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

# 基金业绩与风险评价工具白名单
FUND_PERF_RISK_TOOLS = [
    "tushare_fund_nav",      # NAV history for return/risk calculation (primary)
    "tushare_fund_daily",    # Daily price for trading metrics (ETFs)
    "tushare_fund_adj",      # Adjustment factors for accurate returns
    "tushare_kline",         # K-line for technical overlay
    "tushare_fund_basic",    # Fund basic info / benchmark
]


def _extract_code(fund_code: str) -> str:
    """提取纯基金代码（去除交易所前缀和后缀）"""
    return fund_code.replace("sh.", "").replace("sz.", "").replace(
        ".SH", "").replace(".SZ", "").replace(".OF", "").strip()


async def _noop_result(text: str) -> str:
    """返回固定文本的占位协程，用于工具不可用时的占位"""
    return text


async def _call_tool_safe(tool, kwargs: dict, label: str) -> str:
    """调用单个 MCP 工具，带超时和异常保护"""
    try:
        result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=TOOL_TIMEOUT)
        text = str(result).strip()
        if len(text) > 20:
            logger.info(f"{SUCCESS_ICON} FundPerfRiskAgent: {label} 获取成功 ({len(text)} 字符)")
            return text
        logger.warning(f"FundPerfRiskAgent: {label} 返回过短 ({len(text)} 字符)")
        return f"[{label}] 数据不可用（返回过短）"
    except asyncio.TimeoutError:
        logger.warning(f"FundPerfRiskAgent: {label} 超时({TOOL_TIMEOUT}s)")
        return f"[{label}] 数据不可用（超时）"
    except Exception as e:
        logger.warning(f"FundPerfRiskAgent: {label} 调用失败: {e}")
        return f"[{label}] 数据不可用（调用失败: {str(e)[:80]}）"


async def fund_perf_risk_agent(state: AgentState) -> AgentState:
    """
    两阶段基金业绩与风险评价：
    Phase 1: 并行获取全部白名单工具的数据
    Phase 2: 单次 LLM 深度分析（Kimi K2.6, thinking=enabled）
    """
    logger.info(f"{WAIT_ICON} FundPerfRiskAgent: Starting two-phase performance & risk analysis.")

    execution_logger = get_execution_logger()
    agent_name = "fund_perf_risk_agent"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})

    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    cache_code = current_data.get("fund_code", "")

    # 缓存检查（TTL=3天，由 cache_utils 控制）
    if not skip_cache and cache_date and cache_code:
        cached = read_cache("fund_perf_risk", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} FundPerfRiskAgent: 命中缓存，跳过分析 ({cache_code})")
            current_data["fund_perf_risk"] = cached
            current_metadata["fund_perf_risk_executed"] = True
            current_metadata["fund_perf_risk_cached"] = True
            return {"data": current_data,
                    "messages": current_messages + [{"role": "assistant", "content": "基金业绩与风险分析已完成（缓存）"}],
                    "metadata": current_metadata}

    execution_logger.log_agent_start(agent_name, {
        "fund_code": cache_code,
        "fund_name": current_data.get("fund_name"),
        "input_data_keys": list(current_data.keys()),
        "architecture": "two-phase",
    })

    fund_code = current_data.get("fund_code", "")
    fund_name = current_data.get("fund_name", "")
    current_time_info = current_data.get("current_time_info", "未知时间")
    current_date = current_data.get("current_date", "未知日期")
    clean_code = _extract_code(fund_code) if fund_code else ""

    if not fund_code or not clean_code:
        logger.error(f"{ERROR_ICON} FundPerfRiskAgent: Fund code is missing.")
        current_data["fund_perf_risk_error"] = "Fund code is missing."
        execution_logger.log_agent_complete(agent_name, current_data, 0, False, "Fund code is missing")
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    agent_start_time = time.time()

    try:
        # 模型配置：Kimi K2.6
        model_cfg = get_model_config_for_agent("fund_perf_risk_agent", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]

        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} FundPerfRiskAgent: Missing OpenAI environment variables.")
            current_data["fund_perf_risk_error"] = "Missing OpenAI environment variables."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "Missing env vars")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        # ── Phase 1: 并行数据预取 ──────────────────────────────
        logger.info(f"{WAIT_ICON} FundPerfRiskAgent: Phase 1 — 并行获取 {len(FUND_PERF_RISK_TOOLS)} 个工具数据...")
        phase1_start = time.time()

        try:
            all_tools = await get_mcp_tools(tool_filter=FUND_PERF_RISK_TOOLS)
        except Exception as e:
            logger.error(f"{ERROR_ICON} FundPerfRiskAgent: 获取 MCP 工具失败: {e}")
            all_tools = []

        if not all_tools:
            logger.error(f"{ERROR_ICON} FundPerfRiskAgent: No MCP tools available.")
            current_data["fund_perf_risk_error"] = "No MCP tools available."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "No tools")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        tool_map = {t.name: t for t in all_tools}
        logger.info(f"{SUCCESS_ICON} FundPerfRiskAgent: 已加载 {len(all_tools)}/{len(FUND_PERF_RISK_TOOLS)} 个工具")

        # 构建并行任务列表
        tasks = []
        labels = []
        tool_infos = []  # (tool, kwargs)，占位任务为 None

        def _add(task, label, ti=None):
            tasks.append(task)
            labels.append(label)
            tool_infos.append(ti)

        # Helper: 注册占位任务（工具不可用）
        def _placeholder(label, msg):
            labels.append(label)
            tasks.append(_noop_result(msg))
            tool_infos.append(None)

        # --- NAV 净值数据（主要数据源，5年历史） ---
        if "tushare_fund_nav" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_nav"], {"code": clean_code, "days": 1250}, "基金净值(5年)"),
                 "基金净值(5年)", (tool_map["tushare_fund_nav"], {"code": clean_code, "days": 1250}))
        else:
            _placeholder("基金净值(5年)", "[tushare_fund_nav] 工具不可用")

        # --- K线数据（技术面叠加，5年） ---
        if "tushare_kline" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_kline"], {"code": clean_code, "days": 1250}, "K线数据(5年)"),
                 "K线数据(5年)", (tool_map["tushare_kline"], {"code": clean_code, "days": 1250}))
        else:
            _placeholder("K线数据(5年)", "[tushare_kline] 工具不可用")

        # --- 基金基本信息（含业绩基准） ---
        if "tushare_fund_basic" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_basic"], {"code": clean_code}, "基金基本信息"),
                 "基金基本信息", (tool_map["tushare_fund_basic"], {"code": clean_code}))
        else:
            _placeholder("基金基本信息", "[tushare_fund_basic] 工具不可用")

        # --- ETF日线数据（兼容） ---
        if "tushare_fund_daily" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_daily"], {"code": clean_code, "days": 1250}, "ETF日线(5年)"),
                 "ETF日线(5年)", (tool_map["tushare_fund_daily"], {"code": clean_code, "days": 1250}))
        else:
            _placeholder("ETF日线(5年)", "[tushare_fund_daily] 工具不可用")

        # --- 复权因子 ---
        if "tushare_fund_adj" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_adj"], {"code": clean_code, "days": 1250}, "复权因子(5年)"),
                 "复权因子(5年)", (tool_map["tushare_fund_adj"], {"code": clean_code, "days": 1250}))
        else:
            _placeholder("复权因子(5年)", "[tushare_fund_adj] 工具不可用")

        # 并行执行所有任务（第一轮）
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as gather_err:
            logger.error(f"{ERROR_ICON} FundPerfRiskAgent: Phase 1 并行调用异常: {gather_err}")
            results = [f"并行调用异常: {gather_err}"] * len(tasks)

        # 空数据重试
        results = await retry_failed_fetches(
            results, tool_infos, labels, _call_tool_safe,
            agent_label="FundPerfRiskAgent",
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
        logger.info(f"{SUCCESS_ICON} FundPerfRiskAgent: Phase 1 完成 ({phase1_elapsed:.1f}s, {success_count}/{total_real} 个工具有效数据)")

        # 聚合数据
        data_sections = []
        for label, result in zip(labels, safe_results):
            data_sections.append(f"### [{label}]\n{result}")
        raw_data_text = "\n\n".join(data_sections)

        # ── Phase 2: LLM 深度分析 ──────────────────────────────
        logger.info(f"{WAIT_ICON} FundPerfRiskAgent: Phase 2 — LLM 深度分析 (model={model_name}, thinking=enabled)...")
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

        analysis_prompt = f"""请以资深基金分析师的标准，对{fund_name}（基金代码：{fund_code}）进行全面的业绩与风险评价分析。

当前时间：{current_time_info}
当前日期：{current_date}

## 原始数据

以下是通过工具获取的全部原始数据，请基于这些数据进行深度分析：

{raw_data_text}

## 分析要求

请进行以下维度的深度量化分析（每个维度都需要基于上述原始数据，引用具体数字）：

### 一、收益表现分析
1. **近1年年化收益率**：基于最近250个交易日的净值数据计算
2. **近3年年化收益率**：基于最近750个交易日的净值数据计算（如数据不足3年则标注数据不足）
3. **成立以来年化收益率**：基于全部可用净值数据计算
4. **累计总回报**：从数据起始日至今的累计收益率
5. **收益分布特征**：月度/季度收益的均值、中位数、标准差、偏度、峰度
6. **正收益月份占比**：统计月度收益为正的比例
7. **滚动1年收益**：滚动1年收益的均值、中位数、最小值、最大值，用于评估收益稳定性

### 二、风险指标分析
1. **最大回撤（Maximum Drawdown）**：
   - 最大回撤幅度及其发生时间区间（起止日期）
   - 回撤后恢复到前高所需天数（恢复时间）
   - 当前是否处于回撤状态及当前回撤幅度
2. **年化波动率（Annualized Volatility）**：基于日收益率年化计算的波动率
3. **下行标准差（Downside Deviation）**：仅计算负收益的波动
4. **最大回撤恢复能力**：
   - 历次5%以上回撤的平均恢复天数
   - 最长恢复天数
   - 回撤深度分布统计（<5% / 5-10% / 10-20% / >20%各档位次数）

### 三、风险调整后收益
1. **夏普比率（Sharpe Ratio）**：假设无风险利率为2%（年化），计算公式 = (年化收益率 - 2%) / 年化波动率
2. **Calmar比率（Calmar Ratio）**：年化收益率 / 最大回撤的绝对值
3. **索提诺比率（Sortino Ratio）**：如可计算（基于下行标准差），公式 = (年化收益率 - 2%) / 下行标准差
4. **信息比率（Information Ratio）**：相对于业绩基准的超额收益 / 跟踪误差（如基准数据可用）
5. **收益-风险比综合评估**：将上述各项比率与同类基金平均水平对比

### 四、相对基准表现
1. **超额收益（Excess Return / Alpha）**：基金累计收益减去基准累计收益
2. **跟踪误差（Tracking Error）**：基金与基准日收益差异的年化标准差
3. **信息比率**：年化超额收益 / 跟踪误差
4. **上行/下行捕获率**：
   - 上行捕获率：基准上涨时基金的相对表现
   - 下行捕获率：基准下跌时基金的相对表现
5. **相对强弱**：基金相对基准的累计超额收益曲线趋势

### 五、不同市场阶段表现
请根据A股主要市场阶段（如数据覆盖），将净值数据分为以下阶段分别分析：
1. **牛市阶段**（如2019-2021年、2024年924行情等）：期间收益率、超额收益
2. **熊市阶段**（如2018年、2022年等）：期间收益率、最大回撤
3. **震荡市阶段**（如2023年等）：期间收益率、波动率
4. **各阶段风控表现对比**：不同市场环境下的回撤控制能力

### 六、滚动排名稳定性（如可比数据可用）
1. 滚动1年收益率在同类基金中的分位数变化趋势
2. 排名稳定性评估（波动大 vs 持续稳定）
3. 如果同类排名数据不可用，请说明并基于绝对收益分析

### 七、综合评分

请给出以下量化评分（满分100分）：

1. **业绩评分（满分100分）**：
   - 基于近1年/3年/成立以来年化收益表现
   - 收益稳定性（滚动收益的一致性）
   - 相对于基准的超额收益

2. **风险评分（满分100分）**：
   - 基于最大回撤水平（越小越好）
   - 波动率控制（越小越好）
   - 回撤恢复能力（恢复越快越好）
   - 下行保护能力（熊市表现）

3. **综合评分**：业绩与风险的平衡得分

请明确给出评分及评分依据。

---

## 重要限制

- 请专注于基金历史业绩和风险指标的量化分析，不要分析持仓标的的基本面或新闻事件
- 分析必须有数据支撑，引用上述原始数据中的具体净值/价格数字进行计算
- 如果某些数据无法获取（如基准数据、同类排名数据），请明确说明原因，不要编造
- **严禁使用模型训练数据中的知识来补充数据事实** — 所有计算必须基于上述原始数据
- 如果原始数据不足以支撑某项分析，请如实标注"数据不足，无法分析"

## ⚠️ 监管合规声明

根据中国证监会《公开募集证券投资基金销售机构监督管理办法》及相关法规要求，基金分析报告需明确提示：

**"历史业绩不代表未来表现（Past performance does not predict future results）。基金的过往业绩及其净值高低并不预示其未来业绩表现。投资者应当认真阅读基金合同、招募说明书等法律文件，了解基金的风险收益特征，根据自身风险承受能力审慎决策。"**

---

## ⛔ 输出格式要求（防幻觉机制）

请将分析输出严格分为两个区域：

## 📊 数据事实区
列出上述原始数据中的关键客观数据，每条标注数据来源标签（如 [基金净值(5年)]）：
- [标签] 具体数值（如：近1年年化收益=15.2%，年化波动率=18.5%，最大回撤=-25.3%）
- [标签] 关键时间节点（如：最大回撤区间=2022-01-05至2022-04-26）
- ...
如果某项数据在上述原始数据中标注为「数据不可用」，必须在此如实声明，不得推测。

## 🔍 分析判断区
基于上述数据事实进行分析和推断。每个判断必须：
1. 引用数据事实区的具体数值
2. 使用「【基于数据的计算】」或「【分析师经验推断】」标注计算/推断性质
3. 如果某个结论无法从数据中直接计算得出，必须声明「此为分析师推断」
4. 不得在任何地方编造数据事实区没有的数值
5. 不得编造原始数据中不存在的市场阶段事件或新闻"""

        messages = [
            {"role": "system", "content": "你是一位资深基金分析师，专注于公募基金/ETF的业绩归因和量化风险评价。擅长使用净值数据和市场数据进行科学的业绩分析，严格遵守监管合规要求，所有分析均有数据支撑。"},
            {"role": "user", "content": analysis_prompt},
        ]

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=float(LLM_TIMEOUT)
            )
            final_output = response.content.strip() if hasattr(response, 'content') else str(response)
            phase2_elapsed = time.time() - phase2_start
            logger.info(f"FundPerfRiskAgent: Phase 2 完成 ({phase2_elapsed:.1f}s, {len(final_output)} 字符)")
            llm_success = True
        except asyncio.TimeoutError:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundPerfRiskAgent: LLM 超时 ({phase2_elapsed:.0f}s)")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析判断区\n\n⚠️ LLM分析超时，以上为原始数据，请人工分析。\n"
            llm_success = False
        except Exception as llm_err:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundPerfRiskAgent: LLM 失败: {llm_err}")
            final_output = f"## 📊 数据事实区\n\n{raw_data_text[:3000]}\n\n## 🔍 分析判断区\n\n⚠️ LLM分析失败({str(llm_err)[:100]})，以上为原始数据，请人工分析。\n"
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
        logger.info(f"{SUCCESS_ICON if llm_success else WAIT_ICON} FundPerfRiskAgent: 分析完成 (总耗时 {total_time:.1f}s, Phase1={phase1_elapsed:.1f}s, Phase2={phase2_elapsed:.1f}s)")

        # 更新状态
        current_data["fund_perf_risk"] = final_output
        if not skip_cache and cache_date and cache_code:
            write_cache("fund_perf_risk", cache_code, cache_date, final_output)
        current_metadata["fund_perf_risk_executed"] = True
        current_metadata["fund_perf_risk_timestamp"] = str(time.time())
        current_metadata["fund_perf_risk_execution_time"] = f"{total_time:.2f} seconds"

        execution_logger.log_agent_complete(agent_name, {
            "fund_perf_risk_length": len(final_output),
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
            "messages": current_messages + [{"role": "assistant", "content": "基金业绩与风险分析已完成"}],
            "metadata": current_metadata,
        }

    except Exception as e:
        logger.error(f"{ERROR_ICON} FundPerfRiskAgent: Error: {e}", exc_info=True)
        current_data["fund_perf_risk_error"] = f"Error: {e}"
        current_data["fund_perf_risk"] = f"基金业绩与风险分析过程中出现错误: {str(e)}"
        current_metadata["fund_perf_risk_error"] = str(e)
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}


# 本地测试函数
async def test_fund_perf_risk_agent():
    """基金业绩与风险评价 Agent的测试函数"""
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
            "fund_code": "510050",
            "fund_name": "华夏上证50ETF",
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat()
        },
        metadata={}
    )

    result = await fund_perf_risk_agent(test_state)
    print("Fund Performance & Risk Analysis Result:")
    print(result.get("data", {}).get("fund_perf_risk", "No analysis found"))
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fund_perf_risk_agent())
