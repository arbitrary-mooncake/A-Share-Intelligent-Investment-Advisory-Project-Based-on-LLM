"""
FundBenchmark Agent: 两阶段架构 — 并行数据预取 + 单次 LLM 深度分析。
Phase 1: asyncio.gather 并行获取白名单工具的数据
Phase 2: 将所有原始数据喂给 LLM 一次性完成基准一致性/风格漂移分析（thinking 开启，跟踪误差/信息比率/风格漂移需要深度推理）
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

# 基准一致性/风格漂移分析白名单
FUND_BENCHMARK_TOOLS = [
    "tushare_fund_basic",       # 基金基本信息（含benchmark基准字段，如：沪深300指数收益率×80%+中证全债指数收益率×20%）
    "tushare_fund_nav",         # 基金净值（用于计算基金实际收益）
    "tushare_fund_daily",       # 基金日线行情
    "tushare_index_daily",      # 基准指数日线数据（用于对比）
    "tushare_etf_index",        # ETF跟踪指数信息
]

# 基准指数代码映射：常见基准名 → Tushare 指数代码
BENCHMARK_INDEX_MAP = {
    "沪深300": "000300.SH",
    "中证500": "000905.SH",
    "中证800": "000906.SH",
    "中证1000": "000852.SH",
    "上证50": "000016.SH",
    "上证180": "000010.SH",
    "上证综指": "000001.SH",
    "深证成指": "399001.SZ",
    "深证100": "399330.SZ",
    "创业板指": "399006.SZ",
    "创业板50": "399673.SZ",
    "科创50": "000688.SH",
    "科创100": "000698.SH",
    "中证全债": "H11001.CSI",
    "中证国债": "H11006.CSI",
    "中证信用": "H11073.CSI",
    "中证转债": "000832.SH",
    "恒生指数": "HSI",
    "标普500": "SPX",
    "纳斯达克100": "NDX",
    "中证红利": "000922.CSI",
    "中证银行": "399986.SZ",
    "中证军工": "399967.SZ",
    "中证医药": "000933.SH",
    "中证消费": "000932.SH",
    "中证科技": "931087.CSI",
    "中证新能源": "399808.SZ",
    "中证半导体": "990001.CSI",
    "中证酒": "399987.SZ",
    "国证芯片": "980017.CNI",
    "中证传媒": "399971.SZ",
    "中证环保": "000827.SH",
    "中证煤炭": "399998.SZ",
    "中证有色": "000823.SH",
    "中证钢铁": "930606.CSI",
    "中证农业": "000949.SH",
    "中证汽车": "930607.CSI",
}


def _extract_fund_code(fund_code: str) -> str:
    """提取纯数字/字母基金代码，去除前缀后缀"""
    code = str(fund_code).strip()
    # 去除常见前缀后缀
    for prefix in ["sh.", "sz.", "of.", "SH.", "SZ.", "OF.", "bj."]:
        if code.lower().startswith(prefix.lower()):
            code = code[len(prefix):]
    for suffix in [".OF", ".SH", ".SZ", ".BJ", ".of", ".sh", ".sz", ".bj"]:
        if code.lower().endswith(suffix.lower()):
            code = code[:-(len(suffix))]
    return code.strip()


def _parse_benchmark_indices(benchmark_text: str) -> List[Dict[str, str]]:
    """
    从基金基准文本中解析出包含的指数名称及大致权重。

    示例输入: "沪深300指数收益率×80%+中证全债指数收益率×20%"
    输出: [{"name": "沪深300指数", "weight": 0.8}, {"name": "中证全债指数", "weight": 0.2}]
    """
    import re
    if not benchmark_text or benchmark_text == "未知":
        return []

    indices = []
    # 匹配模式：指数名 + 收益率/指数 × 百分比(%)
    # 更宽松的匹配，先提取所有"×数字%"的组合
    parts = re.split(r'\+', benchmark_text)
    for part in parts:
        part = part.strip()
        # 提取权重
        weight_match = re.search(r'[×xX]\s*(\d+(?:\.\d+)?)\s*%', part)
        weight = float(weight_match.group(1)) / 100.0 if weight_match else 0.0

        # 提取指数名（在×之前的部分，去除"收益率""指数收益率"等后缀）
        name_part = re.split(r'[×xX]', part)[0].strip() if '×' in part or 'x' in part else part
        name_part = re.sub(r'(指数)?收益率.*$', '', name_part).strip()
        # 如果还有空，直接用整个part
        if not name_part:
            name_part = part

        indices.append({
            "name": name_part,
            "weight": weight,
            "raw": part,
        })

    return indices


def _match_index_code(index_name: str) -> Optional[str]:
    """根据指数名称匹配 Tushare 指数代码"""
    # 精确匹配
    if index_name in BENCHMARK_INDEX_MAP:
        return BENCHMARK_INDEX_MAP[index_name]

    # 模糊匹配：遍历映射表
    for key, code in BENCHMARK_INDEX_MAP.items():
        if key in index_name or index_name in key:
            return code

    # 更宽松的匹配：去掉"指数"后缀后再试
    clean_name = index_name.replace("指数", "").replace("收益率", "").strip()
    for key, code in BENCHMARK_INDEX_MAP.items():
        if key in clean_name or clean_name in key:
            return code

    return None


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
            logger.info(f"{SUCCESS_ICON} FundBenchmarkAgent: {label} 获取成功 ({len(text)} 字符)")
            await set_cached_tool_result(tool_name, kwargs, text)
            return text
        logger.warning(f"FundBenchmarkAgent: {label} 返回过短 ({len(text)} 字符)")
        return f"[{label}] 数据不可用（返回过短）"
    except asyncio.TimeoutError:
        logger.warning(f"FundBenchmarkAgent: {label} 超时({TOOL_TIMEOUT}s)")
        return f"[{label}] 数据不可用（超时）"
    except Exception as e:
        logger.warning(f"FundBenchmarkAgent: {label} 调用失败: {e}")
        return f"[{label}] 数据不可用（调用失败: {str(e)[:80]}）"


async def fund_benchmark_agent(state: AgentState) -> AgentState:
    """
    两阶段基准一致性/风格漂移分析：
    Phase 1: 并行获取基金基本信息和基准指数数据
    Phase 2: 单次 LLM 深度分析（Model 3: Qwen3.7-Plus, thinking=enabled，跟踪误差/信息比率/风格漂移需深度推理）

    分析维度：
    - 业绩比较基准定义
    - 基准一致性（基金表现是否与基准匹配）
    - 跟踪误差（指数基金）或主动偏离度
    - 信息比率
    - 风格稳定性与风格漂移检测
    - 2026年证监会基准指引合规性
    """
    logger.info(f"{WAIT_ICON} FundBenchmarkAgent: 启动两阶段基准一致性分析。")

    execution_logger = get_execution_logger()
    agent_name = "fund_benchmark"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})

    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    # 优先使用 fund_code，回退到 stock_code
    cache_code = current_data.get("fund_code") or current_data.get("stock_code", "")

    # 缓存检查（TTL=7天，由 cache_utils 控制）
    if not skip_cache and cache_date and cache_code:
        cached = read_cache("fund_benchmark", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} FundBenchmarkAgent: 命中缓存，跳过分析 ({cache_code})")
            current_data["fund_benchmark"] = cached
            from src.utils.cache_utils import read_signal_pack_cache
            cached_sp = read_signal_pack_cache("fund_benchmark", cache_code, cache_date)
            if cached_sp:
                current_data["fund_benchmark_signal_pack"] = cached_sp
            current_metadata["fund_benchmark_executed"] = True
            current_metadata["fund_benchmark_cached"] = True
            return {"data": current_data,
                    "messages": current_messages + [{"role": "assistant", "content": "基准一致性分析已完成（缓存）"}],
                    "metadata": current_metadata}

    execution_logger.log_agent_start(agent_name, {
        "fund_code": cache_code,
        "fund_name": current_data.get("fund_name"),
        "current_date": cache_date,
        "input_data_keys": list(current_data.keys()),
        "architecture": "two-phase",
    })

    agent_start_time = time.time()

    try:
        # 模型配置：Model 3（Qwen3.7-Plus）
        model_cfg = get_model_config_for_agent("fund_benchmark_agent", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]

        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} FundBenchmarkAgent: 缺少 OpenAI 环境变量。")
            current_data["fund_benchmark_error"] = "缺少 OpenAI 环境变量。"
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "Missing env vars")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        fund_code = current_data.get("fund_code") or current_data.get("stock_code", "Unknown")
        fund_name = current_data.get("fund_name") or current_data.get("company_name", "Unknown")
        current_time_info = current_data.get("current_time_info", "未知时间")
        current_date = current_data.get("current_date", "未知日期")
        clean_code = _extract_fund_code(fund_code) if fund_code else ""

        # ── Phase 1: 并行数据预取 ──────────────────────────────
        logger.info(f"{WAIT_ICON} FundBenchmarkAgent: Phase 1 — 并行获取 {len(FUND_BENCHMARK_TOOLS)} 个工具数据...")
        phase1_start = time.time()

        try:
            all_tools = await get_mcp_tools(tool_filter=FUND_BENCHMARK_TOOLS)
        except Exception as e:
            logger.error(f"{ERROR_ICON} FundBenchmarkAgent: 获取 MCP 工具失败: {e}")
            all_tools = []

        if not all_tools:
            logger.error(f"{ERROR_ICON} FundBenchmarkAgent: 没有可用的 MCP 工具。")
            current_data["fund_benchmark_error"] = "没有可用的 MCP 工具。"
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "No tools")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        tool_map = {t.name: t for t in all_tools}
        logger.info(f"{SUCCESS_ICON} FundBenchmarkAgent: 已加载 {len(all_tools)}/{len(FUND_BENCHMARK_TOOLS)} 个工具")

        # 构建并行任务列表
        tasks = []
        labels = []
        tool_infos = []  # (tool, kwargs)，占位任务为 None

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
            # fund_basic 通常用 ts_code（如 159919.SZ）查询
            _add(_call_tool_safe(tool_map["tushare_fund_basic"], {"code": clean_code}, "基金基本信息"),
                 "基金基本信息",
                 (tool_map["tushare_fund_basic"], {"code": clean_code}))
        else:
            _placeholder("基金基本信息", "[tushare_fund_basic] 工具不可用")

        # --- 基金净值 ---
        if "tushare_fund_nav" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_nav"], {"code": clean_code, "days": 500}, "基金净值(500天)"),
                 "基金净值",
                 (tool_map["tushare_fund_nav"], {"code": clean_code, "days": 500}))
        else:
            _placeholder("基金净值", "[tushare_fund_nav] 工具不可用")

        # --- 基金日线 ---
        if "tushare_fund_daily" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_fund_daily"], {"code": clean_code, "days": 500}, "基金日线(500天)"),
                 "基金日线",
                 (tool_map["tushare_fund_daily"], {"code": clean_code, "days": 500}))
        else:
            _placeholder("基金日线", "[tushare_fund_daily] 工具不可用")

        # --- ETF跟踪指数 ---
        if "tushare_etf_index" in tool_map:
            _add(_call_tool_safe(tool_map["tushare_etf_index"], {"code": clean_code}, "ETF跟踪指数信息"),
                 "ETF跟踪指数",
                 (tool_map["tushare_etf_index"], {"code": clean_code}))
        else:
            _placeholder("ETF跟踪指数", "[tushare_etf_index] 工具不可用")

        # --- 基准指数日线数据 ---
        # 这里我们先占位，等 fund_basic 数据返回后从基准文本中解析出指数代码再请求
        # 但因为我们使用并行架构，需要在 Phase 2 的 LLM 提示中包含基准信息
        # 同时，我们会尝试用 fund_basic 的返回结果来推断基准指数代码

        # 先尝试获取 index_daily 工具
        index_daily_tool = tool_map.get("tushare_index_daily")
        kline_tool = tool_map.get("tushare_kline")

        if not index_daily_tool and not kline_tool:
            _placeholder("基准指数日线", "[tushare_index_daily] 和 [tushare_kline] 均不可用，无法获取基准指数行情")
        else:
            # 占位：实际指数代码需从 fund_basic 返回的 benchmark 字段解析
            # 这里先占位，让 LLM 在 Phase 2 基于 fund_basic 数据进行判断
            _placeholder("基准指数日线(待解析)", "[提示] 基准指数代码需从基金基本信息(benchmark字段)中解析。"
                         "请LLM在Phase 2根据fund_basic返回的业绩基准文本自行匹配指数。"
                         "如fund_basic返回了基准文本（如：沪深300×80%+中证全债×20%），"
                         "则对应指数代码为：沪深300→000300.SH，中证全债→H11001.CSI")

        # 并行执行所有任务（第一轮）
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as gather_err:
            logger.error(f"{ERROR_ICON} FundBenchmarkAgent: Phase 1 并行调用异常: {gather_err}")
            results = [f"并行调用异常: {gather_err}"] * len(tasks)

        # 空数据重试
        results = await retry_failed_fetches(
            results, tool_infos, labels, _call_tool_safe,
            agent_label="FundBenchmarkAgent",
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
        logger.info(f"{SUCCESS_ICON} FundBenchmarkAgent: Phase 1 完成 ({phase1_elapsed:.1f}s, {success_count}/{total_real} 个工具有效数据)")

        # 聚合数据
        data_sections = []
        for label, result in zip(labels, safe_results):
            data_sections.append(f"### [{label}]\n{result}")
        raw_data_text = "\n\n".join(data_sections)

        # ── Phase 2: LLM 基准一致性分析 ──────────────────────────────
        logger.info(f"{WAIT_ICON} FundBenchmarkAgent: Phase 2 — LLM 基准一致性分析 (model={model_name}, thinking=enabled)...")
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

        analysis_prompt = f"""请以基金研究分析师的标准，对"{fund_name}"（基金代码：{fund_code}）进行基准一致性与风格漂移分析。

当前时间：{current_time_info}
当前日期：{current_date}

## 原始数据

以下是通过工具获取的全部原始数据：

{raw_data_text}

## 分析要求

请基于上述原始数据进行以下维度的深度分析（每个维度必须引用原始数据中的具体数值，使用「【基于数据的推断】」标注推断性质）：

### 1. 业绩比较基准定义
- 从 tushare_fund_basic 的 benchmark 字段中提取该基金的业绩比较基准
- 明确基准的构成：包含哪些指数、各自的权重
- 分析该基准是否合理匹配基金的投资目标和策略类型
- 按2026年证监会基准指引要求，判断基准是否"反映产品投资目标并约束管理人行为"

### 2. 基准一致性分析
- 对比基金实际净值走势（fund_nav/fund_daily）与业绩基准的理论走势
- 计算或估算基金回报与基准回报的偏差方向和幅度
- 判断基金表现是否在逻辑上与基准保持一致
- 分析超额收益或负超额收益的来源

### 3. 跟踪误差评估
- **指数基金/ETF**：计算或估算跟踪误差（基金净值收益率与基准指数收益率之差的标准差）
  - 日跟踪偏离度平均值
  - 年化跟踪误差（理想 < 2%，一般 < 4%）
  - 分析跟踪误差的来源（费率、打新收益、大额申赎、现金拖累等）
- **主动管理基金**：计算或估算主动偏离度（active share 等效指标）
  - 行业偏离度
  - 风格暴露偏离

### 4. 信息比率
- 计算或估算信息比率 = 年化超额收益 / 年化跟踪误差
- 信息比率 > 0.5 为良好，> 1.0 为优秀
- 判断超额收益是否具备持续性和统计显著性

### 5. 风格稳定性分析
- 基于基金历史净值数据，分析基金的风格暴露是否稳定
- 检查是否存在明显的风格转换（如从价值转向成长、从大盘转向小盘）
- 分析基金在不同市场环境下的表现一致性

### 6. 风格漂移检测
- 判断是否存在显著的风格漂移
- **量化漂移指标**：
  - 滚动跟踪误差是否在近期显著增大
  - 基金收益率与基准收益率的相关性是否下降
  - 特定阶段（如牛熊转换期）基金表现是否与基准背离
- 风格漂移的严重程度评估（轻微/中度/严重）

### 7. 偏离合理性判断
- 如果存在偏离，分析偏离是否可被基金策略合理解释
- 区分"主动管理创造alpha"和"风格漂移"的边界
- 评估偏离是否在基金合同允许范围内

### 8. 短期排名压力 vs 产品定位
- 分析基金是否存在因短期业绩排名压力而偏离基准的行为
- 评估基金公司是否牺牲长期基准一致性换取短期排名
- 对该基金"说到做到"的文化和制度进行评估

### 9. "说到做到"综合评估
- 综合上述所有维度，对基金是否"说到做到"给出综合判断
- 即：基金的实际投资行为是否与招募说明书中的投资目标和基准保持一致
- 给出最终的基准一致性评级（优秀/良好/一般/较差/严重漂移）

### 10. 监管合规视角（2026年CSRC基准指引）
- 根据2026年证监会发布的公募基金业绩比较基准指引：
  - 基准应当反映产品的投资目标和策略
  - 基准指数应当具备可投资性和代表性
  - 基准变更需充分披露并经持有人大会审议
  - 实际投资不应长期大幅偏离基准
- 对照上述要求，逐条评估该基金的合规情况

## 重要限制
- 请专注于基准一致性和风格漂移分析，不要分析新闻信息
- 分析必须有数据支撑，引用上述原始数据中的具体数值，避免空洞的定性描述
- 如果某些数据无法获取（如无fund_basic、无fund_nav等），请说明原因并基于可用数据提供分析
- 不要使用模型训练数据中的知识来补充数据事实
- 如果数据不足以完成某项分析，必须如实说明"数据不足，无法完成该维度分析"

## 输出格式要求
请将分析输出严格分为两个区域：

## 数据事实区
列出上述原始数据中的关键客观数据：
- 基金基本信息：基金代码、名称、类型、投资目标
- 业绩比较基准：完整基准文本、各成分及权重
- 净值数据关键点：最新净值、近期收益率、波动率
- 基准指数关键点：基准指数涨跌幅、波动率
- 如果某项数据标注为「数据不可用」，必须在此如实声明，不得推测。

## 分析判断区
基于上述数据事实进行分析和推断。每个判断必须：
1. 引用数据事实区的具体数值
2. 使用「【基于数据的推断】」或「【行业知识补充】」标注推断性质
3. 如果某个结论无法从数据中直接得出，必须声明「此为分析师推断」
4. 不得在任何地方编造数据事实区没有的数值
5. 不得编造原始数据中不存在的新闻或事件

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
            {"role": "system", "content": "你是一位资深基金研究分析师，专注于公募基金的基准一致性、风格漂移和业绩归因分析，熟悉2026年证监会基准指引要求。"},
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
                    sp["agent_name"] = "fund_benchmark"
                    sp["as_of_date"] = current_date
                except Exception:
                    pass
            if sp is None:
                from src.utils.analysis_package_builder import text_to_signal_pack
                sp = text_to_signal_pack(final_output, "fund_benchmark", current_date)
            current_data["fund_benchmark_signal_pack"] = sp
            phase2_elapsed = time.time() - phase2_start
            logger.info(f"FundBenchmarkAgent: Phase 2 完成 ({phase2_elapsed:.1f}s, {len(final_output)} 字符)")
            llm_success = True
        except asyncio.TimeoutError:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundBenchmarkAgent: LLM 超时 ({phase2_elapsed:.0f}s)")
            final_output = f"## 数据事实区\n\n{raw_data_text[:3000]}\n\n## 分析判断区\n\n⚠️ LLM分析超时，以上为原始数据，请人工分析。\n"
            llm_success = False
        except Exception as llm_err:
            phase2_elapsed = time.time() - phase2_start
            logger.error(f"{ERROR_ICON} FundBenchmarkAgent: LLM 失败: {llm_err}")
            final_output = f"## 数据事实区\n\n{raw_data_text[:3000]}\n\n## 分析判断区\n\n⚠️ LLM分析失败({str(llm_err)[:100]})，以上为原始数据，请人工分析。\n"
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
        logger.info(f"{SUCCESS_ICON if llm_success else WAIT_ICON} FundBenchmarkAgent: 分析完成 (总耗时 {total_time:.1f}s, Phase1={phase1_elapsed:.1f}s, Phase2={phase2_elapsed:.1f}s)")

        # 更新状态
        current_data["fund_benchmark"] = final_output
        if not skip_cache and cache_date and cache_code:
            write_cache("fund_benchmark", cache_code, cache_date, final_output)
            if "fund_benchmark_signal_pack" in current_data:
                from src.utils.cache_utils import write_signal_pack_cache
                write_signal_pack_cache("fund_benchmark", cache_code, cache_date, current_data["fund_benchmark_signal_pack"])
        current_metadata["fund_benchmark_executed"] = True
        current_metadata["fund_benchmark_timestamp"] = str(time.time())
        current_metadata["fund_benchmark_execution_time"] = f"{total_time:.2f} seconds"

        execution_logger.log_agent_complete(agent_name, {
            "fund_benchmark_length": len(final_output),
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
            "messages": current_messages + [{"role": "assistant", "content": "基准一致性分析已完成"}],
            "metadata": current_metadata,
        }

    except Exception as e:
        logger.error(f"{ERROR_ICON} FundBenchmarkAgent: 错误: {e}", exc_info=True)
        current_data["fund_benchmark_error"] = f"错误: {e}"
        current_data["fund_benchmark"] = f"基准一致性分析过程中出现错误: {str(e)}"
        current_metadata["fund_benchmark_error"] = str(e)
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}


# 本地测试函数
async def test_fund_benchmark_agent():
    """基准一致性分析 Agent的测试函数"""
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
            "query": "分析黄金ETF的基准一致性",
            "fund_code": "sh.518880",
            "fund_name": "黄金ETF",
            "company_name": "黄金ETF",
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat(),
            "is_etf": True,
        },
        metadata={}
    )

    result = await fund_benchmark_agent(test_state)
    print("Fund Benchmark Analysis Result:")
    print(result.get("data", {}).get("fund_benchmark", "No analysis found"))
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fund_benchmark_agent())
