"""
TechnicalAnalysis Agent: Performs technical analysis of a stock using ReAct Agent framework.
技术分析 Agent：使用ReAct Agent框架对股票进行技术分析
"""
import asyncio
import os
import json
from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langgraph.prebuilt import create_react_agent
import time

from src.utils.state_definition import AgentState
from src.tools.mcp_client import get_mcp_tools
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.utils.cache_utils import read_cache, write_cache
from src.utils.model_config import get_model_config_for_agent, get_thinking_body
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv(override=True)

logger = setup_logger(__name__)


# ── 技术指标计算 ──────────────────────────────────

def _compute_ema(data: List[float], period: int) -> List[float]:
    """计算指数移动平均线"""
    if len(data) < period:
        return [float('nan')] * len(data)
    result = [float('nan')] * (period - 1)
    k = 2.0 / (period + 1)
    ema = sum(data[:period]) / period  # SMA作为初始EMA
    result.append(ema)
    for val in data[period:]:
        ema = val * k + ema * (1 - k)
        result.append(ema)
    return result


def _compute_sma(data: List[float], period: int) -> List[float]:
    """计算简单移动平均线"""
    if len(data) < period:
        return [float('nan')] * len(data)
    result = []
    for i in range(len(data)):
        if i < period - 1:
            result.append(float('nan'))
        else:
            result.append(round(sum(data[i - period + 1:i + 1]) / period, 4))
    return result


def _compute_rsi(closes: List[float], period: int = 14) -> List[float]:
    """计算RSI指标"""
    if len(closes) < period + 1:
        return [float('nan')] * len(closes)
    result = [float('nan')] * period
    gains = []
    losses = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        result.append(round(rsi, 2))
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    return result


def _compute_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
    """计算MACD指标，返回(DIF, DEA, MACD柱)三元组"""
    ema_fast = _compute_ema(closes, fast)
    ema_slow = _compute_ema(closes, slow)
    dif = [f - s if not (isinstance(f, float) and isinstance(s, float) and (f != f or s != s)) else float('nan')
           for f, s in zip(ema_fast, ema_slow)]
    dea = _compute_ema([d if d == d else 0 for d in dif], signal)
    macd_bar = [(d - e) * 2 if d == d and e == e else float('nan')
                for d, e in zip(dif, dea)]
    return dif, dea, macd_bar


def _build_indicator_summary(kline_json_str: str, stock_code: str) -> str:
    """从K线JSON数据计算技术指标摘要，返回格式化字符串"""
    try:
        data = json.loads(kline_json_str)
        if not isinstance(data, list) or len(data) < 30:
            return ""

        closes = []
        highs = []
        lows = []
        vols = []
        dates = []
        opens = []

        for row in data:
            try:
                close_val = float(row.get("close", row.get("收盘价", 0)))
                high_val = float(row.get("high", row.get("最高价", 0)))
                low_val = float(row.get("low", row.get("最低价", 0)))
                vol_val = float(row.get("vol", row.get("成交量", 0)))
                open_val = float(row.get("open", row.get("开盘价", 0)))
                date_val = str(row.get("trade_date", row.get("日期", "")))
                if close_val > 0:
                    closes.append(close_val)
                    highs.append(high_val)
                    lows.append(low_val)
                    vols.append(vol_val)
                    dates.append(date_val)
                    opens.append(open_val)
            except (ValueError, TypeError):
                continue

        if len(closes) < 30:
            return ""

        # 计算均线
        ma5 = _compute_sma(closes, 5)
        ma10 = _compute_sma(closes, 10)
        ma20 = _compute_sma(closes, 20)
        ma60 = _compute_sma(closes, 60)
        ma200 = _compute_sma(closes, 200)

        # 计算MACD
        dif, dea, macd_bar = _compute_macd(closes)

        # 计算RSI
        rsi14 = _compute_rsi(closes, 14)
        rsi6 = _compute_rsi(closes, 6)

        # 计算成交量均线
        vol_ma5 = _compute_sma(vols, 5)
        vol_ma20 = _compute_sma(vols, 20)

        # 当前最新值
        last_close = closes[-1]
        last_ma5 = ma5[-1] if len(ma5) > 0 and ma5[-1] == ma5[-1] else None
        last_ma10 = ma10[-1] if len(ma10) > 0 and ma10[-1] == ma10[-1] else None
        last_ma20 = ma20[-1] if len(ma20) > 0 and ma20[-1] == ma20[-1] else None
        last_ma60 = ma60[-1] if len(ma60) > 0 and ma60[-1] == ma60[-1] else None
        last_ma200 = ma200[-1] if len(ma200) > 0 and ma200[-1] == ma200[-1] else None
        last_dif = dif[-1] if len(dif) > 0 and dif[-1] == dif[-1] else None
        last_dea = dea[-1] if len(dea) > 0 and dea[-1] == dea[-1] else None
        last_macd = macd_bar[-1] if len(macd_bar) > 0 and macd_bar[-1] == macd_bar[-1] else None
        last_rsi14 = rsi14[-1] if len(rsi14) > 0 and rsi14[-1] == rsi14[-1] else None
        last_rsi6 = rsi6[-1] if len(rsi6) > 0 and rsi6[-1] == rsi6[-1] else None

        # 20日高点/低点
        recent_20 = closes[-20:] if len(closes) >= 20 else closes
        high_20d = max(recent_20)
        low_20d = min(recent_20)

        # 成交量判断
        last_vol = vols[-1] if vols else 0
        vol_ratio = round(last_vol / vol_ma20[-1], 2) if vol_ma20[-1] and vol_ma20[-1] > 0 else None

        # 涨跌幅
        pct_5d = round((closes[-1] / closes[-5] - 1) * 100, 2) if len(closes) >= 6 else None
        pct_20d = round((closes[-1] / closes[-20] - 1) * 100, 2) if len(closes) >= 21 else None
        pct_60d = round((closes[-1] / closes[-60] - 1) * 100, 2) if len(closes) >= 61 else None

        lines = [
            "## 📈 预计算技术指标汇总",
            f"以下指标基于 {len(closes)} 交易日K线数据自动计算（最新日期: {dates[-1] if dates else 'N/A'}）：",
            "",
            "### 均线系统",
            f"- MA5 = {last_ma5} | MA10 = {last_ma10} | MA20 = {last_ma20} | MA60 = {last_ma60}",
            f"- 当前收盘价 {last_close}",
            f"  - 与MA5关系: {'上方' if last_ma5 and last_close > last_ma5 else '下方' if last_ma5 else 'N/A'}",
            f"  - 与MA20关系: {'上方' if last_ma20 and last_close > last_ma20 else '下方' if last_ma20 else 'N/A'}",
            f"  - 与MA60关系: {'上方' if last_ma60 and last_close > last_ma60 else '下方' if last_ma60 else 'N/A'}",
            f"- 均线排列: {'多头' if last_ma5 and last_ma20 and last_ma5 > last_ma20 and last_ma20 > last_ma60 else '交叉/空头'}",
            "",
            "### MACD指标",
            f"- DIF = {last_dif} | DEA = {last_dea} | MACD柱 = {last_macd}",
            f"- MACD状态: {'金叉/多头' if last_dif and last_dea and last_dif > last_dea else '死叉/空头' if last_dif and last_dea else 'N/A'}",
            f"- 柱状线: {'红柱(多头)' if last_macd and last_macd > 0 else '绿柱(空头)' if last_macd and last_macd < 0 else 'N/A'}",
            "",
            "### RSI指标",
            f"- RSI(6) = {last_rsi6} | RSI(14) = {last_rsi14}",
            f"- RSI(14)区域: {'超买(>70)' if last_rsi14 and last_rsi14 > 70 else '超卖(<30)' if last_rsi14 and last_rsi14 < 30 else '中性(30-70)' if last_rsi14 else 'N/A'}",
            "",
            "### 量价关系",
            f"- 最新成交量 = {last_vol} | 5日均量 = {vol_ma5[-1] if vol_ma5 and len(vol_ma5) > 0 else 'N/A'} | 20日均量 = {vol_ma20[-1] if vol_ma20 and len(vol_ma20) > 0 else 'N/A'}",
            f"- 量比(相对20日均量) = {vol_ratio}",
            "",
            "### 近期价格统计",
            f"- 20日最高价 = {high_20d} | 20日最低价 = {low_20d}",
            f"- 5日涨跌幅 = {pct_5d}% | 20日涨跌幅 = {pct_20d}% | 60日涨跌幅 = {pct_60d}%",
            "",
            "请在你的技术分析中直接引用以上预计算数值，无需重复计算。如果某项显示N/A则表示数据不足。",
        ]
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"技术指标预计算失败: {e}")
        return ""


async def technical_agent(state: AgentState) -> AgentState:
    """
    使用ReAct框架进行技术分析，直接集成MCP工具
    
    Args:
        state: 包含用户查询的当前 Agent状态

    Returns:
        更新后的AgentState，包含技术分析结果
    """
    logger.info(f"{WAIT_ICON} TechnicalAgent: Starting technical analysis using ReAct framework.")

    # 获取执行日志记录器，用于记录 Agent的执行过程
    execution_logger = get_execution_logger()
    agent_name = "technical_agent"

    # 从状态中提取当前数据、消息和元数据
    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})
    user_query = current_data.get("query")

    # 记录 Agent开始执行，包含关键信息
    # 检查中间产物缓存（快筛模式跳过）
    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    cache_code = current_data.get("stock_code", "")
    if not skip_cache and cache_date and cache_code:
        cached = read_cache("technical_analysis", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} TechnicalAgent: 命中缓存，跳过分析 ({cache_code})")
            current_data["technical_analysis"] = cached
            current_metadata["technical_agent_executed"] = True
            current_metadata["technical_agent_cached"] = True
            from src.utils.cache_utils import read_signal_pack_cache
            cached_sp = read_signal_pack_cache("technical_analysis", cache_code, cache_date)
            if cached_sp:
                current_data["technical_signal_pack"] = cached_sp
            else:
                from src.utils.analysis_package_builder import text_to_signal_pack
                import re, json
                # Try to re-extract SIGNAL_PACK from cached LLM output
                sp = None
                tag_match = re.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', cached)
                if tag_match:
                    try:
                        sp = json.loads(tag_match.group(1))
                        sp["agent_name"] = "technical"
                        sp["as_of_date"] = cache_date
                    except Exception:
                        pass
                if sp is None:
                    sp = text_to_signal_pack(cached, "technical", cache_date)
                current_data["technical_signal_pack"] = sp
            return {"data": current_data,
                    "messages": current_messages + [{"role": "assistant", "content": "技术分析已完成（缓存）"}],
                    "metadata": current_metadata}

    execution_logger.log_agent_start(agent_name, {
        "user_query": user_query,
        "stock_code": current_data.get("stock_code"),
        "company_name": current_data.get("company_name"),
        "input_data_keys": list(current_data.keys())
    })

    # 验证用户查询是否存在
    if not user_query:
        logger.error(f"{ERROR_ICON} TechnicalAgent: User query is missing in state data.")
        current_data["technical_analysis_error"] = "User query is missing."
        execution_logger.log_agent_complete(agent_name, current_data, 0, False, "User query is missing")
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    # 记录 Agent开始时间，用于计算执行时长
    agent_start_time = time.time()

    try:
        # 模型配置：优先 state 覆盖（快筛），否则使用 agent 分配的模型 (Qwen3.6-Plus)
        model_cfg = get_model_config_for_agent("technical_agent", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]

        # 验证必要的环境变量是否存在
        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} TechnicalAgent: Missing OpenAI environment variables.")
            current_data["technical_analysis_error"] = "Missing OpenAI environment variables."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "Missing OpenAI environment variables")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        logger.info(f"{WAIT_ICON} TechnicalAgent: Creating ChatOpenAI with model {model_name}")
        # 创建LLM实例，设置合适的参数
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.6,
            request_timeout=360,
            max_tokens=16000,
            extra_body=get_thinking_body(base_url, False)  # ReAct工具调用模式，关闭思考以提速
        )

        # 2. 获取MCP工具集
        logger.info(f"{WAIT_ICON} TechnicalAgent: Fetching MCP tools...")
        try:
            mcp_tools = await get_mcp_tools(tool_filter=[
                "tushare_kline", "tushare_daily_basic", "tushare_stock_info",
                "tushare_latest_trading_date",
                "tushare_moneyflow", "tushare_pe_percentile",
                "tushare_hsgt_flow", "tushare_adj_factor",
                "get_market_analysis_timeframe", "get_stock_analysis",
            ])
            if not mcp_tools:
                logger.error(f"{ERROR_ICON} TechnicalAgent: No MCP tools available.")
                current_data["technical_analysis_error"] = "No MCP tools available."
                execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "No MCP tools available")
                return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

            logger.info(f"{SUCCESS_ICON} TechnicalAgent: Successfully loaded {len(mcp_tools)} tools.")

            # 打印可用工具列表，便于调试
            tool_names = [tool.name for tool in mcp_tools]
            logger.info(f"Available tools: {tool_names}")

            # ── 预计算技术指标 ──────────────────────────────
            stock_code = current_data.get('stock_code', 'Unknown')
            company_name = current_data.get('company_name', 'Unknown')
            current_time_info = current_data.get('current_time_info', '未知时间')
            current_date = current_data.get('current_date', '未知日期')
            clean_code = stock_code.replace("sh.", "").replace("sz.", "").replace("bj.", "").replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()

            indicator_summary = ""
            tool_map = {t.name: t for t in mcp_tools}
            if "tushare_kline" in tool_map:
                try:
                    kline_result = await asyncio.wait_for(
                        tool_map["tushare_kline"].ainvoke({"code": clean_code, "days": 250}),
                        timeout=30.0,
                    )
                    kline_str = str(kline_result) if kline_result else ""
                    indicator_summary = _build_indicator_summary(kline_str, stock_code)
                    if indicator_summary:
                        logger.info(f"{SUCCESS_ICON} TechnicalAgent: 预计算指标完成 ({len(indicator_summary)} 字符)")
                except Exception as precompute_err:
                    logger.warning(f"技术指标预计算失败: {precompute_err}")

            # 3. 创建ReAct Agent - 只传入LLM和工具
            logger.info(f"{WAIT_ICON} TechnicalAgent: Creating ReAct agent...")
            agent = create_react_agent(llm, mcp_tools)

            # 4. 准备输入数据，构建详细的分析请求
            # 构建详细的技术分析请求，包含多个分析维度
            precompute_block = (
                f"\n\n{indicator_summary}\n\n"
                "以上预计算指标已经提供了MACD、RSI、均线的精确数值，"
                "请直接在分析中引用这些数值，无需再用工具重复计算。\n"
            ) if indicator_summary else ""

            agent_input = f"""请以券商分析师的标准，对{company_name}（股票代码：{stock_code}）进行技术面分析。

当前时间：{current_time_info}
当前日期：{current_date}
{precompute_block}
请进行以下技术分析（每个维度都需要基于实际价格数据，引用具体数字）：

1. 价格趋势与形态分析
   - 使用获取的数据判断当前处于上升趋势、下降趋势还是横盘整理
   - 识别近期重要K线形态（如突破、反转、头肩顶/底、双顶/底等）

2. 均线系统分析
   - 短期均线（5日、10日、20日）与长期均线（60日、200日）的位置关系
   - 均线排列情况（多头排列/空头排列/交叉）
   - 当前价格在各主要均线上方还是下方

3. 技术指标分析
   - MACD：请引用上面预计算的DIF/DEA/MACD柱数值，说明金叉/死叉状态
   - RSI：请引用上面预计算的RSI(6/14)数值，判断是否超买/超卖
   - 成交量：近期成交量变化、量比情况

4. 量价关系分析
   - 上涨/下跌时的成交量变化（放量/缩量）
   - 是否存在量价背离
   - 主力资金流向趋势

5. 支撑位与阻力位
   - 近期关键支撑价位（前期低点、成交密集区下沿等）
   - 近期关键阻力价位（前期高点、成交密集区上沿等）
   - 突破关键位的可能性和信号

6. 综合技术评估
   - 短期(1-2周)走势判断
   - 中期(1-3个月)走势判断
   - 技术面风险提示

重要限制：
- 请专注于价格数据和技术指标分析，不要使用crawl_news工具获取新闻信息
- 分析必须有数据支撑，引用具体的价格、指标数值，避免空洞的定性描述
- 技术分析应基于实际获取的K线数据，不要使用假设数据

请使用可用的工具获取实际数据进行分析，而不是基于假设。

⛔ 输出格式要求（防幻觉机制）：
请将分析输出严格分为两个区域：

## 📊 数据事实区
列出通过工具调取到的所有客观数据，每条标注数据来源：
- [K线数据] 具体价格/指标数值（如：最新收盘价=XXX元，MACD DIF=XX）
- [工具名] 具体数值
- ...
如果某项数据工具无法获取，必须标注「数据不可用」而不是推测。

## 🔍 分析判断区
基于上述数据事实进行分析和推断。每个判断必须：
1. 引用数据事实区的具体数值
2. 使用「【基于数据的推断】」或「【行业知识补充】」标注推断性质
3. 如果某个结论无法从数据中直接得出，必须声明「此为分析师推断」
4. 不得在任何地方编造数据事实区没有的数值

⛔ 结构化输出要求：
在完成上述分析的「🔍 分析判断区」之后，请额外输出一个 JSON block：

<SIGNAL_PACK>
{{
    "bias": "bullish"|"neutral"|"bearish",
    "confidence": 0.0-1.0,
    "key_points": ["关键结论1", "关键结论2"] (最多5条),
    "signals": [
        {{
            "factor": "因子名(如:均线多头排列/MACD金叉/量价背离)",
            "direction": 1|-1|0,
            "strength": 0-100,
            "time_horizon": ["short","medium"],
            "source_level": "structured",
            "risk_flags": [],
            "freshness": "daily",
            "note": "一句话说明"
        }}
    ] (最多5条),
    "risk_flags": ["liquidity_risk"],
    "missing_data": ["未获取到某些指标"],
    "source_summary": "Tushare行情+预计算技术指标(MACD/RSI/均线/量价)"
}}
</SIGNAL_PACK>

请确保SIGNAL_PACK内的JSON完全有效。"""

            logger.info(f"Agent input: {agent_input}")

            # 5. 调用ReAct Agent - 使用正确的messages格式
            logger.info(f"{WAIT_ICON} TechnicalAgent: Calling ReAct agent...")
            start_time = time.time()

            # LangGraph ReAct Agent需要messages格式的输入
            input_data = {
                "messages": [HumanMessage(content=agent_input)]
            }

            # 调用 Agent执行分析
            response = await agent.ainvoke(input_data, config={"recursion_limit": 30})

            end_time = time.time()
            execution_time = end_time - start_time

            logger.info(f"ReAct agent execution completed in {execution_time:.2f} seconds")

            # 6. 提取分析结果
            final_output = "No analysis generated."
            
            if "messages" in response and isinstance(response["messages"], list):
                messages = response["messages"]
                # 查找最后一条AI消息，这通常包含最终的分析结果
                ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
                if ai_messages:
                    last_ai_message = ai_messages[-1]
                    final_output = last_ai_message.content
                    logger.info(f"Successfully extracted analysis from AI message.")
                else:
                    logger.warning("No AI messages found in response")
                    # 如果没有AI消息，尝试获取所有消息的内容
                    all_content = []
                    for msg in messages:
                        if hasattr(msg, 'content') and msg.content:
                            all_content.append(str(msg.content))
                    if all_content:
                        final_output = "\n".join(all_content)
            else:
                logger.error(f"Unexpected response format: {type(response)}")
                logger.error(f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")

            logger.info(f"Final extracted analysis length: {len(final_output)} characters")
            print(f"TECHNICALAGENT: {final_output}")

            # ═══ signal_pack 提取 ═══
            import json as _json_sp
            import re as _re_sp
            from src.utils.analysis_package_builder import text_to_signal_pack

            sp = None
            tag_match = _re_sp.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', final_output)
            if tag_match:
                try:
                    sp = _json_sp.loads(tag_match.group(1))
                    sp["agent_name"] = "technical"
                    sp["as_of_date"] = current_date
                    sp.setdefault("analysis_text", final_output[:500])
                    sp.setdefault("data_quality_score", 0.7)
                except Exception:
                    pass
            if sp is None:
                sp = text_to_signal_pack(final_output, "technical", current_date)
            current_data["technical_signal_pack"] = sp

            # 7. 记录LLM交互，用于后续分析和优化
            model_config = {
                "model": model_name,
                "temperature": 0.6,
                "max_tokens": 16000,
                "thinking": "disabled",
                "api_base": base_url
            }
            
            execution_logger.log_llm_interaction(
                agent_name=agent_name,
                interaction_type="react_agent",
                input_messages=[{"role": "user", "content": agent_input}],
                output_content=final_output,
                model_config=model_config,
                execution_time=execution_time
            )

            logger.info(f"{SUCCESS_ICON} TechnicalAgent: Successfully completed technical analysis.")
            
            # 8. 更新状态，保存分析结果和元数据
            current_data["technical_analysis"] = final_output
            if not skip_cache and cache_date and cache_code:
                write_cache("technical_analysis", cache_code, cache_date, final_output)
                if "technical_signal_pack" in current_data:
                    from src.utils.cache_utils import write_signal_pack_cache
                    write_signal_pack_cache("technical_analysis", cache_code, cache_date, current_data["technical_signal_pack"])
            current_metadata["technical_agent_executed"] = True
            current_metadata["technical_agent_timestamp"] = str(time.time())
            current_metadata["technical_agent_execution_time"] = f"{execution_time:.2f} seconds"

            # 9. 添加消息记录，保持对话历史
            new_message = {"role": "assistant", "content": "技术分析已完成"}
            updated_messages = current_messages + [new_message]

            # 记录 Agent执行成功
            total_execution_time = time.time() - agent_start_time
            execution_logger.log_agent_complete(agent_name, {
                "technical_analysis_length": len(final_output),
                "analysis_preview": final_output[:500] if len(final_output) > 500 else final_output,
                "llm_execution_time": execution_time,
                "total_execution_time": total_execution_time
            }, total_execution_time, True)

            return {
                "data": current_data,
                "messages": updated_messages,
                "metadata": current_metadata
            }

        except Exception as e:
            logger.error(f"{ERROR_ICON} TechnicalAgent: Error in MCP or agent execution: {e}", exc_info=True)
            current_data["technical_analysis_error"] = f"Error in MCP or agent execution: {e}"
            current_data["technical_analysis"] = f"技术分析过程中出现错误: {str(e)}"
            current_metadata["technical_agent_error"] = str(e)
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    except Exception as e:
        logger.error(f"{ERROR_ICON} TechnicalAgent: Error during execution: {e}", exc_info=True)
        current_data["technical_analysis_error"] = f"Error during execution: {e}"
        current_metadata["technical_agent_error"] = str(e)
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}


# 本地测试函数
async def test_technical_agent():
    """技术分析 Agent的测试函数"""
    from src.utils.state_definition import AgentState
    from datetime import datetime

    # 准备测试数据，包含当前时间信息
    current_datetime = datetime.now()
    current_date_cn = current_datetime.strftime("%Y年%m月%d日")
    current_date_en = current_datetime.strftime("%Y-%m-%d")
    current_weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][current_datetime.weekday()]
    current_time = current_datetime.strftime("%H:%M:%S")
    current_time_info = f"{current_date_cn} ({current_date_en}) {current_weekday_cn} {current_time}"

    # 创建测试状态，模拟真实的用户查询
    test_state = AgentState(
        messages=[],
        data={
            "query": "分析嘉友国际的技术指标",
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

    # 运行 Agent并输出结果
    result = await technical_agent(test_state)
    print("Technical Analysis Result:")
    print(result.get("data", {}).get("technical_analysis", "No analysis found"))

    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_technical_agent()) 