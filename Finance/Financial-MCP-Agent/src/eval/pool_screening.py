"""
精筛股票池四层筛选管线 — 严格按照评分智能体开发总纲 §4 实现。

四层管线:
  Layer 0: 纯代码硬筛 (去ST/*ST/新股/低流动/北交所/B股)
  Layer 1: M1/M3 批量粗筛分档 (强烈推荐→白名单, 推荐→初筛池, 中性/回避→丢弃, 卖出→黑名单)
  Layer 2: M2 快筛过滤初筛池 (便宜模型, 淘汰明显不行的)
  Layer 3: 1:1.2差额精筛 (白名单+初筛通过→正式7Agent+3Scorer→LLM动态阈值→最终精筛池)
"""
import asyncio
import heapq
import json
import logging
import math
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable, Optional

from src.eval.pool_manager import PoolManager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Streaming Dual-Heap Top-α Selector (流式双堆 top-α 选择器)
# ═══════════════════════════════════════════════════════════

def calc_alpha(whitelist_count: int, recommended_total: int,
               target_size: int, ratio: float = 1.2) -> float:
    """动态计算 Layer 2 的 α 比例。

    α = (target_size × ratio - whitelist_count) / recommended_total

    白名单越多, α 越小; 推荐越多, α 越小。
    返回值限制在 [0, 1]。
    """
    layer3_capacity = int(target_size * ratio)
    layer2_slots = max(0, layer3_capacity - whitelist_count)
    if recommended_total == 0:
        return 0.0
    return min(1.0, layer2_slots / recommended_total)


class StreamingTopAlphaHeap:
    """流式双堆 top-α 选择器。

    维护两个堆:
      - High: 小顶堆, 存当前 top-α 候选 (分数最高的 α 比例)
      - Low:  大顶堆, 存其余股票

    τ = High.min() = top-α 下界阈值。

    Dispatch 策略:
      连续 stable_batches 批 τ 波动 < epsilon → 视为 τ 收敛
      → 将 High 中尚未 dispatch 的股票 dispatch 到 Layer 3。
    """

    def __init__(self, alpha_fn: Callable[[], float],
                 epsilon: float = 3.0, stable_batches: int = 3):
        self.alpha_fn = alpha_fn
        self.epsilon = epsilon
        self.stable_batches = stable_batches
        self.high: List[tuple] = []   # min-heap: (score, stock_code)
        self.low: List[tuple] = []    # max-heap: (-score, stock_code)
        self._dispatched: set = set()
        self._tau_history: List[float] = []
        self._total = 0
        self._stock_map: Dict[str, Any] = {}  # code → stock dict

    @property
    def tau(self) -> float:
        """当前 top-α 阈值."""
        return self.high[0][0] if self.high else 0.0

    @property
    def total(self) -> int:
        return self._total

    def feed(self, score: float, stock: Dict[str, Any]):
        """喂入一个新分数和对应的股票."""
        code = stock.get("code", str(id(stock)))
        self._stock_map[code] = stock
        self._total += 1

        # Step 1: 粗分
        if not self.high or score >= self.high[0][0]:
            heapq.heappush(self.high, (score, code))
        else:
            heapq.heappush(self.low, (-score, code))

        # Step 2: 按比例重平衡
        k = max(1, math.ceil(self.alpha_fn() * self._total))
        while len(self.high) > k:
            s, c = heapq.heappop(self.high)
            heapq.heappush(self.low, (-s, c))
        while len(self.high) < k and self.low:
            s, c = heapq.heappop(self.low)
            heapq.heappush(self.high, (-s, c))

        # Step 3: 修正跨堆次序
        while self.low and self.high and -self.low[0][0] > self.high[0][0]:
            a_s, a_c = heapq.heappop(self.low)
            b_s, b_c = heapq.heappop(self.high)
            heapq.heappush(self.high, (-a_s, a_c))
            heapq.heappush(self.low, (-b_s, b_c))

    def batch_done(self):
        """一批数据喂完后调用, 记录 τ 历史."""
        self._tau_history.append(self.tau)
        if len(self._tau_history) > self.stable_batches * 3:
            self._tau_history = self._tau_history[-self.stable_batches * 2:]

    def is_tau_stable(self) -> bool:
        """τ 是否连续 stable_batches 批稳定 (波动 < epsilon)."""
        if len(self._tau_history) < self.stable_batches:
            return False
        recent = self._tau_history[-self.stable_batches:]
        return max(recent) - min(recent) < self.epsilon

    def get_undispatched(self) -> List[Dict[str, Any]]:
        """获取 High 中尚未 dispatch 的股票列表."""
        result = []
        for _, code in self.high:
            if code not in self._dispatched and code in self._stock_map:
                result.append(self._stock_map[code])
                self._dispatched.add(code)
        return result

    def finalize(self) -> List[Dict[str, Any]]:
        """Layer 1 完成后调用, dispatch High 中所有剩余的."""
        return self.get_undispatched()

    @property
    def high_size(self) -> int:
        return len(self.high)

    @property
    def low_size(self) -> int:
        return len(self.low)


# ═══════════════════════════════════════════════════════════
# Layer 0: Hard Screen (纯代码, 无LLM)
# ═══════════════════════════════════════════════════════════

def _is_st_name(name: str) -> bool:
    """检测股票名称是否含ST"""
    return bool(re.search(r'\*?ST', str(name), re.IGNORECASE))


def _is_low_volume(daily_amounts: list, min_amount: float) -> bool:
    """判断日均成交额是否低于阈值。无数据时保守保留(返回False)。"""
    if not daily_amounts:
        return False
    avg = sum(daily_amounts) / len(daily_amounts)
    return avg < min_amount


def _is_recent_ipo(list_date: str, min_days: int = 60) -> bool:
    """检测是否上市不满min_days天"""
    if not list_date or len(str(list_date)) < 8:
        return False
    try:
        dt = datetime.strptime(str(list_date)[:8], "%Y%m%d")
        return (datetime.now() - dt).days < min_days
    except ValueError:
        return False


def _is_stock_excluded(stock: Dict[str, Any]) -> bool:
    """判断股票是否应被硬筛排除（ST/BJ/B/新股）"""
    code = stock.get("ts_code", "")
    name = stock.get("name", "")

    if code.endswith(".BJ"):
        return True
    # B股: 上交所900xxx, 深交所200xxx (Tushare ts_code格式: 900901.SH / 200002.SZ)
    # B股: 上交所900xxx.SH, 深交所200xxx.SZ
    if code.startswith("900") and code.endswith(".SH"):
        return True
    if code.startswith("200") and code.endswith(".SZ"):
        return True
    if _is_st_name(name):
        return True
    if _is_recent_ipo(stock.get("list_date", "")):
        return True
    return False


async def hard_screen() -> List[Dict[str, Any]]:
    """
    Layer 0: 从Tushare获取全A股 → 硬筛(ST/BJ/B/新股/低成交额) → 返回候选列表。

    Returns:
        [{"ts_code": "603871.SH", "name": "嘉友国际", "industry": "物流", ...}, ...]
    """
    from src.eval.data_fetcher import _call as tushare_call
    from src.eval.config import get as eval_get

    # Tushare daily.amount 单位是千元, 20000千元 = 2000万元
    min_daily_amount = eval_get("hard_screen_min_daily_amount", 20000)  # 千元
    logger.info("[Layer 0] 硬筛开始: 最低日均成交额=%d万元 (%.0f千元)",
                min_daily_amount * 1000 // 10000, min_daily_amount)

    # 获取全A股列表 (不传exchange参数=全市场)
    stock_list = tushare_call("stock_basic", {
        "list_status": "L",
    }, fields="ts_code,name,list_date,industry")
    if not stock_list or "items" not in stock_list:
        raise RuntimeError("Tushare stock_basic查询失败")
    if len(stock_list["items"]) < 100:
        logger.error("stock_basic返回异常: 仅%d只 (预期>5000), 可能Tushare限流或参数错误",
                     len(stock_list["items"]))

    fields = stock_list["fields"]
    stocks = []
    for row in stock_list["items"]:
        item = dict(zip(fields, row))
        stocks.append(item)

    total = len(stocks)
    logger.info("  Tushare返回: %d只", total)

    # ── 阶段A: ST/BJ/B/新股硬筛 ──
    filtered = []
    st_removed = 0
    for s in stocks:
        if _is_stock_excluded(s):
            if _is_st_name(s.get("name", "")):
                st_removed += 1
            continue
        filtered.append(s)

    logger.info("  硬筛后: %d只 (排除ST/BJ/B/新股共%d只, 其中ST%d只)",
                len(filtered), total - len(filtered), st_removed)

    # ── 阶段B: 近20日日均成交额筛选 ──
    today_str = datetime.now().strftime("%Y%m%d")
    start_str = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

    logger.info("  成交量筛选: %d只 → 过滤日均成交额<%d万 (批量查询模式)",
                len(filtered), min_daily_amount // 10000)

    # 获取近20个交易日
    trade_cal = tushare_call("trade_cal", {
        "exchange": "SSE",
        "start_date": start_str,
        "end_date": today_str,
    }, fields="cal_date,is_open")
    trading_days = []
    if trade_cal and "items" in trade_cal:
        for row in trade_cal["items"]:
            item = dict(zip(trade_cal["fields"], row))
            if item.get("is_open") == 1:
                trading_days.append(item["cal_date"])
    trading_days.sort(reverse=True)
    trading_days = trading_days[:20]  # 最多取最近20个交易日
    logger.info("  近20个交易日: %s", trading_days[:5] if trading_days else "空")

    # 批量查询: 按交易日逐日查 daily（每只股票单日一行，替代逐只查询）
    volume_map: Dict[str, List[float]] = {}  # {ts_code: [amount1, amount2, ...]}
    for day_idx, day in enumerate(trading_days):
        if day_idx % 5 == 0:
            logger.info("  批量成交量查询进度: %d/%d 天", day_idx, len(trading_days))
        try:
            daily_bulk = tushare_call("daily", {
                "trade_date": day,
            }, fields="ts_code,amount")
            if daily_bulk and "items" in daily_bulk:
                fields_d = daily_bulk["fields"]
                for row in daily_bulk["items"]:
                    item = dict(zip(fields_d, row))
                    code = item.get("ts_code", "")
                    try:
                        amt = float(item.get("amount", 0))
                    except (ValueError, TypeError):
                        continue
                    if code not in volume_map:
                        volume_map[code] = []
                    volume_map[code].append(amt)
        except Exception:
            pass

    logger.info("  成交量批量查询完成: %d只股票有数据", len(volume_map))

    # 基于内存 map 筛选
    volume_filtered = []
    for s in filtered:
        ts_code = s.get("ts_code", "")
        amounts = volume_map.get(ts_code, [])
        if len(amounts) >= 5:
            if not _is_low_volume(amounts, min_daily_amount):
                volume_filtered.append(s)
        else:
            volume_filtered.append(s)  # 无足够交易记录, 保守保留

    logger.info("  成交量筛选后: %d只 (剔除%d只低成交额)",
                len(volume_filtered), len(filtered) - len(volume_filtered))
    return volume_filtered


# ═══════════════════════════════════════════════════════════
# Layer 1: M1/M3 Batch Scoring (生产模型批量粗筛)
# ═══════════════════════════════════════════════════════════

# 总纲 §4.1 第1层分类标准 — 与 batch_scorer.LEVELS 对齐
# 注意: 此列表必须与 _BATCH_SYSTEM_PROMPT 中的分类一致，
# 否则 parse_batch_response 的 normalization 会静默改写 LLM 输出。
LAYER1_LEVELS = ["强烈推荐", "推荐", "中性", "回避", "卖出"]

# 总纲5级 → 3档 映射 (只有"卖出"进黑名单, "中性"/"回避"丢弃)
LEVEL_TO_TIER = {
    "强烈推荐": "whitelist",
    "推荐": "initial_pool",
    "中性": "initial_pool",
    "回避": "initial_pool",
    "卖出": "blacklist",
}

# Layer 1 使用 batch_scorer 内置的 _BATCH_SYSTEM_PROMPT
# 自定义 prompt 已废弃——与 batch_scorer 共用同一套 prompt 保证分类一致


def classify_batch_result(score: Dict[str, str]) -> str:
    """将批量打分结果按总纲标准分类为 whitelist / initial_pool / blacklist"""
    level = score.get("level", "中性")
    return LEVEL_TO_TIER.get(level, "initial_pool")


def _prepare_stocks_for_batch(ts_stocks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """将Tushare格式股票转换为batch_scorer所需的格式"""
    result = []
    for s in ts_stocks:
        ts_code = s.get("ts_code", "")
        name = s.get("name", "")
        if ts_code.endswith(".SH"):
            internal = f"sh.{ts_code[:6]}"
        elif ts_code.endswith(".SZ"):
            internal = f"sz.{ts_code[:6]}"
        else:
            continue
        result.append({"code": internal, "name": name})
    return result


async def batch_score_layer1(
    ts_stocks: List[Dict[str, Any]],
    term: str,
    on_progress: Optional[Callable] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Layer 1: 用M1/M3生产模型批量粗筛 ~4500只股票 → 按总纲分4档。

    Args:
        ts_stocks: Layer 0 输出的股票列表
        term: short/medium/long
        on_progress: 进度回调(completed, total)

    Returns:
        {"whitelist": [...], "initial_pool": [...], "blacklist": [...]}
    """
    from src.api.batch_scorer import fetch_batch, score_batch

    # 模型选择: short→M3, medium→M1, long→M1
    model_suffix = "_3" if term == "short" else ""

    logger.info("[Layer 1] 批量粗筛开始: %d只, term=%s, model_suffix=%s",
                len(ts_stocks), term, model_suffix)

    if on_progress:
        on_progress(0, len(ts_stocks))

    # Step A: 格式转换
    batch_input = _prepare_stocks_for_batch(ts_stocks)
    if not batch_input:
        return {"whitelist": [], "initial_pool": [], "blacklist": []}

    # Step B: 数据获取 (Tushare + HTTP)
    batch_stocks = await fetch_batch(
        batch_input,
        semaphore=12,
        on_progress=lambda c, t: on_progress(c * 3 // 4, t) if on_progress else None,
    )

    # Step C: 批量LLM打分 (传入总纲分类标准)
    scored = await score_batch(
        batch_stocks,
        horizon=term,
        semaphore=8,
        model_suffix=model_suffix,
        custom_levels=LAYER1_LEVELS,
        on_progress=lambda c, t: on_progress(t * 3 // 4 + c // 4, t) if on_progress else None,
    )

    # Step D: 分档
    whitelist, initial_pool, blacklist = [], [], []
    for s in scored:
        score_data = s.get("score", {})
        if isinstance(score_data, dict):
            tier = classify_batch_result(score_data)
            entry = {
                "code": s.get("code", ""),
                "name": s.get("name", ""),
                "layer1_level": score_data.get("level", ""),
                "layer1_confidence": score_data.get("confidence", ""),
                "layer1_reason": score_data.get("reason", ""),
                "layer1_risk": score_data.get("risk", ""),
            }
            if tier == "whitelist":
                whitelist.append(entry)
            elif tier == "initial_pool":
                initial_pool.append(entry)
            else:
                blacklist.append(entry)

    logger.info(
        "[Layer 1] 分档完成: 白名单%d只, 初筛池%d只, 黑名单%d只",
        len(whitelist), len(initial_pool), len(blacklist)
    )

    if on_progress:
        on_progress(len(ts_stocks), len(ts_stocks))

    return {"whitelist": whitelist, "initial_pool": initial_pool, "blacklist": blacklist}


async def batch_score_layer1_stream(
    ts_stocks: List[Dict[str, Any]],
    term: str,
    on_progress: Optional[Callable] = None,
    progress_touch: Optional[Callable] = None,
):
    """Layer 1 流式版: 数据一次性获取, LLM 打分分批产出。

    每批 CHUNK_SIZE 只(默认100) 打分完成后 yield 该批的分类结果列表。
    用于 run_pool_update_v3 的流水线调度——Layer 1 边跑边分叉到 Layer 2/3。

    progress_touch: 可选, 数据获取阶段定期调用以刷新心跳/防卡顿。
    """
    from src.api.batch_scorer import fetch_batch, score_batch

    CHUNK_SIZE = 100  # 每次 score_batch 处理的股票数 (20批×5只)
    model_suffix = "_3" if term == "short" else ""

    logger.info("[Layer 1-Stream] 流式批量粗筛: %d只, term=%s, chunk=%d",
                len(ts_stocks), term, CHUNK_SIZE)

    if on_progress:
        on_progress(0, len(ts_stocks))

    # Step A+B: 格式转换 + 数据获取 (一次性)
    batch_input = _prepare_stocks_for_batch(ts_stocks)
    if not batch_input:
        return

    # 数据获取阶段: 定期回调以刷新心跳/防卡顿, 并推进进度估算
    def _fetch_progress(completed, total):
        if progress_touch:
            progress_touch()

    batch_stocks = await fetch_batch(
        batch_input, semaphore=12,
        on_progress=_fetch_progress,
    )
    valid = [s for s in batch_stocks if s.get("status") == "fetched" and s.get("data")]
    logger.info("[Layer 1-Stream] 数据获取完成: %d/%d 有效", len(valid), len(batch_stocks))

    # Step C: 分批 LLM 打分, 逐批 yield
    # yield 同时携带原始 data 字典, 供 Layer 2 复用 (避免冗余 fetch_batch)。
    total_valid = len(valid)
    scored_total = 0
    for chunk_start in range(0, total_valid, CHUNK_SIZE):
        chunk = valid[chunk_start:chunk_start + CHUNK_SIZE]
        scored = await score_batch(
            chunk, horizon=term, semaphore=8,
            model_suffix=model_suffix,
            custom_levels=LAYER1_LEVELS,
        )
        # 产出本批分类结果 (含 raw data, 供 Layer 2 直接复用, 避免二次 fetch)
        batch_results = []
        for s in scored:
            sd = s.get("score", {})
            if isinstance(sd, dict):
                entry = {
                    "code": s.get("code", ""),
                    "name": s.get("name", ""),
                    "layer1_level": sd.get("level", ""),
                    "layer1_confidence": sd.get("confidence", ""),
                    "layer1_reason": sd.get("reason", ""),
                    "layer1_risk": sd.get("risk", ""),
                    "_raw_data": s.get("data"),
                }
                batch_results.append(entry)

        scored_total += len(chunk)
        if on_progress:
            progress = min(scored_total, total_valid)
            on_progress(progress, len(ts_stocks))
        logger.info("[Layer 1-Stream] chunk %d/%d: %d results",
                    chunk_start // CHUNK_SIZE + 1,
                    (total_valid + CHUNK_SIZE - 1) // CHUNK_SIZE,
                    len(batch_results))
        yield batch_results

    if on_progress:
        on_progress(len(ts_stocks), len(ts_stocks))


# ═══════════════════════════════════════════════════════════
# Layer 2: DSV4Pro 流式排序打分 (输出 0-100 连续分数)
# ═══════════════════════════════════════════════════════════

# Layer 2 专用 System Prompt: 输出 0-100 连续数值分数, 用于精确排序
_LAYER2_SYSTEM_PROMPT = """你是 A 股精筛排序器。你的任务不是判断"能不能投"，而是对已确认"有投资价值"的股票给出 0-100 的连续分数，用于精确排序。

## 评分标准 (0-100 连续值)
- 90-100: 极优质——基本面/估值/成长/风险全方位领先同类
- 75-89:  优秀——多项指标突出，建议优先考虑
- 60-74:  良好——整体不错，部分指标一般
- 40-59:  一般——中规中矩，无明显亮点也无严重缺陷
- 20-39:  偏弱——存在明显瑕疵但不至于淘汰
- 0-19:   弱——勉强合格，不推荐

## 核心原则
1. **精确区分**: 这批股票 Layer 1 都已判为"推荐"，彼此差别可能很细微。请精确使用 0-100 连续值，65 和 68 的差异要有依据，不要所有人都在 70-80。
2. **数据引用规则**: 只能引用股票数据中实际出现的数字，严禁编造。
3. **跨行业可比**: 参考行业估值基准，银行 PE 6 倍合理 ≠ 科技 PE 60 倍高估。
4. **宁可保守**: 数据不足时给 40-50 的中性分，不要猜测。

## 输出格式
严格输出 JSON 数组（不要 markdown 代码块标记）：

[{"code": "sh.603871", "score": 78, "reason": "低估值+高ROE+行业景气(30字内)", "risk": "原材料涨价(30字内)"},
 ...]
"""

_HORIZON_CONTEXT_L2 = {
    "short": """## 当前评分维度: 短线 (1-5 交易日)
重点关注: 量价关系、技术信号、短期资金情绪、近期涨跌幅。基本面权重降低。""",

    "medium": """## 当前评分维度: 中线 (1-3 个月)
重点关注: 基本面质量(ROE/毛利率/增速)、估值水平(vs行业)、技术趋势、风险评估。""",

    "long": """## 当前评分维度: 长线 (1-3 年)
重点关注: 商业护城河、行业景气度、ROE 持续性、估值安全边际。短线波动可忽略。""",
}


async def score_layer2_batch(
    recommended_stocks: List[Dict[str, Any]],
    term: str,
    pre_fetched_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """对一批推荐股票用 DeepSeek V4 Pro 打连续 0-100 分 (5只/批, 8并发)。

    使用自定义 prompt (_LAYER2_SYSTEM_PROMPT) 让 LLM 输出 0-100 连续数值,
    而非 5 级分类。分数用于 StreamingTopAlphaHeap 的精确排序。

    Args:
        recommended_stocks: [{code, name, ...}, ...]
        term: short/medium/long
        pre_fetched_data: 可选 {code: raw_data_dict}. 若提供且命中率高, 跳过 fetch_batch
            (Layer 1 已预先抓取过数据, 复用可省 10-20min/cold-start)。

    Returns:
        [{code, score: float, reason, risk}, ...]
    """
    from src.utils.llm_clients import OpenAICompatibleClient
    import os

    if not recommended_stocks:
        return []

    # 数据获取: 优先复用 Layer 1 已抓取的 raw_data, 仅对缺失部分走 fetch_batch
    pre_hit = 0
    valid = []
    if pre_fetched_data:
        for s in recommended_stocks:
            code = s.get("code", "")
            raw = pre_fetched_data.get(code)
            if raw and isinstance(raw, dict):
                valid.append({"code": code, "name": s.get("name", ""), "data": raw, "status": "fetched"})
                pre_hit += 1
        missing = [s for s in recommended_stocks
                   if not (pre_fetched_data.get(s.get("code", "")))]
        if missing:
            from src.api.batch_scorer import fetch_batch
            batch_input = [{"code": s["code"], "name": s.get("name", "")} for s in missing]
            fetched = await fetch_batch(batch_input, semaphore=12)
            valid.extend(f for f in fetched if f.get("status") == "fetched" and f.get("data"))
    else:
        from src.api.batch_scorer import fetch_batch
        batch_input = [{"code": s["code"], "name": s.get("name", "")} for s in recommended_stocks]
        fetched = await fetch_batch(batch_input, semaphore=12)
        valid = [f for f in fetched if f.get("status") == "fetched" and f.get("data")]

    if not valid:
        logger.warning("[Layer 2] 无有效股票数据")
        return []

    logger.info("[Layer 2] 数据复用: %d/%d 来自 L1 pre-fetched, %d 需重新获取",
                pre_hit, len(recommended_stocks), len(valid) - pre_hit)

    # 分块 + LLM 评分
    from src.api.batch_scorer import chunk_stocks
    chunks = chunk_stocks(valid, chunk_size=5)
    total_chunks = len(chunks)
    logger.info("[Layer 2] DSV4Pro 打分: %d只, %d 批次, 8并发", len(valid), total_chunks)

    # 模型配置
    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY_6", "")
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL_6", "")
    model = os.getenv("OPENAI_COMPATIBLE_MODEL_6", "deepseek-v4-pro")
    if not all([api_key, base_url, model]):
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY", "")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")
        model = os.getenv("OPENAI_COMPATIBLE_MODEL", "deepseek-v4-pro")

    from src.utils.model_config import get_thinking_body
    llm = OpenAICompatibleClient(
        api_key=api_key, base_url=base_url, model=model, env_prefix="",
        extra_body=get_thinking_body(base_url, enabled=True),
        http_timeout=180, http_connect_timeout=10,
    )

    horizon_context = _HORIZON_CONTEXT_L2.get(term, _HORIZON_CONTEXT_L2["medium"])

    def _build_l2_prompt(stocks_data: list) -> str:
        from src.utils.industry_knowledge import identify_industry, get_industry_info
        stock_blocks = []
        for s in stocks_data:
            fields = []
            for lbl, key in [("PE", "pe"), ("PB", "pb"), ("ROE(%)", "roe"),
                            ("毛利率(%)", "gross_margin"), ("营收增速(%)", "revenue_growth"),
                            ("利润增速(%)", "profit_growth"), ("负债率(%)", "debt_ratio"),
                            ("市值", "market_cap"), ("行业", "industry")]:
                v = s.get(key, "")
                if v and str(v).strip() not in ("", "None", "N/A"):
                    fields.append(f"  {lbl}: {v}")
            pc = s.get("price_changes", {}) or {}
            for period, label in [("1m", "近1月"), ("3m", "近3月"), ("1y", "近1年")]:
                v = pc.get(period, "")
                if v and v != "N/A":
                    fields.append(f"  涨跌({label}): {v}")
            blocks = "\n".join(fields) if fields else "  (无数据)"
            stock_blocks.append(f"### {s.get('name','')} ({s.get('code','')})\n{blocks}")
        return f"""{horizon_context}

## 股票数据

{chr(10).join(stock_blocks)}

## 输出要求
对上述 {len(stocks_data)} 只股票，返回 JSON 数组（不要 markdown 代码块）。
每只股票输出: code, score(0-100连续值), reason(30字内), risk(30字内)

只返回 JSON 数组:"""

    sem = asyncio.Semaphore(8)
    score_lock = asyncio.Lock()
    results = []

    async def _score_chunk(chunk):
        stocks_data = [s["data"] for s in chunk]
        prompt = _build_l2_prompt(stocks_data)
        try:
            response = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None, lambda: llm.get_completion(
                        [{"role": "system", "content": _LAYER2_SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}], max_retries=1
                    )
                ), timeout=300.0
            )
            if response:
                import re
                text = str(response).strip()
                start = text.find('[')
                if start >= 0:
                    for end in range(len(text), start, -1):
                        try:
                            parsed = json.loads(text[start:end])
                            if isinstance(parsed, list):
                                async with score_lock:
                                    for item in parsed:
                                        if isinstance(item, dict):
                                            code = item.get("code", "")
                                            if re.match(r'^(sh|sz)\.\d{5,6}$', code):
                                                results.append({
                                                    "code": code,
                                                    "score": float(item.get("score", 50)),
                                                    "reason": item.get("reason", ""),
                                                    "risk": item.get("risk", ""),
                                                })
                                break
                        except json.JSONDecodeError:
                            continue
        except asyncio.TimeoutError:
            logger.warning("[Layer 2] 批次超时")
        except Exception as e:
            logger.error("[Layer 2] 批次失败: %s", e)

    await asyncio.gather(*[_score_chunk(c) for c in chunks])

    logger.info("[Layer 2] DSV4Pro 打分完成: %d 只有效分数", len(results))
    return results


# ═══════════════════════════════════════════════════════════
# Layer 2: M2 Quick Screen (快筛模型过滤初筛池) [保留旧版兼容]
# ═══════════════════════════════════════════════════════════

async def quick_screen_layer2(
    initial_pool: List[Dict[str, Any]],
    term: str,
    threshold: int = 50,
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    Layer 2: 用M2(Qwen3.6-Flash)快筛初筛池，淘汰明显不行的。

    总纲要求: 快筛目的是"去掉明显不行的"，不是"选出最好的"，阈值不宜过高。

    Args:
        initial_pool: Layer 1 输出的初筛池股票
        term: short/medium/long
        threshold: 快筛分数阈值，低于此分则淘汰
        on_progress: 进度回调

    Returns:
        快筛通过的股票列表（含quick_screen_score字段）
    """
    from src.api.batch_scorer import fetch_batch, score_batch
    from src.eval.config import get as eval_get

    threshold = eval_get(f"quick_screen_threshold_{term}", threshold)

    logger.info("[Layer 2] 快筛开始: %d只, term=%s, threshold=%d",
                len(initial_pool), term, threshold)

    if not initial_pool:
        return []

    if on_progress:
        on_progress(0, len(initial_pool))

    # 数据获取
    batch_input = [{"code": s["code"], "name": s.get("name", "")} for s in initial_pool]
    batch_stocks = await fetch_batch(
        batch_input, semaphore=12,
        on_progress=lambda c, t: on_progress(c // 2, t) if on_progress else None,
    )

    # M2 快筛打分 (使用默认batch_scorer分类体系: 强烈推荐/推荐/中性/回避/卖出)
    scored = await score_batch(
        batch_stocks, horizon=term, semaphore=8, model_suffix="_2",
        on_progress=lambda c, t: on_progress(t // 2 + c // 2, t) if on_progress else None,
    )

    # 低于阈值的淘汰
    level_map = {"强烈推荐": 90, "推荐": 75, "中性": 60, "回避": 40, "卖出": 20}
    passing = []
    eliminated = 0
    for s in scored:
        score_data = s.get("score", {})
        level = score_data.get("level", "中性") if isinstance(score_data, dict) else "中性"
        numeric_score = level_map.get(level, 50)

        if numeric_score >= threshold:
            s["quick_screen_score"] = numeric_score
            s["quick_screen_level"] = level
            passing.append(s)
        else:
            eliminated += 1

    logger.info("[Layer 2] 快筛完成: 通过%d只, 淘汰%d只", len(passing), eliminated)

    if on_progress:
        on_progress(len(initial_pool), len(initial_pool))

    return passing


# ═══════════════════════════════════════════════════════════
# Layer 3: Formal Scoring (1:1.2差额 → 正式7Agent+3Scorer → LLM动态阈值)
# ═══════════════════════════════════════════════════════════

def calculate_candidate_quota(
    whitelist_count: int,
    initial_passing_count: int,
    target_size: int,
    ratio: float = 1.2,
) -> Dict[str, int]:
    """
    计算白名单和初筛通过各自进入精筛的配额（总纲: 白名单:初筛通过 ≈ 1:1.2）。

    规则:
      - 白名单全部进入精筛候选
      - 若白名单不足(低于理想比例), 差额由初筛补
      - 若白名单过多, 按理想比例截断
      - 总候选数 ≈ target_size × ratio

    Returns: {"whitelist_slots": N, "initial_slots": M, "total_candidates": N+M}
    """
    total_candidates = int(target_size * ratio)
    ideal_whitelist = int(total_candidates * 1.0 / (1.0 + ratio))

    # 白名单 ≤ 理想配额: 全部进入; 白名单 > 理想配额: 按理想配额截断
    whitelist_slots = min(whitelist_count, ideal_whitelist)

    initial_slots = total_candidates - whitelist_slots
    initial_slots = min(initial_slots, initial_passing_count)

    return {
        "whitelist_slots": whitelist_slots,
        "initial_slots": initial_slots,
        "total_candidates": whitelist_slots + initial_slots,
    }


async def _dynamic_threshold(scores: List[float], target_size: int) -> float:
    """
    用内置LLM根据得分分布动态设定淘汰阈值。
    波动范围: 使筛选后股票数在目标池大小的90%~110%之间。
    """
    if not scores:
        return 0.0

    scores_sorted = sorted(scores, reverse=True)
    default_cutoff_idx = min(target_size, len(scores_sorted))
    default_threshold = max(scores_sorted[default_cutoff_idx - 1] - 1, 0) if default_cutoff_idx > 0 else 0

    # 用LLM微调阈值
    try:
        from src.utils.llm_clients import OpenAICompatibleClient
        from src.utils.model_config import get_eval_model_config, get_thinking_body

        # 精筛池阈值设定属于评测调度逻辑，使用 eval_orchestrator profile (DeepSeek V4 Pro)
        # 而非生产模型，避免污染生产缓存与模型配额
        model_cfg = get_eval_model_config("eval_orchestrator")

        llm = OpenAICompatibleClient(
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
            model=model_cfg["model_name"],
            env_prefix="",
            extra_body=get_thinking_body(model_cfg["base_url"], enabled=True),
            http_timeout=60,
        )

        score_dist = scores_sorted[:target_size + 20]
        prompt = f"""你是精筛池阈值设定器。根据得分分布设定一个合理淘汰阈值。

目标池大小: {target_size}只
得分分布 (前{len(score_dist)}名): {score_dist}

要求:
1. 阈值应使通过股票数在{int(target_size * 0.9)}~{int(target_size * 1.1)}之间
2. 优先考虑得分"断层"处（相邻股票分差>5的位置）
3. 默认阈值={default_threshold}，可以调整±10分

只输出一个数字（阈值分数），不要任何其他文字:"""

        response = llm.get_completion([
            {"role": "user", "content": prompt}
        ], max_retries=1)

        nums = re.findall(r'\d+\.?\d*', str(response))
        if nums:
            threshold = float(nums[0])
            threshold = max(0, min(threshold, 90))
            logger.info("  LLM动态阈值: %.1f (默认=%.1f)", threshold, default_threshold)
            return threshold
    except Exception as e:
        logger.warning("  LLM动态阈值失败, 使用默认: %s", e)

    return default_threshold


async def formal_score_layer3(
    whitelist: List[Dict[str, Any]],
    initial_passing: List[Dict[str, Any]],
    term: str,
    target_size: int = 100,
    ratio: float = 1.2,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Layer 3: 1:1.2差额组建候选 → 正式7Agent+3Scorer打分 → LLM动态阈值淘汰 → 最终精筛池。

    Returns:
        {"pool": [...], "whitelist": [...], "blacklist": [...], "stats": {...}}
    """
    from src.stock_pool.scoring_engine import ScoringEngine

    logger.info(
        "[Layer 3] 精筛定稿开始: 白名单%d只, 初筛通过%d只, target=%d, ratio=%.1f",
        len(whitelist), len(initial_passing), target_size, ratio
    )

    # Step A: 计算配额
    quota = calculate_candidate_quota(
        len(whitelist), len(initial_passing), target_size, ratio
    )
    logger.info("  配额: 白名单%d + 初筛%d = %d",
                quota["whitelist_slots"], quota["initial_slots"], quota["total_candidates"])

    # Step B: 组建候选列表 (白名单优先 + 初筛按快筛分数排序)
    candidates = list(whitelist[:quota["whitelist_slots"]])
    initial_sorted = sorted(
        initial_passing,
        key=lambda x: x.get("quick_screen_score", 50),
        reverse=True
    )
    candidates += initial_sorted[:quota["initial_slots"]]

    logger.info("  精筛候选: %d只", len(candidates))

    if on_progress:
        on_progress(0, len(candidates))

    # Step C: 分批并发运行正式7Agent+3Scorer
    # 每批至多20只, semaphore=3并发, 批次间汇报进度, 支持增量展示
    BATCH_SIZE = 20
    scored_candidates = []
    whitelist_final = []
    blacklist_final = []
    candidates_lock = asyncio.Lock()
    sem = asyncio.Semaphore(3)
    completed_count = 0

    term_key = {"short": "short_term_score", "medium": "medium_term_score", "long": "long_term_score"}

    async def _score_one(stock: Dict[str, Any]) -> None:
        nonlocal completed_count
        from src.utils.cache_utils import read_cache, write_cache
        code = stock.get("code", "")
        name = stock.get("name", code)
        safe_code = code.replace(".", "_").replace("/", "_")
        entry = {**stock, "final_score": 50, "recommendation": ""}

        # 检查 full_analysis 缓存 (1天TTL)
        cache_date = datetime.now().strftime("%Y-%m-%d")
        cached_full = None
        try:
            cached_full = read_cache("full_pool_analysis", safe_code, cache_date)
        except Exception:
            pass
        if cached_full:
            try:
                cache_data = json.loads(cached_full)
            except (json.JSONDecodeError, TypeError, ValueError):
                cache_data = None
            sd = cache_data.get("score_data", {}) if cache_data else {}
            if sd:
                term_score = sd.get(term_key.get(term, "medium_term_score"), {})
                actual_score = term_score.get("score") if isinstance(term_score, dict) else sd.get("score", 50)
                if actual_score is None:
                    actual_score = sd.get("score", 50)
                rec = sd.get("recommendation", "")
                entry = {**stock, "final_score": actual_score or 50,
                         "recommendation": rec}
                logger.info("  %s 命中full_analysis缓存, score=%s", code, actual_score)
                async with candidates_lock:
                    if "强烈推荐" in str(rec) or (actual_score or 0) >= 85:
                        whitelist_final.append(entry)
                    elif "卖出" in str(rec) or (actual_score or 50) < 30:
                        blacklist_final.append(entry)
                    scored_candidates.append(entry)
                    completed_count += 1
                return

        engine = ScoringEngine()
        try:
            async with sem:
                result = await engine.score_stock(code, name)
            if result and result.get("score_data"):
                # 写入 full_analysis 缓存
                try:
                    write_cache("full_pool_analysis", safe_code, cache_date,
                                json.dumps({"score_data": result["score_data"]},
                                           ensure_ascii=False, default=str))
                except Exception:
                    pass
                sd = result["score_data"]
                score = sd.get("score") or 50

                term_score = sd.get(term_key.get(term, "medium_term_score"), {})
                actual_score = term_score.get("score") if isinstance(term_score, dict) else score
                if actual_score is None:
                    actual_score = score
                recommendation = sd.get("recommendation", "")

                entry = {**stock, "final_score": actual_score, "recommendation": recommendation}
        except Exception as e:
            logger.warning("  正式评分失败 %s: %s", code, e)
            entry = {**stock, "final_score": 40, "recommendation": ""}

        async with candidates_lock:
            if "强烈推荐" in str(entry.get("recommendation", "")) or entry.get("final_score", 0) >= 85:
                whitelist_final.append(entry)
                scored_candidates.append(entry)
            elif "卖出" in str(entry.get("recommendation", "")) or entry.get("final_score", 50) < 30:
                blacklist_final.append(entry)
            else:
                scored_candidates.append(entry)
            completed_count += 1
            if on_progress and completed_count % 5 == 0:
                on_progress(completed_count, len(candidates))

    # 分批执行: 每批 BATCH_SIZE 只, 批次间汇报进度
    total_candidates = len(candidates)
    for batch_start in range(0, total_candidates, BATCH_SIZE):
        batch = candidates[batch_start:batch_start + BATCH_SIZE]
        batch_idx = batch_start // BATCH_SIZE + 1
        total_batches = (total_candidates + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info("  精筛批次 %d/%d: %d只 (%d~%d)",
                    batch_idx, total_batches, len(batch),
                    batch_start + 1, min(batch_start + BATCH_SIZE, total_candidates))
        await asyncio.gather(*[_score_one(s) for s in batch])
        if on_progress:
            on_progress(min(batch_start + BATCH_SIZE, total_candidates), total_candidates)

    if on_progress:
        on_progress(len(candidates), len(candidates))

    # Step D: LLM动态阈值
    all_scores = [s.get("final_score", 50) for s in scored_candidates]
    threshold = await _dynamic_threshold(all_scores, target_size)

    # Step E: 按阈值筛选
    pool_final = [s for s in scored_candidates if s.get("final_score", 0) >= threshold]
    pool_final.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    if len(pool_final) > target_size:
        pool_final = pool_final[:target_size]

    stats = {
        "candidates": len(candidates),
        "scored": len(all_scores),
        "whitelist_final": len(whitelist_final),
        "blacklist_final": len(blacklist_final),
        "pool_size": len(pool_final),
        "dynamic_threshold": round(threshold, 1),
        "score_range": {
            "min": min(all_scores) if all_scores else 0,
            "max": max(all_scores) if all_scores else 0,
        },
    }

    logger.info("[Layer 3] 精筛定稿完成: 池%d只, 阈值=%.1f, 白%d, 黑%d",
                stats["pool_size"], threshold, stats["whitelist_final"], stats["blacklist_final"])

    return {
        "pool": pool_final,
        "whitelist": whitelist_final,
        "blacklist": blacklist_final,
        "stats": stats,
    }


# ═══════════════════════════════════════════════════════════
# Pipeline Progress Monitor (心跳 + 进度 + 卡死检测 + ETA)
# ═══════════════════════════════════════════════════════════

class PipelineProgress:
    """流水线进度追踪器: 多阶段进度、ETA、心跳、卡死检测。"""

    def __init__(self, on_progress: Optional[Callable] = None):
        self.on_progress = on_progress
        self.start_time = datetime.now()
        self.stages: Dict[str, Dict[str, Any]] = {
            "0_hard_screen":    {"label": "硬筛", "total": 1, "done": 0, "start": None, "end": None},
            "1_batch_score":    {"label": "批量粗筛", "total": 1, "done": 0, "start": None, "end": None},
            "2_stream_heap":    {"label": "流式排序", "total": 1, "done": 0, "start": None, "end": None},
            "3_formal_score":   {"label": "精筛打分", "total": 1, "done": 0, "start": None, "end": None},
        }
        self.queue_depth = 0       # Layer 3 队列当前深度
        self.last_progress_time = datetime.now()
        self.heartbeat_count = 0

    def stage_start(self, stage: str, total: int = 1):
        s = self.stages.get(stage)
        if s:
            s["start"] = datetime.now()
            s["total"] = total
            s["done"] = 0

    def stage_progress(self, stage: str, done: int):
        """更新阶段进度, done 只允许单调递增 (防止 UI 进度条倒退/清零)。

        调用方 (如 _layer3_consumer 的 completed 计数) 本身是累计值, 但若有
        其他路径传入更小值时, 此处兜底忽略。last_progress_time 始终更新以
        保证卡顿检测不误报。
        """
        s = self.stages.get(stage)
        if s:
            if done >= s.get("done", 0):
                s["done"] = done
        self.last_progress_time = datetime.now()

    def stage_end(self, stage: str):
        s = self.stages.get(stage)
        if s:
            s["end"] = datetime.now()
            s["done"] = s["total"]

    @property
    def elapsed_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def overall_progress_pct(self) -> float:
        """整体进度百分比 (加权: L0=2%, L1=40%, L2=5%, L3=53%)."""
        weights = {"0_hard_screen": 0.02, "1_batch_score": 0.40,
                   "2_stream_heap": 0.05, "3_formal_score": 0.53}
        pct = 0.0
        for stage, w in weights.items():
            s = self.stages[stage]
            if s["total"] > 0:
                pct += w * min(s["done"] / s["total"], 1.0)
        return pct

    @property
    def eta_seconds(self) -> float:
        """预估剩余时间 (秒)."""
        pct = self.overall_progress_pct
        if pct < 0.01:
            return 999999  # 还没开始
        elapsed = self.elapsed_seconds
        if elapsed < 1:
            return 999999
        total_est = elapsed / pct
        return max(0, total_est - elapsed)

    @property
    def eta_str(self) -> str:
        s = self.eta_seconds
        if s > 86400:
            return "计算中..."
        if s > 3600:
            return f"{s/3600:.1f}h"
        if s > 60:
            return f"{s/60:.0f}min"
        return f"{s:.0f}s"

    @property
    def stall_seconds(self) -> float:
        """上次进度更新以来经过的秒数."""
        return (datetime.now() - self.last_progress_time).total_seconds()

    def heartbeat(self):
        """生成心跳日志."""
        self.heartbeat_count += 1
        parts = []
        for stage, s in self.stages.items():
            if s["start"] and not s["end"]:
                pct = min(s["done"] / s["total"] * 100, 100) if s["total"] > 0 else 0
                parts.append(f"{s['label']}={pct:.0f}%")
        return (f"[心跳 #{self.heartbeat_count}] 运行{self.elapsed_seconds/60:.0f}min, "
                f"ETA={self.eta_str}, Q={self.queue_depth}, "
                f"进度: {', '.join(parts)}")

    def emit_progress(self):
        """发射结构化进度到回调."""
        if self.on_progress:
            try:
                self.on_progress({
                    "overall_pct": round(self.overall_progress_pct * 100, 1),
                    "elapsed_s": round(self.elapsed_seconds),
                    "eta_s": round(self.eta_seconds),
                    "eta_str": self.eta_str,
                    "queue_depth": self.queue_depth,
                    "stages": {
                        k: {"label": v["label"], "pct": round(
                             min(v["done"] / v["total"] * 100, 100) if v["total"] > 0 else 0, 1),
                            "done": v["done"], "total": v["total"]}
                        for k, v in self.stages.items()
                    },
                    "stall_s": round(self.stall_seconds),
                })
            except Exception:
                pass

    def check_stall(self) -> Optional[str]:
        """检查是否卡死。返回警告消息或 None."""
        stall = self.stall_seconds
        if stall > 900:   # 15分钟
            return f"⚠️ 流水线可能卡死: {stall/60:.0f}分钟无进度更新, 最后活跃层={self._active_stage()}"
        if stall > 300:   # 5分钟
            logger.warning("流水线 %d 分钟无进度, 最后活跃: %s",
                          stall // 60, self._active_stage())
        return None

    def _active_stage(self) -> str:
        for k, v in self.stages.items():
            if v["start"] and not v["end"]:
                return v["label"]
        return "无"


async def _heartbeat_loop(progress: PipelineProgress, interval: float = 30.0):
    """心跳协程: 定期打印进度和ETA."""
    while True:
        await asyncio.sleep(interval)
        stall_msg = progress.check_stall()
        if stall_msg:
            logger.warning(stall_msg)
        logger.info(progress.heartbeat())
        progress.emit_progress()


# ═══════════════════════════════════════════════════════════
# Layer 3: Async Consumer (流式队列消费者)
# ═══════════════════════════════════════════════════════════

async def _layer3_consumer(
    queue: asyncio.Queue,
    results: List[Dict[str, Any]],
    whitelist_final: List[Dict[str, Any]],
    blacklist_final: List[Dict[str, Any]],
    lock: asyncio.Lock,
    term: str,
    progress: Optional[PipelineProgress] = None,
    heartbeat_interval: float = 30.0,
    per_task_timeout: float = 900.0,
):
    """Layer 3 异步消费者: 从队列取股票 → ScoringEngine 打分 → 分类。

    一直运行到收到毒丸信号(None), 由 run_pool_update_v3 控制生命周期。
    内置心跳、超时保护、进度上报。
    """
    from src.stock_pool.scoring_engine import ScoringEngine

    # 共用单个 ScoringEngine: 避免 N 并发各自初始化 MCP client 导致 TaskGroup 崩溃
    # 并发数 5 (2026-07-02 从 8 回退): 冷启动时 8 并发 × 7 agent ≈ 56 路 MCP stdio
    # 调用会严重阻塞 Tushare/MCP 通道, 实测单只股票 Phase 1 从 13s 涨到 739s,
    # 5 并发 × 7 agent ≈ 35 路更稳健, 与 CLAUDE.md 架构文档对齐。
    shared_engine = ScoringEngine()
    sem = asyncio.Semaphore(5)
    completed = 0
    total_dispatched = 0
    last_heartbeat = datetime.now()
    term_key = {"short": "short_term_score", "medium": "medium_term_score",
                 "long": "long_term_score"}
    cache_date = datetime.now().strftime("%Y-%m-%d")

    async def _score_one(stock: Dict[str, Any]):
        nonlocal completed, last_heartbeat
        from src.utils.cache_utils import read_cache, write_cache
        code = stock.get("code", "")
        name = stock.get("name", code)
        safe_code = code.replace(".", "_").replace("/", "_")
        entry = {**stock, "final_score": 50, "recommendation": ""}

        # full_analysis 缓存检查
        cached_full = None
        try:
            cached_full = read_cache("full_pool_analysis", safe_code, cache_date)
        except Exception:
            pass
        if cached_full:
            try:
                cache_data = json.loads(cached_full)
            except (json.JSONDecodeError, TypeError, ValueError):
                cache_data = None
            sd = cache_data.get("score_data", {}) if cache_data else {}
            if sd:
                ts = sd.get(term_key.get(term, "medium_term_score"), {})
                actual_score = ts.get("score") if isinstance(ts, dict) else sd.get("score", 50)
                if actual_score is None:
                    actual_score = sd.get("score", 50)
                rec = sd.get("recommendation", "")
                entry = {**stock, "final_score": actual_score or 50, "recommendation": rec}
                logger.debug("  %s 命中full_analysis缓存, score=%s", code, actual_score)
                async with lock:
                    if "强烈推荐" in str(rec) or (actual_score or 0) >= 85:
                        whitelist_final.append(entry)
                    elif "卖出" in str(rec) or (actual_score or 50) < 30:
                        blacklist_final.append(entry)
                    results.append(entry)
                    completed += 1
                    last_heartbeat = datetime.now()
                    if progress:
                        progress.stage_progress("3_formal_score", completed)
                        progress.queue_depth = queue.qsize()
                queue.task_done()
                return

        try:
            # 共用 engine 实例, 只靠 semaphore 控制并发
            async with sem:
                result = await asyncio.wait_for(
                    shared_engine.score_stock(code, name),
                    timeout=per_task_timeout,
                )
            if result and result.get("score_data"):
                try:
                    write_cache("full_pool_analysis", safe_code, cache_date,
                                json.dumps({"score_data": result["score_data"]},
                                           ensure_ascii=False, default=str))
                except Exception:
                    pass
                sd = result["score_data"]
                score = sd.get("score") or 50
                ts = sd.get(term_key.get(term, "medium_term_score"), {})
                actual_score = ts.get("score") if isinstance(ts, dict) else score
                if actual_score is None:
                    actual_score = score
                rec = sd.get("recommendation", "")
                entry = {**stock, "final_score": actual_score, "recommendation": rec}
            else:
                entry = {**stock, "final_score": 50, "recommendation": ""}
        except asyncio.TimeoutError:
            logger.warning("  Layer3 超时 %s (%.0fs), 使用默认分数", code, per_task_timeout)
            entry = {**stock, "final_score": 40, "recommendation": "超时"}
        except Exception as e:
            logger.warning("  Layer3 评分失败 %s: %s", code, e)
            entry = {**stock, "final_score": 40, "recommendation": ""}

        async with lock:
            if "强烈推荐" in str(entry.get("recommendation", "")) or entry.get("final_score", 0) >= 85:
                whitelist_final.append(entry)
            elif "卖出" in str(entry.get("recommendation", "")) or entry.get("final_score", 50) < 30:
                blacklist_final.append(entry)
            results.append(entry)
            completed += 1
            last_heartbeat = datetime.now()
            if progress:
                progress.stage_progress("3_formal_score", completed)
                progress.queue_depth = queue.qsize()
        queue.task_done()

    # 主循环: 从队列拉取任务, 分派到 _score_one
    while True:
        try:
            stock = await asyncio.wait_for(queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            # 心跳检查
            hb_age = (datetime.now() - last_heartbeat).total_seconds()
            if progress:
                progress.queue_depth = queue.qsize()
                progress.last_progress_time = datetime.now()  # 防卡顿误报
            if hb_age > heartbeat_interval:
                logger.info("[L3心跳] 运行中: 已完成%d/%d, Q=%d, 距上次完成%.0fs",
                           completed, completed + queue.qsize(),
                           queue.qsize(), hb_age)
            continue
        if stock is None:  # 毒丸信号
            queue.task_done()
            if progress:
                progress.queue_depth = queue.qsize()
            break
        total_dispatched += 1
        asyncio.create_task(_score_one(stock))

    # 等待所有 in-flight 任务完成 (queue.join 在 task_done 计数归零后返回)
    logger.info("[L3消费者] 收到毒丸, 等待剩余%d个任务完成...", queue.qsize())
    await queue.join()
    logger.info("[L3消费者] 所有任务完成: 已处理%d只", completed)


# ═══════════════════════════════════════════════════════════
# Safe Queue Put (队列背压保护)
# ═══════════════════════════════════════════════════════════

async def _safe_queue_put(queue: asyncio.Queue, item: Any, label: str = "",
                          timeout: float = 10.0):
    """安全入队: 超时保护, 队列满时等待而非死锁.

    如果队列满, 等待 timeout 秒后仍无法入队, 记录警告并丢弃 (非毒丸).
    毒丸 (item is None) 不会丢弃——必须入队以确保消费者退出.
    """
    try:
        await asyncio.wait_for(queue.put(item), timeout=timeout)
    except asyncio.TimeoutError:
        if item is None:
            logger.warning("[%s] 毒丸入队超时, 重试...", label)
            await queue.put(item)  # 毒丸必须入队, 无限等待
        else:
            logger.warning("[%s] 队列满 %.0fs, 丢弃 %s",
                          label, timeout,
                          item.get("code", str(item)) if isinstance(item, dict) else str(item))


# ═══════════════════════════════════════════════════════════
# Main Orchestrator V3: 流式流水线
# ═══════════════════════════════════════════════════════════

async def run_pool_update_v3(
    term: str = "short",
    on_stage: Optional[Callable] = None,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """精筛池流水线 V3: Layer 0→1→(2+3) 流式并发 + L2 pre-fetch 复用 (2026-07 优化)。

    架构:
      Layer 0: 硬筛 (~30s)
      Layer 1: M1/M3 流式批量粗筛 (100只/批), 逐批分叉:
        → 强烈推荐 → whitelist → Layer 3 立即 dispatch
        → 推荐     → raw_data 累积 → Layer 2 后台流水线 (不阻塞 L1)
        → 中性/回避 → 放弃
        → 卖出     → blacklist
      Layer 2: DSV4Pro 流式双堆 top-α, **复用 Layer 1 raw_data 跳过 fetch_batch**
              (省 10-20min/cold-start). τ 收敛后 dispatch top-α 到 Layer 3.
      Layer 3: 5 并发异步队列消费者, 7Agent+3Scorer 正式评分
      截断: target_size × 1.2 进, target_size 出

    预估冷启动耗时: short ~1.0-1.5h, medium ~1.0-1.5h, long ~0.8-1.0h.
    """
    from src.eval.config import get as eval_get

    pm = PoolManager()
    target_size = {"short": 100, "medium": 80, "long": 60}[term]
    ratio = eval_get("whitelist_pass_ratio", 1.2)

    def _stage(name, msg):
        logger.info("[PoolUpdateV3:%s] %s: %s", term, name, msg)
        if on_stage:
            on_stage(name, msg)

    result = {"term": term, "stages": {}}

    # ── 进度/心跳/ETA监控 ──
    progress = PipelineProgress(on_progress=on_progress)
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(progress, interval=30.0)
    )

    # ── Layer 0: Hard Screen ──
    _stage("0_hard_screen", "从Tushare获取全A股, 硬筛...")
    progress.stage_start("0_hard_screen")
    ts_stocks = await hard_screen()
    progress.stage_end("0_hard_screen")
    result["stages"]["0_hard_screen"] = f"硬筛后{len(ts_stocks)}只"
    if len(ts_stocks) < target_size:
        heartbeat_task.cancel()
        return {"error": f"硬筛后仅{len(ts_stocks)}只"}

    # ── 初始化共享状态 ──
    whitelist: List[Dict] = []
    blacklist: List[Dict] = []
    layer3_results: List[Dict] = []
    whitelist_final: List[Dict] = []
    blacklist_final: List[Dict] = []
    layer3_lock = asyncio.Lock()
    layer3_queue_maxsize = max(200, target_size * 2)
    layer3_queue = asyncio.Queue(maxsize=layer3_queue_maxsize)
    code_to_stock: Dict[str, Dict] = {}
    recommended_score_map: Dict[str, float] = {}  # code → layer2_score (fallback用)
    # Layer 1 预抓取的 raw data (供 Layer 2 复用, 避免二次 fetch)
    layer1_raw_data_map: Dict[str, Dict[str, Any]] = {}
    # Layer 2 后台任务队列 (流水线: L2 与 L1 并行, 避免阻塞)
    layer2_pending_tasks: List[asyncio.Task] = []
    layer2_task_lock = asyncio.Lock()

    # Layer 2 流式堆 + fallback flag
    def alpha_fn():
        return calc_alpha(len(whitelist), stream_heap.total, target_size, ratio)
    stream_heap = StreamingTopAlphaHeap(alpha_fn=alpha_fn, epsilon=3.0, stable_batches=3)
    layer2_fallback = False  # 如果 DSV4Pro 失败, 用 Layer 1 兜底

    # Layer 3 消费者
    _stage("3_start", "启动 Layer 3 消费者 (5并发, 15min超时)...")
    progress.stage_start("3_formal_score", total=target_size * 2)
    l3_task = asyncio.create_task(
        _layer3_consumer(
            layer3_queue, layer3_results, whitelist_final, blacklist_final,
            layer3_lock, term, progress=progress,
            heartbeat_interval=30.0, per_task_timeout=900.0,
        )
    )

    # ── Layer 1: 流式批量粗筛 + 分叉路由 ──
    _stage("1_batch_score", f"流式批量粗筛{len(ts_stocks)}只 (M1/M3)...")
    progress.stage_start("1_batch_score", total=len(ts_stocks))
    layer1_stats = {"whitelist": 0, "recommended": 0, "neutral_avoid": 0, "sell": 0}
    layer1_scored = 0

    # 数据获取阶段进度: 每回调一次推进 total 的 0.15% (约 7 只股票等效)
    _fetch_step = len(ts_stocks) * 0.0015
    _fetch_acc = [0.0]
    def _progress_touch():
        progress.last_progress_time = datetime.now()
        _fetch_acc[0] += _fetch_step
        progress.stage_progress("1_batch_score", int(min(_fetch_acc[0], len(ts_stocks) * 0.35)))

    async for batch in batch_score_layer1_stream(
        ts_stocks, term, on_progress=None,
        progress_touch=_progress_touch
    ):
        layer1_scored += len(batch)
        progress.stage_progress("1_batch_score", layer1_scored)
        progress.emit_progress()

        new_recommended = []
        for stock in batch:
            code = stock.get("code", "")
            # 累积 L1 raw data (供 Layer 2 复用, 避免冗余 fetch)
            raw_data = stock.pop("_raw_data", None)
            if raw_data and code:
                layer1_raw_data_map[code] = raw_data
            code_to_stock[code] = stock
            level = stock.get("layer1_level", "")

            if level == "强烈推荐":
                whitelist.append(stock)
                layer1_stats["whitelist"] += 1
                await _safe_queue_put(layer3_queue, stock, "L3-whitelist")
            elif level == "推荐":
                new_recommended.append(stock)
                layer1_stats["recommended"] += 1
            elif level == "卖出":
                blacklist.append(stock)
                layer1_stats["sell"] += 1
            else:  # 中性/回避
                layer1_stats["neutral_avoid"] += 1

        # 新推荐 → Layer 2 打分 → 喂流式堆
        # 流水线: Layer 2 后台执行, 不阻塞 Layer 1 处理下一批
        if new_recommended:
            chunk_snapshot = list(new_recommended)

            async def _process_l2_chunk(chunk, raw_map_snapshot):
                nonlocal layer2_fallback
                try:
                    layer2_scored = await score_layer2_batch(
                        chunk, term, pre_fetched_data=raw_map_snapshot,
                    )
                    for s in layer2_scored:
                        numeric = s.get("score", 50)
                        if isinstance(numeric, (int, float)):
                            code = s.get("code", "")
                            recommended_score_map[code] = float(numeric)
                            stock = code_to_stock.get(code)
                            if stock:
                                stock["layer2_score"] = float(numeric)
                                stream_heap.feed(float(numeric), stock)
                    stream_heap.batch_done()
                except Exception as e:
                    logger.warning("Layer 2 DSV4Pro 失败, 启用 Layer 1 兜底排序: %s", e)
                    layer2_fallback = True
                    # Layer 1 兜底: 用 confidence 近似分数
                    conf_map = {"高": 80, "中": 50, "低": 30}
                    for stock in chunk:
                        code = stock.get("code", "")
                        conf = stock.get("layer1_confidence", "中")
                        fallback_score = conf_map.get(conf, 50)
                        recommended_score_map[code] = fallback_score
                        stock["layer2_score"] = fallback_score
                        stream_heap.feed(fallback_score, stock)
                    stream_heap.batch_done()

                # τ 稳定 → dispatch
                if stream_heap.is_tau_stable():
                    for stock in stream_heap.get_undispatched():
                        await _safe_queue_put(layer3_queue, stock, "L3-heap")

            # 仅把本批 chunk 对应的 raw_data 快照传给后台任务, 避免 dict 共享竞态
            raw_snapshot = {code: layer1_raw_data_map[code]
                            for code in (s.get("code", "") for s in chunk_snapshot)
                            if code in layer1_raw_data_map}
            task = asyncio.create_task(
                _process_l2_chunk(chunk_snapshot, raw_snapshot)
            )
            async with layer2_task_lock:
                layer2_pending_tasks.append(task)

        progress.queue_depth = layer3_queue.qsize()

    # Layer 1 完成: 等待所有后台 L2 任务结束
    if layer2_pending_tasks:
        _stage("1_wait_l2", f"等待 {len(layer2_pending_tasks)} 个 L2 后台任务完成...")
        await asyncio.gather(*layer2_pending_tasks, return_exceptions=True)
        async with layer2_task_lock:
            layer2_pending_tasks.clear()

    # Layer 1 完成
    progress.stage_end("1_batch_score")
    result["stages"]["1_batch_score"] = (
        f"白名单{layer1_stats['whitelist']}只, 推荐{layer1_stats['recommended']}只, "
        f"放弃{layer1_stats['neutral_avoid']}只, 卖出{layer1_stats['sell']}只"
    )
    logger.info("[Layer 1-Stream] 完成: %s (fallback=%s, L1 pre-fetch 复用 %d 只)",
                result["stages"]["1_batch_score"], layer2_fallback,
                len(layer1_raw_data_map))

    # ── Layer 2 finalize ──
    progress.stage_start("2_stream_heap", total=stream_heap.total)
    _stage("2_finalize", f"Layer 2 流式堆 finalize: {stream_heap.total}只推荐, "
           f"high={stream_heap.high_size}, τ={stream_heap.tau:.1f}, fallback={layer2_fallback}")
    for stock in stream_heap.finalize():
        await _safe_queue_put(layer3_queue, stock, "L3-finalize")
    progress.stage_end("2_stream_heap")
    progress.queue_depth = layer3_queue.qsize()

    # ── Layer 3 收尾 ──
    _stage("3_wait", f"等待 Layer 3 完成 (队列剩余{layer3_queue.qsize()}只)...")
    await asyncio.sleep(2.0)  # 让最后几个入队任务被消费
    await _safe_queue_put(layer3_queue, None, "poison")  # 毒丸
    try:
        await asyncio.wait_for(l3_task, timeout=3600.0)  # 最多再等 1h
    except asyncio.TimeoutError:
        logger.error("Layer 3 消费者超时 (1h), 强制结束")
        l3_task.cancel()
    progress.stage_end("3_formal_score")

    # 停止心跳
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

    # ── 截断到 target_size ──
    async with layer3_lock:
        scored = sorted(layer3_results, key=lambda x: x.get("final_score", 0), reverse=True)
        pool_final = scored[:target_size]

    stats = {
        "candidates": len(layer3_results),
        "scored": len(layer3_results),
        "whitelist_final": len(whitelist_final),
        "blacklist_final": len(blacklist_final),
        "pool_size": len(pool_final),
        "layer2_fallback": layer2_fallback,
        "elapsed_s": round(progress.elapsed_seconds),
        "score_range": {
            "min": min((s.get("final_score", 0) for s in layer3_results), default=0),
            "max": max((s.get("final_score", 0) for s in layer3_results), default=0),
        },
    }

    # ── 持久化 ──
    pm.clean_expired_blacklist()
    pm.update_pool(term, pool_final)

    blacklist_expiry = eval_get("blacklist_expiry_days", 120)
    all_blacklist = blacklist + blacklist_final
    for b in all_blacklist:
        code = b.get("code", "")
        if code and not pm.is_blacklisted(code):
            pm.add_to_blacklist(
                code,
                b.get("layer1_reason", b.get("recommendation", "批量粗筛判定")),
                expiry_days=blacklist_expiry,
            )

    result["pool"] = pool_final
    result["whitelist"] = whitelist_final
    result["blacklist"] = blacklist_final
    result["stats"] = stats
    result["final_pool_size"] = len(pool_final)

    _stage("done", f"精筛池[{term}] V3完成: {len(pool_final)}只, "
           f"耗时{progress.elapsed_seconds/60:.0f}min, "
           f"(白{layer1_stats['whitelist']}, 推{layer1_stats['recommended']}, "
           f"弃{layer1_stats['neutral_avoid']}, 卖{layer1_stats['sell']})")
    progress.emit_progress()
    return result


# ═══════════════════════════════════════════════════════════
# Main Orchestrator (V2 兼容, 保持不变)
# ═══════════════════════════════════════════════════════════

async def run_pool_update(
    term: str = "short",
    on_stage: Optional[Callable] = None,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    完整四层精筛池更新管线 — 严格按照总纲 §4.1 执行。

    管线:
      Layer 0: 硬筛(ST/BJ/B/新股/低成交额)
      Layer 1: M1/M3批量粗筛分档(强烈推荐→白名单, 推荐→初筛池, 中性/回避→丢弃, 卖出→黑名单)
      Layer 2: M2快筛过滤初筛池(便宜模型, 淘汰明显不行的)
      Layer 3: 1:1.2差额 → 正式7Agent+3Scorer → LLM动态阈值 → 最终精筛池

    Args:
        term: short/medium/long
        on_stage: 阶段回调(stage_name: str, message: str)
        on_progress: 进度回调(completed: int, total: int, stage: str)

    Returns:
        完整结果字典 {"term", "stages", "pool", "whitelist", "blacklist", "stats", "final_pool_size"}
    """
    from src.eval.config import get as eval_get

    pm = PoolManager()
    target_size = {"short": 100, "medium": 80, "long": 60}[term]
    ratio = eval_get("whitelist_pass_ratio", 1.2)

    def _stage(name, msg):
        logger.info("[PoolUpdate:%s] %s: %s", term, name, msg)
        if on_stage:
            on_stage(name, msg)

    result = {"term": term, "stages": {}}

    # ── Layer 0: Hard Screen ──
    _stage("0_hard_screen", "从Tushare获取全A股, 硬筛去除ST/新股/BJ/B股/低成交额...")
    ts_stocks = await hard_screen()
    result["stages"]["0_hard_screen"] = f"硬筛后{len(ts_stocks)}只"

    if len(ts_stocks) < target_size:
        return {"error": f"硬筛后仅{len(ts_stocks)}只, 不足目标{target_size}只"}

    # ── Layer 1: Batch Scoring ──
    _stage("1_batch_score", f"用M1/M3批量粗筛{len(ts_stocks)}只...")
    layer1 = await batch_score_layer1(ts_stocks, term, on_progress=on_progress)
    result["stages"]["1_batch_score"] = (
        f"白名单{len(layer1['whitelist'])}只, "
        f"初筛池{len(layer1['initial_pool'])}只, "
        f"黑名单{len(layer1['blacklist'])}只"
    )

    # ── Layer 2: Quick Screen ──
    _stage("2_quick_screen", f"用M2快筛初筛池{len(layer1['initial_pool'])}只...")
    initial_passing = await quick_screen_layer2(layer1["initial_pool"], term, on_progress=on_progress)
    result["stages"]["2_quick_screen"] = f"快筛通过{len(initial_passing)}只"

    # ── Layer 3: Formal Scoring ──
    total_candidates = len(layer1["whitelist"]) + len(initial_passing)
    _stage("3_formal_score",
           f"1:{ratio}差额组建候选({total_candidates}只) → 正式7Agent+3Scorer → LLM动态阈值...")
    layer3 = await formal_score_layer3(
        layer1["whitelist"], initial_passing,
        term=term, target_size=target_size, ratio=ratio,
        on_progress=on_progress,
    )

    # ── Store ──
    # 清理过期黑名单后存储
    pm.clean_expired_blacklist()
    pm.update_pool(term, layer3["pool"])

    # 黑名单入库 (Layer1 + Layer3)
    blacklist_expiry = eval_get("blacklist_expiry_days", 120)
    all_blacklist = layer1["blacklist"] + layer3.get("blacklist", [])
    for b in all_blacklist:
        code = b.get("code", "")
        if code and not pm.is_blacklisted(code):
            pm.add_to_blacklist(
                code,
                b.get("layer1_reason", b.get("recommendation", "批量粗筛判定")),
                expiry_days=blacklist_expiry,
            )

    result["stages"]["3_formal_score"] = (
        f"精筛池{len(layer3['pool'])}只, "
        f"白名单{len(layer3['whitelist'])}只, "
        f"黑名单{len(layer3['blacklist'])}只"
    )
    result["pool"] = layer3["pool"]
    result["whitelist"] = layer3["whitelist"]
    result["blacklist"] = layer3["blacklist"]
    result["stats"] = layer3["stats"]
    result["final_pool_size"] = len(layer3["pool"])

    _stage("done",
           f"精筛池[{term}]更新完成: {len(layer3['pool'])}只 "
           f"(阈值={layer3['stats'].get('dynamic_threshold', 'N/A')})")
    return result
