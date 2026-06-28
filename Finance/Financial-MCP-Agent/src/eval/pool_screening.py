"""
精筛股票池四层筛选管线 — 严格按照评分智能体开发总纲 §4 实现。

四层管线:
  Layer 0: 纯代码硬筛 (去ST/*ST/新股/低流动/北交所/B股)
  Layer 1: M1/M3 批量粗筛分档 (强烈推荐→白名单, 买入/谨慎买入/观望→初筛池, 卖出→黑名单)
  Layer 2: M2 快筛过滤初筛池 (便宜模型, 淘汰明显不行的)
  Layer 3: 1:1.2差额精筛 (白名单+初筛通过→正式7Agent+3Scorer→LLM动态阈值→最终精筛池)
"""
import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable, Optional

from src.eval.pool_manager import PoolManager

logger = logging.getLogger(__name__)


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

    min_daily_amount = eval_get("hard_screen_min_daily_amount", 20000000)
    logger.info("[Layer 0] 硬筛开始: 最低日均成交额=%d万", min_daily_amount // 10000)

    # 获取全A股列表
    stock_list = tushare_call("stock_basic", {
        "list_status": "L",
        "exchange": "",
    }, fields="ts_code,name,list_date,industry")
    if not stock_list or "items" not in stock_list:
        raise RuntimeError("Tushare stock_basic查询失败")

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

# 总纲 §4.1 第1层分类标准
LAYER1_LEVELS = ["强烈推荐", "买入", "谨慎买入", "观望", "卖出"]

# 总纲5级 → 3档 映射 (只有"卖出"进黑名单)
LEVEL_TO_TIER = {
    "强烈推荐": "whitelist",
    "买入": "initial_pool",
    "谨慎买入": "initial_pool",
    "观望": "initial_pool",
    "卖出": "blacklist",
}

# Layer 1 专用 System Prompt (使用总纲分类标准)
_LAYER1_SYSTEM_PROMPT = """你是 A 股批量筛选器。你的任务是对每只股票基于其数据给出 5 级分类和一句话理由。

## 分类标准（总纲 §4.1 精筛池筛选第1层）

| 级别 | 含义 | 判定标准 |
|------|------|---------|
| 强烈推荐 | 白名单 | 基本面优秀 + 估值合理或低估 + 无明显风险信号 |
| 买入 | 初筛池 | 基本面良好 + 估值合理 + 有一定投资价值 |
| 谨慎买入 | 初筛池 | 基本面尚可但有瑕疵 / 估值略偏高 / 存在一定不确定性 |
| 观望 | 初筛池 | 多空因素交织 / 数据不足无法判断 / 需观察后续变化 |
| 卖出 | 黑名单 | 基本面恶化 / 严重高估 / 重大风险 / ST类 |

## 核心原则
1. **宁可保守不可激进**：数据不明确时选"观望"，绝不猜测
2. **数据引用规则**：只能引用下方股票数据中实际出现的数字，严禁编造
3. **跨行业可比**：参考行业估值基准，银行 PE 6 倍可能合理，科技 PE 60 倍可能低估
4. **ST/*ST 股票一律标"卖出"**
5. **"强烈推荐"宁缺毋滥**：只有基本面+估值+成长性+风险四方面都优秀的股票才能给此评级

## 输出格式
严格输出 JSON 数组（不要 markdown 代码块标记）：

[{"code": "sh.603871", "level": "买入", "confidence": "高",
  "reason": "低估值+高ROE+行业景气(30字内)", "risk": "原材料涨价(30字内)"},
 ...]
"""


def classify_batch_result(score: Dict[str, str]) -> str:
    """将批量打分结果按总纲标准分类为 whitelist / initial_pool / blacklist"""
    level = score.get("level", "观望")
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
        semaphore=6,
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


# ═══════════════════════════════════════════════════════════
# Layer 2: M2 Quick Screen (快筛模型过滤初筛池)
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
        batch_input, semaphore=6,
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
# Main Orchestrator
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
      Layer 1: M1/M3批量粗筛分档(强烈推荐→白名单, 买入/谨慎买入/观望→初筛池, 卖出→黑名单)
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
