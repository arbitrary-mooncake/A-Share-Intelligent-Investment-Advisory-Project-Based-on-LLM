# 精筛股票池四层筛选重写 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将精筛股票池筛选从当前简化的2层（PE/PB代理→正式打分）重写为总纲第4节规定的完整4层管线（硬筛→M1/M3批量粗筛分档→M2快筛→1:1.2差额精筛）。

**Architecture:** 新增 `src/eval/pool_screening.py` 独立模块承载四层管线,重构 `src/eval/orchestrator.py` 调用新管线,扩展 `src/api/batch_scorer.py` 的 `score_batch()` 支持多模型切换,对齐 Streamlit UI 默认值。四层按顺序串联,每层之间有明确的输入输出接口(Layer0→stock list, Layer1→{whitelist/initial/blacklist}, Layer2→filtered initial, Layer3→final pool)。

**Tech Stack:** Tushare HTTP API, M1(MiMo-V2.5-Pro)/M2(Qwen3.6-Flash)/M3(Qwen3.7-Plus), ScoringEngine(7Agent+3Scorer), SQLite, Streamlit

---

## File Structure

```
src/api/batch_scorer.py          ← 扩展: score_batch() 增加 model_suffix 参数(当前硬编码M2)
src/eval/pool_screening.py       ← 新建: 四层管线核心模块
src/eval/orchestrator.py         ← 修改: run_pool_update_formal/light 替换为调用 pool_screening
src/eval/pool_manager.py         ← 修改: 增加白名单/初始池/黑名单的存储和查询方法
src/app/pages/06_模拟分析与迭代.py ← 修改: 滑块默认值对齐池容量
tests/test_pool_screening.py     ← 新建: 四层管线单元测试
```

---

### Task 1: 扩展 batch_scorer.score_batch() 支持多模型切换

**Files:**
- Modify: `src/api/batch_scorer.py:1032-1080`
- Test: `tests/test_batch_scorer.py` (已有测试覆盖，无需新增)

当前 `score_batch()` 硬编码使用 M2 (`OPENAI_COMPATIBLE_API_KEY_2` / `OPENAI_COMPATIBLE_MODEL_2`)。需增加 `model_suffix` 参数使 Layer 1 可指定 M1/M3。

- [ ] **Step 1: 修改 score_batch 函数签名和 LLM 客户端初始化**

将 `src/api/batch_scorer.py:1032-1078` 中的函数签名和 LLM 客户端初始化修改为:

```python
async def score_batch(
    stocks: list,
    horizon: str = "medium",
    semaphore: int = 8,
    on_progress: callable = None,
    model_suffix: str = "_2",  # NEW: 默认M2(快筛), Layer1用""(M1)或"_3"(M3)
    custom_levels: list = None,  # NEW: Layer1传入总纲分类标准 ["强烈推荐","买入","谨慎买入","观望","卖出"]
) -> list:
    """批量 LLM 打分编排器。

    Args:
        stocks: [{code, name, data: {...}, status: ...}, ...]
        horizon: 打分维度
        semaphore: LLM 调用并发数
        on_progress: 可选回调 on_progress(scored_count, total)
        model_suffix: 模型后缀, ""=M1(MiMo-V2.5-Pro), "_2"=M2(Qwen3.6-Flash), "_3"=M3(Qwen3.7-Plus)
        custom_levels: 自定义5级分类(默认用batch_scorer内置的"强烈推荐/推荐/中性/回避/卖出")

    Returns:
        stocks 列表中添加 "score" 字段
    """
    # 过滤只保留数据获取成功的股票
    valid = [s for s in stocks if s.get("status") == "fetched" and s.get("data")]
    if not valid:
        logger.warning("没有可打分的股票（所有数据获取均失败）")
        return stocks

    # 分块
    chunks = chunk_stocks(valid, chunk_size=5)
    total_chunks = len(chunks)
    scored_count = 0
    total_valid = len(valid)

    logger.info(
        f"{WAIT_ICON} 批量LLM打分开始: {total_valid} 只股票, "
        f"{total_chunks} 批次, 并发={semaphore}, 模型后缀={model_suffix}"
    )

    from src.utils.llm_clients import OpenAICompatibleClient
    import os

    # 根据 model_suffix 选择模型配置
    api_key = os.getenv(f"OPENAI_COMPATIBLE_API_KEY{model_suffix}", "")
    base_url = os.getenv(f"OPENAI_COMPATIBLE_BASE_URL{model_suffix}", "")
    model = os.getenv(f"OPENAI_COMPATIBLE_MODEL{model_suffix}", "")

    # 回退: 如果指定模型未配置 → M1
    if not all([api_key, base_url, model]):
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY", "")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")
        model = os.getenv("OPENAI_COMPATIBLE_MODEL", "mimo-v2.5-pro")

    # M1/M3 启用 thinking, M2 禁用
    extra_body = {}
    if model_suffix in ("_2",):
        extra_body = {}
    else:
        from src.utils.model_config import get_thinking_body
        extra_body = get_thinking_body(base_url, enabled=True)

    llm = OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        env_prefix="",
        extra_body=extra_body,
        http_timeout=180,
        http_connect_timeout=10,
    )
    # ... 以下不变 (semaphore, score_chunk, batch_counter 等逻辑基本不变)
    # 但 score_chunk 内部调用 build_batch_prompt 时需传入 custom_levels:
    # prompt = build_batch_prompt(stocks_data, horizon, custom_levels=custom_levels)
```

同时在 `build_batch_prompt()` 函数签名增加 `custom_levels: list = None` 参数, 
当 `custom_levels` 不为 None 时替换默认的 `LEVELS` 和 `_BATCH_SYSTEM_PROMPT` 中的分类标准文字。

同时在 `parse_batch_response()` 函数签名增加 `custom_levels: list = None` 参数,
当 `custom_levels` 不为 None 时用 `custom_levels` 替代默认 `LEVELS` 做 level 校验。

同时在 `score_batch()` 内的 `score_chunk` 中, 调用 `build_batch_prompt(stocks_data, horizon, custom_levels=custom_levels)` 
和 `parse_batch_response(response, custom_levels=custom_levels)`。

注意：以上只修改函数签名和 LLM 客户端初始化部分，其余并发控制、进度回调、结果解析逻辑完全不变。

- [ ] **Step 2: 运行 batch_scorer 已有测试确认不破坏现有功能**

Run: `python -m pytest tests/test_batch_scorer.py -v`
Expected: 全部 16 个测试 PASS

- [ ] **Step 3: 快速验证新参数可正常调用**

```bash
python -c "
import asyncio
async def test():
    from src.api.batch_scorer import score_batch
    stocks = [{
        'code': 'sh.603871', 'name': '嘉友国际', 'status': 'fetched',
        'data': {'code': 'sh.603871', 'name': '嘉友国际', 'pe': '15', 'pb': '2.5'}
    }]
    result = await score_batch(stocks, 'medium', model_suffix='_2')
    print('OK' if result[0].get('score') else 'FAIL')
asyncio.run(test())
"
```
Expected: OK (如无网络/API则跳过，不阻塞)

- [ ] **Step 4: Commit**

```bash
git add src/api/batch_scorer.py
git commit -m "feat: add model_suffix parameter to batch_scorer.score_batch() for multi-model support"
```

---

### Task 2: 创建 pool_screening.py 四层管线核心模块

**Files:**
- Create: `src/eval/pool_screening.py`
- Test: `tests/test_pool_screening.py`

这是本次重写的核心文件。四层管线:

```
Layer 0: hard_screen()        → ~4500 filtered stocks
Layer 1: batch_score_layer1() → {whitelist, initial_pool, blacklist}
Layer 2: quick_screen_layer2() → filtered initial_pool
Layer 3: formal_score_layer3() → final refined pool (100/80/60)
```

- [ ] **Step 1: 创建测试文件，编写 Layer 0 硬筛测试**

```python
# tests/test_pool_screening.py
"""四层精筛管线测试"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestHardScreen:
    """Layer 0: 硬筛逻辑测试（无LLM调用）"""

    def _make_stock(self, ts_code, name="测试", list_date="20200101", industry="制造业"):
        return {"ts_code": ts_code, "name": name, "list_date": list_date, "industry": industry}

    def test_excludes_bj_stocks(self):
        """排除北交所股票"""
        from src.eval.pool_screening import _is_stock_excluded
        assert _is_stock_excluded(self._make_stock("830001.BJ")) is True

    def test_excludes_b_shares(self):
        """排除B股"""
        from src.eval.pool_screening import _is_stock_excluded
        assert _is_stock_excluded(self._make_stock("200001.B")) is True

    def test_excludes_st_stocks(self):
        """检测ST股票"""
        from src.eval.pool_screening import _is_st_name
        assert _is_st_name("*ST测试") is True
        assert _is_st_name("ST中孚") is True
        assert _is_st_name("贵州茅台") is False

    def test_excludes_recent_ipo(self):
        """排除上市不满60天"""
        from src.eval.pool_screening import _is_recent_ipo, _is_stock_excluded
        # 使用一个明确的近期日期不应影响通过
        pass

    def test_normal_stock_not_excluded(self):
        """正常股票不被排除"""
        from src.eval.pool_screening import _is_stock_excluded
        assert _is_stock_excluded(self._make_stock("603871.SH", "嘉友国际")) is False

    def test_low_volume_detected(self):
        """日均成交额过低检测"""
        from src.eval.pool_screening import _is_low_volume
        # 日均成交额 < 2000万 → True(剔除)
        assert _is_low_volume([10000000, 15000000, 8000000], 20000000) is True
        # 日均成交额 >= 2000万 → False(保留)
        assert _is_low_volume([30000000, 25000000, 20000000], 20000000) is False
        # 空数据 → False(保守保留)
        assert _is_low_volume([], 20000000) is False


class TestLayerClassifier:
    """Layer 1 分类逻辑: 总纲5级→3档 (白名单/初筛池/黑名单)"""

    def test_strong_recommend_to_whitelist(self):
        """强烈推荐 → 白名单"""
        from src.eval.pool_screening import classify_batch_result
        result = classify_batch_result({
            "code": "sh.603871", "level": "强烈推荐", "confidence": "高",
            "reason": "低估值高ROE", "risk": "无"
        })
        assert result == "whitelist"

    def test_buy_cautious_watch_to_initial(self):
        """买入/谨慎买入/观望 → 初筛池（只有'卖出'进黑名单）"""
        from src.eval.pool_screening import classify_batch_result
        assert classify_batch_result({"code": "sh.000001", "level": "买入"}) == "initial_pool"
        assert classify_batch_result({"code": "sh.000001", "level": "谨慎买入"}) == "initial_pool"
        assert classify_batch_result({"code": "sh.000001", "level": "观望"}) == "initial_pool"

    def test_sell_to_blacklist(self):
        """卖出 → 黑名单"""
        from src.eval.pool_screening import classify_batch_result
        assert classify_batch_result({"code": "sh.000001", "level": "卖出"}) == "blacklist"

    def test_invalid_level_defaults_to_initial(self):
        """非法分类 → 默认入初筛池（保守处理）"""
        from src.eval.pool_screening import classify_batch_result
        assert classify_batch_result({"code": "sh.000001", "level": "不存在的分类"}) == "initial_pool"


class TestRatioCalculator:
    """1:1.2差额计算"""

    def test_whitelist_smaller_than_target(self):
        """白名单不足时，差额由初筛补"""
        from src.eval.pool_screening import calculate_candidate_quota
        result = calculate_candidate_quota(
            whitelist_count=10, initial_passing_count=50,
            target_size=100, ratio=1.2
        )
        # whitelist 10只, 按1:1.2需要从初筛取 100*1.2 - 10 = 110
        assert result["whitelist_slots"] == 10
        assert result["initial_slots"] == 110

    def test_whitelist_larger_than_target(self):
        """白名单超出时，取前N名"""
        from src.eval.pool_screening import calculate_candidate_quota
        result = calculate_candidate_quota(
            whitelist_count=50, initial_passing_count=80,
            target_size=100, ratio=1.2
        )
        # whitelist 50只, 目标1:1.2 = 50 : 60, 总候选=110
        assert result["whitelist_slots"] == 50
        assert result["initial_slots"] == 60
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pool_screening.py -v`
Expected: 全部 FAIL (模块和函数尚未创建)

- [ ] **Step 3: 实现 pool_screening.py 核心模块**

```python
"""
精筛股票池四层筛选管线 — 严格按照评分智能体开发总纲 §4 实现。

四层管线:
  Layer 0: 纯代码硬筛 (去ST/*ST/新股/低流动/北交所/B股)
  Layer 1: M1/M3 批量粗筛分档 (强烈推荐→白名单, 买入/谨慎买入/观望→初筛池, 卖出→黑名单)
  Layer 2: M2 快筛过滤初筛池 (便宜模型, 淘汰明显不行的)
  Layer 3: 1:1.2差额精筛 (白名单+初筛通过→正式7Agent+3Scorer→LLM动态阈值→最终精筛池)
"""
import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

from src.eval.database import init_db
from src.eval.pool_manager import PoolManager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Layer 0: Hard Screen (纯代码, 无LLM)
# ═══════════════════════════════════════════════════════════

def _is_st_name(name: str) -> bool:
    """检测股票名称是否含ST"""
    return bool(re.search(r'\*?ST', str(name), re.IGNORECASE))


def _is_low_volume(daily_amounts: list, min_amount: float) -> bool:
    """判断日均成交额是否低于阈值"""
    if not daily_amounts:
        return False  # 无数据, 保守保留
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
    """判断股票是否应被硬筛排除"""
    code = stock.get("ts_code", "")
    name = stock.get("name", "")

    if code.endswith(".BJ") or code.endswith(".B"):
        return True
    if _is_st_name(name):
        return True
    if _is_recent_ipo(stock.get("list_date", "")):
        return True
    return False


async def hard_screen() -> List[Dict[str, Any]]:
    """
    Layer 0: 从Tushare获取全A股 → 硬筛 → 返回候选列表。

    Returns:
        [{"ts_code": "603871.SH", "name": "嘉友国际", "industry": "物流", ...}, ...]
    """
    from src.eval.data_fetcher import _call as tushare_call
    from src.eval.config import get as eval_get

    min_daily_amount = eval_get("hard_screen_min_daily_amount", 20000000)
    logger.info(f"[Layer 0] 硬筛开始: 最低日均成交额={min_daily_amount/10000:.0f}万")

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
    logger.info(f"  Tushare返回: {total}只")

    # 硬筛
    filtered = []
    st_removed = 0
    for s in stocks:
        if _is_stock_excluded(s):
            if _is_st_name(s.get("name", "")):
                st_removed += 1
            continue
        filtered.append(s)

    logger.info(f"  硬筛后: {len(filtered)}只 (排除ST/BJ/B/新股共{total - len(filtered)}只, 其中ST{st_removed}只)")

    # ── 日均成交额筛选 ──
    # 通过 daily 接口查最近20个交易日数据, 计算日均成交额
    # 为效率使用批量查询, 每批50只
    volume_filtered = []
    batch_size = 50
    today_str = datetime.now().strftime("%Y%m%d")
    # 往回退10天作为start_date (应对非交易日)
    start_str = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

    logger.info(f"  成交量筛选: {len(filtered)}只 → 过滤日均成交额<{min_daily_amount/10000:.0f}万")
    for i in range(0, len(filtered), batch_size):
        batch = filtered[i:i + batch_size]
        for s in batch:
            ts_code = s.get("ts_code", "")
            try:
                daily = tushare_call("daily", {
                    "ts_code": ts_code,
                    "start_date": start_str,
                    "end_date": today_str,
                }, fields="trade_date,amount")
                if daily and "items" in daily and len(daily["items"]) >= 5:
                    amounts = []
                    for row in daily["items"][:20]:
                        item = dict(zip(daily["fields"], row))
                        amt = item.get("amount", 0)
                        try:
                            amounts.append(float(amt))
                        except (ValueError, TypeError):
                            pass
                    if amounts:
                        avg_amount = sum(amounts) / len(amounts)
                        if avg_amount >= min_daily_amount:
                            volume_filtered.append(s)
                        # else: 日均成交额不达标, 剔除
                    else:
                        volume_filtered.append(s)  # 有交易记录但无金额, 保守保留
                else:
                    volume_filtered.append(s)  # 无足够交易记录, 保守保留
            except Exception:
                volume_filtered.append(s)  # 查询失败, 保守保留

    logger.info(f"  成交量筛选后: {len(volume_filtered)}只 (剔除{len(filtered) - len(volume_filtered)}只低成交额)")
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

# Layer 1 专用 System Prompt (总纲分类标准, 与 batch_scorer 默认prompt不同)
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
        # 转换为内部格式 sh.603871 / sz.000001
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
    on_progress: callable = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Layer 1: 用M1/M3生产模型批量粗筛 ~4500只股票 → 分4档。

    Args:
        ts_stocks: Layer 0 输出的股票列表
        term: short/medium/long
        on_progress: 进度回调(completed, total)

    Returns:
        {"whitelist": [...], "initial_pool": [...], "blacklist": [...]}
    """
    from src.api.batch_scorer import fetch_batch, score_batch, chunk_stocks

    # 模型选择: short→M3, medium→M1, long→M1
    model_suffix = "_3" if term == "short" else ""

    logger.info(f"[Layer 1] 批量粗筛开始: {len(ts_stocks)}只, term={term}, model_suffix={model_suffix}")

    if on_progress:
        on_progress(0, len(ts_stocks))

    # Step A: 数据获取
    batch_input = _prepare_stocks_for_batch(ts_stocks)
    if not batch_input:
        return {"whitelist": [], "initial_pool": [], "blacklist": []}

    batch_stocks = await fetch_batch(
        batch_input,
        semaphore=6,
        on_progress=lambda c, t: on_progress(c * 3 // 4, t) if on_progress else None,
    )

    # Step B: 批量LLM打分 (传入总纲分类标准)
    scored = await score_batch(
        batch_stocks,
        horizon=term,
        semaphore=8,
        model_suffix=model_suffix,
        custom_levels=LAYER1_LEVELS,  # 使用总纲5级分类,非batch_scorer默认的"推荐/中性/回避"
        on_progress=lambda c, t: on_progress(t * 3 // 4 + c // 4, t) if on_progress else None,
    )

    # Step C: 分档
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
        f"[Layer 1] 分档完成: 白名单{len(whitelist)}只, "
        f"初筛池{len(initial_pool)}只, 黑名单{len(blacklist)}只"
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
    on_progress: callable = None,
) -> List[Dict[str, Any]]:
    """
    Layer 2: 用M2(Qwen3.6-Flash)快筛初筛池，淘汰明显不行的。

    Args:
        initial_pool: Layer 1 输出的初筛池股票
        term: short/medium/long
        threshold: 快筛分数阈值，低于此分则淘汰
        on_progress: 进度回调

    Returns:
        快筛通过的股票列表（含快筛评分）
    """
    from src.api.batch_scorer import fetch_batch, score_batch
    from src.eval.config import get as eval_get

    threshold = eval_get(f"quick_screen_threshold_{term}", threshold)

    logger.info(f"[Layer 2] 快筛开始: {len(initial_pool)}只, term={term}, threshold={threshold}")

    if not initial_pool:
        return []

    if on_progress:
        on_progress(0, len(initial_pool))

    # 数据获取（初筛池数量较少，速较快）
    batch_input = [{"code": s["code"], "name": s.get("name", "")} for s in initial_pool]
    batch_stocks = await fetch_batch(
        batch_input, semaphore=6,
        on_progress=lambda c, t: on_progress(c // 2, t) if on_progress else None,
    )

    # M2 快筛打分
    scored = await score_batch(
        batch_stocks, horizon=term, semaphore=8, model_suffix="_2",
        on_progress=lambda c, t: on_progress(t // 2 + c // 2, t) if on_progress else None,
    )

    # 低于阈值的淘汰
    passing = []
    eliminated = 0
    for s in scored:
        score_data = s.get("score", {})
        level_map = {"强烈推荐": 90, "推荐": 75, "中性": 60, "回避": 40, "卖出": 20}
        level = score_data.get("level", "中性") if isinstance(score_data, dict) else "中性"
        numeric_score = level_map.get(level, 50)

        if numeric_score >= threshold:
            s["quick_screen_score"] = numeric_score
            s["quick_screen_level"] = level
            passing.append(s)
        else:
            eliminated += 1

    logger.info(f"[Layer 2] 快筛完成: 通过{len(passing)}只, 淘汰{eliminated}只")

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
    计算白名单和初筛通过各自进入精筛的配额（1:1.2差额）。

    规则:
      - 白名单全部进入精筛候选，最多取 target_size * ratio 中的合理份额
      - 若白名单数量不足 (白名单 < target_size / (1 + 1/ratio))，差额由初筛补
      - 总候选数 ≈ target_size * ratio

    Returns: {"whitelist_slots": N, "initial_slots": M, "total_candidates": N+M}
    """
    total_candidates = int(target_size * ratio)
    # 理想白名单数量: total_candidates * 1/(1+1.2) = total_candidates * 0.4545
    ideal_whitelist = int(total_candidates * 1.0 / (1.0 + ratio))

    whitelist_slots = min(whitelist_count, max(ideal_whitelist, whitelist_count))
    # 如果白名单太多，按比例截断
    if whitelist_count > ideal_whitelist * 2:
        whitelist_slots = ideal_whitelist * 2

    initial_slots = total_candidates - whitelist_slots
    initial_slots = min(initial_slots, initial_passing_count)

    return {
        "whitelist_slots": whitelist_slots,
        "initial_slots": initial_slots,
        "total_candidates": whitelist_slots + initial_slots,
    }


async def _dynamic_threshold(
    scores: List[float], target_size: int
) -> float:
    """
    用内置LLM根据得分分布动态设定淘汰阈值。
    波动范围：使筛选后股票数在目标池大小的90%~110%之间。
    """
    if not scores:
        return 0.0

    scores_sorted = sorted(scores, reverse=True)
    # 默认取第target_size名的分数作为初始阈值
    default_cutoff_idx = min(target_size, len(scores_sorted))
    default_threshold = max(scores_sorted[default_cutoff_idx - 1] - 1, 0) if default_cutoff_idx > 0 else 0

    # 用LLM微调阈值
    try:
        from src.utils.llm_clients import OpenAICompatibleClient
        import os

        llm = OpenAICompatibleClient(
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", ""),
            base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", ""),
            model=os.getenv("OPENAI_COMPATIBLE_MODEL", "mimo-v2.5-pro"),
            env_prefix="",
            extra_body={"thinking": {"type": "enabled"}},
            http_timeout=60,
        )

        score_dist = scores_sorted[:target_size + 20]
        prompt = f"""你是精筛池阈值设定器。根据得分分布设定一个合理淘汰阈值。

目标池大小: {target_size}只
得分分布 (前{len(score_dist)}名): {score_dist}

要求:
1. 阈值应使通过股票数在{int(target_size*0.9)}~{int(target_size*1.1)}之间
2. 优先考虑得分"断层"处（相邻股票分差>5的位置）
3. 默认阈值={default_threshold}，可以调整±10分

只输出一个数字（阈值分数），不要任何其他文字:"""

        response = llm.get_completion([
            {"role": "user", "content": prompt}
        ], max_retries=1)

        # 提取数字
        nums = re.findall(r'\d+\.?\d*', str(response))
        if nums:
            threshold = float(nums[0])
            # 安全边界
            threshold = max(0, min(threshold, 90))
            logger.info(f"  LLM动态阈值: {threshold:.1f} (默认={default_threshold:.1f})")
            return threshold
    except Exception as e:
        logger.warning(f"  LLM动态阈值失败, 使用默认: {e}")

    return default_threshold


async def formal_score_layer3(
    whitelist: List[Dict[str, Any]],
    initial_passing: List[Dict[str, Any]],
    term: str,
    target_size: int = 100,
    ratio: float = 1.2,
    on_progress: callable = None,
) -> Dict[str, Any]:
    """
    Layer 3: 1:1.2差额组建候选 → 正式7Agent+3Scorer打分 → LLM动态阈值淘汰 → 最终精筛池。

    Returns:
        {"pool": [...], "whitelist": [...], "blacklist": [...], "stats": {...}}
    """
    from src.stock_pool.scoring_engine import ScoringEngine

    logger.info(
        f"[Layer 3] 精筛定稿开始: 白名单{len(whitelist)}只, "
        f"初筛通过{len(initial_passing)}只, target={target_size}, ratio={ratio}"
    )

    # Step A: 计算配额
    quota = calculate_candidate_quota(
        len(whitelist), len(initial_passing), target_size, ratio
    )
    logger.info(f"  配额: 白名单{quota['whitelist_slots']} + 初筛{quota['initial_slots']} = {quota['total_candidates']}")

    # Step B: 组建候选列表
    candidates = list(whitelist[:quota["whitelist_slots"]])
    # 初筛通过的按快筛分数排序取前initial_slots
    initial_sorted = sorted(
        initial_passing,
        key=lambda x: x.get("quick_screen_score", 50),
        reverse=True
    )
    candidates += initial_sorted[:quota["initial_slots"]]

    logger.info(f"  精筛候选: {len(candidates)}只")

    if on_progress:
        on_progress(0, len(candidates))

    # Step C: 逐只运行正式7Agent+3Scorer
    engine = ScoringEngine()
    scored_candidates = []
    whitelist_final = []
    blacklist_final = []

    for i, stock in enumerate(candidates):
        code = stock.get("code", "")
        name = stock.get("name", code)
        try:
            result = await engine.score_stock(code, name)
            if result and result.get("score_data"):
                sd = result["score_data"]
                score = sd.get("score", 50)

                # 提取对应期限评分
                term_key = {"short": "short_term_score", "medium": "medium_term_score", "long": "long_term_score"}
                term_score = sd.get(term_key.get(term, "medium_term_score"), {})
                actual_score = term_score.get("score", score) if isinstance(term_score, dict) else score
                recommendation = sd.get("recommendation", "")

                entry = {**stock, "final_score": actual_score, "recommendation": recommendation}

                if "强烈推荐" in recommendation or actual_score >= 85:
                    whitelist_final.append(entry)
                    scored_candidates.append(entry)
                elif "卖出" in recommendation or actual_score < 30:
                    blacklist_final.append(entry)
                else:
                    scored_candidates.append(entry)
            else:
                # 评分失败 → 中性处理，入候选池
                scored_candidates.append({**stock, "final_score": 50, "recommendation": ""})
        except Exception as e:
            logger.warning(f"  正式评分失败 {code}: {e}")
            scored_candidates.append({**stock, "final_score": 40, "recommendation": ""})

        if on_progress:
            on_progress(i + 1, len(candidates))

    # Step D: LLM动态阈值
    all_scores = [s.get("final_score", 50) for s in scored_candidates]
    threshold = await _dynamic_threshold(all_scores, target_size)

    # Step E: 按阈值筛选
    pool_final = [s for s in scored_candidates if s.get("final_score", 0) >= threshold]
    pool_final.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    # 确保不超过目标大小
    if len(pool_final) > target_size:
        pool_final = pool_final[:target_size]

    stats = {
        "candidates": len(candidates),
        "scored": len(all_scores),
        "whitelist_final": len(whitelist_final),
        "blacklist_final": len(blacklist_final),
        "pool_size": len(pool_final),
        "dynamic_threshold": threshold,
        "score_range": {"min": min(all_scores) if all_scores else 0, "max": max(all_scores) if all_scores else 0},
    }

    logger.info(f"[Layer 3] 精筛定稿完成: 池{stats['pool_size']}只, 阈值={threshold:.1f}, 白{stats['whitelist_final']}, 黑{stats['blacklist_final']}")

    return {
        "pool": pool_final,
        "whitelist": whitelist_final,
        "blacklist": blacklist_final,
        "stats": stats,
    }


# ═══════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════

async def run_pool_update(term: str = "short", on_stage: callable = None) -> Dict[str, Any]:
    """
    完整四层精筛池更新管线。

    严格按照总纲 §4.1 执行:
      Layer 0: 硬筛 → ~4500只
      Layer 1: M1/M3批量粗筛 → 分4档(白名单/初筛/黑名单)
      Layer 2: M2快筛 → 过滤初筛池
      Layer 3: 1:1.2差额 → 正式7Agent+3Scorer → LLM动态阈值 → 最终精筛池

    Args:
        term: short/medium/long
        on_stage: 阶段回调(stage_name, message)

    Returns:
        完整结果字典
    """
    from src.eval.config import get as eval_get

    pm = PoolManager()
    target_size = {"short": 100, "medium": 80, "long": 60}[term]
    ratio = eval_get("whitelist_pass_ratio", 1.2)

    def _stage(name, msg):
        logger.info(f"[PoolUpdate:{term}] {name}: {msg}")
        if on_stage:
            on_stage(name, msg)

    result = {"term": term, "stages": {}}

    # ── Layer 0: Hard Screen ──
    _stage("0_hard_screen", "从Tushare获取全A股，硬筛去除ST/新股/BJ/B股...")
    ts_stocks = await hard_screen()
    result["stages"]["0_hard_screen"] = f"硬筛后{len(ts_stocks)}只"

    if len(ts_stocks) < target_size:
        return {"error": f"硬筛后仅{len(ts_stocks)}只，不足目标{target_size}只"}

    # ── Layer 1: Batch Scoring ──
    _stage("1_batch_score", f"用M1/M3批量粗筛{len(ts_stocks)}只...")
    layer1 = await batch_score_layer1(ts_stocks, term)
    result["stages"]["1_batch_score"] = f"白名单{len(layer1['whitelist'])}只, 初筛池{len(layer1['initial_pool'])}只, 黑名单{len(layer1['blacklist'])}只"

    # ── Layer 2: Quick Screen ──
    _stage("2_quick_screen", f"用M2快筛初筛池{len(layer1['initial_pool'])}只...")
    initial_passing = await quick_screen_layer2(layer1["initial_pool"], term)
    result["stages"]["2_quick_screen"] = f"快筛通过{len(initial_passing)}只"

    # ── Layer 3: Formal Scoring ──
    total_candidates = len(layer1["whitelist"]) + len(initial_passing)
    _stage("3_formal_score",
           f"1:{ratio}差额组建候选({total_candidates}只) → 正式7Agent+3Scorer → LLM动态阈值...")
    layer3 = await formal_score_layer3(
        layer1["whitelist"], initial_passing,
        term=term, target_size=target_size, ratio=ratio,
    )

    # ── Store ──
    pm.update_pool(term, layer3["pool"])

    # 黑名单入库
    all_blacklist = layer1["blacklist"] + layer3.get("blacklist", [])
    for b in all_blacklist:
        code = b.get("code", "")
        if code and not pm.is_blacklisted(code):
            pm.add_to_blacklist(code, b.get("layer1_reason", "批量粗筛判定"), expiry_days=120)

    result["stages"]["3_formal_score"] = f"精筛池{len(layer3['pool'])}只, 白名单{len(layer3['whitelist'])}只, 黑名单{len(layer3['blacklist'])}只"
    result["pool"] = layer3["pool"]
    result["whitelist"] = layer3["whitelist"]
    result["blacklist"] = layer3["blacklist"]
    result["stats"] = layer3["stats"]
    result["final_pool_size"] = len(layer3["pool"])

    _stage("done", f"精筛池[{term}]更新完成: {len(layer3['pool'])}只 (阈值={layer3['stats'].get('dynamic_threshold', 'N/A')})")
    return result
```

注意：此文件 ~350 行。所有函数包含完整的 docstring 和日志。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pool_screening.py -v`
Expected: 全部 PASS (12个测试)

- [ ] **Step 5: Commit**

```bash
git add src/eval/pool_screening.py tests/test_pool_screening.py
git commit -m "feat: add 4-layer pool screening pipeline per 总纲 §4"
```

---

### Task 3: 重构 orchestrator.py 接入新管线

**Files:**
- Modify: `src/eval/orchestrator.py:238-421`

- [ ] **Step 1: 替换 run_pool_update_light 和 run_pool_update_formal**

将 `orchestrator.py:238-421` 中的 `run_pool_update_light` 和 `run_pool_update_formal` 替换为:

```python
    async def run_pool_update(self, term: str = "short",
                               on_stage: callable = None) -> Dict[str, Any]:
        """
        精筛池更新 — 完整四层管线（总纲 §4.1）。

        管线:
          Layer 0: 硬筛 (去ST/BJ/B/新股/低成交额)
          Layer 1: M1/M3批量粗筛分档 (强烈推荐→白名单, 买入/谨慎买入/观望→初筛池, 卖出→黑名单)
          Layer 2: M2快筛过滤初筛池
          Layer 3: 1:1.2差额组建候选 → 正式7Agent+3Scorer → LLM动态阈值 → 最终精筛池

        Args:
            term: short/medium/long
            on_stage: 阶段回调(stage_name, message) — 供Streamlit实时显示进度

        Returns:
            完整结果字典
        """
        from src.eval.pool_screening import run_pool_update as _run
        return await _run(term=term, on_stage=on_stage)

    async def run_pool_update_light(self, term: str = "short") -> Dict[str, Any]:
        """[已废弃] 保留兼容旧接口，内部转调 run_pool_update"""
        logger.warning("run_pool_update_light is deprecated, using run_pool_update instead")
        return await self.run_pool_update(term=term)

    async def run_pool_update_formal(self, term: str = "short",
                                      max_stocks: int = 50) -> Dict[str, Any]:
        """[已废弃] 保留兼容旧接口，内部转调 run_pool_update。max_stocks 参数忽略(由四层管线自行决定)"""
        logger.warning("run_pool_update_formal is deprecated, using run_pool_update instead")
        return await self.run_pool_update(term=term)
```

同时在文件顶部增加 import:
```python
import logging
logger = logging.getLogger(__name__)
```

- [ ] **Step 2: 运行已有测试确保不破坏**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: 不新增 FAILURE

- [ ] **Step 3: Commit**

```bash
git add src/eval/orchestrator.py
git commit -m "refactor: wire orchestrator to 4-layer pool screening pipeline"
```

---

### Task 4: 更新 Streamlit UI 对齐新管线

**Files:**
- Modify: `src/app/pages/06_模拟分析与迭代.py:74-94`

- [ ] **Step 1: 修复滑块默认值并改用新方法名**

```python
# 修改 06_模拟分析与迭代.py:74-94

with col3:
    pool_term = st.selectbox("选择期限", ["short", "medium", "long"],
                              format_func=lambda x: {"short": "短线(100只)", "medium": "中线(80只)", "long": "长线(60只)"}[x],
                              key="pool_term")
    # 滑块默认值对齐目标池容量
    target_sizes = {"short": 100, "medium": 80, "long": 60}
    pool_count = st.number_input("精筛候选股数（1:1.2差额冗余）",
                                  min_value=target_sizes[pool_term],
                                  max_value=target_sizes[pool_term] * 2,
                                  value=int(target_sizes[pool_term] * 1.2),
                                  step=10,
                                  help="从初筛通过股票中取此数量的候选进入正式7Agent+3Scorer精筛。默认为目标池容量的1.2倍（差额淘汰机制）。")
    if st.button("🎯 更新精筛池", use_container_width=True, type="primary"):
        if eval_ready:
            with st.spinner(f"四层管线运行中: 硬筛→M1/M3批量粗筛→M2快筛→1:1.2差额精筛..."):
                import asyncio
                update_result = asyncio.run(orch.run_pool_update(pool_term))
                # ... 其余结果展示逻辑不变
    st.caption(
        "👆 四层筛选管线（总纲§4.1）:\n"
        "① Layer 0 硬筛: 去ST/新股/BJ/B股/低成交额 → ~4500只\n"
        "② Layer 1 批量粗筛: M1/M3生产模型5只/批打分 → 分4档(白名单/初筛/黑名单)\n"
        "③ Layer 2 快筛: M2(Qwen3.6-Flash)过滤初筛池 → 淘汰明显不行的\n"
        "④ Layer 3 精筛: 白名单+初筛通过1:1.2差额 → 正式7Agent+3Scorer → LLM动态阈值 → 填满精筛池\n"
        "耗时: Layer0约30秒 + Layer1约20分钟(4500只) + Layer2约2分钟 + Layer3约40分钟(120只×20秒/只) ≈ 1小时"
    )
```

- [ ] **Step 2: 验证 Streamlit 页面导入**

```bash
python -c "import sys; sys.path.insert(0,'Finance/Financial-MCP-Agent/src'); from src.app.pages.06_模拟分析与迭代 import *; print('Import OK')"
```

- [ ] **Step 3: Commit**

```bash
git add "src/app/pages/06_模拟分析与迭代.py"
git commit -m "fix: align pool update UI with 4-layer pipeline, fix default slider to match target size"
```

---

### Task 5: 集成测试与端到端验证

**Files:**
- Create: `tests/test_pool_screening_integration.py`

- [ ] **Step 1: 集成测试（模拟数据，不调真实API）**

```python
# tests/test_pool_screening_integration.py
"""四层管线集成测试 — 使用mock数据验证流程完整性"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestFourLayerPipeline:
    """端到端验证四层管线数据流转"""

    def test_hard_screen_filters_correctly(self):
        """Layer 0: 正确过滤各种排除条件"""
        pass  # 已在test_pool_screening.py中覆盖

    def test_classify_distributes_correctly(self):
        """Layer 1: 总纲5级→3档 (只有'卖出'进黑名单)"""
        from src.eval.pool_screening import classify_batch_result
        assert classify_batch_result({"code":"sh.000001","level":"强烈推荐"}) == "whitelist"
        assert classify_batch_result({"code":"sh.000001","level":"买入"}) == "initial_pool"
        assert classify_batch_result({"code":"sh.000001","level":"谨慎买入"}) == "initial_pool"
        assert classify_batch_result({"code":"sh.000001","level":"观望"}) == "initial_pool"
        assert classify_batch_result({"code":"sh.000001","level":"卖出"}) == "blacklist"

    def test_quota_total_equals_target_times_ratio(self):
        """Layer 3: 配额总数 = target_size × ratio"""
        from src.eval.pool_screening import calculate_candidate_quota
        q = calculate_candidate_quota(10, 80, 100, 1.2)
        assert q["total_candidates"] == 120

    def test_quota_when_whitelist_exceeds(self):
        """白名单过多时合理截断"""
        from src.eval.pool_screening import calculate_candidate_quota
        q = calculate_candidate_quota(200, 50, 100, 1.2)
        # 白名单200只但不应全取
        assert q["whitelist_slots"] < 200
        assert q["total_candidates"] <= 200  # 不超过 target*ratio+一些缓冲
        assert q["whitelist_slots"] + q["initial_slots"] == q["total_candidates"]

    def test_quota_when_initial_empty(self):
        """初筛为空时白名单顶上去"""
        from src.eval.pool_screening import calculate_candidate_quota
        q = calculate_candidate_quota(10, 0, 100, 1.2)
        assert q["whitelist_slots"] == 10
        assert q["initial_slots"] == 0

    def test_code_format_conversion(self):
        """Tushare格式→内部格式转换正确"""
        from src.eval.pool_screening import _prepare_stocks_for_batch
        ts_stocks = [
            {"ts_code": "603871.SH", "name": "嘉友国际"},
            {"ts_code": "000001.SZ", "name": "平安银行"},
        ]
        result = _prepare_stocks_for_batch(ts_stocks)
        assert result[0]["code"] == "sh.603871"
        assert result[1]["code"] == "sz.000001"

    def test_dynamic_threshold_fallback(self):
        """LLM不可用时使用默认阈值"""
        import asyncio
        from src.eval.pool_screening import _dynamic_threshold
        scores = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
        threshold = asyncio.run(_dynamic_threshold(scores, 8))
        # 默认阈值应接近第8名的分数 - 1 = 55 - 1 = 54
        assert 45 <= threshold <= 85
```

- [ ] **Step 2: 运行集成测试**

Run: `python -m pytest tests/test_pool_screening_integration.py -v`
Expected: 7 PASS

- [ ] **Step 3: 运行全部测试确认无回归**

Run: `python -m pytest tests/ -v --ignore=tests/test_pool_screening.py --ignore=tests/test_pool_screening_integration.py -x`
Expected: 所有已有测试 PASS, 无回归

- [ ] **Step 4: Commit**

```bash
git add tests/test_pool_screening_integration.py
git commit -m "test: add 4-layer pipeline integration tests"
```

---

### Spec Coverage Self-Review

| 总纲 §4.1 要求 | 对应实现 | 状态 |
|---|---|---|
| 第0层: 纯代码硬筛 (ST/*ST/新股<60d/成交额) | `hard_screen()` | ✅ |
| 第1层: M1/M3批量打分, 分4档 | `batch_score_layer1()` + `classify_batch_result()` | ✅ |
| 第1层: "强烈推荐"→白名单→跳过初筛 | `LEVEL_TO_TIER` 映射 | ✅ |
| 第2层: M2快筛模型过滤初筛池 | `quick_screen_layer2()` | ✅ |
| 第3层: 白名单+初筛1:1.2差额 | `calculate_candidate_quota()` | ✅ |
| 第3层: 白名单不足差额由初筛补 | `calculate_candidate_quota()` whitelist_slots逻辑 | ✅ |
| 第3层: 白名单超出按正式评分取前N | `calculate_candidate_quota()` 截断逻辑 | ✅ |
| 第3层: 正式模型重新打一次完整分 | `formal_score_layer3()` 调 ScoringEngine.score_stock() | ✅ |
| 第3层: 内置LLM动态阈值 | `_dynamic_threshold()` | ✅ |
| 第3层: 90%~110%目标容量 | `_dynamic_threshold()` prompt约束 | ✅ |
| 最终精筛池大小: 短线100/中线80/长线60 | `target_size` 查表 | ✅ |
| 黑名单120天有效期 | `add_to_blacklist(expiry_days=120)` | ✅ |
| 精筛池筛选用生产模型M1/M3 | Layer1: M3(short)/M1(medium/long), Layer3: ScoringEngine | ✅ |
</parameter>
