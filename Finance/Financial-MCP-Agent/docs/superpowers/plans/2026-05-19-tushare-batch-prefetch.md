# Tushare 批量预取优化

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task.

**Goal:** 将批量打分的数据获取阶段从 37 分钟降到 10-15 分钟，同时消除因僵尸锁导致的数据获取失败。

**Architecture:** 将 Tushare 调用从"与 HTTP 并行交错在线程池中"改为"主线程顺序预取 → 内存合并 → HTTP 纯并行"。stock_basic 全量 1 次、fina_indicator 批量 40 只/次、daily_basic 逐只但无锁无超时。

**Tech Stack:** Python, Tushare HTTP API, asyncio + ThreadPoolExecutor

---

### Task 1: 新增 `get_stock_info_batch` 到 tushare_client.py

**Files:**
- Modify: `src/utils/tushare_client.py` (add after `get_stock_info`)

- [ ] **Step 1: 添加函数**

在 `get_stock_info` 之后添加：

```python
def get_stock_info_batch(ts_codes: List[str]) -> Dict[str, Dict]:
    """批量获取股票基本信息。
    
    Args:
        ts_codes: ['603871.SH', '000858.SZ', ...]
    
    Returns:
        {ts_code: {name, industry, list_date, area}, ...}
        缺失的 ts_code 不出现在返回中。
    """
    if not ts_codes:
        return {}
    
    ts_str = ",".join(ts_codes)
    r = _call("stock_basic", {"ts_code": ts_str},
              "ts_code,name,industry,list_date,area")
    items = _items_to_dicts(r)
    return {d["ts_code"]: d for d in items if d.get("ts_code")}
```

- [ ] **Step 2: 验证**

```bash
python -c "from src.utils.tushare_client import get_stock_info_batch; r = get_stock_info_batch(['603871.SH','000858.SZ','600519.SH']); print(len(r), list(r.keys()))"
```

预期: `3 ['603871.SH', '000858.SZ', '600519.SH']`

---

### Task 2: 新增 `get_fina_indicator_batch` 到 tushare_client.py

**Files:**
- Modify: `src/utils/tushare_client.py` (add after `get_fina_indicator`)

- [ ] **Step 1: 添加批量函数 + 分块辅助**

```python
_FINA_BATCH_SIZE = 40


def _chunk_list(lst: list, size: int) -> list:
    """将列表拆分为固定大小的批次"""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def get_fina_indicator_batch(ts_codes: List[str], years: int = 2) -> Dict[str, Dict]:
    """批量获取最新财务指标，每批最多 40 只。
    
    Args:
        ts_codes: ['603871.SH', '000858.SZ', ...]
        years: 查询最近几年的数据
    
    Returns:
        {ts_code: {roe, grossprofit_margin, ...}, ...}
        只返回每个 ts_code 的最新一期数据。
    """
    if not ts_codes:
        return {}
    
    end_date = datetime.now().strftime("%Y1231")
    start_date = f"{datetime.now().year - years}0101"
    fields = ("ts_code,ann_date,end_date,roe,roe_dt,roa,grossprofit_margin,"
              "netprofit_margin,debt_to_assets,current_ratio,quick_ratio,"
              "inv_turn,ar_turn,assets_turn,or_yoy,profit_yoy,ocf_yoy,ocf_netprofit")
    
    result = {}
    chunks = _chunk_list(ts_codes, _FINA_BATCH_SIZE)
    
    for chunk in chunks:
        ts_str = ",".join(chunk)
        r = _call("fina_indicator", {
            "ts_code": ts_str,
            "start_date": start_date,
            "end_date": end_date,
        }, fields)
        items = _items_to_dicts(r)
        
        # 按 ts_code 分组，取最新一期 (end_date 最大)
        by_code = {}
        for d in items:
            code = d.get("ts_code", "")
            if code not in by_code or d.get("end_date", "") > by_code[code].get("end_date", ""):
                by_code[code] = d
        result.update(by_code)
    
    return result
```

- [ ] **Step 2: 验证**

```bash
python -c "from src.utils.tushare_client import get_fina_indicator_batch; r = get_fina_indicator_batch(['603871.SH','000858.SZ']); print(len(r)); [print(f'{k}: ROE={v.get(\"roe\")}') for k,v in r.items()]"
```

预期: 2 只股票各返回最新财务指标

---

### Task 3: 新增 `_prefetch_tushare_batch` 到 batch_scorer.py

**Files:**
- Modify: `src/api/batch_scorer.py` (add before `_fetch_light_stock_data_sync`)

- [ ] **Step 1: 删除旧的 Tushare 锁和 enrich 函数，添加预取函数**

删除：
- `_tushare_lock` (line 52)
- `_TUSHARE_LOCK_TIMEOUT` (line 53)
- `_try_acquire_tushare_lock` 函数 (lines 224-230)
- `_enrich_light_tushare` 函数 (lines 233-305)

添加：

```python
# ──────────────────────────────────────────────
# Tushare 批量预取 (主线程顺序执行，无需锁)
# ──────────────────────────────────────────────

_BATCH_PREFETCH_ENABLED = True

_TS_FIELDS_MAP = {
    # daily_basic → 输出字段映射
    "pe": "pe_ttm",
    "pb": "pb",
    "ps": "ps_ttm",
    # fina_indicator → 输出字段映射
    "roe": "roe",
    "gross_margin": "grossprofit_margin",
    "net_margin": "netprofit_margin",
    "debt_ratio": "debt_to_assets",
    "revenue_growth": "or_yoy",
    "profit_growth": "profit_yoy",
}


def _prefetch_tushare_batch(stocks: List[Dict[str, str]]) -> Dict[str, Dict]:
    """批量预取 Tushare 数据 (主线程，顺序执行)。

    三步: stock_basic 全量 → daily_basic 逐只 → fina_indicator 批量(40只/批)
    返回: {ts_code: {industry, pe, pb, ps, roe, gross_margin, ...}}
    缺失字段留空字符串。
    """
    from src.utils.tushare_client import (
        get_all_stocks, get_daily_basic, get_fina_indicator_batch,
    )

    if not stocks:
        return {}

    ts_codes = [_to_tushare_code(s["code"]) for s in stocks]
    cache: Dict[str, Dict] = {tc: {} for tc in ts_codes}
    total = len(stocks)

    # ── Step A: stock_basic 全量 (1次调用) ──
    logger.info(f"  预取 Step A: stock_basic 全量 (1 次调用)")
    try:
        all_info = get_all_stocks()
        info_map = {d["ts_code"]: d for d in all_info if d.get("ts_code")}
        for tc in ts_codes:
            if tc in info_map:
                info = info_map[tc]
                cache[tc]["industry"] = info.get("industry", "") or ""
            else:
                cache[tc]["industry"] = ""
        hit = sum(1 for tc in ts_codes if tc in info_map)
        logger.info(f"  预取 Step A 完成: 行业命中 {hit}/{total}")
    except Exception as e:
        logger.warning(f"  预取 Step A 失败: {e}")

    # ── Step B: daily_basic 逐只 (不能批量) ──
    logger.info(f"  预取 Step B: daily_basic 逐只 ({total} 次调用, {total*0.35/60:.0f}分预计)")
    done_b = 0
    for tc in ts_codes:
        done_b += 1
        if done_b % 200 == 0:
            logger.info(f"  预取 Step B 进度: {done_b}/{total}")
        try:
            basics = get_daily_basic(tc, days=5)
            if basics and isinstance(basics, list) and basics:
                latest = basics[0]
                cache[tc]["pe"] = str(latest.get("pe_ttm", "") or "")
                cache[tc]["pb"] = str(latest.get("pb", "") or "")
                cache[tc]["ps"] = str(latest.get("ps_ttm", "") or "")
        except Exception:
            pass
    pe_hit = sum(1 for tc in ts_codes if cache[tc].get("pe"))
    logger.info(f"  预取 Step B 完成: PE 命中 {pe_hit}/{total}")

    # ── Step C: fina_indicator 批量 (每批40只) ──
    batch_count = (total + 39) // 40
    logger.info(f"  预取 Step C: fina_indicator 批量 ({batch_count} 批, 每批40只)"
                f"[{total} 只]")
    try:
        fina_map = get_fina_indicator_batch(ts_codes, years=2)
        for tc in ts_codes:
            if tc in fina_map:
                f = fina_map[tc]
                cache[tc]["roe"] = str(f.get("roe", "") or "")
                cache[tc]["gross_margin"] = str(f.get("grossprofit_margin", "") or "")
                cache[tc]["net_margin"] = str(f.get("netprofit_margin", "") or "")
                cache[tc]["debt_ratio"] = str(f.get("debt_to_assets", "") or "")
                cache[tc]["revenue_growth"] = str(f.get("or_yoy", "") or "")
                cache[tc]["profit_growth"] = str(f.get("profit_yoy", "") or "")
    except Exception as e:
        logger.warning(f"  预取 Step C 失败: {e}")
    roe_hit = sum(1 for tc in ts_codes if cache[tc].get("roe"))
    logger.info(f"  预取 Step C 完成: ROE 命中 {roe_hit}/{total}")

    return cache
```

- [ ] **Step 2: 验证 (手动脚本)**

```bash
python -c "
from src.api.batch_scorer import _prefetch_tushare_batch
stocks = [{'code': 'sh.603871'}, {'code': 'sz.000858'}, {'code': 'sh.600519'}]
cache = _prefetch_tushare_batch(stocks)
for code, data in cache.items():
    print(f'{code}: industry={data.get(\"industry\")}, pe={data.get(\"pe\")}, roe={data.get(\"roe\")}')
"
```

---

### Task 4: 修改 `_fetch_light_stock_data_sync` 接受缓存参数

**Files:**
- Modify: `src/api/batch_scorer.py` (`_fetch_light_stock_data_sync` function)

- [ ] **Step 1: 增加 `tushare_cache` 参数，移除 `_enrich_light_tushare` 调用**

```python
def _fetch_light_stock_data_sync(
    stock_code: str,
    tushare_cache: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """获取单只股票的轻量数据: HTTP 实时行情 + K线 + Tushare 缓存合并。

    与旧版区别: Tushare 数据从预取缓存中合并，不再单独调 API。
    """
    import requests as req

    pure_code = stock_code.replace("sh.", "").replace("sz.", "")
    tx_code = f"{_get_exchange_prefix(pure_code)}{pure_code}"
    result = {"code": stock_code, "name": "", "status": "fetched"}

    # ── 1. Tencent 实时行情 ──
    try:
        resp = req.get(
            f"https://qt.gtimg.cn/q={tx_code}",
            timeout=(3, 12),
            headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        match = re.search(r'"(.+)"', resp.text)
        if match:
            fields = match.group(1).split("~")
            if len(fields) >= 47:
                result["name"] = fields[1]
                result["last_price"] = fields[3]
                result["pct_chg"] = fields[32]
                result["pe"] = fields[39]
                result["market_cap"] = _format_market_cap(fields[44]) if fields[44] else ""
                result["pb"] = fields[46]
                if fields[38] and fields[38] != "0.00":
                    result["turnover_rate"] = fields[38]
    except Exception:
        pass

    # 1b. 名称回退: akshare 缓存
    if not result.get("name"):
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            if df is not None and not df.empty:
                row = df[df["code"] == pure_code]
                if not row.empty:
                    result["name"] = str(row.iloc[0].get("name", ""))
        except Exception:
            pass

    # ── 2. K线数据: Tencent 优先 ──
    kline_data = []
    try:
        from datetime import timedelta
        start = (datetime.now() - timedelta(days=1130)).strftime("%Y-%m-%d")
        k_url = (
            f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={tx_code},day,{start},,1100,qfq"
        )
        k_resp = req.get(k_url, timeout=(3, 20),
                        headers={"User-Agent": "Mozilla/5.0"})
        k_resp.raise_for_status()
        k_raw = k_resp.json()
        day_data = None
        if "data" in k_raw:
            stock_info = k_raw["data"]
            for key in [tx_code] + list(stock_info.keys()):
                if key in stock_info:
                    dd = stock_info[key]
                    day_data = dd.get("qfqday") or dd.get("day") or dd.get("qfq")
                    if day_data:
                        break
        if day_data:
            prev_close = None
            for item in day_data:
                close_val = float(item[2]) if len(item) > 2 else 0
                pct = ""
                if prev_close and prev_close > 0:
                    pct = f"{((close_val - prev_close) / prev_close * 100):.2f}"
                prev_close = close_val
                kline_data.append({"date": item[0], "close": str(close_val), "pctChg": pct})
    except Exception:
        pass

    # ── 3. 涨跌幅计算 ──
    result["price_changes"] = _compute_price_changes(kline_data)

    recent = []
    for bar in kline_data[-60:]:
        recent.append({
            "date": bar.get("date", ""),
            "close": bar.get("close", ""),
            "pctChg": bar.get("pctChg", ""),
        })
    result["recent_kline"] = recent

    # ── 4. 从 Tushare 缓存合并数据 ──
    if tushare_cache:
        ts_code = _to_tushare_code(stock_code)
        cached = tushare_cache.get(ts_code, {})
        if cached:
            # 行业优先用 Tushare 的 (比 Tencent 返回的可靠)
            if cached.get("industry"):
                result["industry"] = cached["industry"]
            # 估值字段: 缓存优先于 Tencent (Tushare 更准)
            for key in ("pe", "pb", "ps", "roe", "gross_margin", "net_margin",
                        "debt_ratio", "revenue_growth", "profit_growth"):
                val = cached.get(key, "")
                if val:
                    result[key] = val
            result["_enrich_errors"] = cached.get("_enrich_errors", [])
        else:
            result["_enrich_errors"] = [f"ts_code {ts_code} 未在 Tushare 缓存命中"]
    else:
        result["_enrich_errors"] = ["无 Tushare 缓存 (预取阶段可能失败)"]

    return result
```

- [ ] **Step 2: 确认 HTTP + K线逻辑无变化**

原有 HTTP 行情和 K 线获取逻辑完全不变，仅替换了最后的 `_enrich_light_tushare` 调用为缓存合并。

---

### Task 5: 修改 `_fetch_round` 和 `fetch_batch` 传入缓存

**Files:**
- Modify: `src/api/batch_scorer.py` (`_fetch_round` and `fetch_batch` functions)

- [ ] **Step 1: `_fetch_round` 增加 `tushare_cache` 参数**

修改函数签名和 `fetch_one` 内部调用：

```python
async def _fetch_round(
    stocks: List[Dict[str, str]],
    total: int,
    semaphore: int,
    on_progress: callable = None,
    only_failed: bool = False,
    round_label: str = "",
    base_completed: int = 0,
    tushare_cache: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
```

将 `fetch_one` 内的：
```python
_fetch_light_stock_data_sync, stock["code"]
```
改为：
```python
_fetch_light_stock_data_sync, stock["code"], tushare_cache
```

- [ ] **Step 2: `fetch_batch` 先预取再传入缓存**

```python
async def fetch_batch(
    stocks: List[Dict[str, str]],
    semaphore: int = 6,
    on_progress: callable = None,
    max_retry_rounds: int = 5,
) -> List[Dict]:
    total = len(stocks)
    
    # ── 阶段1: Tushare 批量预取 (主线程，无锁) ──
    tushare_cache = {}
    if _BATCH_PREFETCH_ENABLED:
        logger.info(f"Tushare 批量预取开始: {total} 只股票")
        t_start = time.time()
        try:
            tushare_cache = _prefetch_tushare_batch(stocks)
            elapsed = time.time() - t_start
            logger.info(
                f"Tushare 批量预取完成: {len(tushare_cache)} 只, "
                f"耗时 {elapsed:.0f}s"
            )
        except Exception as e:
            logger.error(f"Tushare 批量预取异常: {e}")
            tushare_cache = {}
    
    # ── 阶段2: HTTP 并行获取 ──
    stocks = await _fetch_round(
        stocks, total, semaphore, on_progress,
        round_label="R1",
        tushare_cache=tushare_cache,
    )
    
    # ── 重试轮次 ──
    for retry_idx in range(1, max_retry_rounds + 1):
        failed = [s for s in stocks if s.get("status") != "fetched"]
        if not failed:
            break
        
        retry_sem = max(2, semaphore // (retry_idx + 1))
        success_so_far = total - len(failed)
        logger.warning(
            f"  重试轮次 {retry_idx}/{max_retry_rounds}: "
            f"{len(failed)} 只数据获取失败，以并发={retry_sem} 重试"
        )
        
        stocks = await _fetch_round(
            stocks, total, retry_sem, on_progress,
            only_failed=True,
            round_label=f"R{retry_idx + 1}",
            base_completed=success_so_far,
            tushare_cache=tushare_cache,
        )
    
    final_success = sum(1 for s in stocks if s.get("status") == "fetched")
    if final_success < total:
        logger.warning(
            f"{WAIT_ICON} 数据获取完成: {final_success}/{total} 成功 "
            f"({total - final_success} 只最终失败)"
        )
    else:
        logger.info(
            f"{SUCCESS_ICON} 批量数据获取完成: {final_success}/{total} 成功"
        )
    return stocks
```

---

### Task 6: 运行完整测试套件

- [ ] **Step 1: 运行全部测试**

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/ -v
```

预期: 31 passed

---

### Task 7: 端到端批量预取验证

- [ ] **Step 1: 小规模批量预取测试 (3只股票)**

```bash
cd Finance/Financial-MCP-Agent && python -c "
import asyncio
from src.api.batch_scorer import fetch_batch

async def test():
    stocks = [
        {'code': 'sh.603871', 'name': '嘉友国际'},
        {'code': 'sz.000858', 'name': '五粮液'},
        {'code': 'sh.600519', 'name': '贵州茅台'},
    ]
    result = await fetch_batch(stocks)
    for s in result:
        d = s.get('data', {})
        sc = s.get('score', {})
        print(f'{s[\"code\"]} {d.get(\"name\")}: PE={d.get(\"pe\")}, PB={d.get(\"pb\")}, '
              f'ROE={d.get(\"roe\")}, industry={d.get(\"industry\")}')
        enrich_errs = d.get('_enrich_errors', [])
        if enrich_errs:
            print(f'  enrich_errors: {enrich_errs}')

asyncio.run(test())
"
```

预期: 3 只股票全部有 PE/PB/ROE/industry 数据，无 enrich_errors。

- [ ] **Step 2: 数据一致性验证 (对比新旧方式)**

```bash
cd Finance/Financial-MCP-Agent && python -c "
# 对比单只 Tushare 调用 vs 批量预取的结果一致性
from src.utils.tushare_client import get_stock_info, get_daily_basic, get_fina_indicator
from src.api.batch_scorer import _prefetch_tushare_batch, _to_tushare_code

# 旧方式: 单只调
ts = '603871.SH'
old_industry = get_stock_info(ts).get('industry', 'N/A')
old_pe = get_daily_basic(ts, days=5)[0].get('pe_ttm', 'N/A')
old_roe = get_fina_indicator(ts, years=2)[0].get('roe', 'N/A')

# 新方式: 批量预取
cache = _prefetch_tushare_batch([{'code': 'sh.603871'}])
new_data = cache.get('603871.SH', {})

print(f'industry: old={old_industry} new={new_data.get(\"industry\", \"N/A\")}')
print(f'pe_ttm:   old={old_pe} new={new_data.get(\"pe\", \"N/A\")}')
print(f'roe:      old={old_roe} new={new_data.get(\"roe\", \"N/A\")}')

assert str(old_industry) == str(new_data.get('industry', '')), 'industry mismatch!'
assert str(old_pe) == str(new_data.get('pe', '')), 'pe mismatch!'
assert str(old_roe) == str(new_data.get('roe', '')), 'roe mismatch!'
print('OK: 新旧数据一致')
"
```

---

### Task 8: Commit

```bash
git add src/utils/tushare_client.py src/api/batch_scorer.py
git commit -m "perf: Tushare 批量预取优化，数据阶段从37分钟降至约10分钟"
```
