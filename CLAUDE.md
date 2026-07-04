# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A stock investment advisor agent system for A-share (Chinese stock market) and fund/ETF analysis. Uses LangGraph for multi-agent workflow orchestration, MCP (Model Context Protocol) for data access, and LLMs for analysis/scoring. Runs on Linux/WSL and Windows.

## Architecture

### Two-Layer Architecture

**Layer 1: MCP Servers** (`Finance/a-share-mcp-server/`)
- `tushare_mcp_server.py` — Tushare MCP server (primary data source for A-share data)
- `mcp_server.py` — Legacy AKShare MCP server (minimal use, being phased out)
- `web_search_mcp_server.py` — Web search MCP server (news/sentiment, supplementary data)
- `yfinance_mcp_server.py` — Yahoo Finance MCP server (international/macro/commodity data)
- All run over stdio via FastMCP; in-memory tool-level cache (5-min TTL)

**Layer 2: Financial MCP Agent** (`Finance/Financial-MCP-Agent/`)
- Consumes MCP servers via `langchain-mcp-adapters` (`MultiServerMCPClient`)
- LangGraph workflows orchestrate analysis → scoring/report pipelines
- LLM calls via OpenAI-compatible API (6-model architecture in `.env`, M4 currently unused)

### A-Stock Pipeline (v2 — 2026-06 upgrade)

**Single-stock analysis** (`src/main.py`):
```
start_node → [fundamental, technical, value, news, event, quality_risk, moneyflow] (7 parallel)
          → summarizer → Markdown + PDF report (9-section format)
```

**Stock pool scoring** (`src/stock_pool/scoring_engine.py`):
```
start_node → [7 agents parallel]
  short_term_scorer  ← [technical, news, event, moneyflow] (4 agents only — streaming)
  medium_term_scorer ← [all 7]
  long_term_scorer   ← [all 7]
          → risk_gate post-processing → stock_pool.json
```

### Fund Analysis Pipeline (v2 — 2026-06 upgrade)

```
start_node → [fund_product_doc, fund_perf_risk, fund_holdings, fund_manager,
              fund_benchmark, fund_fee, fund_event] (7 parallel)
          → fund_merge_node (non-LLM, reads signal_packs + regex fallback)
              ├── fund_report_agent
              └── fund_scoring_agent → fund_risk_gate
```

### Structured Evidence Architecture (shared by both pipelines)

Each analysis agent outputs TWO artifacts into `state.data`:
1. **Text analysis** (`{agent}_analysis`): Natural language, backward-compatible
2. **Signal pack** (`{agent}_signal_pack`): Structured JSON with `bias`, `confidence`, `signals[]` (factor/direction/strength/source_level), `risk_flags[]`, `missing_data[]`

A-stock agents merge via `analysis_package_builder.py` → `AnalysisPackage` (conflict detection, source priority, compact context). Fund agents merge via `fund_merge_node.py` (signal_pack preferred, regex fallback, bias conflict detection, source-weighted scoring).

### Agent State

`src/utils/state_definition.py` defines `AgentState` TypedDict:
- `messages`: `Annotated[Sequence[BaseMessage], operator.add]` — append-only
- `data`: `Annotated[Dict[str, Any], merge_dicts]` — shallow merge (safe for parallel writes)
- `metadata`: `Annotated[Dict[str, Any], merge_dicts]`

Key data fields (v2): `{agent}_analysis` + `{agent}_signal_pack` (7 agents × 2), `analysis_package`, `short/medium/long_term_score` (each with embedded `risk_gate`).

## A-Stock Agents (11 total)

| Agent | Model | Thinking | Mode | Role |
|-------|-------|----------|------|------|
| **fundamental** | M1 MiMo-V2.5-Pro | ✅ | Two-phase (fetch→LLM) | Profit quality, cash flow, balance sheet health, growth |
| **technical** | M3 Qwen3.7-Plus | ❌ | ReAct Agent | Price trends, volume-price, indicators (MACD/RSI/MA) |
| **value** | M1 MiMo-V2.5-Pro | ✅ | Two-phase | Industry-relative PE/PB, historical percentile, safety margin |
| **news** | M3 Qwen3.7-Plus | ✅ | Multi-source→LLM | Media sentiment, narrative strength (NOT factual events) |
| **event** | M3 Qwen3.7-Plus | ✅ | Two-phase | Catalysts: earnings, buybacks, M&A, penalties, pledges |
| **quality_risk** | M1 MiMo-V2.5-Pro | ✅ | Two-phase | Cash flow quality, goodwill/impairment, governance, risk flags |
| **moneyflow** | M3 Qwen3.7-Plus | ✅ | Two-phase | Margin trading, block trades, top list, volume confirmation |
| **short_term_scorer** | M3 Qwen3.7-Plus | ✅ | Direct LLM | Depends on 4 agents; weights: tech(25), volume(20), capital(20), event(20), sentiment(15) |
| **medium_term_scorer** | M1 MiMo-V2.5-Pro | ✅ | Direct LLM | Depends on all 7; weights: fundamental(20), valuation(15), quality(20), event(15), tech(10), industry(10), sentiment(10) |
| **long_term_scorer** | M1 MiMo-V2.5-Pro | ✅ | Direct LLM | Depends on all 7; weights: returns(25), quality(20), valuation(15), moat(15), capital(10), policy(10), tech(5) |
| **summary_agent** | M1 MiMo-V2.5-Pro | ✅ | Direct LLM | 9-section report: conclusion→signal overview→bullish→bearish→timeline→S/M/L judgment→risks→confidence→disclaimer |

### Scoring Risk Gate

`src/utils/risk_gate.py` applies 4 post-scoring rules:
1. Critical risk flags → score cap + action downgrade (audit_risk=60, delist_risk=50, etc.)
2. News-only narrative with no factual support → cap at 55 for medium/long
3. ≥2 missing agents + data_quality < 0.4 → abstain
4. Short-term liquidity issues → cap at 50

Score cap is enforced in `scoring_nodes.py` via `min(score, gate.score_cap)`.

### Evidence Priority (source_level)

`official_like` (正式公告) > `structured` (数值工具) > `news` (媒体) > `derived` (推断) > `proxy` (代理).
When `news` conflicts with `event`, `event` wins. Reports must surface conflicts.

## Fund Agents (10 total)

| Agent | Model | Thinking | Role |
|-------|-------|----------|------|
| fund_product_doc | M3 Qwen3.7-Plus | ✅ | Fund type, benchmark, strategy |
| fund_perf_risk | M1 MiMo-V2.5-Pro | ✅ | NAV performance, drawdown, risk metrics |
| fund_holdings | M1 MiMo-V2.5-Pro | ✅ | Portfolio composition, concentration |
| fund_manager | M3 Qwen3.7-Plus | ✅ | Manager experience, style, stability |
| fund_benchmark | M3 Qwen3.7-Plus | ✅ | Tracking error, style drift |
| fund_fee | M3 Qwen3.7-Plus | ✅ | Fee structure vs peers |
| fund_event | M3 Qwen3.7-Plus | ✅ | Fund news, manager changes, dividends |
| fund_merge_node | — (non-LLM) | — | Merges 7 outputs into `fund_analysis_package` |
| fund_scoring_agent | M1 MiMo-V2.5-Pro | ✅ | Scores on 6 dimensions; applies `fund_risk_gate` |
| fund_report_agent | M1 MiMo-V2.5-Pro | ✅ | Generates markdown fund report |

### Fund Risk Gate

`src/utils/fund_risk_gate.py`: 10 fund-specific risk flags (manager_just_changed=55, frequent_manager_change=50, tiny_fund_size=55, etc.). Applied in `fund_scoring_agent.py`.

## Shared Infrastructure

### All agents have these capabilities:

| Capability | Mechanism |
|------------|-----------|
| **Structured JSON output** | `<SIGNAL_PACK>` tag in LLM prompt → 3-layer extraction (tag→regex→text fallback) |
| **MCP tool-level cache** | `src/utils/tool_cache.py`: 5-min in-memory cache, MD5 keys, asyncio.Lock, 500-entry cap |
| **Data retry** | `src/utils/fetch_utils.py`: `retry_failed_fetches()` (3 rounds, 8→4→2 concurrency) with `alt_kwargs_list` for fallback queries |
| **Signal pack cache persistence** | `cache_utils.py`: `read/write_signal_pack_cache()` alongside text cache |
| **Anti-hallucination** | Two-zone output (📊数据事实区 / 🔍分析判断区), `[数据]`/`[判断]` tags, graded missing-data language |
| **Industry knowledge** | `src/utils/industry_knowledge.py` (25 Shenwan industries), `src/utils/fund_type_knowledge.py` (ETF/MM/bond/hybrid/equity/QDII) |

### Anti-Hallucination Patterns

- Analysis agents: `📊 数据事实区` (only tool-returned data) + `🔍 分析判断区` (labeled as `【基于数据的推断】` or `【行业知识补充】`)
- Scorers: Graded missing-data language — "无法评估" (core missing, ≤40pts), "基于不完整数据" (partial, ≤65pts), "存在数据缺口" (minor)
- Summarizer: Every statement tagged `[数据]` or `[判断]`, must declare conflicts, must have counter-evidence

### Cache TTL

| Agent | TTL | Rationale |
|-------|-----|-----------|
| fundamental_analysis | 15 days | Quarterly data |
| value_analysis | 7 days | Valuation framework stable |
| quality_risk_analysis | 7 days | Quarterly financial quality |
| technical_analysis | 1 day | Daily price changes |
| news_analysis | 1 day | News timeliness |
| event_analysis | 1 day | Event freshness |
| moneyflow_analysis | 1 day | Daily market data |

Cache hit restores both text AND structured signal_pack (via `read_signal_pack_cache` → fallback re-extraction from cached LLM text).

### Two Tushare Access Paths

The system accesses Tushare data through two distinct layers — important for debugging:

**MCP tools (FastMCP stdio):** Used by agents during normal single-stock analysis. Each tool call goes through the MCP protocol → Tushare HTTP API. Has `tool_cache.py` in-memory cache (5-min TTL, MD5 keys).

**Direct HTTP (`tushare_client.py` / `data_fetcher.py`):** Used by `batch_scorer.py` and eval system for bulk operations. `tushare_client.py` wraps Tushare HTTP API with explicit rate limiting (200 calls/min, 0.35s interval). `data_fetcher.py` is eval-specific, raises `TushareUnavailableError` on failure.

`pool_screening.py` Layer 0 (`hard_screen`) uses `eval/data_fetcher._call()` directly for per-stock `daily` queries.

**`_BATCH_PREFETCH_ENABLED`** (`batch_scorer.py:55`): Toggles Tushare bulk pre-fetch before HTTP data fetch in `fetch_batch()`. Disable to isolate HTTP fetch issues from Tushare issues during debugging.

## Eval/Backtest System (V2 — 2026-06 upgrade)

The eval system (`src/eval/`) is an **evaluation control tower** surrounding the main advisory system. It runs simulation trading (14 lines) and historical backtesting (23 lines) to measure agent contributions via ablation experiments.

Design spec: `评分智能体开发总纲.md` at repo root. All eval design decisions reference this document.

### Line Architecture (总纲 §3)

**14 live simulation lines** (`src/eval/line_manager.py`):
- Short-term (10 lines): S-L0 (ablation baseline, all 7 agents) → S-L7 (minus moneyflow), S-L8 (longhold with mature short-term strategy), S-L9 (LLM free)
- Medium-term (2 lines): M-L0 (all agents), M-L1 (LLM free)
- Long-term (2 lines): L-L0 (all agents), L-L1 (LLM free)

**23 backtest lines** (`src/eval/replay_backtest_engine.py`):
- SB-L0~SB-L6 (7 short), MB-L0~MB-L7 (8 medium), LB-L0~LB-L7 (8 long)
- MB-L6/LB-L6 and MB-L7/LB-L7 are degenerate lines (remove news/event, already disabled in backtest) → consistency checks
- SB-L6 is a reference line (continuous holding with ShortLongHoldStrategy), NOT used in ablation ΔLoss

### Core Pipeline

```
run_full_check() → detect_missed_days() → catch_up (≤7 days) →
  settle_historical() → run_daily_rebalance() → run_daily_settlement() →
  ReportBuilder → ReportWriterAgent (DeepSeek V4 Pro) → MemoryManager
```

### Key Modules

| Module | Role |
|--------|------|
| `orchestrator.py` | Central scheduler: full check, rebalance, settlement, catch-up, pool update |
| `market_simulator.py` | Shared trade execution: commission, stamp tax, slippage, volume constraints, limit up/down, suspension, T+1 |
| `settlement_engine.py` | Daily P&L settlement: realized/unrealized gains, position tracking, NAV calculation |
| `loss_engine.py` | Multi-dim Loss: L_total = w_effect×L_effect + w_stability×L_stability + w_efficiency×L_efficiency |
| `contribution_engine.py` | Agent ablation ΔLoss with Bootstrap CI, Cluster Bootstrap (short-term), Permutation Test |
| `experiment_engine.py` | Manages ablation experiment configuration and execution across line combinations |
| `replay_backtest_engine.py` | PIT backtest: per-anchor agent analysis via cache, regime slicing (bull/bear/ranging) |
| `pool_manager.py` | 3 refined pools (short/medium/long) with health monitoring (5 trigger conditions from 总纲 §4.2) |
| `pool_screening.py` | 4-layer screening pipeline: hard screen → batch score (M1/M3) → quick screen (M2) → formal scoring |
| `fidelity_engine.py` | Detects score drift between eval model (M5) and production models (M1/M3) |
| `report_builder.py` | Aggregates settlement/loss/contribution data into structured report payloads |
| `report_writer_agent.py` | LLM report writing via DeepSeek V4 Pro with anti-hallucination verification |
| `memory_manager.py` | Long-term trend storage: score/loss/contribution/fidelity/runtime histories |
| `chart_service.py` | Trend data generation for Streamlit charts (5 tabs) |
| `check_runner.py` | Executes daily full-check cycle: orchestrates rebalance + settlement + report generation |
| `config.py` | Priority-based config: env vars > `config/eval/*.json` > hardcoded defaults |
| `database.py` | SQLite connection manager for `eval.db` (eval results, settlements, positions) |
| `repositories.py` | Data access layer: CRUD for settlements, positions, scores, losses, contributions |

### Eval Config System

Eval system uses a three-tier priority config (`src/eval/config.py`):
1. **Environment variables** (highest priority)
2. **JSON config files** in `config/eval/*.json` (pool sizes, thresholds, model profiles)
3. **Hardcoded defaults** in `config.py`

Key configurable parameters (via JSON or env):
- `stock_pool_short_size` (100), `stock_pool_medium_size` (80), `stock_pool_long_size` (60)
- `hard_screen_min_daily_amount` (20000 = 2000万元), `hard_screen_min_list_days` (60)
- `quick_screen_threshold_{short,medium,long}` (50)
- `grading_enabled` (True), `monday_full_run` (True)
- `score_change_upgrade_threshold` (10.0), `score_stable_downgrade_threshold` (3.0)

### Strategies (总纲 §5)

All strategies are pure code, no LLM calls (except LLM Free). Implemented in `src/eval/strategies/`:
- **ShortAblationStrategy** (`short_ablation.py`): Daily clear + Top-N by score, for ablation lines S-L0~S-L7
- **ShortLongHoldStrategy** (`short_longhold.py`): Continuous holding with multi-signal protection, for S-L8
- **MediumTermStrategy** (`medium_term.py`): Value + trend, weekly rebalance, cross-term synergy with long scores
- **LongTermStrategy** (`long_term.py`): Deep value, monthly rebalance, buy-more-on-dips, very restrained selling
- **LLMFreeStrategy** (`llm_free.py`): DeepSeek V4 Pro autonomous decisions using raw market data ONLY (NO Agent scores per 总纲 §3.5)

### Adapter Layer

`src/eval/adapters/stock_pipeline_adapter.py`:
- `run_stock_analysis(stock_code, company_name, as_of_date, eval_mode)` — non-invasive wrapper around `ScoringEngine.score_stock()`
- Returns scores + signal_packs + analysis_texts for all 3 terms
- `_build_decision_pack()` — maps scorer JSON output to `DecisionPack` dataclass

### Cache Architecture (总纲 §14)

**Dual cache system — production and eval are physically isolated:**

| | Production Cache | Eval Cache |
|---|---|---|
| Directory | `data/intermediate_cache/` | `data/eval/cache/` |
| Module | `cache_utils.py` | `eval/cache.py` |
| Key suffix | (none) | `_eval` |
| Model | M1/M3 | M5 (MiMo-V2.5) |

**Cache namespace switching** (`cache_utils.set_cache_namespace()`):
- Pool screening (M1/M3) → namespace=None → writes to `intermediate_cache/` → **shared with production**
- Daily simulation/backtest (M5) → namespace="eval" → writes to `eval/cache/` → **strictly isolated**
- `finish_batch()` auto-resets namespace to None

**Per-agent independent caching**: Each of the 7 agents caches its signal_pack separately with its own TTL (fundamental=15d, value=7d, quality_risk=7d, technical/news/event/moneyflow=1d). When all 7 agent caches are fresh, the LLM pipeline is skipped.

**Graded scoring frequency** (总纲 §14.4): Stocks analyzed at different cadences based on score tier:
- Holdings + high-volatility candidates: daily
- Stable high-score (score>75, low vol): every 3 days
- Mid-range (score 45-65): every 5 days
- Stable low-score (score<45): every 7 days
- Monday full coverage for all tiers

Implementation: `orchestrator._get_scoring_frequency_tier()` (line ~877) computes tiers, `_should_analyze_today()` (line ~910) applies day_of_year modulo rules. Non-analysis-day stocks fall back to cached pool scores.

**Eval orchestrator L2 cache check** (`orchestrator.py:770-822`, `_get_real_scores()`): Before running the full pipeline, reads all 7 agent signal_pack caches via `read_signal_pack_cache()`. If ALL 7 are fresh within their TTL → skips the entire LLM pipeline and calls `_assemble_from_agent_caches()` to reconstruct the result from cached data. If any one agent's cache is stale or missing → runs `run_stock_analysis()` for the full pipeline, then decomposes and writes per-agent caches individually.

### Pool Screening Performance

Cold start (first run, no caches exist) — V3 流水线 (2026-07 优化后):
- Layer 0: ~2min (20 批量 `daily` API queries, trade_date-keyed)
- Layer 1: ~20-30min (Tushare prefetch + HTTP parallel `semaphore=24` + LLM × ~900 batches of 5, 100只/批流式产出 raw_data)
- Layer 2: ~3-5min (**复用 L1 raw_data, 跳过 fetch_batch**; DSV4Pro 流式双堆 top-α 后台流水线)
- Layer 3: ~30-60min (~100 stocks × ~200s/stock, **5 并发** async queue consumer, shared ScoringEngine)
- **Total cold start: ~1.0-1.5h** (short/medium), ~0.8-1.0h (long)

V3 vs V2 savings:
- 消除 Layer 2 冗余 `fetch_batch` (省 ~10-20min)
- L2 后台流水线不阻塞 L1 循环 (省 ~2-5min)
- L3 并发从 5 提升到 8 (省 ~15-20min)

Hot cache (subsequent runs within TTL):
- Agents with fresh caches skip LLM (fundamental=15d, value/quality_risk=7d, tech/news/event/moneyflow=1d)
- Eval orchestrator skips entire pipeline when all 7 signal_packs hit
- Layer 3 bottleneck shifts from agent LLM time to scorer LLM time only

Key bottleneck (V3): Layer 3 的 5 并发 ScoringEngine × 7 agent × MCP stdio server。Layer 0 的 per-day 批量 `daily` 查询 (20 次 Tushare API) 已优化, 不再是 ~4500 次逐只查询。

### Eval-Specific Constraints

- **Max catch-up: 7 trading days** (总纲 §7.1). If missed days > 7, reset to latest trading day with empty positions.
- **Tushare unavailable → hard error** (`TushareUnavailableError` in `data_fetcher.py`). Never fall back to simulated/estimated data. 总纲 §20.2 第10条.
- **LLM Free lines**: DeepSeek V4 Pro makes autonomous decisions using raw Tushare MCP data ONLY. Agent scores/signal_packs/analysis_packages are FORBIDDEN in LLM Free prompts (总纲 §3.5).
- **Market regime slicing**: Backtest can optionally run `run_regime_analysis()` to compare agent contributions in bull/bear/ranging markets.
- **10-batch minimum for trend charts**: Trend curves only render after ≥10 batches accumulated, otherwise show progress bar.
- **Progress bars**: All operations >1 second MUST show `st.progress()` + `st.status()` in Streamlit (总纲 §16.1.9).

## Streamlit Web UI

The web UI is in `src/app/` with `Home.py` as the entry point and 7 pages in `src/app/pages/`:

| Page | File | Purpose |
|------|------|---------|
| 股票查询 | `01_股票查询.py` | Single-stock search and quick analysis |
| 股票池 | `02_股票池.py` | Pool management: add/remove stocks, view scores |
| 批量打分 | `03_批量打分.py` | Batch scoring for multiple stocks |
| 智能问答 | `04_智能问答.py` | Natural language Q&A about stocks/markets |
| 基金专区 | `05_基金专区.py` | Fund/ETF analysis and pool management |
| 模拟分析与迭代 | `06_模拟分析与迭代.py` | Eval system UI: line status, trends, pool updates, reports |
| 智能投顾 | `07_智能投顾.py` | Investment advisor dashboard with multi-turn chat, persistent session history, collapsible history sidebar |

**UI components** in `src/app/components/`: reusable Streamlit components (charts, tables, status indicators).
**API client** (`src/app/api_client.py`): HTTP client for communicating with the FastAPI backend when running in API mode.
**Theme** (`src/app/theme.py`): centralized color scheme and styling.

**精筛池更新跨重启存活**: The "🎯 更新精筛池" operation runs as a detached subprocess (see `src/eval/job_manager.py` and `src/eval/pool_update_worker.py`), decoupled from Streamlit lifecycle.

## Data Directory Structure

`data/` under `Finance/Financial-MCP-Agent/`:

| Directory | Purpose |
|-----------|---------|
| `intermediate_cache/` | Production LLM cache (M1/M3), no `_eval` suffix |
| `eval/cache/` | Eval system LLM cache (M5), `_eval` suffix — **physically isolated from production** |
| `eval/` | Eval DB (`eval.db`), settlement records, position snapshots, refined pool JSON |
| `pool_update_jobs/` | Job state files + worker logs for detached pool updates (1-year retention) |
| `batch_jobs/` | Batch scoring job state |
| `advisory_sessions/` | Multi-turn advisory chat sessions (JSON persistence) |
| `advisory_settlements/` | Advisory system settlement records |
| `advisory_llm_free/` | LLM Free strategy decision logs |
| `portfolios/` | User portfolio data |
| `qa_sessions/` | Q&A engine session state |
| `reports/` | Generated Markdown/PDF reports |
| `strategies/` | Strategy configuration files |
| `user_profiles/` | User preference profiles |

## Commands

All commands run from `Finance/Financial-MCP-Agent/`.

### Setup
**Windows first time**: `setup.bat`
**Linux/WSL**: `cd Finance && pip install -r requirements.txt && cd Financial-MCP-Agent && cp .env.example .env`

### Web UI
```bash
./run.sh start|stop|status|restart   # Linux/WSL
.\run.ps1 start|stop|status|restart  # Windows PowerShell
run.bat start|stop|status|restart    # Windows CMD
```

### CLI
```bash
python -m src.main --command "分析嘉友国际"     # Single-stock analysis
python -m src.main_pool add 603871 嘉友国际    # Add to pool (default: medium)
python -m src.main_pool score 603871          # Score (all 3 terms)
python -m src.main_pool report 603871         # View score details
python -m src.fund_main --command "分析华夏上证50ETF"  # Fund analysis

# Eval system
python -m src.eval check              # One-click daily check (rebalance + settlement)
python -m src.eval status             # View all line status
python -m src.eval pool status --term short  # View refined pool
python -m src.eval pool update --term short --mode full  # Full pool update (4-layer pipeline)
python -m src.eval backtest --term medium --start 2024-01-01 --end 2025-12-31  # Run backtest
python -m src.eval report --latest    # View latest evaluation report
python -m src.eval trends --metric score --term medium --days 90  # View trends
python -m src.eval agent-contribution --term medium --source backtest  # Agent contribution data
python -m src.eval optimize --analyze  # Generate optimization suggestions
```

### Testing
```bash
python -m pytest tests/ -v                          # All tests
python -m pytest tests/test_analysis_package.py -v  # Schema + builder
python -m pytest tests/test_risk_gate.py -v         # Risk gate rules
python -m pytest tests/test_market_simulator.py -v  # Trade execution logic
python -m pytest tests/test_loss_engine.py -v       # Multi-dim loss calculation
python -m pytest tests/test_contribution.py -v      # Agent ablation + Bootstrap CI
python -m pytest tests/test_orchestrator.py -v      # Eval scheduler
python -m pytest tests/test_pool_screening.py -v    # 4-layer screening pipeline
python -m pytest tests/test_eval_cache.py -v        # Eval cache isolation
python -m pytest tests/test_job_manager.py -v       # Pool update job lifecycle
python -m pytest tests/test_backtest.py -v          # Replay backtest engine
python -m pytest tests/test_settlement.py -v        # Daily settlement logic
python -m pytest tests/test_qa.py -v                # Q&A engine
```

## Environment Variables

`.env` file. Six-model architecture via `src/utils/model_config.py`:

| Model | Suffix | Default Model | Used By |
|-------|--------|---------------|---------|
| M1 | (none) | MiMo-V2.5-Pro | summary, medium/long scorer, fund scorer/report/perf/holdings, fundamental, value, quality_risk |
| M2 | `_2` | Qwen3.6-Flash | Quick query, quick-screen, batch scoring |
| M3 | `_3` | Qwen3.7-Plus | technical, news, short scorer, event, moneyflow, fund manager/event/fee/doc/benchmark |
| M4 | `_4` | (unused — migrated to M1) | Previously Kimi K2.6 for fundamental/value |
| M5 | `_5` | MiMo-V2.5 | qa_engine; complex → M1 (qa_engine_pro). **Also used by eval system** for daily simulation/backtest agent analysis (cost-effective). |
| M6 | `_6` | DeepSeek V4 Pro | **Eval system orchestrator**: ablation analysis, root cause diagnosis, optimization suggestions, report writing, LLM free investment lines. Must use anti-hallucination verification (§12 of 总纲). |

Env var naming: `OPENAI_COMPATIBLE_API_KEY{_N}`, `OPENAI_COMPATIBLE_BASE_URL{_N}`, `OPENAI_COMPATIBLE_MODEL{_N}`.
`get_thinking_body()` handles DashScope (`enable_thinking`) vs OpenAI-compatible (`thinking.type`) parameter formats.

## Key Utility Modules

- `src/utils/analysis_schema.py` — Signal, SignalPack, AnalysisPackage, RiskGateResult dataclasses; SourceLevel enum; FALLBACK_SIGNAL_PACK
- `src/utils/analysis_package_builder.py` — Merges 7 signal_packs into AnalysisPackage: conflict detection, source priority sorting, compact_prompt_context generation
- `src/utils/risk_gate.py` — 4-rule post-scoring risk gate for A-stock
- `src/utils/fund_risk_gate.py` — 10-rule risk gate for fund scoring
- `src/utils/tool_cache.py` — In-memory MCP tool result cache (5-min TTL, asyncio.Lock, 500-entry cap)
- `src/utils/cache_utils.py` — File-based intermediate cache with per-agent TTL; `read/write_signal_pack_cache()`. V2 adds `set_cache_namespace("eval" | None)` for production vs eval cache isolation (thread-safe). Eval namespace routes writes to `data/eval/cache/` with `_eval` suffix, production writes to `data/intermediate_cache/` without suffix.
- `src/utils/fetch_utils.py` — `retry_failed_fetches()` with `alt_kwargs_list` support
- `src/utils/model_config.py` — Centralized agent→model mapping; `get_model_config_for_agent()`, `get_thinking_body()`
- `src/utils/industry_knowledge.py` — 25 Shenwan industry PE/PB/ROE benchmarks
- `src/utils/fund_type_knowledge.py` — Fund-type-specific scoring guidance
- `src/utils/tushare_client.py` — Tushare API wrappers (stock_info, daily_basic, fina_indicator, PE percentile, etc.)
- `src/utils/stock_data_cache.py` — Stock data caching layer for frequently accessed market data
- `src/stock_pool/stock_pool_manager.py` — 5 per-term pools (short/medium/long/quick_screen/fine) in `stock_pool.json`

## Anti-Patterns to Avoid

- **Do NOT add new agents without signal_pack output** — every analysis agent must produce `<SIGNAL_PACK>` JSON
- **Do NOT skip tool_cache in _call_tool_safe** — all agents that call MCP tools must use tool-level cache
- **A股数据优先 Tushare，不要为已有 Tushare 覆盖的 A 股数据新增 AKShare 依赖。**
  但国际市场/宏观/商品数据（美国CPI/PMI/非农、COMEX库存、国际期货等）Tushare 无覆盖，
  可使用 AKShare 对应函数 + Yahoo Finance + Web Search 补充。
- **Do NOT assume LLM JSON is perfectly typed** — always normalize `strength→int`, `confidence→float`, `data_quality_score→float`
- **Do NOT hardcode model names** — use `get_model_config_for_agent()` from `model_config.py`
- **Do NOT skip the cache-hit signal_pack fallback** — every agent must restore signal_pack on cache hit (try `read_signal_pack_cache` → regex re-extraction → `text_to_signal_pack`)
- **Do NOT remove the `merge_dicts` reducer** — parallel agent writes depend on it
- **Do NOT contaminate production cache with M5 results** — daily simulation/backtest MUST use `cache_namespace="eval"` to isolate M5 outputs in `data/eval/cache/`. Only pool screening (M1/M3) writes to production cache.
- **Do NOT use Agent scores in LLM Free prompts** — total纲 §3.5 prohibits passing `{agent}_score`, `{agent}_signal_pack`, or `analysis_package` to LLM free lines.
- **Do NOT fall back to simulated/estimated data when Tushare is down** — raise `TushareUnavailableError` and stop. 总纲 §20.2 第10条.
- **Do NOT skip eval cache (_eval suffix)** — all eval agent cache keys must use the `_eval` suffix to prevent cross-contamination with production cache.
- **Do NOT modify 总纲 without updating CLAUDE.md** — the two documents must stay in sync on architecture, constraints, and design decisions.

## Windows-Specific Notes

- All entry points set `WindowsSelectorEventLoopPolicy` — required for MCP stdio subprocess
- `src/tools/mcp_config.py` uses `sys.executable` (not hardcoded `python`)
- WSL-created venvs are incompatible; recreate with `python -m venv venv`
- `launch.ps1` uses `taskkill /f /t` for tree-kill of uvicorn/streamlit
