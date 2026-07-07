# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A stock investment advisor agent system for A-share (Chinese stock market) and fund/ETF analysis. Uses LangGraph for multi-agent workflow orchestration, MCP (Model Context Protocol) for data access, and LLMs for analysis/scoring. Runs on Linux/WSL and Windows.

## Quick Start for Developers

```bash
# 1. Install dependencies
cd Finance
pip install -r requirements.txt

# 2. Configure environment
cd Financial-MCP-Agent
cp .env.example .env
# Edit .env with API keys (see "Environment Variables" section)

# 3. Start web UI
./run.sh start                    # Linux/WSL
.\run.ps1 start                   # Windows PowerShell
# Open http://localhost:8501

# 4. Run a single test
python -m pytest tests/test_analysis_package.py -v

# 5. Run single-stock analysis via CLI
python -m src.main --command "分析 603871"
```

## Feature Maturity

| Feature | Status | Notes |
|---------|--------|-------|
| **股票查询 (Stock Query)** | ★ Stable | Most mature feature - 7-agent parallel analysis + 9-section reports |
| **股票池 (Stock Pool)** | ★ Stable | Pool management + 3-term scoring |
| **批量打分 (Batch Scoring)** | Stable | Excel upload + batch scoring |
| **智能问答 (Q&A)** | Stable | Multi-turn chat, L1-L4 complexity levels |
| **基金专区 (Fund Analysis)** | Stable | 7-agent fund pipeline + 6-dimension scoring |
| **模拟分析与迭代 (Eval)** | ⚠ In Development | 14 sim lines + 23 backtest lines - some statistical tests need ≥10 batches |
| **智能投顾 (Advisory)** | ⚠ In Development | Requires pool update first (~1.5-2h cold start per pool) |

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

## Dual-Version Architecture (Lite/Full)

The system supports two operating modes controlled by `APP_MODE` in `.env`:

| | Lite (`APP_MODE=lite`) | Full (`APP_MODE=full`, default) |
|---|---|---|
| **LLM** | 1 DeepSeek API Key → all agents | 6 specialized models (M1-M6) |
| **Data** | Free Tushare (120 pts) + AKShare fallback | Tushare 5000+ pts |
| **Pages** | 5/7 (模拟分析与迭代 + 智能投顾 disabled) | All 7 |
| **Use case** | Trial, learning, zero-cost entry | Production research |

### Mode Manager

`src/utils/mode_manager.py` provides `is_lite_mode()`, `is_full_mode()`, `is_page_enabled()`, `set_mode()`. Mode is read from `os.getenv("APP_MODE", "full")` at runtime.

### Lite Mode Model Routing

`src/utils/model_config.py` — `get_model_config_for_agent()` checks `APP_MODE` first. In Lite mode, `_get_lite_model_config()` routes all agents to DeepSeek V4 Pro (or Flash for `quick_query`/`quick_screen`). Full mode code path is completely unchanged.

### AKShare Fallback in MCP Server

`tushare_mcp_server.py` — When `_IS_LITE_MODE=True` and a Tushare API call fails, `_call_akshare_fallback()` tries AKShare equivalents. Covers 8 API types: `fina_indicator`, `income`, `balancesheet`, `cashflow`, `moneyflow`, `top10_holders`, `daily`, `daily_basic`. AKShare results are converted to Tushare-compatible `{fields, items}` format via `src/data/data_adapter.py`. **Agents are completely unaware** — they call the same MCP tools in both modes.

### Unified Data Layer

`src/data/unified_data_layer.py` — Higher-level async data access with automatic Tushare→AKShare fallback. Wraps both `tushare_client._call()` and `akshare_client.call_akshare()`. Currently used as a utility; the primary fallback path for agents goes through the MCP server.

### Feature Gating

- `src/app/components/shared_sidebar.py`: Lite mode renders disabled pages as gray text with 🔒 (no `st.page_link`)
- `src/app/pages/06_模拟分析与迭代.py` and `07_智能投顾.py`: `is_lite_mode()` check at top → locked message + `st.stop()`
- `src/app/Home.py`: Navigation cards grayed out in Lite mode

### Onboarding

`src/app/onboarding.py`: First-run detection (`.env` missing or contains `your_` template placeholders). Presents version selection → API key wizard → writes `.env` + updates `os.environ` → `st.rerun()`.

### Mode Switch

`src/app/components/mode_panel.py`: Sidebar mode indicator + switch dialog. `set_mode()` in `mode_manager.py` writes `APP_MODE` to `.env` and updates `os.environ`. MCP server subprocess requires app restart to pick up changes.

### Critical Constraint

**Full version must remain completely unchanged.** All Lite logic is inside `if is_lite_mode()` branches. Full mode code paths have been verified to execute identically to the pre-dual-version state. If a Lite design conflicts with Full version behavior, the Lite design must change.

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

## Q&A Engine (智能问答子系统)

The Q&A module (`src/qa/`) powers the "智能问答" page — a ChatGPT-style multi-turn conversational interface for ad-hoc stock/market queries. **Distinct from the single-stock analysis pipeline** — uses lightweight models (M5) for fast responses, with runtime upgrade to M1 for complex questions.

### Architecture

```
process_question() (qa_engine.py)
  → complexity_analyzer.py (L1~L4 复杂度分级)
  → task_planner.py (数据域确定 + 标的提取 + 工具选择)
  → evidence_assembler.py (并行数据拉取 + 证据装配)
  → answer_generator.py (流式回答生成, SSE)
```

### Key Modules

| Module | Role |
|--------|------|
| `qa_engine.py` | 总编排器，协调复杂度分析→任务规划→证据装配→运行时升级→回答生成 |
| `complexity_analyzer.py` | L1~L4 复杂度分级，决定是否启用 ReAct 或运行时升级模型 |
| `task_planner.py` | 数据域确定（基本面/技术面/资金面等）、标的提取（股票代码/行业主题）、工具选择 |
| `evidence_assembler.py` | 并行调用 MCP 工具拉取数据，装配为 `EvidencePackage` |
| `answer_generator.py` | 流式回答生成（SSE 格式），支持 `[数据]`/`[判断]` 标签的反幻觉标注 |
| `session_manager.py` | 多轮对话会话管理，持久化到 `data/qa_sessions/` |
| `anti_hallucination.py` | Q&A 专用的反幻觉验证层 |

### Complexity Levels & Runtime Upgrade

| Level | Description | Model | ReAct |
|-------|-------------|-------|-------|
| L1 | 简单数据查询（单只股票单一指标） | M5 | ❌ |
| L2 | 中等查询（单只股票多指标对比） | M5 | ❌ |
| L3 | 复杂分析（多股票对比/行业分析） | M5 → 可升级 M1 | 可选 |
| L4 | 深度研究（策略讨论/投资组合） | M1 | ✅ |

运行时升级（`try_runtime_upgrade()`）：L3/L4 问题自动切换到 M1（`qa_engine_pro`）以获得更强的推理能力。

### Topic-to-Stock Mapping

`task_planner.py` 内置 `TOPIC_STOCK_MAP`，支持无股票代码时的主题匹配：
- 黄金 → 黄金ETF (159934, 518880) + 代表性个股（中金黄金、紫金矿业）
- 半导体 → 半导体ETF + 代表性个股（中芯国际、寒武纪）
- 白酒、银行、新能源、医药等 15+ 主题

## Advisory System (智能投顾子系统)

The advisory module (`src/advisory/`) powers the "智能投顾" page with portfolio management, strategy execution, and multi-turn advisory sessions. **Distinct from the eval system's strategies** — these are user-facing trading strategies for portfolio construction, not simulation trading rules.

### Architecture

```
AdvisorySessionManager (多轮对话持久化)
  ↓
StrategyEngine (策略信号计算, 纯Python无LLM)
  ↓
StrategyRegistry (20+ 内置策略注册)
  ↓
PortfolioManager (持仓管理) + BacktestRunner (策略回测)
  ↓
SimulationRunner (模拟运行) + ReportGenerator (收益报告)
```

### Key Components

| Module | Role |
|--------|------|
| `strategy_engine.py` | 策略信号计算（静态方法，无LLM），返回 (signal, reason) 元组，signal: 1=买入, -1=卖出, 0=不动 |
| `strategy_registry.py` | 策略注册表，管理内置策略和用户自定义策略的加载/查询 |
| `strategies/builtin/*.py` | 20+ 内置策略（ma_cross, macd, rsi, kdj, boll, turtle, adx_macd, triple_ma 等），继承 `TradingStrategy` 基类 |
| `advisory_session_manager.py` | 多轮对话持久化到 `data/advisory_sessions/sessions.json`，线程安全，历史压缩（超 max_turns 轮早期对话压缩为摘要） |
| `portfolio_manager.py` | 持仓组合管理（虚拟持仓，非真实交易） |
| `backtest_runner.py` | 策略历史回测（与 eval 系统的回测线路独立） |
| `simulation_runner.py` | 策略模拟运行（实盘模拟，非 eval 系统的模拟盘线路） |
| `report_generator.py` | 收益归因报告生成 |
| `custom_strategy_loader.py` | 用户自定义策略加载器 |

### Strategy Implementation Pattern

所有内置策略继承 `src/advisory/strategies/strategy_base.py` 的 `TradingStrategy` 基类：
- 必须实现 `compute_signal(df_daily, params, context) → (signal, reason)`
- 可选实现 `risk_exit(context)` 用于风控止损逻辑
- 通过 `StrategyRegistry.register()` 注册后可通过 `StrategyEngine.compute_signal(strategy_name, ...)` 调用

### Dependency on Eval System

智能投顾功能依赖 eval 系统的精筛池数据（`data/eval/` 下的 refined pool JSON）。**使用智能投顾前必须先在"模拟分析与迭代"页面全量更新精筛池**（首次冷启动约 1.5~2 小时/池）。

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
**Shared sidebar** (`src/app/components/shared_sidebar.py`): custom navigation replacing Streamlit auto-generated sidebar; supports Lite mode feature gating (2 pages grayed out with 🔒).
**Mode panel** (`src/app/components/mode_panel.py`): sidebar mode indicator (⚡ Lite / 🚀 Full) + switch dialog.
**Onboarding** (`src/app/onboarding.py`): first-run wizard with version comparison, API key configuration, `.env` writing.
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

### Development Workflow

```bash
# Initial setup
cd Finance && pip install -r requirements.txt
cd Financial-MCP-Agent && cp .env.example .env
# Edit .env with API keys

# Start development server
./run.sh start                    # Linux/WSL
.\run.ps1 start                   # Windows PowerShell
run.bat start                     # Windows CMD

# Stop server
./run.sh stop

# Check server status
./run.sh status

# Restart after code changes
./run.sh restart
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_analysis_package.py -v

# Run single test function
python -m pytest tests/test_risk_gate.py::test_critical_risk_cap -v

# Run tests matching pattern
python -m pytest tests/ -v -k "cache"

# Run with coverage (if pytest-cov installed)
python -m pytest tests/ --cov=src --cov-report=term-missing

# Key test files by component:
python -m pytest tests/test_analysis_package.py -v  # Signal pack schema + builder
python -m pytest tests/test_risk_gate.py -v         # A-stock risk gate (4 rules)
python -m pytest tests/test_fund_risk_gate.py -v    # Fund risk gate (10 rules)
python -m pytest tests/test_eval_cache.py -v        # Cache isolation (production vs eval)
python -m pytest tests/test_pool_screening.py -v    # 4-layer screening pipeline
python -m pytest tests/test_contribution.py -v      # Agent ablation + Bootstrap CI
python -m pytest tests/test_market_simulator.py -v  # Trade execution logic
python -m pytest tests/test_backtest.py -v          # Replay backtest engine
python -m pytest tests/test_qa.py -v                # Q&A engine
```

### CLI Operations

**Single-Stock Analysis:**
```bash
python -m src.main --command "分析嘉友国际"     # By name
python -m src.main --command "分析 603871"      # By code
```

**Stock Pool Management:**
```bash
python -m src.main_pool add 603871 嘉友国际              # Add to medium pool (default)
python -m src.main_pool add 603871 嘉友国际 --term short # Add to short pool
python -m src.main_pool score 603871                     # Score all 3 terms
python -m src.main_pool report 603871                    # View score details
python -m src.main_pool list --term medium               # List pool contents
```

**Fund Analysis:**
```bash
python -m src.fund_main --command "分析华夏上证50ETF"
python -m src.fund_main --command "分析 510050"
```

**Eval System (Simulation & Backtest):**
```bash
# Daily operations
python -m src.eval check                                    # One-click: rebalance + settlement
python -m src.eval status                                   # View all 14 sim + 23 backtest lines

# Pool management
python -m src.eval pool status --term short                 # View refined pool status
python -m src.eval pool update --term short --mode full     # Full 4-layer pool update (~1.5h)
python -m src.eval pool update --term short --mode quick    # Quick update (Layer 2 only)

# Backtesting
python -m src.eval backtest --term medium --start 2024-01-01 --end 2025-12-31
python -m src.eval backtest --term short --start 2025-01-01 --regime-slicing  # With regime analysis

# Analysis & reporting
python -m src.eval report --latest                          # View latest evaluation report
python -m src.eval trends --metric score --term medium --days 90
python -m src.eval agent-contribution --term medium --source backtest
python -m src.eval optimize --analyze                       # Generate optimization suggestions
```

### Maintenance

```bash
# Clear production cache (M1/M3 results)
rm -rf data/intermediate_cache/*

# Clear eval cache (M5 results)
rm -rf data/eval/cache/*

# View pool update job logs
ls -lt data/pool_update_jobs/
cat data/pool_update_jobs/{job_id}.log

# Check eval database
sqlite3 data/eval/eval.db "SELECT COUNT(*) FROM settlements;"

# View cache statistics
python -c "from src.utils.cache_utils import get_cache_stats; print(get_cache_stats())"
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

**Lite mode** adds: `APP_MODE=lite`, `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` (default `https://api.deepseek.com/v1`), `DEEPSEEK_MODEL_PRO` (default `deepseek-v4-pro`), `DEEPSEEK_MODEL_FLASH` (default `deepseek-v4-flash`). When `APP_MODE=lite`, all agents route to DeepSeek regardless of `OPENAI_COMPATIBLE_*` values.

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
- `src/utils/mode_manager.py` — Dual-version mode detection: `is_lite_mode()`, `is_full_mode()`, `is_page_enabled()`, `set_mode()`
- `src/data/akshare_client.py` — Async AKShare client with rate limiting (2/s), 5-min cache, exception safety
- `src/data/data_adapter.py` — AKShare→Tushare format conversion (12 adapter functions for financial statements, money flow, daily data, etc.)
- `src/data/unified_data_layer.py` — Unified async data access with automatic Tushare→AKShare fallback based on Tushare point level
- `src/app/onboarding.py` — First-run onboarding: version selection, API key wizard, `.env` writing + `os.environ` update
- `src/app/components/mode_panel.py` — Sidebar mode indicator and switch dialog component

## Adding New Components

### Adding a New Analysis Agent

1. Create agent file in `src/agents/` following the pattern of existing agents (e.g., `fundamental_agent.py`)
2. **Must output `<SIGNAL_PACK>` JSON** in LLM prompt with structure:
   ```json
   {
     "bias": "bullish|bearish|neutral",
     "confidence": 0.0-1.0,
     "signals": [{"factor": "name", "direction": "up|down|flat", "strength": 1-5, "source_level": "official_like|structured|news|derived|proxy"}],
     "risk_flags": ["flag1", "flag2"],
     "missing_data": ["field1", "field2"]
   }
   ```
3. Use `_call_tool_safe()` with tool_cache for all MCP tool calls
4. Add agent to LangGraph workflow in `src/main.py` (A-stock) or `src/fund_main.py` (fund)
5. Register in `src/utils/model_config.py` with appropriate model (M1/M2/M3)
6. Add cache TTL to `src/utils/cache_utils.py` TTL table

### Adding a New Trading Strategy (Advisory System)

1. Create strategy file in `src/advisory/strategies/builtin/`
2. Inherit from `TradingStrategy` base class (`src/advisory/strategies/strategy_base.py`)
3. Implement `compute_signal(df_daily, params, context) → (signal, reason)`:
   - `signal`: 1 (buy), -1 (sell), 0 (hold)
   - `reason`: human-readable explanation
4. Optionally implement `risk_exit(context)` for stop-loss logic
5. Register via `StrategyRegistry.register()` in strategy file's module-level code
6. Strategy becomes available via `StrategyEngine.compute_signal(strategy_name, ...)`

### Adding a New Eval Line

1. Define line ID in `src/eval/line_manager.py` (follow naming: S-L#, M-L#, L-L# for sim; SB-L#, MB-L#, LB-L# for backtest)
2. Specify which agents to include/exclude (ablation design)
3. Choose strategy from `src/eval/strategies/` or create new one inheriting from base
4. Register in `LINE_CONFIGS` dict in `line_manager.py`
5. If backtest line, add to `replay_backtest_engine.py` line iteration
6. Update `评分智能体开发总纲.md` if changing ablation structure (keep CLAUDE.md in sync)

## Debugging Common Issues

### Cache-Related Issues

**Symptom**: Agent analysis returns stale data or doesn't reflect new market data

**Solution**:
```bash
# Clear production cache (M1/M3 results)
rm -rf data/intermediate_cache/*

# Clear eval cache (M5 results)
rm -rf data/eval/cache/*

# Clear specific agent cache
rm -f data/intermediate_cache/fundamental_analysis_603871.json
```

**Verify cache behavior**: Check `cache_utils.read_signal_pack_cache()` logs or add debug prints in agent files.

### MCP Server Not Responding

**Symptom**: Tool calls timeout or hang indefinitely

**Solution**:
1. Check MCP server process: `ps aux | grep mcp_server` (Linux) or Task Manager (Windows)
2. Restart web UI: `./run.sh restart` or `.\run.ps1 restart`
3. Check stderr logs in terminal where Streamlit was started
4. Verify `src/tools/mcp_config.py` uses `sys.executable` (not hardcoded `python`)

### Tushare API Errors

**Symptom**: `TushareUnavailableError` or rate limit errors

**Solutions**:
- Check Tushare token points (need ≥2000 for most APIs)
- Verify rate limiting in `tushare_client.py` (200 calls/min, 0.35s interval)
- For batch operations, check `_BATCH_PREFETCH_ENABLED` flag in `batch_scorer.py:55`
- If API is down, system will raise `TushareUnavailableError` by design (no fallback to estimated data)

### Model API Errors

**Symptom**: LLM calls fail or return malformed JSON

**Solutions**:
- Verify `.env` has correct `OPENAI_COMPATIBLE_API_KEY{_N}` and `BASE_URL`
- Check model availability with provider (MiMo/Qwen/DeepSeek)
- For malformed JSON, check signal_pack extraction in agent files (3-layer fallback: tag→regex→text)
- Add `print()` statements in `_extract_signal_pack()` to debug extraction

### Eval System Issues

**Symptom**: Pool update hangs or worker crashes

**Solutions**:
- Check job state: `ls data/pool_update_jobs/` for job status files
- View worker logs: `cat data/pool_update_jobs/{job_id}.log`
- Cold start takes ~1.5-2h per pool (Layer 3 bottleneck: 5 concurrent ScoringEngine × 7 agents)
- If worker dies, check for orphaned processes: `ps aux | grep pool_update_worker`
- Job state persists across Streamlit restart (detached subprocess design)

### Cache Contamination

**Symptom**: Eval results (M5) appear in production cache or vice versa

**Solution**:
- Verify `cache_utils.set_cache_namespace("eval")` is called before eval operations
- Check cache keys: production has no suffix, eval has `_eval` suffix
- `finish_batch()` auto-resets namespace to None - ensure it's called
- Never manually call agent analysis with wrong namespace

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
- **Do NOT put Lite mode logic outside `if is_lite_mode()` branches** — Full mode code paths must remain completely untouched. All Lite additions are additive branches, never modifications to existing Full mode logic.
- **Do NOT skip `os.environ` update after writing `.env`** — `onboarding.py` and `set_mode()` must update `os.environ` directly after writing `.env`, because `load_dotenv()` runs at module import time and `st.rerun()` does not re-import modules.
- **Do NOT modify agent code for Lite mode** — agents call MCP tools in both modes. AKShare fallback is handled transparently in `tushare_mcp_server.py._call_akshare_fallback()`. Agents should never import `mode_manager` or check `APP_MODE`.

## Windows-Specific Notes

- All entry points set `WindowsSelectorEventLoopPolicy` — required for MCP stdio subprocess
- `src/tools/mcp_config.py` uses `sys.executable` (not hardcoded `python`)
- WSL-created venvs are incompatible; recreate with `python -m venv venv`
- `launch.ps1` uses `taskkill /f /t` for tree-kill of uvicorn/streamlit
