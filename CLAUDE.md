# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

A stock investment advisor agent system for A-share (Chinese stock market) and fund/ETF analysis. Uses LangGraph for multi-agent workflow orchestration, MCP (Model Context Protocol) for data access, and LLMs for analysis/scoring. Runs on Linux/WSL and Windows.

## Architecture

### Two-Layer Architecture

**Layer 1: MCP Servers** (`Finance/a-share-mcp-server/`)
- `tushare_mcp_server.py` — Tushare MCP server (primary data source, all agents use Tushare tools)
- `mcp_server.py` — Legacy AKshare MCP server (minimal use, being phased out)
- Both run over stdio via FastMCP; in-memory tool-level cache (5-min TTL)

**Layer 2: Financial MCP Agent** (`Finance/Financial-MCP-Agent/`)
- Consumes MCP servers via `langchain-mcp-adapters` (`MultiServerMCPClient`)
- LangGraph workflows orchestrate analysis → scoring/report pipelines
- LLM calls via OpenAI-compatible API (5-model architecture in `.env`)

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
```

### Testing
```bash
python -m pytest tests/ -v                          # All tests
python -m pytest tests/test_analysis_package.py -v  # Schema + builder
python -m pytest tests/test_risk_gate.py -v         # Risk gate rules
```

## Environment Variables

`.env` file. Five-model architecture via `src/utils/model_config.py`:

| Model | Suffix | Default Model | Used By |
|-------|--------|---------------|---------|
| M1 | (none) | MiMo-V2.5-Pro | summary, medium/long scorer, fund scorer/report/perf/holdings, fundamental, value, quality_risk |
| M2 | `_2` | Qwen3.6-Flash | Quick query, quick-screen, batch scoring |
| M3 | `_3` | Qwen3.7-Plus | technical, news, short scorer, event, moneyflow, fund manager/event/fee/doc/benchmark |
| M4 | `_4` | (unused — migrated to M1) | Previously Kimi K2.6 for fundamental/value |
| M5 | `_5` | MiMo-V2.5 | qa_engine; complex → M1 (qa_engine_pro) |

Env var naming: `OPENAI_COMPATIBLE_API_KEY{_N}`, `OPENAI_COMPATIBLE_BASE_URL{_N}`, `OPENAI_COMPATIBLE_MODEL{_N}`.
`get_thinking_body()` handles DashScope (`enable_thinking`) vs OpenAI-compatible (`thinking.type`) parameter formats.

## Key Utility Modules

- `src/utils/analysis_schema.py` — Signal, SignalPack, AnalysisPackage, RiskGateResult dataclasses; SourceLevel enum; FALLBACK_SIGNAL_PACK
- `src/utils/analysis_package_builder.py` — Merges 7 signal_packs into AnalysisPackage: conflict detection, source priority sorting, compact_prompt_context generation
- `src/utils/risk_gate.py` — 4-rule post-scoring risk gate for A-stock
- `src/utils/fund_risk_gate.py` — 10-rule risk gate for fund scoring
- `src/utils/tool_cache.py` — In-memory MCP tool result cache (5-min TTL, asyncio.Lock, 500-entry cap)
- `src/utils/cache_utils.py` — File-based intermediate cache with per-agent TTL; `read/write_signal_pack_cache()`
- `src/utils/fetch_utils.py` — `retry_failed_fetches()` with `alt_kwargs_list` support
- `src/utils/model_config.py` — Centralized agent→model mapping; `get_model_config_for_agent()`, `get_thinking_body()`
- `src/utils/industry_knowledge.py` — 25 Shenwan industry PE/PB/ROE benchmarks
- `src/utils/fund_type_knowledge.py` — Fund-type-specific scoring guidance
- `src/utils/tushare_client.py` — Tushare API wrappers (stock_info, daily_basic, fina_indicator, PE percentile, etc.)
- `src/stock_pool/stock_pool_manager.py` — 5 per-term pools (short/medium/long/quick_screen/fine) in `stock_pool.json`

## Anti-Patterns to Avoid

- **Do NOT add new agents without signal_pack output** — every analysis agent must produce `<SIGNAL_PACK>` JSON
- **Do NOT skip tool_cache in _call_tool_safe** — all agents that call MCP tools must use tool-level cache
- **Do NOT add new AkShare tool dependencies** — migrate to Tushare equivalents; AkShare tools are deprecated
- **Do NOT assume LLM JSON is perfectly typed** — always normalize `strength→int`, `confidence→float`, `data_quality_score→float`
- **Do NOT hardcode model names** — use `get_model_config_for_agent()` from `model_config.py`
- **Do NOT skip the cache-hit signal_pack fallback** — every agent must restore signal_pack on cache hit (try `read_signal_pack_cache` → regex re-extraction → `text_to_signal_pack`)
- **Do NOT remove the `merge_dicts` reducer** — parallel agent writes depend on it

## Windows-Specific Notes

- All entry points set `WindowsSelectorEventLoopPolicy` — required for MCP stdio subprocess
- `src/tools/mcp_config.py` uses `sys.executable` (not hardcoded `python`)
- WSL-created venvs are incompatible; recreate with `python -m venv venv`
- `launch.ps1` uses `taskkill /f /t` for tree-kill of uvicorn/streamlit
