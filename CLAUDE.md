# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

A stock investment advisor agent system for A-share (Chinese stock market) analysis. Uses LangGraph for multi-agent workflow orchestration, MCP (Model Context Protocol) for data access, and LLMs for analysis/scoring. Runs on Linux/WSL and Windows.

## Architecture

### Two-Layer Architecture

**Layer 1: A-Share MCP Server** (`Finance/a-share-mcp-is-just-i-need/`)
- FastMCP server over stdio; 8 tool categories for stock data
- Composite data source: `BaostockDataSource` (primary) → `AkshareDataSource` (fallback)
- Fallback triggered by `NoDataFoundError`, `DataSourceError`, or unexpected exceptions
- Entry: `python mcp_server.py`

**Layer 2: Financial MCP Agent** (`Finance/Financial-MCP-Agent/`)
- Consumes MCP server via `langchain-mcp-adapters` (`MultiServerMCPClient`)
- LangGraph workflows orchestrate 4 analysis agents + 1 summarizer (single stock) or 3 scorers (stock pool)
- LLM calls via OpenAI-compatible API configured in `.env`

### LangGraph Workflows

**Single-stock analysis** (`src/main.py`):
```
start_node → [fundamental, technical, value, news] (4 parallel) → summarizer → END
```

**Stock pool scoring** (`src/stock_pool/scoring_engine.py`):
```
start_node → [fundamental, technical, value, news] (4 parallel) → [short, medium, long_term_scorer] (3 parallel after deps met) → END
```

Scorer dependency rules built into edges:
- `short_term_scorer`: waits for `technical_analyst` + `news_analyst` only
- `medium_term_scorer` / `long_term_scorer`: wait for all 4 analysis agents

### Agent State

Defined in `src/utils/state_definition.py` as `AgentState` TypedDict with three keys:
- `messages`: `Annotated[Sequence[BaseMessage], operator.add]` — LangChain message history
- `data`: `Annotated[Dict[str, Any], merge_dicts]` — analysis outputs from each agent
- `metadata`: `Annotated[Dict[str, Any], merge_dicts]` — execution metadata

Each analysis agent writes to `state.data.{fundamental_analysis, technical_analysis, value_analysis, news_analysis}`. Scorers read these and write to `state.data.{short_term_score, medium_term_score, long_term_score}`. The `merge_dicts` reducer ensures parallel writes don't clobber each other.

### State Data Flow for Scoring

```
initial state (stock_code, company_name, query, dates)
  → 4 analysis agents write str outputs to data.{*}_analysis
  → scoring_nodes.py wrappers extract analysis strings, call scorer functions
  → scorers return Dict with: score, sub_scores, rating, reasoning, risk_warning, suggested_action, time_horizon
  → wrappers write to data.{short,medium,long}_term_score
  → ScoringEngine reads final state, updates StockPoolManager
```

### Scorer Outputs

All 3 scorers return the same shape. Medium/long add `moat_type`:
```python
{"score": int, "sub_scores": {"fundamental_quality": int, ...}, "rating": str,
 "reasoning": str, "risk_warning": str, "suggested_action": str, "time_horizon": str}
```

### Scoring Frameworks (all out of 100)

| | Short (1-5 days) | Medium (1-3 months) | Long (1-3 years) |
|---|---|---|---|
| **Depends on** | technical + news only | all 4 agents | all 4 agents |
| **Top weights** | 量价关系 30, 情绪资金 25, 技术信号 25 | 基本面质量 25, 估值 20, 风险 15 | 商业护城河 30, 行业景气 15, 估值安全边际 12 |
| **thinking mode** | enabled | enabled | enabled |
| **max_tokens** | 16000 | 16000 | 16000 |

### Industry-Adaptive Scoring

`src/utils/industry_knowledge.py` contains 32 Shenwan (申万) industry benchmarks. Each scorer calls `identify_industry()` to auto-detect the stock's industry from analysis text, then injects industry-specific PE/PB medians, typical ROE/growth rates, and scoring guidance into the LLM prompt. This ensures cross-industry fair scoring — e.g., bank stocks aren't penalized for naturally low PE.

### Stock Pool Manager Data Architecture

`src/stock_pool/stock_pool_manager.py` maintains three independent per-term pools in `stock_pool.json`:
```json
{"pools": {"short": {"stocks": {}}, "medium": {"stocks": {}}, "long": {"stocks": {}}}}
```
Each stock entry: `stock_code, company_name, score, recommendation, term_score, last_updated, status, score_history[]`. Status progresses: `pending → scoring → scored/failed`. The `list_stocks()` and `get_stock()` methods merge across pools for CLI backward compatibility.

## Commands

All commands run from `Finance/Financial-MCP-Agent/`.

### Setup

**Windows first time**: `setup.bat` (creates venv, installs deps)
**Linux/WSL**: `cd Finance && pip install -r requirements.txt && cd Financial-MCP-Agent && cp .env.example .env`

### Web UI
```bash
# Linux/WSL
./run.sh start|stop|status|restart

# Windows PowerShell
.\run.ps1 start|stop|status|restart

# Windows CMD
run.bat start|stop|status|restart

# Windows desktop shortcut (end-user launcher)
# Shortcut runs: powershell -ExecutionPolicy Bypass -NoExit -File launch.ps1
# Or double-click launch.bat
# Press Enter to stop all services

# Manual:
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
streamlit run src/app/Home.py --server.port 8501
```

### CLI (Terminal Mode)
```bash
python -m src.main                          # Single-stock analysis, interactive
python -m src.main --command "分析嘉友国际"   # Single-stock, CLI arg
python -m src.main_pool                     # Stock pool management, interactive
python -m src.main_pool add 603871 嘉友国际  # Add stock (default: medium-term pool)
python -m src.main_pool score 603871        # Score a stock (all 3 terms)
python -m src.main_pool score-all           # Score all pending stocks
python -m src.main_pool list                # List all stocks
python -m src.main_pool report 603871       # View score details
```

### Logs
```bash
python -m src.utils.log_viewer --list            # List execution logs
python -m src.utils.log_viewer --show <exec_id>  # Show detailed log
```

## Environment Variables

`.env` file (see `.env.example`). Four-model architecture managed by `src/utils/model_config.py`:

| Model | Env Var Suffix | Default Model | Used By |
|-------|-----------|--------------|---------|
| Model 1 | (none) | MiMo-V2.5-Pro | summary_agent, medium/long_term_scorer (thinking enabled) |
| Model 2 | `_2` | Qwen3.6-Flash | Quick query, quick-screen pool, batch scoring (thinking disabled) |
| Model 3 | `_3` | Qwen3.6-Plus | technical_agent, news_agent, short_term_scorer |
| Model 4 | `_4` | Kimi K2.6 | fundamental_agent, value_agent |

Env var naming: `OPENAI_COMPATIBLE_API_KEY{_N}`, `OPENAI_COMPATIBLE_BASE_URL{_N}`, `OPENAI_COMPATIBLE_MODEL{_N}`. Model selection per agent is centralized in `src/utils/model_config.py`. Direct usage (like `app.py` quick-screen) passes env vars explicitly and sets `env_prefix=""`.

- `GEMINI_API_KEY` / `GEMINI_MODEL` — legacy fallback (GeminiClient)
- `USE_LOCAL_MODEL` — set to `api`

The LLM client factory in `src/utils/llm_clients.py` auto-detects which client to use. `OpenAICompatibleClient` supports `extra_body` for thinking mode control and `env_prefix` for multi-key switching.

## Frontend

FastAPI (`src/api/app.py`) + Streamlit (`src/app/Home.py` → pages/). REST endpoints:
- `POST /api/query` — quick stock lookup (Tencent/Sina HTTP data + LLM analysis, uses Model 2)
- `POST /api/report`, `GET /api/report/{task_id}` — async deep report generation (full LangGraph pipeline, 35-min timeout)
- `GET/POST/DELETE /api/pool/{term}` — CRUD per-term stock pool (term: short/medium/long/quick_screen)
- `POST /api/score/{term}/{stock_code}` — trigger full pipeline scoring
- `POST /api/quick-screen/score/{term}/{stock_code}` — quick-screen scoring (HTTP+LLM, bypasses MCP ReAct)
- `POST /api/batch-score/upload` — upload Excel, start batch scoring
- `GET /api/batch-score/{id}/progress`, `GET /api/batch-score/{id}/results` — batch job polling
- `GET /api/cache/{stock_code}`, `GET /api/health`

Intermediate cache at `data/intermediate_cache/` with 7-day sliding window.

Streamlit pages (import `src/app/` into `sys.path`):
- `src/app/Home.py` — navigation hub
- `src/app/pages/01_股票查询.py` — quick stock lookup (real-time data + LLM analysis)
- `src/app/pages/02_股票池.py` — stock pool management (add/remove/score across 4 pool terms)
- `src/app/pages/03_批量打分.py` — batch scoring UI (Excel upload, progress, color-coded results table)
- `src/app/config.py` — API URL, timeouts (including batch upload timeout ~900s)
- `src/app/api_client.py` — async httpx wrappers
- `src/app/components/` — reusable UI (query_form, result_card, pool_table, score_display, quick_screen_table, batch_progress)

Per-tab `st.session_state` keys are scoped per-term: `_pool_action_result_{short,medium,long}` to prevent cross-tab contamination.

## Batch Scoring (批量打分)

Upload an Excel file (stock codes + names, up to 500-1000 stocks). Backend classifies all stocks into a 5-level recommendation tier in ~8 min for 500 stocks.

### Architecture

```
Excel Upload → Parse → Stage 1: Parallel Data Fetch (Semaphore 6)
  → Stage 2: Batched LLM Scoring (5 stocks/call, Semaphore 5, Model 2, no thinking)
  → Results persistence → Frontend polling (2-3s interval)
```

Key files:
- `src/api/batch_scorer.py` — orchestrator (parse_excel, lookup_names, fetch_batch, score_batch, chunk_stocks)
- `src/app/pages/03_批量打分.py` — Streamlit upload page
- `src/app/components/batch_progress.py` — progress display

### Endpoints

- `POST /api/batch-score/upload` — upload .xlsx, start job (horizon: short/medium/long/all)
- `GET /api/batch-score/{id}/progress` — polling (fetched_count, scored_count, progress_pct)
- `GET /api/batch-score/{id}/results` — live results (returns data as available, not just at completion)

### 5-Level Output

```json
{"code": "sh.603871", "level": "强烈推荐", "confidence": "高",
 "reason": "低估值+高ROE+行业景气(30字内)", "risk": "原材料涨价风险(30字内)"}
```

Levels: 强烈推荐 → 推荐 → 中性 → 回避 → 卖出. Color mapping: green/teal/yellow/orange/red.

### Three Scoring Paths (Summary)

| Path | Latency | Purpose |
|------|---------|---------|
| Full Pipeline (LangGraph + MCP ReAct) | 10-40 min/stock | Definitive pool decisions |
| Quick-Screen (HTTP + direct LLM) | 1-5 min/stock | Fast pool pre-screening |
| Batch (parallel HTTP + batched LLM) | ~1s/stock amortized | Mass initial screening |

### Excel Format

| 股票代码 | 股票名称 |
|---------|---------|
| 603871 | 嘉友国际 |
| 000858 | 五粮液 |

Code normalization: 6-prefix → sh.*, 0/3-prefix → sz.*. Names auto-looked up via akshare cache if missing.

### Performance Budget

| Stage | 500 stocks | 1000 stocks |
|-------|-----------|------------|
| Data fetch (6 concurrent) | ~5 min | ~10 min |
| LLM scoring (5 concurrent, 5/call) | ~3 min | ~6 min |
| **Total** | **~8 min** | **~16 min** |

## Testing

```bash
# Run all pytest tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_batch_scorer.py -v
python -m pytest tests/test_batch_frontend.py -v

# Standalone extraction test (no pytest needed)
python test_extraction.py
```

Test files:
- `tests/test_batch_scorer.py` — prompt builder, response parser, chunking (TDD: Phase 2)
- `tests/test_batch_frontend.py` — API protocol shapes, progress calculation, color mapping, data normalization (TDD: Phase 3)
- `test_extraction.py` — stock code/name extraction from natural language queries (standalone, no deps)

No pytest config file — defaults work. Tests add `src/` or `src/app/` to `sys.path` for imports.

## Key Supporting Modules

- `src/utils/model_config.py` — centralized model-to-agent assignment (4-model architecture); all agents call `get_model_for_agent(agent_name)` to get the right `OpenAICompatibleClient` instance
- `src/utils/tushare_client.py` — Tushare API wrappers (get_stock_info, get_daily_basic, get_fina_indicator, get_dividend, compute_pe_percentile, compute_ev_ebitda, get_stock_news_em); used by `app.py`'s `_enrich_with_tushare()` for data enrichment and by batch scorer
- `src/utils/logging_config.py` — unified logging with emoji icons (SUCCESS_ICON, ERROR_ICON, WAIT_ICON)
- `src/utils/log_viewer.py` — CLI log viewer for execution history

## Windows-Specific Notes

- All entry points set `WindowsSelectorEventLoopPolicy` — required for MCP stdio subprocess on Windows
- `src/tools/mcp_config.py` uses `sys.executable` (not hardcoded `python`) and merges `PYTHONPATH` with `os.pathsep`
- Console UTF-8 encoding configured in entry points (with try/except fallback)
- WSL-created venvs (`venv/bin/python`) are incompatible; delete and recreate with `python -m venv venv`
- `launch.ps1` uses `taskkill /f /t` for tree-kill to ensure uvicorn/streamlit child processes die on exit
- The desktop shortcut calls `powershell.exe -ExecutionPolicy Bypass -NoExit -File launch.ps1`
