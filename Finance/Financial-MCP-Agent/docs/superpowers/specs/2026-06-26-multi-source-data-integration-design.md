# Multi-Source Data Integration for QA System

**Date**: 2026-06-26
**Status**: Approved
**Goal**: Add international macro, commodity, and web search data sources to the QA system so it can analyze gold, commodities, and macro topics that Tushare cannot cover.

## Problem

Tushare is A-share focused. For cross-border topics (gold, crude oil, Fed policy, US Treasury yields, geopolitical events), the QA system has zero direct data — it can only proxy through A-share gold ETFs and mining stocks. This produces answers worse than a general AI with web search.

## Architecture

Three new data sources layered onto the existing MCP client:

```
QA Engine → get_mcp_tools()
  ├── Tushare MCP (existing, 65 tools) — A股 primary
  ├── AKShare MCP v2 (existing, extended) — international macro
  ├── Web Search MCP (NEW Phase 1) — real-time news/macro
  └── Yahoo Finance MCP (NEW Phase 2) — international quotes
```

All servers communicate over stdio via FastMCP, registered in `mcp_config.py`. The QA system loads all tools via `MultiServerMCPClient` and filters by `tool_filter`.

## Phase 1

### 1A. Web Search MCP Server

**File**: `Finance/a-share-mcp-server/web_search_mcp_server.py`

A standalone FastMCP server with two tools:

| Tool | Input | Output | Timeout |
|------|-------|--------|---------|
| `web_search` | `query: str`, `max_results: int = 10` | Title + URL + snippet per result, Markdown formatted | 15s |
| `web_fetch` | `url: str` | Full page text content, truncated at 8000 chars | 20s |

Backend: DuckDuckGo Instant Answer API via `duckduckgo-search` library (pip install). No API key needed.

```python
from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 10) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return format_as_markdown(results)

def web_fetch(url: str) -> str:
    import httpx
    resp = httpx.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    return clean_html(resp.text)[:8000]
```

Error handling: graceful degradation — if search fails, return error text rather than crashing.

### 1B. AKShare International Tools Extension

**File**: `Finance/a-share-mcp-server/src/tools/international.py` (NEW)

Expose existing AKShare `macro_usa_*` and commodity functions as MCP tools. AKShare 1.18.60 is already installed. No new dependencies.

| Tool | AKShare Function |
|------|-----------------|
| `get_us_cpi` | `macro_usa_cpi_yoy` |
| `get_us_pmi` | `macro_usa_ism_pmi` |
| `get_us_non_farm` | `macro_usa_non_farm` |
| `get_us_unemployment` | `macro_usa_unemployment_rate` |
| `get_us_gdp` | `macro_usa_gdp_monthly` |
| `get_us_retail_sales` | `macro_usa_retail_sales` |
| `get_comex_inventory` | `futures_comex_inventory` |
| `get_spot_gold_sge` | `spot_golden_benchmark_sge` |

Register in `mcp_server.py` via `register_international_tools(app)`.

### 1C. QA System Integration

1. **`task_planner.py`**: Add `MACRO_TOOLS` list. When `matched_topic_name` is in `{"黄金", "白银", "原油", "房地产"}` (the existing `macro_topics` set), append international tools + `web_search` to `task_plan.tools`.

2. **`evidence_assembler.py`**: Add kwargs mappings for new tools in `_build_tool_kwargs`. Add new tools to `_NO_CODE_TOOLS` so they work without a stock code.

3. **`qa_engine.py`**: When `topic_context` is injected for macro topics, add a hint for the LLM to use `web_search` for current events.

4. **`answer_generator.py`**: No changes needed — system prompt already handles multi-source data.

5. **CLAUDE.md**: Update anti-pattern "Do NOT add new AkShare tool dependencies" → "For A-share data, prefer Tushare. For international macro/commodities, AKShare is permitted as Tushare has no equivalents."

## Phase 2

### 2. Yahoo Finance MCP Server

**File**: `Finance/a-share-mcp-server/yfinance_mcp_server.py`

A standalone FastMCP server for international financial data via `yfinance` (pip install).

| Tool | Symbol | Data |
|------|--------|------|
| `get_commodity_price` | `symbol: str` | OHLCV for GC=F (gold), SI=F (silver), CL=F (oil) |
| `get_us_treasury_yield` | `tenor: str = "10y"` | ^TNX (10Y), ^FVX (5Y), ^IRX (3M) |
| `get_dollar_index` | — | DX-Y.NYB (DXY) |
| `get_gold_etf` | `symbol: str = "GLD"` | GLD, IAU holdings + price |

Uses `yfinance.Ticker(symbol).history(period="6mo")` for consistent output format. Returns OHLCV as Markdown table.

## Files Changed

| File | Action | Phase |
|------|--------|-------|
| `a-share-mcp-server/web_search_mcp_server.py` | CREATE | 1 |
| `a-share-mcp-server/src/tools/international.py` | CREATE | 1 |
| `a-share-mcp-server/mcp_server.py` | EDIT (add import + register) | 1 |
| `a-share-mcp-server/yfinance_mcp_server.py` | CREATE | 2 |
| `Financial-MCP-Agent/src/tools/mcp_config.py` | EDIT (add server configs) | 1+2 |
| `Financial-MCP-Agent/src/qa/task_planner.py` | EDIT (MACRO_TOOLS, topic tool injection) | 1 |
| `Financial-MCP-Agent/src/qa/evidence_assembler.py` | EDIT (tool kwargs, NO_CODE_TOOLS) | 1 |
| `Financial-MCP-Agent/src/qa/qa_engine.py` | EDIT (macro topic context) | 1 |
| `CLAUDE.md` | EDIT (update anti-pattern) | 1 |
| `requirements.txt` | EDIT (add yfinance, duckduckgo-search) | 1+2 |

## Dependencies

```
duckduckgo-search>=7.0.0   # Phase 1
yfinance>=0.2.50           # Phase 2
httpx                       # Already installed
```

## Testing

- Unit tests for each new MCP tool (mock the underlying library)
- Integration test: gold topic query → task plan includes international tools
- End-to-end: "深度分析黄金走势" → verify evidence includes non-A-share data

## Non-Goals

- FRED API integration (deferred per user request)
- Real-time streaming price feeds
- Historical backfill of international data
- Replacing Tushare for A-share data
