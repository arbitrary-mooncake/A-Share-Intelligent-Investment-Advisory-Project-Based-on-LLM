# ST Risk Warning Feature - Design Spec

**Goal:** Add ST (Special Treatment) risk warning section to both single-stock reports and stock pool scoring reports.

**Architecture:** Three-layer: (1) New MCP tools for ST data, (2) Enhanced fundamental/news agents with ST analysis, (3) New ST risk section in summary report and scorers.

**Tech Stack:** Python, AkShare/Sina, Tushare API, FastMCP, LangGraph

## Data Layer

Two new MCP tools with failover:

| Priority | Tool | Source | Data | Timeout |
|----------|------|--------|------|---------|
| P0 | `tushare_st_status` | Tushare `stock_st` | ST history with dates, type, reason | 10s |
| P1 | `get_st_risk_data` | AkShare/Sina `stock_info_a_code_name` | Name-based ST detection | 5s |

Agent strategy: fundamental_agent tries P0 first, falls back to P1. news_agent uses P1 only.

## Agent Changes

- **fundamental_agent**: Add `tushare_st_status`, `get_st_risk_data` tools; add ST analysis dimension to prompt
- **news_agent**: Add `get_st_risk_data` tool; add ST awareness to system prompt
- **scoring_nodes**: Extract ST data for scorers (they already have ST scoring dimensions)

## Report Changes

New "ST Risk Warning" section in summary report (between Risk Factors and Investment Recommendation):
- Current ST status, type, start date
- Trigger reason, delisting risk level
- Trading restrictions, impact assessment

## Failover

All new MCP calls: timeout protection + exception catch + Tushare→AkShare→skip degradation path
