# Eval System 9 Critical Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 9 issues identified across audit report, gap analysis, and code review — ranging from critical scoring bugs to documentation sync.

**Architecture:** P0 fixes modify orchestrator cache-hit path and loss calibration. P1 fixes modify LLM Free strategy term injection, MarketData population, and 总纲 doc. P2 fixes clean dead code, add phase labels, integrate PROMPT_PATCH verification, and create missing test files.

**Tech Stack:** Python 3.13, async/await, LangGraph, Tushare API, Streamlit

## Global Constraints

- Do NOT modify agent code (src/agents/*) except for reading scorer signatures
- Do NOT change the LangGraph workflow graph structure
- All eval cache writes MUST use `_eval` suffix namespace
- Do NOT remove the `merge_dicts` reducer from AgentState
- Preserve backward compatibility with existing refined_pools.json data

---

## Task 1: Fix _assemble_from_agent_caches returning hardcoded 50.0

**Files:**
- Modify: `src/eval/orchestrator.py:889-909` (the `_assemble_from_agent_caches` method)
- Modify: `src/eval/orchestrator.py:818-824` (the cache-hit path in `_get_real_scores`)

**Problem:** When all 7 agent signal_pack caches are fresh, the method returns `score: 50.0` instead of running scorers with cached data.

**Fix:** Add a new method `_run_scorers_from_cached_packs` that directly calls scorer functions with cached signal_packs (bypassing the 7 agents). Call this from the cache-hit path.

---

## Task 2: Fix Loss calibration formula scale

**Files:**
- Modify: `src/eval/loss_engine.py:209` (the calibration expected value)

**Problem:** `(low+high)/200.0` produces expected returns 0.05-0.45 (5%-45%) vs actual daily returns ±2%.

**Fix:** Use a bounded mapping that reflects realistic daily expected returns.

---

## Task 3: LLM Free term injection from line definition

**Files:**
- Modify: `src/eval/strategies/llm_free.py:78-89,115,166`
- Modify: `src/eval/strategies/base.py` (add term support to BaseStrategy)
- Modify: `src/eval/orchestrator.py` (pass term when creating strategies)

**Problem:** `_infer_term` guesses term from holding count. Should be passed from line definition.

**Fix:** Add `term` parameter to BaseStrategy.__init__ and select_stocks kwargs; override in LLM Free.

---

## Task 4: Populate MarketData missing fields

**Files:**
- Modify: `src/eval/data_fetcher.py:170-247` (build_market_data_map)

**Problem:** Fields like roe, revenue_growth, dividend_yield, industry_pe_75th are never populated.

**Fix:** Enhance build_market_data_map to fetch additional fields from Tushare (fina_indicator for ROE/growth, stock_basic for industry).

---

## Task 5: Update 总纲 documentation

**Files:**
- Modify: `评分智能体开发总纲.md` (6 locations)

**Problem:** Code changed but doc not updated for blacklist TTL, disclosure_date, dynamic frequency, etc.

---

## Task 6: Remove _check_volume dead code

**Files:**
- Modify: `src/eval/market_simulator.py:210-219`

---

## Task 7: Add Phase labels to L_stability/L_efficiency

**Files:**
- Modify: `src/eval/loss_engine.py:461-465,550-554` (comments)

---

## Task 8: Integrate PROMPT_PATCH verification

**Files:**
- Modify: `src/eval/optimizer/prompt_patcher.py` (connect verify_prompt_change to backtest engine)

---

## Task 9: Create 3 missing test files

**Files:**
- Create: `tests/test_experiment_engine.py`
- Create: `tests/test_label_builder.py` (settlement-related)
- Create: `tests/test_manual_package_builder.py`
