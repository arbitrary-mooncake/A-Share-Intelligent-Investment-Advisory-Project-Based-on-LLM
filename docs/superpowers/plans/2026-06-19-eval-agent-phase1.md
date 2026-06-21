# 评分智能体 Phase 1 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> **Spec:** 评分智能体开发总纲.md (20 chapters)
> **Key constraint:** Do NOT break existing 5 functions. Minimal changes to existing files.

**Goal:** Build the evaluation agent foundation — data schemas, storage, market simulator, strategies, cache, adapter, CLI, and UI integration.

**Architecture:** New code in `src/eval/` with 5 independent subsystems. Existing files receive minimal non-breaking additions (analysis_schema.py, model_config.py, cache_utils.py, Home.py, new streamlit page).

**Tech Stack:** Python 3.11+, SQLite, Streamlit, LangChain (ChatOpenAI), Tushare MCP

---

## Subsystem A: Data Foundation (独立，无依赖)

### Task A1: DecisionPack schema
**Files:** Modify `src/utils/analysis_schema.py` (add ~60 lines)
**Test:** `tests/test_decision_pack.py`

Add `DecisionPack` dataclass with all fields from 总纲 §15.4, plus `normalize()` static method and `to_json()`. Must coexist with existing `Signal`, `SignalPack`, `AnalysisPackage`, `RiskGateResult`, `FALLBACK_SIGNAL_PACK`.

### Task A2: Eval data schemas
**Files:** Create `src/eval/schemas.py`
**Test:** `tests/test_eval_schemas.py`

Define `EvalBatch`, `PredictionSnapshot`, `RealizedLabel`, `ExperimentRun`, `ModuleLoss`, `AgentContribution`, `OptimizationTicket` dataclasses.

### Task A3: SQLite database + repositories
**Files:** Create `src/eval/database.py`, `src/eval/repositories.py`
**Test:** `tests/test_eval_storage.py`

SQLite schema per 总纲 §15.3 (8 tables + indexes). Repository layer with CRUD for all entities. Must handle Windows paths.

### Task A4: Configuration layer
**Files:** Create `src/eval/config.py`, `config/eval/defaults.json`, `config/eval/strategy_defaults.json`
**Test:** `tests/test_eval_config.py`

All configurable parameters from 总纲 with V1 defaults. Model profiles. Strategy parameters. Must read from env vars with fallbacks.

---

## Subsystem B: Market Simulator (独立，无依赖)

### Task B1: Market simulator
**Files:** Create `src/eval/market_simulator.py`
**Test:** `tests/test_market_simulator.py`

Implements 总纲 §6: Order/ExecutedOrder dataclasses, commission calculation (0.027% buy, 0.127% sell), slippage (0.05% default, 0.02% large cap, 0.1% small cap), volume constraint (5%/10% daily turnover), limit up/down check, suspension handling, T+1 tracking.

---

## Subsystem C: Trading Strategies (依赖B)

### Task C1: Strategy base class
**Files:** Create `src/eval/strategies/__init__.py`, `src/eval/strategies/base.py`
**Test:** `tests/test_strategies.py`

Abstract `BaseStrategy` with `select_stocks()`, `generate_sell_orders()`, `size_positions()`. Order/SellOrder dataclasses.

### Task C2: Short ablation strategy
**Files:** Create `src/eval/strategies/short_ablation.py`

Per 总纲 §5.2: daily clear, top 10 by score, equal weight.

### Task C3: Short long-hold strategy
**Files:** Create `src/eval/strategies/short_longhold.py`

Per 总纲 §5.3: multi-signal selection, tiered stop-loss, position sizing, sector diversification.

### Task C4: Medium-term strategy
**Files:** Create `src/eval/strategies/medium_term.py`

Per 总纲 §5.4: value + trend, weekly rebalance, 8 positions, cross-term signal synergy.

### Task C5: Long-term strategy
**Files:** Create `src/eval/strategies/long_term.py`

Per 总纲 §5.5: deep value, monthly rebalance, 5 positions, buy-more-on-dip.

### Task C6: Strategy factory
**Files:** Create `src/eval/strategies/factory.py`

Registry mapping (term, strategy_type) → strategy class.

---

## Subsystem D: Cache + Adapter (依赖现有项目)

### Task D1: Eval cache layer
**Files:** Create `src/eval/cache.py`

Memory-level dict cache (L1) + async disk persistence (L2). Key format: `{agent}_{code}_{date}_eval`. Cross-line sharing. Startup preload from disk.

### Task D2: Stock pipeline adapter
**Files:** Create `src/eval/adapters/__init__.py`, `src/eval/adapters/stock_pipeline_adapter.py`

Non-invasive wrapper around existing `scoring_engine.py`. Injects `as_of_date`, `eval_mode`, `model_profile_override`. Extracts signal_packs and maps scorer output to DecisionPack.

---

## Subsystem E: CLI + UI Integration

### Task E1: CLI skeleton
**Files:** Create `src/eval/__init__.py`, `src/eval/cli.py`

Commands: `check`, `rebalance`, `settle`, `status`, `pool status/update`, `backtest`, `report`, `trends`, `optimize --analyze/--apply/--rollback`.

### Task E2: Streamlit page (6th function)
**Files:** Create `src/app/pages/06_模拟分析与迭代.py`

Minimal V1 page: operation buttons (调仓/结算/池更新), line status table, refined pool display.

### Task E3: Home page update
**Files:** Modify `src/app/Home.py` (add ~5 lines)

Add 6th function card: "📈 模拟分析与迭代" linking to `pages/06_模拟分析与迭代.py`.

### Task E4: Model config update
**Files:** Modify `src/utils/model_config.py` (add ~15 lines)

Add `EVAL_PROFILES` dict with `eval_analysis`, `eval_orchestrator`, `eval_llm_free` per 总纲 §13.3.

---

## Execution Order

```
Wave 1 (parallel): A1, A2, B1, D1, E1
Wave 2 (parallel): A3, A4, C1
Wave 3 (parallel): C2, C3, C4, C5, C6, D2
Wave 4 (sequential): E2, E3, E4 (need A+C+D done)
```
