"""Regression tests for P0 score-failure boundaries."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest


def _valid_signal_packs():
    return {
        name: {"agent_name": name, "validity": "valid"}
        for name in (
            "fundamental", "technical", "value", "news", "event",
            "quality_risk", "moneyflow",
        )
    }


def _detailed_signal_packs():
    return {
        name: {
            "agent_name": name,
            "validity": "valid",
            "bias": "neutral",
            "confidence": 0.7,
            "data_quality_score": 1.0,
            "signals": [],
            "risk_flags": [],
            "missing_data": [],
        }
        for name in (
            "fundamental", "technical", "value", "news", "event",
            "quality_risk", "moneyflow",
        )
    }


def _valid_batch_stock():
    return {
        "code": "sh.600000",
        "name": "浦发银行",
        "status": "fetched",
        "data": {
            "code": "sh.600000",
            "name": "浦发银行",
            "last_price": "10.25",
            "price_changes": {"1d": "0.5%"},
            "pe": "6.2",
            "pb": "0.6",
        },
    }


def test_empty_score_does_not_create_hold_decision():
    from src.eval.adapters.stock_pipeline_adapter import _build_decision_pack

    decision = _build_decision_pack(
        "sh.600000", "浦发银行", "medium", "2026-07-19", {}, True
    )

    assert decision is None


def test_runtime_score_contract_preserves_failure_as_invalid():
    from src.eval.adapters.stock_pipeline_adapter import _attach_runtime_score_contract

    result = _attach_runtime_score_contract(
        {"score": 50.0, "_scorer_failed": True},
        _valid_signal_packs(),
        "medium",
    )

    assert result["validity"] == "invalid"
    assert result["error_type"] == "scorer_failed"
    assert "score" not in result


def test_runtime_score_contract_emits_coverage_and_validity():
    from src.eval.adapters.stock_pipeline_adapter import _attach_runtime_score_contract

    result = _attach_runtime_score_contract(
        {"score": 72, "rating": "推荐"}, _valid_signal_packs(), "medium"
    )

    assert result["validity"] == "valid"
    assert result["coverage"] == 1.0
    assert result["missing_core_fields"] == []


def test_runtime_score_contract_rejects_score_without_any_evidence():
    from src.eval.adapters.stock_pipeline_adapter import _attach_runtime_score_contract

    result = _attach_runtime_score_contract(
        {"score": 72, "rating": "推荐"}, {}, "medium"
    )

    assert result["validity"] == "invalid"
    assert result["coverage"] == 0.0
    assert result["missing_core_fields"] == ["analysis_evidence"]
    assert "score" not in result


def test_runtime_score_contract_preserves_explicit_invalid():
    from src.eval.adapters.stock_pipeline_adapter import _attach_runtime_score_contract

    result = _attach_runtime_score_contract(
        {
            "score": 50,
            "validity": "invalid",
            "error_type": "provider_failure",
            "error_message": "upstream unavailable",
        },
        _valid_signal_packs(),
        "medium",
    )

    assert result["validity"] == "invalid"
    assert result["error_type"] == "provider_failure"
    assert "score" not in result


def test_new_production_score_cache_has_explicit_contract():
    from src.agents.scoring_nodes import _score_cache_contract

    package = SimpleNamespace(
        available_agents=["technical", "news", "event", "moneyflow"],
        missing_agents=[],
    )
    gate = SimpleNamespace(abstain=False)

    contract = _score_cache_contract(package, gate, 4)

    assert contract == {
        "validity": "valid",
        "coverage": 1.0,
        "missing_core_fields": [],
        "missing_optional_fields": [],
    }


def test_pool_score_boundary_rejects_legacy_and_stale_scores():
    from src.eval.orchestrator import EvalOrchestrator

    current = "2026-07-19"
    assert EvalOrchestrator._validated_pool_score(
        {"code": "sh.600000", "final_score": 70, "scored_at": current}, current
    ) is None

    stale = (datetime.fromisoformat(current) - timedelta(days=8)).isoformat()
    assert EvalOrchestrator._validated_pool_score(
        {
            "code": "sh.600000",
            "final_score": 70,
            "validity": "valid",
            "coverage": 1.0,
            "missing_core_fields": [],
            "missing_optional_fields": [],
            "scored_at": stale,
        },
        current,
    ) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error_code"),
    [
        ("", "empty_llm_response"),
        ("[]", "missing_llm_item"),
    ],
)
async def test_batch_scoring_marks_empty_or_missing_response_invalid(
    monkeypatch, response, error_code
):
    from src.api import batch_scorer
    from src.utils import llm_clients

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_completion(self, messages, max_retries=1):
            return response

    monkeypatch.setattr(llm_clients, "OpenAICompatibleClient", FakeClient)
    stocks = [_valid_batch_stock()]

    result = await batch_scorer.score_batch(stocks, semaphore=1)

    assert result[0]["score"]["validity"] == "invalid"
    assert result[0]["score"]["error_code"] == error_code
    assert result[0]["score"]["level"] is None


@pytest.mark.asyncio
async def test_batch_scoring_marks_timeout_invalid(monkeypatch):
    from src.api import batch_scorer
    from src.utils import llm_clients

    class FakeClient:
        def __init__(self, **kwargs):
            pass

    async def raise_timeout(awaitable, timeout):
        if hasattr(awaitable, "cancel"):
            awaitable.cancel()
        elif hasattr(awaitable, "close"):
            awaitable.close()
        raise batch_scorer.asyncio.TimeoutError

    monkeypatch.setattr(llm_clients, "OpenAICompatibleClient", FakeClient)
    monkeypatch.setattr(batch_scorer.asyncio, "wait_for", raise_timeout)
    stocks = [_valid_batch_stock()]

    result = await batch_scorer.score_batch(stocks, semaphore=1)

    assert result[0]["score"]["validity"] == "invalid"
    assert result[0]["score"]["error_code"] == "llm_timeout"


@pytest.mark.asyncio
async def test_batch_scoring_marks_client_exception_invalid(monkeypatch):
    from src.api import batch_scorer
    from src.utils import llm_clients

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_completion(self, messages, max_retries=1):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(llm_clients, "OpenAICompatibleClient", FakeClient)
    stocks = [_valid_batch_stock()]

    result = await batch_scorer.score_batch(stocks, semaphore=1)

    assert result[0]["score"]["validity"] == "invalid"
    assert result[0]["score"]["error_code"] == "llm_provider_error"


@pytest.mark.asyncio
async def test_batch_scoring_programming_error_is_not_flattened(monkeypatch):
    from src.api import batch_scorer
    from src.utils import llm_clients

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_completion(self, messages, max_retries=1):
            raise RuntimeError("unexpected local invariant failure")

    monkeypatch.setattr(llm_clients, "OpenAICompatibleClient", FakeClient)
    with pytest.raises(RuntimeError, match="invariant"):
        await batch_scorer.score_batch([_valid_batch_stock()], semaphore=1)


@pytest.mark.asyncio
async def test_legacy_scorer_cache_is_bypassed_recomputed_and_refreshed(monkeypatch):
    from src.eval import cache as eval_cache
    from src.eval.orchestrator import EvalOrchestrator
    from src.agents import short_term_scorer as short_module
    from src.agents import medium_term_scorer as medium_module
    from src.agents import long_term_scorer as long_module
    from src.utils import risk_gate as risk_gate_module

    scorer_names = {
        "short_term_scorer", "medium_term_scorer", "long_term_scorer"
    }
    writes = {}
    calls = []

    def fake_read_cache(name, code, as_of_date):
        if name in scorer_names:
            return '{"score": 50}'  # old cache without explicit validity
        return "cached analysis text"

    def fake_write_cache(name, code, as_of_date, content):
        writes[name] = content

    async def short_scorer(**kwargs):
        calls.append("short")
        return {"score": 71, "rating": "推荐"}

    async def medium_scorer(**kwargs):
        calls.append("medium")
        return {"score": 72, "rating": "推荐"}

    async def long_scorer(**kwargs):
        calls.append("long")
        return {"score": 73, "rating": "推荐"}

    gate = SimpleNamespace(
        score_cap=None,
        risk_level="low",
        risk_flags_found=[],
        abstain=False,
        data_quality_score=1.0,
    )
    monkeypatch.setattr(eval_cache, "read_cache", fake_read_cache)
    monkeypatch.setattr(eval_cache, "write_cache", fake_write_cache)
    monkeypatch.setattr(short_module, "short_term_scorer", short_scorer)
    monkeypatch.setattr(medium_module, "medium_term_scorer", medium_scorer)
    monkeypatch.setattr(long_module, "long_term_scorer", long_scorer)
    monkeypatch.setattr(risk_gate_module, "apply_risk_gate", lambda *args: gate)

    orchestrator = object.__new__(EvalOrchestrator)
    result = await orchestrator._assemble_from_agent_caches(
        "sh.600000", "2026-07-19", _detailed_signal_packs(), "medium"
    )

    assert calls == ["short", "medium", "long"]
    assert result["_scorer_cache_hit"] is False
    assert result["medium_term_score"]["score"] == 72
    assert scorer_names <= set(writes)
    assert all('"validity": "valid"' in writes[name] for name in scorer_names)


@pytest.mark.asyncio
async def test_cached_scorer_schema_drift_is_raised(monkeypatch):
    from src.eval import cache as eval_cache
    from src.eval.orchestrator import EvalOrchestrator
    from src.eval.score_assessment import ScoreAssessmentSchemaError

    monkeypatch.setattr(
        eval_cache,
        "read_cache",
        lambda *args: '{"validity":"valid","score":"50","coverage":1}',
    )
    orchestrator = object.__new__(EvalOrchestrator)
    with pytest.raises(ScoreAssessmentSchemaError):
        await orchestrator._assemble_from_agent_caches(
            "sh.600000", "2026-07-19", _detailed_signal_packs(), "medium"
        )


@pytest.mark.asyncio
async def test_batch_scoring_rejects_fetched_shell_before_llm(monkeypatch):
    from src.api import batch_scorer
    from src.utils import llm_clients

    class MustNotRunClient:
        def __init__(self, **kwargs):
            raise AssertionError("LLM must not run without core evidence")

    monkeypatch.setattr(llm_clients, "OpenAICompatibleClient", MustNotRunClient)
    stocks = [{
        "code": "sh.600000",
        "name": "浦发银行",
        "status": "fetched",
        "data": {"code": "sh.600000", "name": "浦发银行"},
    }]

    result = await batch_scorer.score_batch(stocks, semaphore=1)

    assert result[0]["status"] == "data_invalid"
    assert result[0]["score"]["validity"] == "invalid"
    assert result[0]["score"]["error_code"] == "missing_core_data"
    assert set(result[0]["score"]["missing_core_fields"]) == {
        "market_data", "financial_data"
    }


def test_analysis_builder_does_not_count_agent_failure_text_as_available():
    from src.utils.analysis_package_builder import build_analysis_package

    package = build_analysis_package(
        {"fundamental_analysis": "基本面分析失败：上游数据获取失败"},
        "2026-07-19",
    )

    assert "fundamental" in package.missing_agents
    assert "fundamental" not in package.available_agents
    assert "Agent执行失败" in package.global_missing_data
