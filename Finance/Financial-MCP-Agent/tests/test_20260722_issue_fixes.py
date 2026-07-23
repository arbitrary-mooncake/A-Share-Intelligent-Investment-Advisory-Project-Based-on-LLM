"""2026-07-22 问题检测报告对应的回归测试。"""

import asyncio
import importlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_kline_rows(count=35, descending=False):
    rows = [
        {
            "trade_date": f"2026-01-{index + 1:02d}",
            "open": str(10 + index),
            "high": str(11 + index),
            "low": str(9 + index),
            "close": str(10 + index),
            "vol": str(1000 + index),
            "amount": str(100000 + index),
            "pct_chg": "1.0%",
        }
        for index in range(count)
    ]
    return list(reversed(rows)) if descending else rows


def _to_markdown(rows):
    headers = ["trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in headers) + " |")
    return "\n".join(lines)


def test_industry_identification_prefers_strong_semiconductor_evidence():
    from src.utils.industry_knowledge import identify_industry

    text = "证券相关风险仅一处；公司主营半导体，芯片需求和半导体产能持续改善。"
    assert identify_industry("德明利", text) == "电子"


def test_industry_identification_normalizes_and_deduplicates_keywords():
    from src.utils.industry_knowledge import identify_industry

    # 空格/全角标点不会阻断匹配，同一通用词重复出现也只计一次。
    assert identify_industry("", "半 导 体；证券、证券、证券") == "电子"


def test_industry_identification_does_not_use_dictionary_order_for_ties():
    from src.utils.industry_knowledge import identify_industry

    assert identify_industry("", "芯片与证券") is None
    assert identify_industry("", "没有行业线索") is None


def test_indicator_summary_accepts_json_list_and_sorts_latest_date():
    from src.agents.technical_agent import _build_indicator_summary

    rows = _make_kline_rows(descending=True)
    summary = _build_indicator_summary(json.dumps([None, *rows]), "sz.001309")
    assert "最新日期: 2026-01-35" in summary
    assert "当前收盘价 44.0" in summary


def test_indicator_summary_accepts_markdown_table_from_mcp():
    from src.agents.technical_agent import _build_indicator_summary

    summary = _build_indicator_summary(_to_markdown(_make_kline_rows(descending=True)), "sz.001309")
    assert "最新日期: 2026-01-35" in summary
    assert "DIF =" in summary
    assert "RSI(6) =" in summary


def test_indicator_summary_accepts_nested_fields_items_payload_and_placeholders():
    from src.agents.technical_agent import _build_indicator_summary

    rows = _make_kline_rows()
    fields = ["trade_date", "open", "high", "low", "close", "vol"]
    items = [[row[field] for field in fields] for row in rows]
    payload = {"result": {"fields": fields, "items": items}}
    summary = _build_indicator_summary(json.dumps(payload), "sz.001309")
    assert "最新日期: 2026-01-35" in summary

    rows[0]["close"] = "--"
    assert _build_indicator_summary(json.dumps(rows), "sz.001309")
    rows[1]["close"] = "NaN"
    rows[2]["close"] = "inf"
    rows[3]["close"] = "44.0%"
    assert _build_indicator_summary(json.dumps(rows), "sz.001309")


def test_indicator_summary_safely_degrades_on_invalid_or_short_payload():
    from src.agents.technical_agent import _build_indicator_summary

    assert _build_indicator_summary("not-json-and-not-a-table", "sz.001309") == ""
    assert _build_indicator_summary(json.dumps(_make_kline_rows(2)), "sz.001309") == ""


@pytest.mark.parametrize("count", [0, 20, 29, 30, 35, 59, 60, 61])
def test_indicator_summary_handles_short_and_boundary_history_without_type_error(count):
    from src.agents.technical_agent import _build_indicator_summary

    summary = _build_indicator_summary(json.dumps(_make_kline_rows(count)), "sz.001309")
    if count < 30:
        assert summary == ""
    else:
        assert "预计算技术指标汇总" in summary


class _FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.outputs.pop(0))


def _valid_signal_pack_output():
    return "<SIGNAL_PACK>" + json.dumps({
        "bias": "bullish",
        "signals": [{"factor": "PE", "direction": 1, "strength": 70}],
    }) + "</SIGNAL_PACK>"


def test_value_signal_pack_repair_is_limited_to_one_extra_llm_call():
    from src.agents.value_agent import _extract_or_repair_signal_pack

    llm = _FakeLLM([_valid_signal_pack_output()])
    result = asyncio.run(_extract_or_repair_signal_pack(llm, "无结构化产物", "value", "2026-07-22"))
    assert len(llm.calls) == 1
    assert result["agent_name"] == "value"
    assert result["signals"][0]["factor"] == "PE"


def test_value_signal_pack_repair_failure_keeps_text_fallback():
    from src.agents.value_agent import _extract_or_repair_signal_pack

    llm = _FakeLLM(["仍然不是 JSON"])
    result = asyncio.run(_extract_or_repair_signal_pack(llm, "估值数据不可用，无法判断", "value", "2026-07-22"))
    assert len(llm.calls) == 1
    assert isinstance(result, dict)
    assert result["agent_name"] == "value"
    assert isinstance(result["signals"], list)


def test_value_signal_pack_schema_error_triggers_one_repair():
    from src.agents.value_agent import _extract_or_repair_signal_pack

    malformed = "<SIGNAL_PACK>" + json.dumps({
        "bias": "bullish",
        "signals": "not-a-list",
    }) + "</SIGNAL_PACK>"
    llm = _FakeLLM([_valid_signal_pack_output()])
    result = asyncio.run(
        _extract_or_repair_signal_pack(llm, malformed, "value", "2026-07-22")
    )
    assert len(llm.calls) == 1
    assert isinstance(result["signals"], list)


@pytest.mark.parametrize(
    "module_name, function_name",
    [
        ("src.agents.event_analyst_agent", "event_analyst_agent"),
        ("src.agents.moneyflow_analyst_agent", "moneyflow_analyst_agent"),
        ("src.agents.quality_risk_analyst_agent", "quality_risk_analyst_agent"),
    ],
)
def test_agent_failure_path_writes_execution_complete(monkeypatch, module_name, function_name):
    module = importlib.import_module(module_name)
    calls = []

    class _Recorder:
        def log_agent_start(self, *args, **kwargs):
            calls.append(("start", args, kwargs))

        def log_agent_complete(self, *args, **kwargs):
            calls.append(("complete", args, kwargs))

    monkeypatch.setattr(module, "get_execution_logger", lambda: _Recorder())
    monkeypatch.setattr(module, "read_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "get_model_config_for_agent",
        lambda *args, **kwargs: {"api_key": "", "base_url": "", "model_name": ""},
    )

    state = {
        "messages": [],
        "metadata": {},
        "data": {
            "query": "测试",
            "stock_code": "sz.001309",
            "company_name": "德明利",
            "current_date": "2026-07-22",
        },
    }
    result = asyncio.run(getattr(module, function_name)(state))
    assert result["data"]
    completes = [item for item in calls if item[0] == "complete"]
    assert len(completes) == 1
    assert completes[0][1][3] is False


@pytest.mark.parametrize(
    "module_name, function_name, cached_key",
    [
        ("src.agents.event_analyst_agent", "event_analyst_agent", "event_agent_cached"),
        ("src.agents.moneyflow_analyst_agent", "moneyflow_analyst_agent", "moneyflow_agent_cached"),
        ("src.agents.quality_risk_analyst_agent", "quality_risk_analyst_agent", "quality_risk_agent_cached"),
    ],
)
def test_agent_cache_path_writes_execution_complete(monkeypatch, module_name, function_name, cached_key):
    module = importlib.import_module(module_name)
    cache_utils = importlib.import_module("src.utils.cache_utils")
    calls = []

    class _Recorder:
        def log_agent_start(self, *args, **kwargs):
            calls.append(("start", args, kwargs))

        def log_agent_complete(self, *args, **kwargs):
            calls.append(("complete", args, kwargs))

    monkeypatch.setattr(module, "get_execution_logger", lambda: _Recorder())
    monkeypatch.setattr(module, "read_cache", lambda *args, **kwargs: "cached analysis")
    monkeypatch.setattr(
        cache_utils,
        "read_signal_pack_cache",
        lambda *args, **kwargs: {"bias": "neutral", "signals": []},
    )
    state = {
        "messages": [],
        "metadata": {},
        "data": {
            "stock_code": "sz.001309",
            "company_name": "德明利",
            "current_date": "2026-07-22",
        },
    }
    result = asyncio.run(getattr(module, function_name)(state))
    assert result["metadata"].get(cached_key) is True
    completes = [item for item in calls if item[0] == "complete"]
    assert len(completes) == 1
    assert completes[0][1][1]["cached"] is True
    assert completes[0][1][3] is True


@pytest.mark.parametrize(
    "module_name, function_name",
    [
        ("src.agents.event_analyst_agent", "event_analyst_agent"),
        ("src.agents.moneyflow_analyst_agent", "moneyflow_analyst_agent"),
        ("src.agents.quality_risk_analyst_agent", "quality_risk_analyst_agent"),
    ],
)
def test_agent_cancellation_writes_failed_terminal_state(monkeypatch, module_name, function_name):
    module = importlib.import_module(module_name)
    calls = []

    class _Recorder:
        def log_agent_start(self, *args, **kwargs):
            calls.append(("start", args, kwargs))

        def log_agent_complete(self, *args, **kwargs):
            calls.append(("complete", args, kwargs))

    monkeypatch.setattr(module, "get_execution_logger", lambda: _Recorder())
    monkeypatch.setattr(module, "read_cache", lambda *args, **kwargs: None)

    def _cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(module, "get_model_config_for_agent", _cancelled)
    state = {
        "messages": [],
        "metadata": {},
        "data": {
            "stock_code": "sz.001309",
            "company_name": "德明利",
            "current_date": "2026-07-22",
        },
    }
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(getattr(module, function_name)(state))
    completes = [item for item in calls if item[0] == "complete"]
    assert len(completes) == 1
    assert completes[0][1][3] is False
    assert completes[0][1][4] == "cancelled"


def test_data_gateway_logs_miss_categories_without_request_parameters(monkeypatch):
    import src.data.data_gateway as gateway
    import src.tools.mcp_client as mcp_client

    class _Tool:
        def __init__(self, name, result=None, error=None):
            self.name = name
            self.result = result
            self.error = error

        async def ainvoke(self, kwargs):
            if self.error:
                raise self.error
            return self.result

    tools = [
        _Tool("available", "x" * 25),
        _Tool("empty", ""),
        _Tool("failed", error=RuntimeError("boom")),
    ]
    monkeypatch.setattr(
        gateway,
        "build_term_spec",
        lambda *args, **kwargs: [
            ("available", {"code": "SECRET"}),
            ("missing", {"code": "SECRET"}),
            ("empty", {"code": "SECRET"}),
            ("failed", {"code": "SECRET"}),
        ],
    )
    async def _get_tools(**kwargs):
        return tools
    monkeypatch.setattr(mcp_client, "get_mcp_tools", _get_tools)
    async def _no_cache(*args, **kwargs):
        return None
    async def _set_cache(*args, **kwargs):
        return None
    monkeypatch.setattr(gateway, "get_cached_tool_result", _no_cache)
    monkeypatch.setattr(gateway, "set_cached_tool_result", _set_cache)
    logs = []
    monkeypatch.setattr(gateway.logger, "info", lambda message: logs.append(str(message)))

    result = asyncio.run(gateway.prefetch_term_bundle("short", "SECRET"))
    assert result
    miss_logs = [message for message in logs if "预取未命中" in message]
    assert len(miss_logs) == 1
    assert "tool_unavailable=['missing']" in miss_logs[0]
    assert "empty_or_short=['empty']" in miss_logs[0]
    assert "call_failed=['failed']" in miss_logs[0]
    assert "SECRET" not in miss_logs[0]


def test_data_gateway_logs_global_tool_discovery_failure(monkeypatch):
    import src.data.data_gateway as gateway
    import src.tools.mcp_client as mcp_client

    monkeypatch.setattr(
        gateway,
        "build_term_spec",
        lambda *args, **kwargs: [("missing", {"code": "SECRET"})],
    )

    async def _get_tools(**kwargs):
        raise RuntimeError("MCP unavailable")

    monkeypatch.setattr(mcp_client, "get_mcp_tools", _get_tools)
    logs = []
    monkeypatch.setattr(gateway.logger, "info", lambda message: logs.append(str(message)))

    async def _run_in_one_context():
        gateway._run_bundle.set({"stale": "old snapshot"})
        result = await gateway.prefetch_term_bundle("short", "SECRET")
        return result, gateway.get_run_bundle_stats()

    result, bundle_stats = asyncio.run(_run_in_one_context())
    assert result is None
    assert any("call_failed=['missing']" in message for message in logs)
    assert all("SECRET" not in message for message in logs)
    assert bundle_stats == {"bundle_size": 0}


def test_term_workflow_observation_is_added_without_changing_graph(monkeypatch):
    import src.stock_pool.scoring_engine as scoring_engine_module
    from src.stock_pool.scoring_engine import ScoringEngine

    logs = []
    monkeypatch.setattr(scoring_engine_module.logger, "info", lambda message: logs.append(str(message)))
    engine = ScoringEngine(pool_manager=False)
    app = engine._build_term_workflow("short")
    nodes = set(app.get_graph().nodes.keys())
    assert "technical_analyst" in nodes
    assert "short_term_scorer" in nodes
    assert any("期限子图构建完成 term=short" in message for message in logs)
