"""Tests for report builder, memory manager, chart service"""
from src.eval.report_builder import ReportBuilder
from src.eval.memory_manager import MemoryManager
from src.eval.chart_service import ChartService


def test_report_builder_basic():
    rb = ReportBuilder()
    status = {
        "lines": [
            {"line_id": "S-L0", "term": "short", "cumulative_return_pct": 5.2, "max_drawdown_pct": 3.1, "holdings_count": 5},
            {"line_id": "S-L1", "term": "short", "cumulative_return_pct": 3.8, "max_drawdown_pct": 4.2, "holdings_count": 4},
        ],
        "pools": {"short": {"size": 100, "target_size": 100}},
    }
    data = rb.build_batch_report_data("test_batch", status)
    assert data["batch_id"] == "test_batch"
    assert "term_summaries" in data
    assert data["term_summaries"]["short"]["line_count"] == 2


def test_report_builder_markdown():
    rb = ReportBuilder()
    data = {
        "batch_id": "test_001",
        "generated_at": "2026-06-19T00:00:00",
        "executive_summary": {"total_lines": 14, "avg_return_all_lines": 3.5},
        "term_summaries": {},
        "line_details": [],
        "pool_summaries": {},
        "loss_analysis": {},
        "agent_contributions": {},
        "declarations": ["测试声明"],
    }
    md = rb.build_markdown_report(data)
    assert "test_001" in md
    assert "执行摘要" in md
    assert "风险声明" in md


def test_memory_manager_record():
    mm = MemoryManager()
    mm.record_batch("test_batch", {
        "executive_summary": {"avg_return_all_lines": 3.5},
        "term_summaries": {"short": {"avg_cumulative_return_pct": 5.0}},
        "loss_analysis": {"L_total": 0.25, "score_total": 75.0},
    })
    summary = mm.get_summary()
    assert summary["total_batches"] >= 1


def test_memory_manager_trends():
    mm = MemoryManager()
    score_trend = mm.get_score_trend(365)
    assert isinstance(score_trend, list)
    loss_trend = mm.get_loss_trend(365)
    assert isinstance(loss_trend, list)


def test_chart_service_score():
    cs = ChartService()
    data = cs.get_score_trend_data(90)
    assert "data" in data
    assert data["type"] == "line"


def test_chart_service_agent_bar():
    cs = ChartService()
    contributions = [
        {"agent_name": "fundamental", "delta_L_total": 0.042, "ci_95_lower": 0.018, "ci_95_upper": 0.086, "stars": "★★★"},
        {"agent_name": "technical", "delta_L_total": 0.028, "ci_95_lower": 0.005, "ci_95_upper": 0.051, "stars": "★★"},
    ]
    data = cs.get_agent_contribution_bar_data(contributions)
    assert len(data["data"]) == 2
    assert data["type"] == "bar"


def test_chart_service_line_comparison():
    cs = ChartService()
    lines = [
        {"line_id": "S-L0", "cumulative_return_pct": 5.2, "max_drawdown_pct": 3.1, "term": "short"},
        {"line_id": "M-L0", "cumulative_return_pct": 8.1, "max_drawdown_pct": 5.3, "term": "medium"},
    ]
    data = cs.get_line_comparison_data(lines)
    assert len(data["data"]) == 2
