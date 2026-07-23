"""期限子图测试（4.2）：短期打分不得包含财务三件套和另两个 scorer。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stock_pool.scoring_engine import ScoringEngine


@pytest.fixture(scope="module")
def engine():
    return ScoringEngine(pool_manager=False)


def _node_ids(app) -> set:
    return set(app.get_graph().nodes.keys())


def test_short_graph_excludes_financial_agents_and_other_scorers(engine):
    nodes = _node_ids(engine._build_term_workflow("short"))
    for expected in ("technical_analyst", "news_analyst", "event_analyst",
                     "moneyflow_analyst", "short_term_scorer"):
        assert expected in nodes, f"short 子图缺少节点: {expected}"
    for excluded in ("fundamental_analyst", "value_analyst", "quality_risk_analyst",
                     "medium_term_scorer", "long_term_scorer"):
        assert excluded not in nodes, f"short 子图不应包含节点: {excluded}"


def test_medium_graph_has_all_agents_but_only_medium_scorer(engine):
    nodes = _node_ids(engine._build_term_workflow("medium"))
    for expected in ("fundamental_analyst", "technical_analyst", "value_analyst",
                     "news_analyst", "event_analyst", "quality_risk_analyst",
                     "moneyflow_analyst", "medium_term_scorer"):
        assert expected in nodes, f"medium 子图缺少节点: {expected}"
    assert "short_term_scorer" not in nodes
    assert "long_term_scorer" not in nodes


def test_long_graph_has_all_agents_but_only_long_scorer(engine):
    nodes = _node_ids(engine._build_term_workflow("long"))
    assert "long_term_scorer" in nodes
    assert "short_term_scorer" not in nodes
    assert "medium_term_scorer" not in nodes


def test_invalid_term_raises(engine):
    with pytest.raises(ValueError):
        engine._build_term_workflow("weekly")


def test_term_workflows_cached(engine):
    assert engine._build_term_workflow("short") is engine._build_term_workflow("short")
