"""Tests for contribution engine"""
from src.eval.contribution_engine import ContributionEngine, permutation_test
from src.eval.loss_engine import LossEngine


def test_contribution_engine_basic():
    engine = ContributionEngine()

    all_results = {
        "scores": [80, 70, 60, 50, 40, 30, 20, 10],
        "returns": [0.10, 0.08, 0.04, 0.0, -0.02, -0.04, -0.08, -0.10],
    }

    ablation_results = {
        "fundamental": {
            "scores": [75, 65, 55, 45, 35, 25, 15, 5],
        },
        "technical": {
            "scores": [78, 68, 58, 48, 38, 28, 18, 8],
        },
    }

    result = engine.compute_contributions("medium", all_results, ablation_results)

    assert result["term"] == "medium"
    assert len(result["contributions"]) == 2
    assert "delta_L_total" in result["contributions"][0]
    assert "ci_95_lower" in result["contributions"][0]
    assert "stars" in result["contributions"][0]


def test_contribution_scores():
    engine = ContributionEngine()
    contributions = [
        {"agent_name": "fund", "delta_L_total": 0.05},
        {"agent_name": "tech", "delta_L_total": -0.03},
    ]
    result = engine.compute_contribution_scores(contributions)
    assert result[0]["label"] == "正贡献"
    assert result[1]["label"] == "负贡献/拖后腿"


def test_permutation_test():
    all_scores = [80, 70, 60, 50, 40, 30, 20, 10]
    ablated_scores = [75, 65, 55, 45, 35, 25, 15, 5]
    returns = [0.10, 0.08, 0.04, 0.0, -0.02, -0.04, -0.08, -0.10]

    result = permutation_test("medium", all_scores, ablated_scores, returns, n_permutations=500)
    assert "p_value" in result
    assert "observed_delta" in result
