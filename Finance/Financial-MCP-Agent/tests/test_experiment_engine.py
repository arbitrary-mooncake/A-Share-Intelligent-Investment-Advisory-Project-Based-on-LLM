"""
Tests for ExperimentEngine: ablation, gate, consistency, and fidelity experiments.

Covers:
  - Ablation experiment with mock data
  - Gate on/off experiment with mock scores
  - Consistency test with dual-run data
  - Fidelity test delegation
  - Edge cases: empty input, single stock, missing agents
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.eval.experiment_engine import ExperimentEngine


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_ablation_results(n_stocks=5, full_returns=None, noisy=True):
    """Build synthetic ablation results for testing.

    Args:
        n_stocks: number of stocks in pool
        full_returns: optional override for full-system returns
        noisy: if True, ablation lines have slightly worse returns than full
    """
    base = full_returns or [0.05 + i * 0.01 for i in range(n_stocks)]
    full_scores = [75 + i * 2 for i in range(n_stocks)]

    results = {
        "full": {"scores": full_scores, "returns": list(base)},
    }
    agents = ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"]
    for agent in agents:
        if noisy:
            # Ablation = slightly worse
            agent_returns = [r - 0.01 - (hash(agent) % 10) * 0.002 for r in base]
        else:
            agent_returns = list(base)
        agent_scores = [s - 1 for s in full_scores]  # slightly lower scores
        results[f"-{agent}"] = {"scores": agent_scores, "returns": agent_returns}

    return results


# ── Ablation Tests ──────────────────────────────────────────────────────────


def test_run_ablation_experiment_basic():
    """消融实验基本流程：全agent + 逐个移除"""
    engine = ExperimentEngine()
    results = _make_ablation_results(5)
    pool = [f"sh.6000{i:02d}" for i in range(5)]

    report = engine.run_ablation_experiment("medium", pool, results)

    assert report["experiment_type"] == "ablation"
    assert report["term"] == "medium"
    assert report["pool_size"] == 5
    assert "contributions" in report
    assert "run_at" in report


def test_run_ablation_experiment_single_stock():
    """单只股票消融实验（最小池子）"""
    engine = ExperimentEngine()
    results = _make_ablation_results(1)
    pool = ["sh.600000"]

    report = engine.run_ablation_experiment("short", pool, results)
    assert report["experiment_type"] == "ablation"
    assert report["pool_size"] == 1


def test_run_ablation_experiment_empty():
    """空池消融实验默认状态（无crash）"""
    engine = ExperimentEngine()
    results = {"full": {"scores": [], "returns": []}}
    pool = []

    report = engine.run_ablation_experiment("long", pool, results)
    assert report["term"] == "long"
    assert report["pool_size"] == 0


def test_run_ablation_experiment_missing_agents():
    """部分agent缺失时仍能执行（全量baseline不全）"""
    engine = ExperimentEngine()
    results = {
        "full": {"scores": [70, 75, 80], "returns": [0.03, 0.05, 0.08]},
        "-fundamental": {"scores": [69, 74, 79], "returns": [0.02, 0.04, 0.07]},
        "-technical": {"scores": [70, 74, 79], "returns": [0.02, 0.03, 0.06]},
    }
    pool = ["sh.600000", "sh.600001", "sh.600002"]

    report = engine.run_ablation_experiment("short", pool, results)
    assert report["experiment_type"] == "ablation"


# ── Gate Experiment Tests ────────────────────────────────────────────────────


def test_run_gate_experiment_basic():
    """Risk gate开关对比实验：未截断 vs 截断"""
    engine = ExperimentEngine()
    scores_gate_on = [60, 55, 50, 65, 70, 75, 55, 60]    # gate applied
    scores_gate_off = [80, 55, 85, 65, 90, 75, 55, 80]   # raw scores
    returns = [0.10, -0.05, 0.15, -0.02, 0.08, 0.12, -0.15, 0.05]

    report = engine.run_gate_experiment("medium", scores_gate_on, scores_gate_off, returns)

    assert report["experiment_type"] == "gate_experiment"
    assert "L_gate_on" in report
    assert "L_gate_off" in report
    assert "delta_L" in report
    assert "false_kill_rate" in report
    assert "miss_rate" in report
    assert report["sample_size"] == 8


def test_run_gate_experiment_no_false_kills():
    """当gate不产生false kill时，false_kill_rate=0"""
    engine = ExperimentEngine()
    scores_gate_on = [70, 75, 80]
    scores_gate_off = [70, 75, 80]  # identical = no kills
    returns = [0.05, 0.08, 0.12]

    report = engine.run_gate_experiment("short", scores_gate_on, scores_gate_off, returns)
    assert report["false_kill_rate"] == 0.0


def test_run_gate_experiment_all_false_kills():
    """全部被gate截断，但实际收益为正"""
    engine = ExperimentEngine()
    scores_gate_on = [40, 40, 40]   # all capped
    scores_gate_off = [80, 80, 80]  # all high original
    returns = [0.10, 0.10, 0.10]

    report = engine.run_gate_experiment("medium", scores_gate_on, scores_gate_off, returns)
    assert report["false_kill_rate"] == 1.0


def test_run_gate_experiment_empty():
    """空输入返回错误信息"""
    engine = ExperimentEngine()
    report = engine.run_gate_experiment("short", [], [], [])
    assert "error" in report


def test_run_gate_experiment_miss_rate():
    """漏放率：未被gate截断但实际收益<-10%"""
    engine = ExperimentEngine()
    # stock at index 0: gate_on=50 (not capped), return=-0.15 → missed
    # stock at index 2: gate_on=55, return=-0.20 → missed
    scores_gate_on = [50, 40, 55, 70]
    scores_gate_off = [50, 80, 55, 70]
    returns = [-0.15, 0.05, -0.20, 0.10]

    report = engine.run_gate_experiment("short", scores_gate_on, scores_gate_off, returns)
    assert report["miss_rate"] == 0.5  # 2 out of 4


# ── Consistency Test ─────────────────────────────────────────────────────────


def test_run_consistency_test_identical():
    """两次运行输出相同时，diff=0, overlap=1"""
    engine = ExperimentEngine()
    run1 = [70, 75, 80, 85, 90, 65, 60, 55, 50, 45]
    run2 = [70, 75, 80, 85, 90, 65, 60, 55, 50, 45]

    report = engine.run_consistency_test("medium", run1, run2)
    assert report["mean_score_diff"] == 0.0
    assert report["action_flip_rate"] == 0.0
    assert report["top_k_overlap"] == 1.0


def test_run_consistency_test_all_different():
    """两次运行输出完全不同"""
    engine = ExperimentEngine()
    # 使用 20 个元素确保 top-10 集合不同且行动翻转 (n > k, 跨50阈值)
    run1 = list(range(90, 40, -5)) + list(range(40, -10, -5))  # 20 elements: 90..45 + 40..-5
    run2 = list(range(-5, 95, 5))  # 20 elements: -5..90

    report = engine.run_consistency_test("short", run1, run2)
    assert report["mean_score_diff"] > 0
    assert report["top_k_overlap"] == 0.0  # 完全反向，top-10无重叠
    assert report["action_flip_rate"] > 0


def test_run_consistency_test_empty():
    """空输入返回错误"""
    engine = ExperimentEngine()
    report = engine.run_consistency_test("short", [], [])
    assert "error" in report


def test_run_consistency_test_mismatched_lengths():
    """不等长输入取min长度"""
    engine = ExperimentEngine()
    run1 = [70, 75, 80, 85, 90]
    run2 = [70, 75, 80]

    report = engine.run_consistency_test("medium", run1, run2)
    assert report["sample_size"] == 3


# ── Fidelity Test ────────────────────────────────────────────────────────────


def test_run_fidelity_test_basic():
    """保真度测试：快模型 vs 生产模型（本质一致）"""
    engine = ExperimentEngine()
    scores_fast = [70, 75, 80, 85, 90, 65, 60, 55, 50, 45]
    scores_prod = [72, 73, 82, 83, 88, 67, 62, 54, 52, 47]

    report = engine.run_fidelity_test("medium", scores_fast, scores_prod)
    assert "mean_score_diff" in report
    assert "action_flip_rate" in report
    assert report["sample_size"] == 10


def test_run_fidelity_test_perfect_fidelity():
    """完美保真度：两模型输出一致"""
    engine = ExperimentEngine()
    scores_fast = [70, 75, 80, 85, 90, 65, 60, 55, 50, 45]
    scores_prod = [70, 75, 80, 85, 90, 65, 60, 55, 50, 45]

    report = engine.run_fidelity_test("long", scores_fast, scores_prod)
    assert report["mean_score_diff"] == 0.0


# ── Config Injection ─────────────────────────────────────────────────────────


def test_experiment_engine_with_config():
    """自定义config可传递到内部引擎"""
    cfg = {"bootstrap_iterations": 100, "significance_level": 0.90}
    engine = ExperimentEngine(cfg)
    assert engine.config == cfg


def test_experiment_engine_default_config():
    """无config时使用空dict"""
    engine = ExperimentEngine()
    assert engine.config == {}
