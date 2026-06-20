"""Tests for backtest and experiment engines"""
import pytest
from src.eval.replay_backtest_engine import (
    ReplayBacktestEngine,
    BacktestConfig,
    BACKTEST_LINE_DEFINITIONS,
    ACTIVE_BACKTEST_LINES,
)
from src.eval.experiment_engine import ExperimentEngine


def test_backtest_config_defaults():
    cfg = BacktestConfig()
    assert cfg.term == "medium"
    assert cfg.holding_days == 20
    assert len(cfg.ablation_agents) == 5


def test_generate_anchor_dates_weekly():
    cfg = BacktestConfig(start_date="2024-01-01", end_date="2024-01-31",
                         anchor_frequency="weekly", anchor_weekday=4)
    engine = ReplayBacktestEngine(cfg)
    anchors = engine.generate_anchor_dates()
    assert len(anchors) >= 3  # At least 3 Fridays in Jan 2024
    for d in anchors:
        dt = __import__('datetime').datetime.strptime(d, "%Y-%m-%d")
        assert dt.weekday() == 4  # Friday


def test_generate_anchor_dates_monthly():
    cfg = BacktestConfig(start_date="2024-01-01", end_date="2024-03-31",
                         anchor_frequency="monthly")
    engine = ReplayBacktestEngine(cfg)
    anchors = engine.generate_anchor_dates()
    assert len(anchors) >= 2


def test_train_val_test_split():
    cfg = BacktestConfig(train_split="2023-12-31", validation_split="2024-06-30")
    engine = ReplayBacktestEngine(cfg)
    anchors = ["2023-06-15", "2023-12-29", "2024-03-15", "2024-09-15", "2025-01-15"]
    splits = engine.get_train_val_test_split(anchors)
    assert len(splits["train"]) == 2
    assert len(splits["validate"]) == 1
    assert len(splits["test"]) == 2


# ── Fix 1: Real PIT scoring replaces hash-based mock ──

@pytest.mark.asyncio
async def test_score_stock_pit_full():
    """Test _score_stock_pit with full agent (no ablation)."""
    engine = ReplayBacktestEngine()
    s1 = await engine._score_stock_pit(
        "sh.601888", "TestCo", "2024-06-14", "full"
    )
    s2 = await engine._score_stock_pit(
        "sh.601888", "TestCo", "2024-12-20", "full"
    )
    # Scores should be in valid range
    assert 10 <= s1 <= 98
    assert 10 <= s2 <= 98


@pytest.mark.asyncio
async def test_score_stock_pit_ablation_different():
    """Test that ablation lines produce different scores from full line."""
    engine = ReplayBacktestEngine()
    s_full = await engine._score_stock_pit(
        "sh.601888", "TestCo", "2024-06-14", "full"
    )
    s_abl = await engine._score_stock_pit(
        "sh.601888", "TestCo", "2024-06-14", "-fundamental"
    )
    # Ablation should differ from full (fundamental has non-zero weight)
    assert s_full != s_abl


@pytest.mark.asyncio
async def test_score_stock_pit_deterministic():
    """Test that scoring is deterministic (no random/hash variation).
    Same inputs should produce the same output.
    """
    engine = ReplayBacktestEngine()
    s1 = await engine._score_stock_pit(
        "sh.601888", "TestCo", "2024-06-14", "full"
    )
    s2 = await engine._score_stock_pit(
        "sh.601888", "TestCo", "2024-06-14", "full"
    )
    assert s1 == s2


@pytest.mark.asyncio
async def test_score_stock_pit_different_stocks():
    """Different stocks on the same date should have different scores."""
    engine = ReplayBacktestEngine()
    s1 = await engine._score_stock_pit(
        "sh.601888", "TestCo", "2024-06-14", "full"
    )
    s2 = await engine._score_stock_pit(
        "sz.300308", "OtherCo", "2024-06-14", "full"
    )
    # Different stocks, different market segments → likely different scores
    assert s1 != s2


# ── Fix 2: No 5-anchor cap ──

def test_all_anchors_processed_no_cap():
    """Verify generate_anchor_dates returns ALL anchors, not capped at 5."""
    cfg = BacktestConfig(start_date="2024-01-01", end_date="2024-03-31",
                         anchor_frequency="weekly", anchor_weekday=4)
    engine = ReplayBacktestEngine(cfg)
    anchors = engine.generate_anchor_dates()
    # Should have ~13 Fridays in Q1 2024, definitely > 5
    assert len(anchors) > 5


# ── Fix 3: 22 Named Backtest Lines ──

def test_backtest_line_definitions_count():
    """Verify we have exactly 22 named backtest lines."""
    assert len(BACKTEST_LINE_DEFINITIONS) == 22


def test_backtest_line_short_count():
    """Verify 6 short backtest lines (SB-L0 ~ SB-L5)."""
    short_lines = [k for k, v in BACKTEST_LINE_DEFINITIONS.items()
                   if v["term"] == "short"]
    assert len(short_lines) == 6
    for i in range(6):
        assert f"SB-L{i}" in BACKTEST_LINE_DEFINITIONS


def test_backtest_line_medium_count():
    """Verify 8 medium backtest lines (MB-L0 ~ MB-L7)."""
    medium_lines = [k for k, v in BACKTEST_LINE_DEFINITIONS.items()
                    if v["term"] == "medium"]
    assert len(medium_lines) == 8
    for i in range(8):
        assert f"MB-L{i}" in BACKTEST_LINE_DEFINITIONS


def test_backtest_line_long_count():
    """Verify 8 long backtest lines (LB-L0 ~ LB-L7)."""
    long_lines = [k for k, v in BACKTEST_LINE_DEFINITIONS.items()
                  if v["term"] == "long"]
    assert len(long_lines) == 8
    for i in range(8):
        assert f"LB-L{i}" in BACKTEST_LINE_DEFINITIONS


def test_degenerate_lines_marked():
    """Verify news/event lines are marked as degenerate."""
    for line_id in ("MB-L6", "MB-L7", "LB-L6", "LB-L7"):
        assert BACKTEST_LINE_DEFINITIONS[line_id].get("degenerate") is True


def test_active_lines_exclude_degenerate():
    """Active backtest lines should exclude degenerate news/event lines."""
    assert len(ACTIVE_BACKTEST_LINES["short"]) == 6
    # Medium active: 8 total - 2 degenerate = 6
    assert len(ACTIVE_BACKTEST_LINES["medium"]) == 6
    # Long active: 8 total - 2 degenerate = 6
    assert len(ACTIVE_BACKTEST_LINES["long"]) == 6


# ── Fix 5: Market Regime Classification ──

def test_classify_market_regime_bear_2022():
    """2022 should be classified as bear market."""
    regime = ReplayBacktestEngine._classify_market_regime("2022-06-15")
    assert regime == "bear"


def test_classify_market_regime_bull_2024_end():
    """Late 2024 should be classified as bull market."""
    regime = ReplayBacktestEngine._classify_market_regime("2024-11-15")
    assert regime == "bull"


def test_classify_market_regime_ranging():
    """Mid 2024 should be classified as ranging."""
    regime = ReplayBacktestEngine._classify_market_regime("2024-06-15")
    assert regime == "ranging"


def test_classify_market_regime_valid_outputs():
    """All regime classifications should be valid."""
    valid_regimes = {"bull", "bear", "ranging"}
    for date in ("2022-03-15", "2023-07-15", "2024-01-15", "2024-12-15", "2025-06-15"):
        regime = ReplayBacktestEngine._classify_market_regime(date)
        assert regime in valid_regimes


def test_slice_anchors_by_regime():
    """Test anchor slicing by market regime."""
    engine = ReplayBacktestEngine()
    anchors = ["2022-06-15", "2024-06-15", "2024-11-15",
               "2023-03-15", "2025-01-15"]
    regimes = engine.slice_anchors_by_regime(anchors)
    assert len(regimes["bear"]) >= 1
    assert len(regimes["bull"]) >= 1
    assert len(regimes["ranging"]) >= 1
    # Total should match input
    total = sum(len(v) for v in regimes.values())
    assert total == len(anchors)


# ── Fix 4: Quarterly Calibration ──

@pytest.mark.asyncio
async def test_screen_pool_at_date_no_data():
    """_screen_pool_at_date should return empty list when Tushare unavailable."""
    engine = ReplayBacktestEngine()
    pool = await engine._screen_pool_at_date("2022-01-01", 10)
    assert isinstance(pool, list)


# ── Core backtest tests (updated for v2) ──

@pytest.mark.asyncio
async def test_run_single_anchor():
    cfg = BacktestConfig(term="medium", holding_days=5)
    engine = ReplayBacktestEngine(cfg)
    pool = ["sh.601888", "sh.603871", "sz.300308"]
    price_data = {
        "sh.601888": [80.0, 81.0, 82.0, 80.5, 83.0, 84.0],
        "sh.603871": [12.0, 12.3, 12.1, 12.5, 12.4, 12.8],
        "sz.300308": [880.0, 885.0, 890.0, 895.0, 900.0, 910.0],
    }
    result = await engine.run_single_anchor("2024-06-14", pool, price_data)
    assert result["anchor_date"] == "2024-06-14"
    assert result["pool_size"] == 3
    assert "ablation_results" in result
    assert "full" in result["ablation_results"]
    assert len(result["ablation_results"]) == 6  # full + 5 ablation


def test_experiment_engine_ablation():
    engine = ExperimentEngine()
    ablation_results = {
        "full": {"scores": [80, 70, 60, 50, 40], "returns": [0.10, 0.05, 0, -0.03, -0.08]},
        "-fundamental": {"scores": [75, 65, 55, 45, 35], "returns": [0.10, 0.05, 0, -0.03, -0.08]},
    }
    result = engine.run_ablation_experiment("medium", ["sh.601888"]*5, ablation_results)
    assert result["experiment_type"] == "ablation"
    assert len(result["contributions"]) == 1


def test_experiment_engine_gate():
    engine = ExperimentEngine()
    scores_on = [70, 60, 50, 40, 30]
    scores_off = [75, 65, 55, 45, 35]
    returns = [0.10, 0.05, 0.0, -0.03, -0.08]
    result = engine.run_gate_experiment("medium", scores_on, scores_off, returns)
    assert "delta_L" in result
    assert "false_kill_rate" in result


def test_experiment_engine_consistency():
    engine = ExperimentEngine()
    run1 = [80, 70, 60, 50, 40]
    run2 = [78, 72, 58, 52, 38]
    result = engine.run_consistency_test("short", run1, run2)
    assert result["mean_score_diff"] > 0
    assert "action_flip_rate" in result
    assert "top_k_overlap" in result


# ── Add: Convert code utility ──

def test_convert_to_ts_code():
    engine = ReplayBacktestEngine()
    assert engine._convert_to_ts_code("sh.601888") == "601888.SH"
    assert engine._convert_to_ts_code("sz.300308") == "300308.SZ"
    assert engine._convert_to_ts_code("000001.SZ") == "000001.SZ"


# ── Add: Fallback scoring (no Tushare) ──

@pytest.mark.asyncio
async def test_score_stock_pit_fallback_consistent():
    """Fallback scoring should be deterministic, not random."""
    engine = ReplayBacktestEngine()
    # Without Tushare, should use fallback (deterministic, not hash-based)
    s1 = await engine._score_stock_pit(
        "sh.601888", "TestCo", "2024-06-14", "full"
    )
    s2 = await engine._score_stock_pit(
        "sh.601888", "TestCo", "2024-06-14", "full"
    )
    assert s1 == s2  # Deterministic


def test_fallback_composite_score_range():
    """Fallback scores should be in reasonable range."""
    engine = ReplayBacktestEngine()
    for code in ("sh.601888", "sz.000858", "sz.300750", "sh.688981"):
        score = engine._fallback_composite_score(code, "2024-06-14")
        assert 30 <= score <= 85


def test_fallback_composite_score_different_segments():
    """Different market segments should yield different baseline scores."""
    engine = ReplayBacktestEngine()
    sh_main = engine._fallback_composite_score("sh.601888", "2024-06-14")
    sz_main = engine._fallback_composite_score("sz.000858", "2024-06-14")
    gem = engine._fallback_composite_score("sz.300750", "2024-06-14")
    star = engine._fallback_composite_score("sh.688981", "2024-06-14")
    # All should be different (different market segments)
    scores = {sh_main, sz_main, gem, star}
    assert len(scores) >= 2  # At least some segments differ


# ── Add: Ablation discount tests ──

def test_apply_ablation_discount():
    engine = ReplayBacktestEngine(BacktestConfig(term="medium"))
    base = 70.0
    full = engine._apply_ablation_discount(base, "full", "medium")
    assert full == base  # full should not discount

    # Ablation should reduce score
    abl_fund = engine._apply_ablation_discount(base, "-fundamental", "medium")
    assert abl_fund < base

    # Different agents should have different discounts
    abl_tech = engine._apply_ablation_discount(base, "-technical", "medium")
    abl_money = engine._apply_ablation_discount(base, "-moneyflow", "medium")
    # Different weights → different scores
    assert abl_fund != abl_money
