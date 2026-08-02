"""Regime classification.

The ensemble cancelled a live signal on 62.7% of bars because momentum and
mean_reversion measure the same thing with opposite sign. Routing to one
strategy per regime replaces that averaging, so this classifier decides
which strategy runs.
"""

import pytest

from src.agent.regime import (
    RANGING, TRENDING, UNDEFINED, classify, conviction, efficiency_ratio,
)

CFG = {'regime': {'window': 10, 'trend_threshold': 0.35, 'range_threshold': 0.20}}

STRAIGHT_UP = [100.0 + i for i in range(21)]
ZIGZAG = [100.0 + (2 if i % 2 else 0) for i in range(21)]
FLAT = [100.0] * 21


def test_monotonic_series_has_efficiency_ratio_of_one():
    assert efficiency_ratio(STRAIGHT_UP, 10) == pytest.approx(1.0)


def test_zigzag_series_has_efficiency_ratio_near_zero():
    assert efficiency_ratio(ZIGZAG, 10) < 0.15


def test_flat_series_returns_zero_not_a_division_error():
    assert efficiency_ratio(FLAT, 10) == 0.0


def test_short_history_returns_zero():
    assert efficiency_ratio([100.0, 101.0], 10) == 0.0


def test_efficiency_ratio_is_direction_agnostic():
    down = [100.0 - i for i in range(21)]
    assert efficiency_ratio(down, 10) == pytest.approx(1.0)


def test_strong_trend_classifies_as_trending():
    assert classify(STRAIGHT_UP, None, CFG) == TRENDING


def test_chop_classifies_as_ranging():
    assert classify(ZIGZAG, None, CFG) == RANGING


def test_flat_series_is_undefined_not_ranging():
    """A flat series has no measurable character; trading it on a
    mean-reversion signal would be trading noise."""
    assert classify(FLAT, None, CFG) == UNDEFINED


def test_short_history_is_undefined():
    assert classify([100.0, 101.0], None, CFG) == UNDEFINED


def test_dead_band_holds_the_previous_regime():
    """Hysteresis: without it the router flips at the boundary every other
    bar and pays costs for the privilege."""
    band = [100.0, 101.0, 100.5, 101.5, 101.0, 102.0, 101.5, 102.5, 102.0, 103.0, 102.5]
    er = efficiency_ratio(band, 10)
    assert 0.20 < er < 0.35, f'fixture must sit in the dead band, got {er}'
    assert classify(band, TRENDING, CFG) == TRENDING
    assert classify(band, RANGING, CFG) == RANGING


def test_dead_band_with_no_previous_regime_is_undefined():
    band = [100.0, 101.0, 100.5, 101.5, 101.0, 102.0, 101.5, 102.5, 102.0, 103.0, 102.5]
    assert classify(band, None, CFG) == UNDEFINED


def test_crossing_the_opposite_threshold_does_switch():
    assert classify(ZIGZAG, TRENDING, CFG) == RANGING
    assert classify(STRAIGHT_UP, RANGING, CFG) == TRENDING


def test_conviction_is_full_at_an_extreme_regime():
    assert conviction(STRAIGHT_UP, TRENDING, CFG) == pytest.approx(1.0)


def test_conviction_is_floored_at_a_marginal_regime():
    marginal = [100.0] * 11
    assert conviction(marginal, UNDEFINED, CFG) == pytest.approx(0.7)


def test_conviction_stays_within_bounds():
    for series in (STRAIGHT_UP, ZIGZAG, FLAT):
        for regime in (TRENDING, RANGING, UNDEFINED):
            assert 0.7 <= conviction(series, regime, CFG) <= 1.0


def test_defaults_apply_when_config_omits_the_regime_section():
    assert classify(STRAIGHT_UP, None, {}) == TRENDING


def test_efficiency_ratio_negative_window_returns_zero():
    assert efficiency_ratio(STRAIGHT_UP, -5) == 0.0


def test_efficiency_ratio_zero_window_returns_zero():
    assert efficiency_ratio(STRAIGHT_UP, 0) == 0.0


def test_classify_negative_window_is_undefined_not_misclassified():
    """A misconfigured negative window must never silently misclassify.

    Regression for the bug where `closes[-(window + 1):]` with a negative
    window sliced from the wrong end and produced a bogus but plausible
    'ranging' verdict instead of surfacing the bad config as UNDEFINED.
    """
    cfg = {'regime': {'window': -5, 'trend_threshold': 0.35, 'range_threshold': 0.20}}
    assert classify(STRAIGHT_UP, None, cfg) == UNDEFINED
    assert classify(ZIGZAG, None, cfg) == UNDEFINED


def test_classify_zero_window_is_undefined():
    cfg = {'regime': {'window': 0, 'trend_threshold': 0.35, 'range_threshold': 0.20}}
    assert classify(STRAIGHT_UP, None, cfg) == UNDEFINED
