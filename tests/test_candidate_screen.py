"""Screening a symbol for the trading universe."""
import numpy as np
import pandas as pd
import pytest

from src.agent.candidate_screen import (ADD, MARGINAL, REJECT, cost_hurdle,
                                        correlation_with_universe, screen)
from src.utils.price_files import data_quality

CFG = {
    'data_manager': {'nse_symbols': ['SCOM', 'KCB'], 'crypto_symbols': ['BTC-USD']},
    'costs': {
        'us_equity': {'commission_pct': 0.0, 'slippage_pct': 0.0005, 'verified': True},
        'nse': {'commission_pct': 0.017, 'slippage_pct': 0.003, 'verified': False},
        'crypto': {'commission_pct': 0.006, 'slippage_pct': 0.001, 'verified': False},
    },
}


def frame(returns, start=100.0, volume=100_000, freq='D'):
    prices, p = [], start
    for r in returns:
        p *= (1 + r)
        prices.append(p)
    idx = pd.date_range('2020-01-01', periods=len(prices), freq=freq)
    return pd.DataFrame({'close': prices, 'high': [x * 1.01 for x in prices],
                         'low': [x * 0.99 for x in prices],
                         'volume': [volume] * len(prices)}, index=idx)


def steady(n=300, drift=0.001, vol=0.02, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(drift, vol, n)


def barely_moves(n=300):
    """A series whose 20-bar move is a fraction of a percent."""
    return [0.00002 * (1 if i % 2 else -1) for i in range(n)]


# --- cost hurdle ------------------------------------------------------------

def test_us_equity_clears_the_cost_hurdle_easily():
    h = cost_hurdle(frame(steady()), 'AAPL', CFG)
    assert h['round_trip_pct'] == pytest.approx(0.001)
    assert h['move_to_cost_ratio'] > 10


def test_an_nse_symbol_faces_a_far_higher_bar():
    h = cost_hurdle(frame(steady()), 'SCOM', CFG)
    assert h['round_trip_pct'] == pytest.approx(0.04)
    us = cost_hurdle(frame(steady()), 'AAPL', CFG)
    assert h['move_to_cost_ratio'] < us['move_to_cost_ratio'] / 10


def test_insufficient_history_is_reported_not_guessed():
    h = cost_hurdle(frame(steady(10)), 'AAPL', CFG, holding_periods=20)
    assert h.get('insufficient_history')


# --- the central rejection --------------------------------------------------

def test_a_symbol_whose_move_is_smaller_than_its_cost_is_rejected():
    """The check that disqualifies most NSE candidates: a 4% round trip
    against a typical move of a fraction of a percent."""
    r = screen('SCOM', frame(barely_moves()), CFG)
    assert r.verdict == REJECT
    assert any('cannot pay for itself' in x for x in r.reasons)
    assert not r.passed


def test_the_same_series_is_tradeable_where_costs_are_near_zero():
    """Identical price behaviour, different market: cost decides, not the chart.

    A ~1% typical move over 20 bars comfortably clears a 0.1% US round trip
    and cannot come close to a 4% NSE one.
    """
    modest = steady(300, drift=0.0005, vol=0.002, seed=11)
    us = screen('AAPL', frame(modest), CFG)
    nse = screen('SCOM', frame(modest), CFG)
    assert 0.005 < us.metrics['cost']['typical_abs_move'] < 0.03   # ~1%
    assert us.passed
    assert nse.verdict == REJECT
    assert any('cannot pay for itself' in x for x in nse.reasons)


def test_a_healthy_us_symbol_passes_cleanly():
    r = screen('AAPL', frame(steady()), CFG)
    assert r.verdict == ADD
    assert r.reasons == []


def test_marginal_when_the_move_only_just_covers_the_cost():
    # ~1.5% typical 20-bar move against a 4% round trip fails outright;
    # raise the bar instead on a cheap market to get the warning path.
    r = screen('AAPL', frame(steady(vol=0.0008)), CFG, min_move_to_cost=1000)
    assert r.verdict == MARGINAL
    assert any('edge must be exceptional' in w for w in r.warnings)


# --- liquidity --------------------------------------------------------------

def test_thin_volume_is_rejected_when_a_minimum_is_set():
    r = screen('AAPL', frame(steady(), volume=100), CFG, min_avg_volume=50_000)
    assert r.verdict == REJECT
    assert any('average volume' in x for x in r.reasons)


def test_missing_volume_warns_rather_than_rejecting():
    df = frame(steady()).drop(columns=['volume'])
    r = screen('AAPL', df, CFG)
    assert r.passed
    assert any('no volume data' in w for w in r.warnings)


# --- diversification --------------------------------------------------------

def test_correlation_is_reported_against_the_existing_universe():
    base = frame(steady(seed=3))
    corrs = correlation_with_universe(base['close'], {'TWIN': base})
    assert corrs['TWIN'] == pytest.approx(1.0, abs=1e-6)


def test_a_near_duplicate_of_a_held_name_is_flagged():
    base = frame(steady(seed=4))
    r = screen('AAPL', base, CFG, universe={'HELD': base})
    assert r.verdict == MARGINAL
    assert any('doubles an existing bet' in w for w in r.warnings)


def test_short_overlap_is_skipped_rather_than_reported_as_zero():
    long = frame(steady(300, seed=5))
    short = frame(steady(10, seed=6))
    assert correlation_with_universe(long['close'], {'SHORT': short}) == {}


# --- cost verification gates the verdict ------------------------------------

def test_unverified_costs_cap_the_verdict_at_marginal():
    """A clean pass on a guessed fee schedule is not a clean pass."""
    r = screen('BTC-USD', frame(steady(vol=0.05)), CFG)
    assert r.verdict == MARGINAL
    assert any('unverified placeholder' in w for w in r.warnings)


# --- data quality -----------------------------------------------------------

def test_a_stale_repeating_feed_is_flagged():
    """Repeated closes look like zero volatility, which makes stops too tight
    and positions too large."""
    df = frame([0.0] * 300)
    assert any('repeat the previous close' in i for i in data_quality(df)['issues'])


def test_quality_reports_the_span_it_saw():
    q = data_quality(frame(steady(120)))
    assert q['bars'] == 120
    assert q['typical_gap_days'] == 1.0
