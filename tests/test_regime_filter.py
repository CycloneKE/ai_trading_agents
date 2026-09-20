"""Regime classification and the ensemble changes that depend on it."""
import json
import pytest

from src.agent.regime import (RANGING, TRENDING_DOWN, TRENDING_UP, UNKNOWN,
                              RegimeDetector, family_of, filter_signals, is_active)
from src.agent.strategy_manager import StrategyManager


def _rising(n=80, start=100.0, step=1.0):
    return [start + i * step for i in range(n)]


def _falling(n=80, start=180.0, step=1.0):
    return [start - i * step for i in range(n)]


def _oscillating(n=80, mid=100.0, amp=2.0):
    return [mid + (amp if i % 2 else -amp) for i in range(n)]


# --- detector ---------------------------------------------------------------

def test_regime_is_unknown_until_slow_period_bars():
    d = RegimeDetector(fast_period=20, slow_period=50)
    for p in _rising(30):
        d.update('X', p)
    assert d.regime('X') == UNKNOWN
    assert d.bars('X') == 30


def test_detects_uptrend():
    d = RegimeDetector(fast_period=20, slow_period=50)
    d.warm_start('X', _rising())
    assert d.regime('X') == TRENDING_UP


def test_detects_downtrend():
    d = RegimeDetector(fast_period=20, slow_period=50)
    d.warm_start('X', _falling())
    assert d.regime('X') == TRENDING_DOWN


def test_detects_range():
    d = RegimeDetector(fast_period=20, slow_period=50)
    d.warm_start('X', _oscillating())
    assert d.regime('X') == RANGING


def test_detector_ignores_bad_prices():
    d = RegimeDetector()
    for bad in (0, -1, None, 'x'):
        d.update('X', bad)
    assert d.bars('X') == 0


def test_symbols_are_tracked_independently():
    d = RegimeDetector(fast_period=20, slow_period=50)
    d.warm_start('UP', _rising())
    d.warm_start('DOWN', _falling())
    assert d.regime('UP') == TRENDING_UP
    assert d.regime('DOWN') == TRENDING_DOWN


# --- family mapping ---------------------------------------------------------

@pytest.mark.parametrize('name,family', [
    ('momentum', 'momentum'), ('fast_momentum', 'momentum'),
    ('mean_reversion', 'reversion'), ('reversion_v2', 'reversion'),
    ('rsi_strategy', None), ('anything_else', None),
])
def test_family_mapping(name, family):
    assert family_of(name) == family


def test_momentum_acts_in_trends_not_ranges():
    assert is_active('momentum', TRENDING_UP)
    assert is_active('momentum', TRENDING_DOWN)
    assert not is_active('momentum', RANGING)


def test_mean_reversion_acts_in_ranges_not_trends():
    assert is_active('mean_reversion', RANGING)
    assert not is_active('mean_reversion', TRENDING_UP)
    assert not is_active('mean_reversion', TRENDING_DOWN)


def test_unclassified_strategies_always_act():
    for regime in (TRENDING_UP, TRENDING_DOWN, RANGING, UNKNOWN):
        assert is_active('rsi_strategy', regime)


def test_unknown_regime_filters_nobody():
    for name in ('momentum', 'mean_reversion', 'rsi_strategy'):
        assert is_active(name, UNKNOWN)


# --- filtering --------------------------------------------------------------

def test_filter_drops_out_of_regime_votes():
    sigs = {'momentum': {'action': 'buy'}, 'mean_reversion': {'action': 'sell'},
            'rsi_strategy': {'action': 'hold'}}
    kept = filter_signals(sigs, TRENDING_UP)
    assert set(kept) == {'momentum', 'rsi_strategy'}


def test_filter_never_silences_everyone():
    """If the filter would leave no voters, fall back to the unfiltered set."""
    sigs = {'mean_reversion': {'action': 'buy'}}
    assert filter_signals(sigs, TRENDING_UP) == sigs


def test_filter_handles_empty_input():
    assert filter_signals({}, TRENDING_UP) == {}


# --- ensemble integration ---------------------------------------------------

def _cfg(**over):
    with open('config/config.json') as f:
        cfg = json.load(f)
    cfg.update(over)
    return cfg


def test_configured_weights_are_actually_used():
    """The bug: strategy_performance seeds sharpe/win_rate at 0.0, so the
    1.0/0.5 defaults never applied and every strategy scored the 0.1 floor,
    ignoring config weights entirely."""
    m = StrategyManager(_cfg())
    sigs = {n: {'action': 'buy', 'confidence': 1.0, 'position_size': 0.0}
            for n in m.strategies}
    out = m._adaptive_confidence_ensemble(sigs, 'X', {'close': 100, 'open': 100})
    assert out['action'] == 'buy'
    # Unanimous max-confidence buy must score near 1.0. Under the bug every
    # weight collapsed to the same floor, which still normalises to 1.0 here,
    # so assert the weights themselves differ as configured instead.
    assert len(set(m.strategy_weights.values())) > 1
    assert m.strategy_weights['rsi_strategy'] < m.strategy_weights['momentum']


def test_regime_is_reported_on_the_ensemble_result():
    m = StrategyManager(_cfg())
    m.regime_detector.warm_start('X', _rising())
    out = m.generate_signals({'symbol': 'X', 'price': 181.0, 'close': 181.0})
    assert out['regime'] == TRENDING_UP


def test_filter_can_be_disabled():
    cfg = _cfg()
    cfg['regime_filter'] = {**cfg.get('regime_filter', {}), 'enabled': False}
    m = StrategyManager(cfg)
    m.regime_detector.warm_start('X', _rising())
    out = m.generate_signals({'symbol': 'X', 'price': 181.0, 'close': 181.0})
    assert out['regime'] == UNKNOWN          # not consulted


def test_warm_start_seeds_the_detector():
    m = StrategyManager(_cfg())
    m.warm_start({'X': _rising()})
    assert m.regime_detector.regime('X') == TRENDING_UP
