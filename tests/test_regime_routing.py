"""Routing replaces averaging.

The regression this exists to fix: `_weighted_average_ensemble` divides the
winning action's score by total weight, so a 2-vs-1 disagreement lands at
~0.5 by construction. Since position_size = confidence * max_position_size,
every surviving trade was sized at ~2.6% regardless of conviction.
"""

import pytest

from src.agent.regime import RANGING, TRENDING, UNDEFINED
from src.agent.strategy_manager import StrategyManager

CONFIG = {
    'ensemble_method': 'regime_routed',
    'regime': {'window': 10, 'trend_threshold': 0.35, 'range_threshold': 0.20},
    'strategies': {
        'momentum': {'type': 'technical', 'enabled': True, 'weight': 1.0,
                     'lookback_period': 20, 'max_position_size': 0.05},
        'mean_reversion': {'type': 'technical', 'enabled': True, 'weight': 1.0,
                           'lookback_period': 20, 'max_position_size': 0.05},
        'rsi_strategy': {'type': 'technical', 'enabled': True, 'weight': 0.8,
                         'lookback_period': 20, 'max_position_size': 0.05},
    },
}


def feed(mgr, symbol, closes):
    """Push a price series through generate_signals, returning the last result."""
    out = {}
    for price in closes:
        out = mgr.generate_signals({'symbol': symbol, 'price': price, 'close': price})
    return out


def signals(momentum, mean_reversion, rsi):
    return {
        'momentum': {'action': momentum, 'confidence': 0.8, 'position_size': 0.04},
        'mean_reversion': {'action': mean_reversion, 'confidence': 0.9, 'position_size': 0.045},
        'rsi_strategy': {'action': rsi, 'confidence': 0.7, 'position_size': 0.035},
    }


def test_trending_regime_routes_to_momentum():
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0 + i for i in range(21)]
    out = mgr._regime_routed_ensemble(signals('buy', 'sell', 'sell'), 'X', {'symbol': 'X'})
    assert out['regime'] == TRENDING
    assert out['routed_strategy'] == 'momentum'
    assert out['action'] == 'buy'


def test_ranging_regime_routes_to_mean_reversion():
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0 + (2 if i % 2 else 0) for i in range(21)]
    out = mgr._regime_routed_ensemble(signals('buy', 'sell', 'sell'), 'X', {'symbol': 'X'})
    assert out['regime'] == RANGING
    assert out['routed_strategy'] == 'mean_reversion'
    assert out['action'] == 'sell'


def test_ranging_regime_requires_rsi_agreement():
    """RSI holds 88.8% of the time, so as a voter it is inert. As a
    confirmation gate in the one regime where it is meaningful, it earns
    its place."""
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0 + (2 if i % 2 else 0) for i in range(21)]
    out = mgr._regime_routed_ensemble(signals('buy', 'sell', 'buy'), 'X', {'symbol': 'X'})
    assert out['action'] == 'hold'


def test_undefined_regime_produces_hold():
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0] * 21
    out = mgr._regime_routed_ensemble(signals('buy', 'buy', 'buy'), 'X', {'symbol': 'X'})
    assert out['regime'] == UNDEFINED
    assert out['action'] == 'hold'


def test_confidence_is_no_longer_pinned_near_half():
    """THE regression. Under weighted-average this returned ~0.5 whenever the
    strategies disagreed; the diagnostic measured a median of 0.517 with a
    maximum of 0.629 across 306 live signals."""
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0 + i for i in range(21)]
    out = mgr._regime_routed_ensemble(signals('buy', 'sell', 'sell'), 'X', {'symbol': 'X'})
    assert out['confidence'] > 0.7


def test_position_size_scales_with_confidence():
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0 + i for i in range(21)]
    strong = mgr._regime_routed_ensemble(signals('buy', 'sell', 'sell'), 'X', {'symbol': 'X'})
    weak_signals = signals('buy', 'sell', 'sell')
    weak_signals['momentum'] = {'action': 'buy', 'confidence': 0.5, 'position_size': 0.01}
    weak = mgr._regime_routed_ensemble(weak_signals, 'X', {'symbol': 'X'})
    assert strong['position_size'] > weak['position_size']


def test_a_missing_routed_strategy_degrades_to_hold():
    """A config without a momentum strategy must not raise in the signal path."""
    cfg = {k: v for k, v in CONFIG.items()}
    cfg['strategies'] = {'mean_reversion': CONFIG['strategies']['mean_reversion']}
    mgr = StrategyManager(cfg)
    mgr._regime_history['X'] = [100.0 + i for i in range(21)]
    out = mgr._regime_routed_ensemble({'mean_reversion': {'action': 'sell', 'confidence': 0.9}},
                                      'X', {'symbol': 'X'})
    assert out['action'] == 'hold'


def test_generate_signals_builds_its_own_price_history():
    mgr = StrategyManager(CONFIG)
    feed(mgr, 'X', [100.0 + i for i in range(30)])
    assert len(mgr._regime_history['X']) >= 21


def test_price_history_is_bounded():
    """An unbounded buffer leaks in a process that runs for weeks."""
    mgr = StrategyManager(CONFIG)
    feed(mgr, 'X', [100.0 + i for i in range(1200)])
    assert len(mgr._regime_history['X']) <= 500


def test_regime_state_persists_between_calls_for_hysteresis():
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0 + i for i in range(21)]
    mgr._regime_routed_ensemble(signals('buy', 'sell', 'sell'), 'X', {'symbol': 'X'})
    assert mgr._regime_state['X'] == TRENDING


def test_other_ensemble_methods_are_unchanged():
    """The A/B depends on the old path behaving exactly as before."""
    cfg = dict(CONFIG, ensemble_method='weighted_average')
    mgr = StrategyManager(cfg)
    out = mgr._combine_signals(signals('buy', 'sell', 'sell'), {'symbol': 'X'})
    assert out['ensemble_method'] == 'weighted_average'
    assert 'regime' not in out


def test_warm_start_seeds_the_regime_buffer():
    mgr = StrategyManager(CONFIG)
    mgr.warm_start({'X': [100.0 + i for i in range(60)]})
    assert len(mgr._regime_history.get('X', [])) >= 21
