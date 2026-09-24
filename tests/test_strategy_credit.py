"""Crediting trades to the strategies that decided them.

Every blended order used to be journaled as 'ensemble' and every NSE fill as
'nse_manual', so no individual strategy was ever credited with a result and
the performance-weighted ensemble had nothing to learn from.
"""
import json

import pytest

from src.agent.strategy_attribution import compute_attribution, credit_weights


def _row(side, qty, price, t, weights=None, strategy='ensemble', symbol='AAPL'):
    return {'strategy': strategy, 'symbol': symbol, 'side': side, 'filled_quantity': qty,
            'filled_avg_price': price, 'created_at': f'2026-09-24T{t}',
            'strategy_weights': json.dumps(weights) if weights else None}


def test_credit_goes_to_the_strategies_that_voted_by_confidence():
    votes = {'momentum': {'action': 'buy', 'confidence': 0.6},
             'rsi_strategy': {'action': 'buy', 'confidence': 0.2},
             'mean_reversion': {'action': 'sell', 'confidence': 0.9},
             'idle': {'action': 'buy', 'confidence': 0.0}}
    assert credit_weights(votes, 'buy') == {'momentum': 0.75, 'rsi_strategy': 0.25}
    assert credit_weights(votes, 'hold') is None
    assert credit_weights(None, 'buy') is None


def test_a_blended_buy_is_split_and_the_exit_settles_each_share():
    out = compute_attribution([
        _row('buy', 100, 10.0, '10:00', {'momentum': 0.75, 'rsi_strategy': 0.25}),
        _row('sell', 100, 12.0, '11:00', {'mean_reversion': 1.0}),
    ])
    assert out['momentum']['realized_pnl'] == 150.0
    assert out['rsi_strategy']['realized_pnl'] == 50.0
    assert 'mean_reversion' not in out  # it triggered the exit; it owned nothing
    assert out['momentum']['trade_returns'] == [pytest.approx(0.2)]


def test_a_stop_exit_closes_the_holders_books_not_a_stop_book():
    out = compute_attribution([
        _row('buy', 10, 100.0, '10:00', {'momentum': 1.0}),
        _row('sell', 10, 92.0, '11:00', {'stop_loss': 1.0}, strategy='stop_loss'),
    ])
    assert set(out) == {'momentum'}
    assert out['momentum']['realized_pnl'] == -80.0 and out['momentum']['win_rate'] == 0.0


def test_a_partial_exit_settles_holders_in_proportion():
    out = compute_attribution([
        _row('buy', 60, 10.0, '10:00', {'a': 1.0}),
        _row('buy', 40, 11.0, '10:30', {'b': 1.0}),
        _row('sell', 50, 12.0, '11:00', {'a': 1.0}),
    ])
    assert out['a']['realized_pnl'] == pytest.approx(30 * 2.0)
    assert out['b']['realized_pnl'] == pytest.approx(20 * 1.0)
    assert out['a']['open_positions'][0]['quantity'] == 30
    assert out['b']['open_positions'][0]['quantity'] == 20


def test_returns_pool_across_currencies_and_fx_converts_the_amounts():
    rows = [_row('buy', 1000, 36.0, '10:00', {'momentum': 1.0}, symbol='SCOM'),
            _row('sell', 1000, 39.6, '11:00', {'momentum': 1.0}, symbol='SCOM'),
            _row('buy', 10, 100.0, '10:00', {'momentum': 1.0}),
            _row('sell', 10, 110.0, '11:00', {'momentum': 1.0})]
    out = compute_attribution(rows, fx=lambda s: 0.0077 if s == 'SCOM' else 1.0)
    assert out['momentum']['trade_returns'] == [pytest.approx(0.1), pytest.approx(0.1)]
    assert out['momentum']['realized_pnl'] == pytest.approx(3600 * 0.0077 + 100, abs=0.01)


def test_older_rows_without_weights_keep_their_own_books():
    out = compute_attribution([
        _row('buy', 10, 100.0, '10:00', strategy='ensemble'),
        _row('sell', 10, 105.0, '11:00', strategy='ensemble'),
    ])
    assert out['ensemble']['realized_pnl'] == 50.0


def test_strategy_performance_is_measured_in_returns_not_currency():
    from src.agent.strategy_manager import StrategyManager
    sm = StrategyManager.__new__(StrategyManager)
    import threading
    sm.lock = threading.Lock()
    sm.strategy_performance = {'momentum': {}}
    # One KES trade of +3,600 and one USD trade of +100, both +10%: a Sharpe
    # over raw amounts would be dominated by the KES one; over returns they
    # are the same trade.
    sm.update_performance_from_attribution({'momentum': {
        'trade_pnls': [3600.0, 100.0], 'trade_returns': [0.1, 0.1],
        'win_rate': 1.0, 'realized_pnl': 3700.0, 'closed_trades': 2}})
    assert sm.strategy_performance['momentum']['returns'] == [0.1, 0.1]
