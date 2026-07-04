"""Tests for the learning feedback loop: history warm-start and
journal-truth performance metrics."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.technical_strategy import TechnicalStrategy
from src.agent.strategy_manager import StrategyManager
from src.agent.strategy_attribution import compute_attribution

CONFIG = {
    'ensemble_method': 'weighted_average',
    'strategies': {
        'momentum': {'type': 'technical', 'enabled': True, 'weight': 1.0,
                     'lookback_period': 50},
    },
}


def test_seed_history_enables_immediate_signals():
    strat = TechnicalStrategy('momentum', {'lookback_period': 50})
    # Unseeded: first tick is blind (needs 50 bars)
    blind = strat.generate_signals({'symbol': 'AAPL', 'price': 100.0})
    assert blind['confidence'] == 0.0

    strat2 = TechnicalStrategy('momentum', {'lookback_period': 50})
    seeded = strat2.seed_history('AAPL', [100 + i * 0.5 for i in range(60)])
    assert seeded == 50  # trimmed to lookback
    out = strat2.generate_signals({'symbol': 'AAPL', 'price': 130.0})
    assert 'indicators' in out  # full signal path ran on the first live tick


def test_seed_history_rejects_garbage():
    strat = TechnicalStrategy('m', {})
    assert strat.seed_history('AAPL', [0, None, -5]) == 0
    assert 'AAPL' not in strat.historical_data or not strat.historical_data.get('AAPL')


def test_warm_start_through_manager():
    mgr = StrategyManager(CONFIG)
    seeded = mgr.warm_start({'AAPL': [100 + i for i in range(60)],
                             'MSFT': [200 + i for i in range(60)]})
    assert seeded == 2  # one strategy x two symbols


def test_attribution_exposes_trade_pnls():
    fills = [
        {'strategy': 's', 'symbol': 'AAPL', 'side': 'buy',
         'filled_quantity': 10, 'filled_avg_price': 100.0, 'created_at': 'a'},
        {'strategy': 's', 'symbol': 'AAPL', 'side': 'sell',
         'filled_quantity': 10, 'filled_avg_price': 105.0, 'created_at': 'b'},
    ]
    out = compute_attribution(fills)
    assert out['s']['trade_pnls'] == [50.0]


def test_performance_updates_from_real_attribution():
    mgr = StrategyManager(CONFIG)
    attribution = {
        'momentum': {
            'realized_pnl': 30.0, 'closed_trades': 3, 'win_rate': 0.667,
            'trade_pnls': [50.0, -40.0, 20.0], 'unrealized_pnl': 0.0,
            'open_positions': [],
        },
        'killswitch': {'realized_pnl': 1.0, 'trade_pnls': [1.0]},  # not a strategy: ignored
    }
    mgr.update_performance_from_attribution(attribution)
    perf = mgr.strategy_performance['momentum']
    assert perf['source'] == 'order_journal'
    assert perf['total_return'] == 30.0
    assert perf['win_rate'] == 0.667
    assert perf['sharpe_ratio'] != 0.0        # computed from the pnl series
    assert perf['max_drawdown'] == 40.0       # peak 50 -> trough 10
    assert 'killswitch' not in mgr.strategy_performance
