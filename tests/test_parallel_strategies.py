"""Tests for per-strategy signal exposure (parallel execution mode)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.strategy_manager import StrategyManager


CONFIG = {
    'ensemble_method': 'weighted_average',
    'strategies': {
        'momentum': {'type': 'technical', 'enabled': True, 'weight': 1.0},
        'mean_reversion': {'type': 'technical', 'enabled': True, 'weight': 1.0},
    },
}


def test_generate_signals_exposes_per_strategy():
    mgr = StrategyManager(CONFIG)
    data = {'symbol': 'AAPL', 'price': 100.0, 'close': 100.0}
    out = mgr.generate_signals(data)
    assert 'per_strategy' in out
    assert set(out['per_strategy'].keys()) == {'momentum', 'mean_reversion'}
    for sig in out['per_strategy'].values():
        assert 'action' in sig and 'confidence' in sig


def test_collect_strategy_signals_degrades_broken_strategy_to_hold():
    mgr = StrategyManager(CONFIG)

    class Broken:
        def generate_signals(self, data):
            raise RuntimeError('boom')

    mgr.strategies['broken'] = Broken()
    out = mgr.collect_strategy_signals({'symbol': 'AAPL', 'price': 100.0})
    assert out['broken']['action'] == 'hold'
    assert out['broken']['confidence'] == 0.0
    # healthy strategies unaffected
    assert 'momentum' in out and 'mean_reversion' in out


def agreeing_executions(signal_data, action, target_value):
    """Mirror of the execution-plan selection in main._execute_trades."""
    per_strategy = signal_data.get('per_strategy') or {}
    agreeing = [n for n, s in per_strategy.items()
                if s.get('action') == action and s.get('confidence', 0) >= 0.3]
    if agreeing:
        share = target_value / len(agreeing)
        return [(n, share) for n in agreeing]
    return [(signal_data.get('strategy', 'ensemble'), target_value)]


def test_execution_plan_splits_between_agreeing_strategies():
    signal = {
        'action': 'buy',
        'per_strategy': {
            'momentum': {'action': 'buy', 'confidence': 0.8},
            'mean_reversion': {'action': 'sell', 'confidence': 0.9},  # dissenter: skipped
            'rsi': {'action': 'buy', 'confidence': 0.2},              # below threshold
            'breakout': {'action': 'buy', 'confidence': 0.5},
        },
    }
    plan = agreeing_executions(signal, 'buy', 1000.0)
    assert sorted(plan) == [('breakout', 500.0), ('momentum', 500.0)]
    # cap preserved: shares sum to the symbol target
    assert sum(v for _, v in plan) == 1000.0


def test_execution_plan_falls_back_to_ensemble_when_no_agreement():
    signal = {
        'action': 'buy',
        'per_strategy': {'momentum': {'action': 'sell', 'confidence': 0.9}},
    }
    assert agreeing_executions(signal, 'buy', 800.0) == [('ensemble', 800.0)]
