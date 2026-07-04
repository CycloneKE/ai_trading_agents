"""Tests for the anomaly detectors."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import anomaly_scan as A


def dec(symbol='AAPL', action='hold', executed=False, skip='hold', ps=None):
    return {'symbol': symbol, 'action': action, 'executed': executed,
            'skip_reason': skip, 'per_strategy': ps or {}}


def order(symbol='AAPL', side='buy', limit=100.0, fill=100.0, status='filled'):
    return {'symbol': symbol, 'side': side, 'limit_price': limit,
            'filled_avg_price': fill, 'status': status}


def test_blocked_intent_flags_repeated_blocks():
    decisions = [dec(action='buy', executed=False, skip='min_notional') for _ in range(6)]
    out = A.detect_blocked_intent(decisions, threshold=5)
    assert len(out) == 1
    assert out[0]['type'] == 'blocked_intent'
    assert out[0]['detail']['count'] == 6
    assert 'min_notional' in out[0]['message']


def test_blocked_intent_ignores_normal_holds():
    decisions = [dec(action='hold', executed=False, skip='hold') for _ in range(10)]
    assert A.detect_blocked_intent(decisions, threshold=5) == []


def test_blocked_intent_ignores_executed():
    decisions = [dec(action='buy', executed=True, skip=None) for _ in range(6)]
    assert A.detect_blocked_intent(decisions, threshold=5) == []


def test_high_slippage_detects_adverse_fills():
    orders = [order(side='buy', limit=100.0, fill=100.3) for _ in range(3)]  # +30 bps each
    out = A.detect_high_slippage(orders, threshold_bps=20)
    assert len(out) == 1
    assert out[0]['detail']['avg_bps'] == 30.0


def test_high_slippage_ignores_good_fills():
    orders = [order(side='buy', limit=100.0, fill=100.0)]
    assert A.detect_high_slippage(orders, threshold_bps=20) == []


def test_persistent_skip_flags_systemic_reason():
    decisions = [dec(skip='fallback_price') for _ in range(4)]
    out = A.detect_persistent_skip(decisions, threshold=4)
    assert len(out) == 1
    assert out[0]['severity'] == 'high'
    assert out[0]['detail']['reason'] == 'fallback_price'


def test_persistent_skip_ignores_normal_hold():
    decisions = [dec(skip='hold') for _ in range(10)]
    assert A.detect_persistent_skip(decisions, threshold=4) == []


def test_strategy_disagreement_detects_split():
    ps = {'momentum': {'action': 'buy', 'confidence': 0.7},
          'mean_reversion': {'action': 'sell', 'confidence': 0.7}}
    out = A.detect_strategy_disagreement([dec(ps=ps)], conf=0.6)
    assert len(out) == 1
    assert out[0]['type'] == 'strategy_disagreement'


def test_strategy_disagreement_ignores_consensus():
    ps = {'momentum': {'action': 'buy', 'confidence': 0.7},
          'mean_reversion': {'action': 'buy', 'confidence': 0.7}}
    assert A.detect_strategy_disagreement([dec(ps=ps)], conf=0.6) == []


def test_drawdown_warn_and_breach():
    assert A.detect_drawdown({'drawdown': 0.05}, 0.8, 0.10) == []          # below warn
    warn = A.detect_drawdown({'drawdown': 0.085}, 0.8, 0.10)              # within 80% of cap
    assert warn and warn[0]['severity'] == 'medium'
    breach = A.detect_drawdown({'drawdown': 0.11}, 0.8, 0.10)             # over cap
    assert breach and breach[0]['severity'] == 'high'


def test_scan_sorts_by_severity():
    decisions = ([dec(action='buy', executed=False, skip='min_notional') for _ in range(6)] +  # high
                 [dec(symbol='X', skip='min_notional') for _ in range(4)])                     # medium persistent
    out = A.scan(decisions, [], risk_report=None)
    assert out[0]['severity'] == 'high'
    assert all(A.SEVERITY_RANK[out[i]['severity']] <= A.SEVERITY_RANK[out[i+1]['severity']]
               for i in range(len(out) - 1))
