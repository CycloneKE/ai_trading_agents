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
    assert out[0]['detail']['reasons'] == {'min_notional': 6}
    assert out[0]['attention'] and 'minimum order size' in out[0]['message']
    assert out[0]['hint']


def test_signals_held_back_by_the_agents_own_rules_are_information_not_problems():
    decisions = ([dec('NVDA', action='buy', skip='add_not_profitable') for _ in range(58)]
                 + [dec('XLF', action='sell', skip='no_position') for _ in range(35)])
    out = {a['symbol']: a for a in A.detect_blocked_intent(decisions, threshold=5)}
    nvda, xlf = out['NVDA'], out['XLF']
    assert nvda['type'] == 'held_back' and nvda['severity'] == 'info' and not nvda['attention']
    assert nvda['message'].startswith('NVDA: buy signal on 58 checks, held back on purpose')
    assert '5% above the last purchase' in nvda['message']
    assert 'never bets on a fall' in xlf['message'] and 'sell signal' in xlf['message']
    assert 'tried' not in nvda['message'] and 'blocked' not in nvda['message']


def test_a_real_block_among_rule_holds_still_needs_a_look():
    decisions = ([dec('TSLA', action='buy', skip='bias_downgrade') for _ in range(20)]
                 + [dec('TSLA', action='buy', skip='insufficient_cash') for _ in range(6)])
    [a] = A.detect_blocked_intent(decisions, threshold=5)
    assert a['type'] == 'blocked_intent' and a['attention'] and 'paper cash' in a['message']


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


def test_a_repeated_price_problem_says_when_it_last_happened():
    decisions = [{**dec('BTC-USD', skip='fallback_price'), 'ts': f'2026-09-28T0{h}:00:00'} for h in (3, 7, 5, 4)]
    [notice] = A.detect_persistent_skip(decisions, threshold=4)
    assert notice['detail']['last_at'] == '2026-09-28T07:00:00'


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


def test_a_block_is_said_once_not_again_as_a_persistent_skip():
    decisions = [dec('NVDA', action='buy', skip='no_price') for _ in range(14)]
    out = A.scan(decisions, [])
    assert [a['type'] for a in out if a['symbol'] == 'NVDA'] == ['blocked_intent']
    assert 'could not get a current price' in out[0]['message']


def test_only_recent_checks_count_and_the_last_one_is_shown():
    from datetime import datetime, timedelta
    now = datetime(2026, 9, 27, 4, 0)
    old = [dict(dec('TSLA', action='buy', skip='no_price'), ts=(now - timedelta(hours=20)).isoformat())
           for _ in range(12)]
    new = [dict(dec('NVDA', action='buy', skip='add_not_profitable'), ts=(now - timedelta(minutes=m)).isoformat())
           for m in range(30, 0, -5)]
    out = A.scan(old + new, [], now=now)
    assert [a['symbol'] for a in out] == ['NVDA']                          # the fixed problem has dropped off
    assert out[0]['detail']['last_at'] == (now - timedelta(minutes=5)).isoformat()
