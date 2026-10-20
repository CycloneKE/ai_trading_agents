# tests/test_verdict_scoreboard.py
from typing import Dict

from src.agent.verdict_scoreboard import score_decisions, summary_line


def test_buy_that_appreciated_is_a_hit():
    decisions = [
        {'symbol': 'SCOM', 'action': 'buy', 'executed': True, 'price': 30.0},
        {'symbol': 'SCOM', 'action': 'buy', 'executed': True, 'price': 40.0},
        {'symbol': 'SCOM', 'action': 'hold', 'executed': False, 'price': 31.0},
    ]
    scores = score_decisions(decisions, lambda s: 34.0)
    assert scores['SCOM']['evaluated'] == 2      # holds/skips not scored
    assert scores['SCOM']['hits'] == 1           # 30->34 hit, 40->34 miss
    assert scores['SCOM']['hit_rate'] == 0.5


def test_summary_line_mentions_rate():
    line = summary_line({'SCOM': {'evaluated': 10, 'hits': 6, 'hit_rate': 0.6}}, 'SCOM')
    assert '60' in line and 'SCOM' in line


def test_unknown_symbol_gives_neutral_line():
    assert 'No scored history' in summary_line({}, 'EQTY')


def test_sell_that_dropped_is_a_hit():
    decisions = [
        {'symbol': 'KCB', 'action': 'sell', 'executed': True, 'price': 40.0},
        {'symbol': 'KCB', 'action': 'sell', 'executed': True, 'price': 30.0},
    ]
    scores = score_decisions(decisions, lambda s: 34.0)
    assert scores['KCB']['evaluated'] == 2       # both executed sells scored
    assert scores['KCB']['hits'] == 1            # 40->34 hit (now < price), 30->34 miss
    assert scores['KCB']['hit_rate'] == 0.5


def test_price_lookup_is_memoized_per_symbol():
    calls: Dict[str, int] = {}

    def price_lookup(sym):
        calls[sym] = calls.get(sym, 0) + 1
        return 34.0

    decisions = [
        {'symbol': 'SCOM', 'action': 'buy', 'executed': True, 'price': 30.0},
        {'symbol': 'SCOM', 'action': 'sell', 'executed': True, 'price': 40.0},
        {'symbol': 'SCOM', 'action': 'buy', 'executed': True, 'price': 20.0},
    ]
    score_decisions(decisions, price_lookup)
    assert calls['SCOM'] == 1


def test_none_price_lookup_symbol_is_skipped_without_crash():
    decisions = [
        {'symbol': 'SCOM', 'action': 'buy', 'executed': True, 'price': 30.0},
        {'symbol': 'MISSING', 'action': 'buy', 'executed': True, 'price': 10.0},
    ]

    def price_lookup(sym):
        return None if sym == 'MISSING' else 34.0

    scores = score_decisions(decisions, price_lookup)
    assert 'SCOM' in scores
    assert 'MISSING' not in scores
