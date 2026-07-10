# tests/test_verdict_scoreboard.py
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
