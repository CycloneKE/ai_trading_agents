"""Scoring the AI's trade reviews (src/agent/ai_scorecard.py)."""
from datetime import date, timedelta

import pytest

from src.agent import ai_scorecard as sc


def _dec(sym, day, action, verdict, price=100.0, skip=None, reasoning='because', hour=10):
    return {'symbol': sym, 'ts': f'{day}T{hour:02d}:00:00', 'action': action, 'price': price,
            'skip_reason': skip,
            'llm_verdict': {'action': verdict, 'reasoning': reasoning} if verdict else {}}


def _closes(start, values):
    d0 = date.fromisoformat(start)
    return {d0 + timedelta(days=i): v for i, v in enumerate(values)}


def test_only_reviewed_directional_signals_count_once_a_day():
    decisions = [
        _dec('SCOM', '2026-09-01', 'buy', 'buy'),
        _dec('SCOM', '2026-09-01', 'buy', 'buy'),               # same signal again: once
        _dec('KCB', '2026-09-01', 'buy', 'hold', skip='llm_veto'),
        _dec('EQTY', '2026-09-01', 'hold', None),                # never reviewed
        _dec('ABSA', '2026-09-01', 'buy', None),                 # no verdict recorded
        # The model was down and the signal passed through: not an approval.
        _dec('COOP', '2026-09-01', 'buy', 'buy', reasoning=None),
    ]
    got = sc.reviewed_signals(decisions)
    assert [(s['symbol'], s['outcome']) for s in got] == [('SCOM', 'approved'), ('KCB', 'vetoed')]


def test_forward_returns_run_close_to_close_and_wait_for_enough_days():
    closes = _closes('2026-09-01', [100, 101, 102, 103, 104, 105, 110])
    assert sc.forward_return(closes, date(2026, 9, 1), 5) == pytest.approx(0.05)
    assert sc.forward_return(closes, date(2026, 9, 5), 5) is None
    # No close on the decision day (a weekend): start from the last one before.
    gap = {date(2026, 9, 4): 100.0, date(2026, 9, 7): 90.0, date(2026, 9, 8): 120.0}
    assert sc.forward_return(gap, date(2026, 9, 5), 1) == pytest.approx(-0.10)
    assert sc.forward_return(gap, date(2026, 9, 3), 1) is None


def test_a_split_in_the_price_series_is_not_a_return():
    # The series is split-adjusted; the decision price was not. Measuring from
    # the decision price would show a 50% loss on a flat stock.
    adjusted = _closes('2026-09-01', [50.0] * 10)
    card = sc.scorecard([_dec('AAPL', '2026-09-01', 'buy', 'buy', price=100.0)], lambda s: adjusted)
    assert card['horizons']['5']['approved']['avg_return_pct'] == pytest.approx(0.0)


def test_an_evening_us_decision_is_dated_by_new_york_time():
    late = _dec('AAPL', '2026-09-02', 'buy', 'buy', hour=1)   # 21:00 on 1 Sep in New York
    (sig,) = sc.reviewed_signals([late], tz_for=lambda s: 'America/New_York')
    assert sig['day'] == date(2026, 9, 1)
    (utc,) = sc.reviewed_signals([late])
    assert utc['day'] == date(2026, 9, 2)


def test_a_sell_followed_by_a_fall_counts_as_a_gain():
    closes = _closes('2026-09-01', [100] + [90] * 25)
    card = sc.scorecard([_dec('SCOM', '2026-09-01', 'sell', 'sell')], lambda s: closes)
    assert card['horizons']['5']['approved']['avg_return_pct'] == pytest.approx(10.0)


def test_the_verdict_waits_for_enough_vetoes_then_says_whether_they_helped():
    rising = _closes('2026-01-01', [100 + i for i in range(400)])
    falling = _closes('2026-01-01', [400 - i for i in range(400)])
    decisions = []
    for i in range(35):
        day = (date(2026, 1, 1) + timedelta(days=i)).isoformat()
        decisions.append(_dec('UP', day, 'buy', 'buy', price=100 + i))
        decisions.append(_dec('DOWN', day, 'buy', 'hold', price=400 - i, skip='llm_veto'))
    card = sc.scorecard(decisions, lambda s: rising if s == 'UP' else falling)
    assert card['approved'] == 35 and card['vetoed'] == 35
    assert card['horizons']['20']['veto_edge_pct'] > 0
    assert card['verdict'].startswith('The review helps')
    few = sc.scorecard(decisions[:10], lambda s: rising if s == 'UP' else falling)
    assert few['verdict'].startswith('Too early to judge')


def test_the_journal_returns_only_reviewed_buy_and_sell_rows(tmp_path):
    from src.agent.decision_journal import DecisionJournal
    j = DecisionJournal(str(tmp_path / 'd.db'))
    j.record({'symbol': 'A', 'cycle': 1, 'action': 'hold'})
    j.record({'symbol': 'B', 'cycle': 1, 'action': 'buy', 'llm_verdict': {}})
    j.record({'symbol': 'C', 'cycle': 1, 'action': 'buy',
              'llm_verdict': {'action': 'hold', 'reasoning': 'no'}, 'skip_reason': 'llm_veto'})
    j.record({'symbol': 'D', 'cycle': 1, 'action': 'sell',
              'llm_verdict': {'action': 'sell', 'reasoning': 'yes'}})
    assert sorted(r['symbol'] for r in j.reviewed()) == ['C', 'D']
    assert j.reviewed(since='2999-01-01') == []
    j.close()
