"""Research that earns its place.

Before: broker ratings were never scored, never expired and reached the AI
with no date; and a free second model could veto Claude's approved trades
with no record that its vetoes helped.
"""
import json
import sqlite3
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from src.agent import broker_calls as bc
from src.agent.escalation_manager import EscalationManager


# ------------------------------------------------------------ broker calls

def test_ratings_map_to_a_direction_or_none():
    assert bc.direction('Buy') == 'buy' and bc.direction('ACCUMULATE') == 'buy'
    assert bc.direction('Outperform') == 'buy' and bc.direction('strong-buy') == 'buy'
    assert bc.direction('Market Perform') is None
    assert bc.direction('REDUCE') == 'sell' and bc.direction('strong sell') == 'sell'
    assert bc.direction('HOLD') is None and bc.direction('NEUTRAL') is None and bc.direction(None) is None


def test_a_sheet_repeating_a_rating_counts_once():
    calls = [{'symbol': 'kcb', 'recommendation': 'BUY', 'day': '2026-06-01', 'source': 'AIB'},
             {'symbol': 'KCB', 'recommendation': 'Buy', 'day': '2026-06-01', 'source': 'AIB'},
             {'symbol': 'KCB', 'recommendation': 'BUY', 'day': '2026-06-02', 'source': 'AIB'},
             {'symbol': 'KCB', 'recommendation': 'HOLD', 'day': '2026-06-03', 'source': 'AIB'},
             {'symbol': 'KCB', 'recommendation': 'BUY', 'day': 'n/a', 'source': 'AIB'}]
    sigs = bc.to_signals(calls, {'data_manager': {'nse_symbols': ['KCB']}})
    assert [(s['day'].isoformat(), s['action'], s['market']) for s in sigs] == [
        ('2026-06-01', 'buy', 'nse'), ('2026-06-02', 'buy', 'nse')]


def _trading_days(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_publishers_are_scored_against_what_prices_then_did():
    days = _trading_days(date(2026, 1, 5), 200)
    rising = {d: 100 * (1.002 ** i) for i, d in enumerate(days)}
    falling = {d: 100 * (0.998 ** i) for i, d in enumerate(days)}
    closes = {'UP': rising, 'DOWN': falling}
    calls = ([{'symbol': 'UP', 'recommendation': 'BUY', 'day': days[i].isoformat(), 'source': 'Good Research'}
              for i in range(0, 50)] +
             [{'symbol': 'UP', 'recommendation': 'SELL', 'day': days[i].isoformat(), 'source': 'Bad Research'}
              for i in range(0, 50)] +
             [{'symbol': 'DOWN', 'recommendation': 'BUY', 'day': days[i].isoformat(), 'source': 'Bad Research'}
              for i in range(0, 50)])
    board = bc.scoreboard(bc.to_signals(calls), lambda s: closes[s], min_calls=20)
    at60 = {r['source']: r for r in board['horizons']['60']}
    assert at60['Good Research']['hit_rate'] == 1.0 and at60['Good Research']['enough']
    assert at60['Bad Research']['hit_rate'] == 0.0
    line = bc.record_line(board, 'Good Research')
    assert line.startswith('of 50 past Good Research calls, 100% moved the called way over 60 trading days')
    assert bc.record_line(board, 'Nobody') is None


def test_a_short_record_is_not_shown_to_the_ai():
    days = _trading_days(date(2026, 1, 5), 200)
    closes = {d: 100 * (1.002 ** i) for i, d in enumerate(days)}
    calls = [{'symbol': 'UP', 'recommendation': 'BUY', 'day': days[i].isoformat(), 'source': 'New Broker'}
             for i in range(5)]
    board = bc.scoreboard(bc.to_signals(calls), lambda s: closes, min_calls=20)
    assert board['horizons']['60'][0]['n'] == 5
    assert bc.record_line(board, 'New Broker') is None


def test_calls_are_read_from_uploaded_notes_with_their_publisher(tmp_path):
    em = EscalationManager(str(tmp_path / 'esc.db'))
    upload = em.record_upload('ms.pdf', 'Morgan Stanley')
    em.record_signal(upload, {'symbol': 'NVDA', 'market': 'international', 'current_price': 120.0,
                              'target_price': 150.0, 'recommendation': 'BUY', 'rationale': 'demand'})
    calls = bc.note_calls(str(tmp_path / 'esc.db'))
    assert [(c['symbol'], c['recommendation'], c['source']) for c in calls] == [('NVDA', 'BUY', 'Morgan Stanley')]
    assert date.fromisoformat(calls[0]['day'][:10]) == datetime.utcnow().date()


def test_scores_refresh_in_the_background_and_answer_from_the_latest(tmp_path):
    days = _trading_days(date(2026, 1, 5), 200)
    closes = {d: 100 * (1.002 ** i) for i, d in enumerate(days)}
    calls = [{'symbol': 'UP', 'recommendation': 'BUY', 'day': days[i].isoformat(), 'source': 'Good'}
             for i in range(30)]
    clock = {'t': 1000.0}
    scores = bc.BrokerCallScores({}, str(tmp_path / 'esc.db'), closes_for=lambda s: closes,
                                 calls=lambda: calls, clock=lambda: clock['t'])
    assert scores.maybe_refresh() is True
    scores._thread.join(10)
    assert 'of 30 past Good calls' in scores.record_for('Good')
    assert scores.maybe_refresh() is False            # fresh: not again for 12 hours
    clock['t'] += 13 * 3600
    assert scores.maybe_refresh() is True
    scores._thread.join(10)


# ---------------------------------------------------- ratings expire, with dates

def test_old_ratings_stop_reaching_the_ai(tmp_path):
    em = EscalationManager(str(tmp_path / 'esc.db'))
    em.add_to_watchlist('KCB', 'kenyan', 'AIB-AXYS', 'BUY', 50.0, 'cheap')
    em.add_to_watchlist('EQTY', 'kenyan', 'AIB-AXYS', 'BUY', 60.0, 'old call')
    old = (datetime.utcnow() - timedelta(days=120)).isoformat()
    em._conn.execute("UPDATE position_watchlist SET added_at = ?, last_updated = ? WHERE symbol = 'EQTY'",
                     (old, old))
    em._conn.commit()
    assert {w['symbol'] for w in em.get_active_watchlist()} == {'KCB', 'EQTY'}       # the dashboard's list
    assert {w['symbol'] for w in em.get_active_watchlist(max_age_days=90)} == {'KCB'}


def test_the_ai_sees_when_a_rating_was_given_and_the_publishers_record(monkeypatch):
    from src.agent.llm_orchestrator import LLMOrchestrator
    from src.agent.main import TradingAgent
    agent = SimpleNamespace(config={'research_ingest': {'rating_max_age_days': 90}},
                            broker_scores=SimpleNamespace(record_for=lambda s: 'of 40 past AIB-AXYS calls, 60% ...'))
    assert TradingAgent._rating_max_age(agent) == 90
    ctx = TradingAgent._dated_rating(agent, {'symbol': 'KCB', 'source': 'AIB-AXYS', 'recommendation': 'BUY',
                                             'target_price': 50.0, 'rationale': 'cheap',
                                             'added_at': '2026-09-01T10:00:00', 'last_updated': '2026-10-02T09:00:00'})
    assert ctx['rated_on'] == '2026-10-02' and ctx['source_record'].startswith('of 40 past')

    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    orch = LLMOrchestrator({'primary_llm_provider': 'gemini'})
    seen = {}
    monkeypatch.setattr(orch, '_complete', lambda s, u, fb, model_override=None, purpose='volume':
                        seen.setdefault('user', u) and {'action': 'buy', 'confidence': 0.6, 'reasoning': 'ok'})
    orch.validate_trade('KCB', {'action': 'buy', 'confidence': 0.6}, {'close': 45.0}, research_context=ctx)
    assert 'Broker Research Context (AIB-AXYS, rated 2026-10-02): recommendation BUY' in seen['user']
    assert 'Broker track record: of 40 past AIB-AXYS calls' in seen['user']


# ------------------------------------------- the second opinion, measured

def test_the_scorecard_compares_agreed_and_disputed_trades():
    from src.agent.ai_scorecard import scorecard
    days = _trading_days(date(2026, 1, 5), 60)
    closes = {'GOOD': {d: 100 * (1.01 ** i) for i, d in enumerate(days)},
              'BAD': {d: 100 * (0.99 ** i) for i, d in enumerate(days)}}

    def decision(symbol, day, agreed):
        return {'symbol': symbol, 'ts': f'{day.isoformat()}T15:00:00', 'action': 'buy', 'price': 100.0,
                'skip_reason': None, 'llm_verdict': {'action': 'buy', 'reasoning': 'ok',
                                                     'consensus_verified': agreed}}
    decisions = [decision('GOOD', days[i], True) for i in range(30)] + \
                [decision('BAD', days[i], False) for i in range(30)]
    card = scorecard(decisions, lambda s: closes[s], horizons=(5, 20))
    row = card['consensus']['20']
    assert row['agreed']['n'] == 30 and row['disputed']['n'] == 30
    assert row['dispute_edge_pct'] > 0
    assert card['consensus_verdict'].startswith('Disputed trades did')
    assert 'worth listening to' in card['consensus_verdict']
