"""The AI self-review (self_assessment.py): advice only, only on closed trades,
and only when new ones have closed since it last looked. The proposals below
are the ones the unchecked review put in the operator's approval queue in
September 2026, with no trades to judge by."""
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-dashboard-honesty-tests')

import bcrypt
import pytest

from src.agent import self_assessment as sa
from src.agent.escalation_manager import EscalationManager
from src.agent.self_assessment import SelfAssessmentEngine

CONFIG = {
    "risk_limits": {"stop_loss_pct": 0.05, "trailing_stop_pct": 0.03,
                    "stop_loss_atr_mult": 2.5, "trailing_stop_atr_mult": 3.0},
    "execution": {"min_order_size": 100, "max_order_size": 10000},
    "self_assessment": {"assessment_interval": 390},
}

# What the unchecked review proposed, with no trades to judge by.
SEEN = {
    "parameter_adjustments": [
        {"target": "risk_limits.trailing_stop_pct", "proposed": 0.05, "reasoning": "widen"},
        {"target": "risk_limits.stop_loss_pct", "proposed": 0.08, "reasoning": "wider"},
        {"target": "risk_limits.trailing_stop_pct", "proposed": 0.035, "reasoning": "lock in gains"},
        {"target": "risk_limits.trailing_stop_pct", "proposed": 0.02, "reasoning": "tighter"},
        {"target": "risk_limits.stop_loss_pct", "proposed": 0.04, "reasoning": "cut losses earlier"},
        {"target": "strategy_weights.trend_following", "proposed": 0.5, "reasoning": "new component"},
        {"target": "execution.max_order_size", "proposed": 50000, "reasoning": "bigger"},
        {"target": "risk_limits.stop_loss_atr_mult", "proposed": "wide", "reasoning": "?"},
    ],
    "symbol_actions": [{"symbol": "ALL", "action": "pause", "reasoning": "0% accuracy"},
                       {"symbol": "ABC", "action": "remove", "reasoning": "slippage"},
                       {"symbol": "DEF", "action": "add", "reasoning": "mean reversion"}],
    "reasoning_summary": "No trades were executed.",
}


def _fill(side, price, minute, symbol='AAPL', strategy='ensemble', qty=10.0):
    return {'symbol': symbol, 'side': side, 'filled_quantity': qty, 'filled_avg_price': price,
            'strategy': strategy, 'created_at': f'2026-09-2{minute // 1440 + 8}T{(minute % 1440) // 60:02d}:{minute % 60:02d}:00'}


def _round_trips(n, win=110.0, exit_strategy='ensemble', start=0):
    """n closed round trips, each a buy at 100 and a sell at `win`, an hour apart."""
    rows = []
    for i in range(n):
        t = start + i * 120
        rows += [_fill('buy', 100.0, t), _fill('sell', win, t + 60, strategy=exit_strategy)]
    return rows


class _Orders:
    """The order journal, as far as the review reads it."""

    def __init__(self, fills=()):
        self.fills = list(fills)

    def filled_orders(self):
        return list(self.fills)


@pytest.fixture
def engine(tmp_path):
    llm = MagicMock()
    llm.enabled = True
    llm.propose_json.return_value = SEEN
    em = MagicMock()
    e = SelfAssessmentEngine(llm, em, json.loads(json.dumps(CONFIG)), str(tmp_path / 'improvements.db'))
    yield e
    e.close()


def test_the_review_waits_for_closed_trades_and_asks_no_ai_until_then(engine):
    review = engine.run_assessment(None, _Orders(_round_trips(3)), None)
    assert review['skipped'] == '3 closed trades so far; the review waits for 15 before suggesting changes'
    engine.llm.propose_json.assert_not_called()
    assert engine.latest_review()['skipped'].startswith('3 closed trades')
    assert engine.latest_review()['suggestions'] == []
    # Buys alone are not trades: fifty open positions is still no evidence.
    only_buys = [_fill('buy', 100.0, i) for i in range(50)]
    assert engine.run_assessment(None, _Orders(only_buys), None)['closed_trades'] == 0
    engine.llm.propose_json.assert_not_called()


def test_only_existing_stop_settings_within_range_are_suggested():
    accepted, rejected = sa.check_proposals(SEEN, CONFIG)
    assert [(a['target'], a['current'], a['proposed']) for a in accepted] == [
        ('risk_limits.trailing_stop_pct', 0.03, 0.035),
        ('risk_limits.stop_loss_pct', 0.05, 0.04)]
    reasons = {(r['target'], str(r['proposed'])): r['reason'] for r in rejected}
    assert reasons[('risk_limits.trailing_stop_pct', '0.05')] == 'a change of more than half the current value'
    assert reasons[('risk_limits.stop_loss_pct', '0.08')] == 'a change of more than half the current value'
    assert reasons[('risk_limits.trailing_stop_pct', '0.02')] == 'suggested twice'
    assert reasons[('strategy_weights.trend_following', '0.5')] == 'not a setting the review may change'
    assert reasons[('execution.max_order_size', '50000')] == 'not a setting the review may change'
    assert reasons[('risk_limits.stop_loss_atr_mult', 'wide')] == 'no number given'
    assert reasons[('pause ALL', 'None')] == reasons[('add DEF', 'None')] == 'symbol changes are not part of the review'
    out_of_range = {'parameter_adjustments': [{'target': 'risk_limits.stop_loss_pct', 'proposed': 0.5}]}
    assert sa.check_proposals(out_of_range, CONFIG)[1][0]['reason'] == 'outside the allowed range 0.02 to 0.15'


def test_with_evidence_the_review_is_advice_and_changes_nothing(engine):
    before = json.loads(json.dumps(engine.config))
    review = engine.run_assessment(None, _Orders(_round_trips(16)), None)
    assert review['skipped'] is None and len(review['suggestions']) == 2
    context = json.loads(engine.llm.propose_json.call_args[0][1])
    assert context['closed_trades'] == 16 and 'executed_trades' not in context
    assert context['current_settings']['risk_limits.trailing_stop_pct'] == 0.03
    assert context['win_rate_pct'] == 100.0 and context['exits']['signal']['n'] == 16
    assert engine.config == before                                        # nothing applied
    engine.escalation_manager.create_escalation.assert_not_called()       # nothing queued
    latest = engine.latest_review()
    assert latest['closed_trades'] == 16 and latest['suggestions'][0]['proposed'] == 0.035
    assert latest['at'].endswith('+00:00')
    assert 'apply_improvements' not in dir(engine)


def test_a_review_the_ai_did_not_answer_says_so(engine):
    engine.llm.propose_json.return_value = None
    assert engine.run_assessment(None, _Orders(_round_trips(16)), None)['skipped'] == 'the AI did not answer'


def test_without_an_ai_the_review_says_so_and_asks_nothing(engine):
    engine.llm.enabled = False
    review = engine.run_assessment(None, _Orders(_round_trips(16)), None)
    assert review['skipped'] == 'no AI service is set up'
    engine.llm.propose_json.assert_not_called()


# ---- what the AI is shown: closed trades, not the next journal row's price

def test_closed_trades_are_matched_first_in_first_out_and_only_sells_close_them():
    fills = [_fill('buy', 100.0, 0, qty=10), _fill('buy', 120.0, 30, qty=10),
             _fill('sell', 110.0, 60, qty=15)]                            # closes lot 1 (+10%) and half of lot 2 (-8.3%)
    ev = sa.closed_trade_evidence(fills, {})
    assert ev['closed_trades'] == 2
    assert ev['win_rate_pct'] == 50.0
    assert ev['avg_win_pct'] > 9.0 and ev['avg_loss_pct'] < -8.0
    assert sa.closed_trade_evidence([_fill('buy', 100.0, 0)], {}) == {'closed_trades': 0}
    assert sa.closed_trade_evidence([_fill('sell', 110.0, 0)], {}) == {'closed_trades': 0}   # nothing held to sell


def test_each_exit_is_tagged_by_what_closed_it():
    fills = (_round_trips(2, win=110.0)
             + _round_trips(3, win=95.0, exit_strategy='stop_loss', start=1000)
             + _round_trips(1, win=104.0, exit_strategy='trailing_stop', start=2000))
    exits = sa.closed_trade_evidence(fills, {})['exits']
    assert {k: v['n'] for k, v in exits.items()} == {'signal': 2, 'stop_loss': 3, 'trailing_stop': 1}
    assert exits['stop_loss']['avg_return_pct'] < 0 < exits['trailing_stop']['avg_return_pct']


def test_returns_are_after_the_costs_a_fill_price_leaves_out():
    """Alpaca's paper fills carry no fee: a flat round trip in BTC-USD is a loss
    once the modelled fee and slippage (0.35% a side) are counted."""
    flat = [_fill('buy', 100.0, 0, symbol='BTC-USD'), _fill('sell', 100.0, 60, symbol='BTC-USD')]
    ev = sa.closed_trade_evidence(flat, {})
    assert ev['expectancy_pct'] < -0.6 and ev['win_rate_pct'] == 0.0
    assert ev['avg_hold_days'] == pytest.approx(60 / 1440, abs=0.01)


# ---- asked when there is something new, and not otherwise

def test_a_repeat_pass_with_no_new_trades_asks_no_ai_and_keeps_the_review(engine):
    fills = _round_trips(16)
    first = engine.run_assessment(None, _Orders(fills), None)
    assert engine.llm.propose_json.call_count == 1 and len(first['suggestions']) == 2
    for _ in range(3):                                                     # e.g. every redeploy resets the cycle count
        again = engine.run_assessment(None, _Orders(fills), None)
        assert again['suggestions'] == first['suggestions'] and again['skipped'] is None
    assert engine.llm.propose_json.call_count == 1
    assert engine.latest_review()['suggestions'] == first['suggestions']  # still on the dashboard


def test_it_asks_again_only_after_enough_new_trades_have_closed(engine):
    engine.run_assessment(None, _Orders(_round_trips(16)), None)
    engine.run_assessment(None, _Orders(_round_trips(20)), None)           # 4 new: under 5
    assert engine.llm.propose_json.call_count == 1
    engine.run_assessment(None, _Orders(_round_trips(21)), None)           # 5 new
    assert engine.llm.propose_json.call_count == 2
    engine.run_assessment(None, _Orders(_round_trips(21)), None)
    assert engine.llm.propose_json.call_count == 2


def test_a_waiting_review_is_recorded_once_not_every_pass(engine):
    fills = _round_trips(3)
    engine.run_assessment(None, _Orders(fills), None)
    engine.run_assessment(None, _Orders(fills), None)
    rows = engine._conn.execute('SELECT COUNT(*) FROM assessments').fetchone()[0]
    assert rows == 1
    engine.run_assessment(None, _Orders(_round_trips(4)), None)            # something new to say
    assert engine._conn.execute('SELECT COUNT(*) FROM assessments').fetchone()[0] == 2


def test_a_failed_ask_is_not_repeated_until_more_trades_close(engine):
    """An AI outage must not turn into a call every cycle."""
    engine.llm.propose_json.side_effect = RuntimeError('provider down')
    fills = _round_trips(16)
    assert engine.run_assessment(None, _Orders(fills), None)['skipped'] == 'the AI did not answer'
    engine.run_assessment(None, _Orders(fills), None)
    assert engine.llm.propose_json.call_count == 1


def test_the_old_unchecked_proposals_are_withdrawn_from_the_queue(tmp_path):
    em = EscalationManager(str(tmp_path / 'esc.db'))
    em.create_escalation(None, 'SYSTEM', 'config_change:risk_limits.stop_loss_pct',
                         'Proposed adjustment: risk_limits.stop_loss_pct to 0.08 (Current: 0.05). Rationale: x', 'high')
    em.create_escalation(None, 'ALL', 'pause_symbol',
                         "Retrospective recommendation: pause tracking for symbol 'ALL'. Rationale: x", 'medium')
    em.create_escalation(None, 'KAPC', 'add_symbol', 'Up 8% today on high volume', 'low')   # universe scout
    upload = em.record_upload('KCB_note.pdf')
    sid = em.record_signal(upload, {'symbol': 'KCB', 'recommendation': 'BUY', 'rationale': 'note'})
    em.create_escalation(sid, 'KCB', 'follow', 'analyst BUY', 'medium')
    reopened = EscalationManager(str(tmp_path / 'esc.db'))                  # runs on every start
    assert sorted(e['symbol'] for e in reopened.get_pending_escalations()) == ['KAPC', 'KCB']
    assert reopened.withdraw_unchecked_ai_proposals() == 0


def _api(tmp_path, monkeypatch, em, engine=None):
    import src.api.auth as auth
    from src.api.api_server import TradingAPI
    from src.api.auth import create_token
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'tester': {
        'password': bcrypt.hashpw(b'unused', bcrypt.gensalt()).decode(), 'role': 'operator'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    agent = SimpleNamespace(components={'escalation_manager': em, 'self_assessment': engine,
                                        'llm_orchestrator': None},
                            config={'data_manager': {'symbols': [], 'nse_symbols': []}})
    api = TradingAPI(agent, {})
    return api.app.test_client(), {'Authorization': f"Bearer {create_token('tester', 'operator')}"}


def test_approving_an_old_proposal_puts_nothing_on_the_watchlist(tmp_path, monkeypatch):
    em = EscalationManager(str(tmp_path / 'esc.db'))
    eid = em.create_escalation(None, 'SYSTEM', 'config_change:risk_limits.stop_loss_pct', 'an old proposal', 'high')
    client, headers = _api(tmp_path, monkeypatch, em)
    r = client.post(f'/api/operator/escalations/{eid}/resolve', json={'status': 'approved'}, headers=headers)
    assert r.status_code == 200
    assert em.get_active_watchlist() == []


def test_the_latest_review_reaches_the_dashboard(tmp_path, monkeypatch, engine):
    engine.run_assessment(None, _Orders(_round_trips(16)), None)
    client, headers = _api(tmp_path, monkeypatch, EscalationManager(str(tmp_path / 'esc.db')), engine)
    body = client.get('/api/ai/health', headers=headers).get_json()
    assert body['review']['suggestions'][1]['target'] == 'risk_limits.stop_loss_pct'
