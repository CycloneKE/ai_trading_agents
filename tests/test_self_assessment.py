"""The daily AI self-review (self_assessment.py): advice only, and only on
evidence. The proposals below are the ones the unchecked review put in the
operator's approval queue in September 2026."""
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


class _Journal:
    def __init__(self, executed, total=40):
        self.rows = [{'symbol': 'AAPL', 'action': 'buy' if i < executed else 'hold', 'price': 100.0 + i,
                      'ts': f'2026-09-27T10:{i:02d}:00', 'executed': 1 if i < executed else 0,
                      'skip_reason': None} for i in range(total)]

    def recent(self, limit):
        return self.rows[:limit]


class _Orders:
    def filled_orders(self):
        return []


@pytest.fixture
def engine(tmp_path):
    llm = MagicMock()
    llm.enabled = True
    llm.propose_json.return_value = SEEN
    em = MagicMock()
    e = SelfAssessmentEngine(llm, em, json.loads(json.dumps(CONFIG)), str(tmp_path / 'improvements.db'))
    yield e
    e.close()


def test_the_review_waits_for_real_trades_and_asks_no_ai_until_then(engine):
    review = engine.run_assessment(_Journal(executed=3), _Orders(), None)
    assert review['skipped'] == '3 trades in the period reviewed; the review waits for 20 before suggesting changes'
    engine.llm.propose_json.assert_not_called()
    assert engine.latest_review()['skipped'].startswith('3 trades')
    assert engine.latest_review()['suggestions'] == []


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
    review = engine.run_assessment(_Journal(executed=25), _Orders(), None)
    assert review['skipped'] is None and len(review['suggestions']) == 2
    context = json.loads(engine.llm.propose_json.call_args[0][1])
    assert context['executed_trades'] == 25
    assert context['current_settings']['risk_limits.trailing_stop_pct'] == 0.03
    assert engine.config == before                                        # nothing applied
    engine.escalation_manager.create_escalation.assert_not_called()       # nothing queued
    latest = engine.latest_review()
    assert latest['executed_trades'] == 25 and latest['suggestions'][0]['proposed'] == 0.035
    assert latest['at'].endswith('+00:00')
    assert 'apply_improvements' not in dir(engine)


def test_a_review_the_ai_did_not_answer_says_so(engine):
    engine.llm.propose_json.return_value = None
    assert engine.run_assessment(_Journal(executed=25), _Orders(), None)['skipped'] == 'the AI did not answer'


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
    engine.run_assessment(_Journal(executed=25), _Orders(), None)
    client, headers = _api(tmp_path, monkeypatch, EscalationManager(str(tmp_path / 'esc.db')), engine)
    body = client.get('/api/ai/health', headers=headers).get_json()
    assert body['review']['suggestions'][1]['target'] == 'risk_limits.stop_loss_pct'


def test_analyze_prediction_accuracy(engine):
    decisions = [
        {"symbol": "SCOM", "action": "buy", "price": 30.0, "ts": "2026-07-09T10:00:00", "executed": 1, "skip_reason": None},
        {"symbol": "SCOM", "action": "hold", "price": 31.0, "ts": "2026-07-09T10:01:00", "executed": 0, "skip_reason": None},
        {"symbol": "KCB", "action": "buy", "price": 80.0, "ts": "2026-07-09T10:00:00", "executed": 1, "skip_reason": None},
        {"symbol": "KCB", "action": "hold", "price": 79.0, "ts": "2026-07-09T10:01:00", "executed": 0, "skip_reason": None},
        {"symbol": "EQTY", "action": "buy", "price": 50.0, "ts": "2026-07-09T10:00:00", "executed": 0,
         "skip_reason": "llm_veto", "per_strategy_json": "{'momentum': {'action': 'buy'}}"},
        {"symbol": "EQTY", "action": "hold", "price": 48.0, "ts": "2026-07-09T10:01:00", "executed": 0, "skip_reason": None},
    ]
    metrics = engine._analyze_prediction_accuracy(decisions, [])
    assert metrics["buy_signals_count"] == 2 and metrics["buy_accuracy_pct"] == 50.0
    assert metrics["llm_vetos_count"] == 1 and metrics["veto_accuracy_pct"] == 100.0
