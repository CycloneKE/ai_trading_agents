"""The Notifications page's source: GET /api/agent-alerts (agent alerts and
what is wrong right now). Viewers see headlines; only operators see detail."""
import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-dashboard-honesty-tests')

import bcrypt
import pytest

import src.api.auth as auth
from src.agent.alerts import AlertLog, EmailAlerter
from src.api.api_server import TradingAPI
from src.api.auth import create_token

NOW = datetime.now(timezone.utc)


def iso(minutes_ago):
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


@pytest.fixture
def api(tmp_path, monkeypatch):
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({role: {'password': bcrypt.hashpw(b'x', bcrypt.gensalt()).decode(), 'role': role}
                                 for role in ('viewer', 'operator')}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    log = AlertLog(tmp_path / 'alerts.jsonl')
    for entry in (
            {'ts': iso(3 * 24 * 60), 'key': 'heal:api_server:restarted', 'severity': 'warning',
             'subject': 'api_server died and was restarted', 'body': 'It was restarted.', 'status': 'sent'},
            {'ts': iso(60), 'key': 'heal:halt', 'severity': 'critical', 'subject': 'trading has been halted',
             'body': 'Reason: daily loss limit', 'status': 'not_configured'},
            {'ts': iso(10), 'key': 'heal:degraded_prices', 'severity': 'warning', 'subject': '2 symbols have had no real price',
             'body': 'EUR_USD (40 min), SCOM (35 min)', 'status': 'failed'},
            {'ts': iso(20 * 24 * 60), 'key': 'heal:old', 'severity': 'info', 'subject': 'ancient', 'body': '', 'status': 'sent'}):
        log.add(entry)
    healing = SimpleNamespace(status=lambda: {
        'workers': [{'worker': 'nse_scraper', 'alive': False}, {'worker': 'api_server', 'alive': True}],
        'symbols_without_price': {'EUR_USD': 40.0}, 'consecutive_loop_errors': 3,
        'last_loop_error': 'KeyError: x', 'halt_clears_itself': False})
    alerter = EmailAlerter({}, env={}, log=log)
    agent = SimpleNamespace(components={}, config={'test_mode': True}, self_healing=healing, alerter=alerter)
    client = TradingAPI(agent, {}).app.test_client()
    header = lambda role: {'Authorization': f"Bearer {create_token(role, role)}"}
    return client, header


def test_the_operator_gets_recent_alerts_newest_first_with_their_detail(api):
    client, header = api
    data = client.get('/api/agent-alerts', headers=header('operator')).get_json()
    assert [a['key'] for a in data['alerts']] == ['heal:degraded_prices', 'heal:halt', 'heal:api_server:restarted']
    assert data['alerts'][1]['body'] == 'Reason: daily loss limit'
    assert [a['emailed'] for a in data['alerts']] == [False, None, True]      # failed, not set up, sent
    assert data['alerts'][0]['id'] == f"{data['alerts'][0]['ts']}|heal:degraded_prices"


def test_a_viewer_sees_the_headlines_but_not_the_detail(api):
    client, header = api
    data = client.get('/api/agent-alerts', headers=header('viewer')).get_json()
    assert len(data['alerts']) == 3 and all('body' not in a for a in data['alerts'])
    assert data['alerts'][1]['subject'] == 'trading has been halted'
    assert data['live']['last_loop_error'] == ''


def test_what_is_wrong_right_now_comes_with_it_and_the_window_can_widen(api):
    client, header = api
    live = client.get('/api/agent-alerts', headers=header('operator')).get_json()['live']
    assert live == {'workers_down': ['nse_scraper'], 'symbols_without_price': {'EUR_USD': 40.0},
                    'loop_errors': 3, 'last_loop_error': 'KeyError: x', 'halt_clears_itself': False}
    wide = client.get('/api/agent-alerts?days=30', headers=header('operator')).get_json()['alerts']
    assert 'heal:old' in [a['key'] for a in wide]
    assert len(client.get('/api/agent-alerts?limit=1', headers=header('operator')).get_json()['alerts']) == 1
    assert client.get('/api/agent-alerts?days=abc', headers=header('operator')).status_code == 200


def test_it_says_whether_email_is_on_and_needs_a_login(api):
    client, header = api
    assert client.get('/api/agent-alerts', headers=header('viewer')).get_json()['email']['configured'] is False
    assert client.get('/api/agent-alerts').status_code in (401, 403)


def test_an_agent_without_self_healing_still_answers(tmp_path, monkeypatch):
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'operator': {'password': bcrypt.hashpw(b'x', bcrypt.gensalt()).decode(), 'role': 'operator'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    monkeypatch.setattr('src.utils.paths.DATA_DIR', tmp_path)
    client = TradingAPI(SimpleNamespace(components={}, config={'test_mode': True}), {}).app.test_client()
    data = client.get('/api/agent-alerts', headers={'Authorization': f"Bearer {create_token('operator', 'operator')}"}).get_json()
    assert data['alerts'] == [] and data['live']['workers_down'] == []
