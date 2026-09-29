"""Closed round trips (round_trips.py): one definition of a closed trade for
the self-review and for the dashboard. The dashboard's Agent Track Record used
to read a list the agent never filled, so it said 0 closed trades and no win
rate however many had closed."""
import os
from types import SimpleNamespace

os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-dashboard-honesty-tests')

import bcrypt
import pytest

from src.agent.round_trips import closed_round_trips, trade_counts


def fill(side, price, minute, symbol='AAPL', strategy='ensemble', qty=10.0):
    return {'symbol': symbol, 'side': side, 'filled_quantity': qty, 'filled_avg_price': price,
            'strategy': strategy, 'created_at': f'2026-09-29T{minute // 60:02d}:{minute % 60:02d}:00'}


def test_only_a_sell_of_something_held_closes_a_trade():
    assert closed_round_trips([fill('buy', 100, 0), fill('buy', 101, 1)]) == []
    assert closed_round_trips([fill('sell', 110, 0)]) == []                     # nothing held: no short
    [trip] = closed_round_trips([fill('buy', 100, 0), fill('sell', 110, 60)], {})
    assert trip['symbol'] == 'AAPL' and trip['exit'] == 'signal' and trip['ret'] > 0.09


def test_a_sell_closing_two_lots_is_two_trades_matched_oldest_first():
    fills = [fill('buy', 100, 0), fill('buy', 120, 30), fill('sell', 110, 60, qty=20)]
    trips = closed_round_trips(fills, {})
    assert [t['ret'] > 0 for t in trips] == [True, False]                       # 100 -> 110 wins, 120 -> 110 loses


def test_forced_exits_are_named_and_costs_count():
    fills = [fill('buy', 100, 0, symbol='BTC-USD'), fill('sell', 100, 30, symbol='BTC-USD', strategy='stop_loss')]
    [trip] = closed_round_trips(fills, {})
    assert trip['exit'] == 'stop_loss' and trip['ret'] < 0                      # a flat trade loses to costs
    assert trip['days'] == pytest.approx(30 / 1440, abs=1e-6)


def test_the_headline_counts():
    assert trade_counts([]) == {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0.0}
    counts = trade_counts([{'ret': 0.02}, {'ret': -0.01}, {'ret': 0.0}, {'ret': 0.05}])
    assert counts == {'total_trades': 4, 'winning_trades': 2, 'losing_trades': 2, 'win_rate': 0.5}


# ---- the dashboard's numbers

class _Journal:
    def __init__(self, fills):
        self.fills = fills

    def filled_orders(self):
        return self.fills


def _client(tmp_path, monkeypatch, journal, analytics_metrics=None):
    import src.api.auth as auth
    from src.api.api_server import TradingAPI
    from src.api.auth import create_token
    import json
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'tester': {
        'password': bcrypt.hashpw(b'unused', bcrypt.gensalt()).decode(), 'role': 'operator'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    report = {'metrics': dict(analytics_metrics or {'total_return': 0.0, 'total_trades': 0, 'win_rate': 0,
                                                      'sharpe_ratio': 0.4, 'max_drawdown': -0.01}),
              'summary': {'current_value': 100000.0}}
    agent = SimpleNamespace(
        components={'nse_order_queue': None, 'nse_paper_account': None, 'data_manager': None},
        config={'data_manager': {'symbols': [], 'nse_symbols': []}}, order_journal=journal,
        performance_analytics=SimpleNamespace(generate_performance_report=lambda period: report))
    api = TradingAPI(agent, {})
    return api.app.test_client(), {'Authorization': f"Bearer {create_token('tester', 'operator')}"}


def test_the_track_record_counts_the_trades_that_closed(tmp_path, monkeypatch):
    fills = [fill('buy', 100, 0), fill('sell', 110, 60),                        # a win
             fill('buy', 50, 90, symbol='COOP'), fill('sell', 45, 120, symbol='COOP', strategy='stop_loss'),
             fill('buy', 20, 150, symbol='EQTY')]                               # still open
    client, headers = _client(tmp_path, monkeypatch, _Journal(fills))
    body = client.get('/api/performance', headers=headers).get_json()
    assert body['total_trades'] == body['metrics']['total_trades'] == 2
    assert body['win_rate'] == body['metrics']['win_rate'] == 0.5
    assert body['metrics']['winning_trades'] == 1 and body['metrics']['losing_trades'] == 1
    assert body['metrics']['sharpe_ratio'] == 0.4                               # still from the equity curve


def test_the_dashboard_keeps_zero_when_nothing_has_closed_and_survives_no_journal(tmp_path, monkeypatch):
    client, headers = _client(tmp_path, monkeypatch, _Journal([fill('buy', 100, 0)]))
    body = client.get('/api/performance', headers=headers).get_json()
    assert body['total_trades'] == 0 and body['win_rate'] == 0
    client, headers = _client(tmp_path, monkeypatch, None, {'total_return': 0.0, 'total_trades': 7, 'win_rate': 0.3})
    body = client.get('/api/performance', headers=headers).get_json()
    assert body['total_trades'] == 7                                            # no journal: the engine's own figure
