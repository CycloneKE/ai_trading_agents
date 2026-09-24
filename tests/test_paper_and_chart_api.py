"""The NSE paper account view and the price charts (step A of the dashboard)."""
import csv
import json
import os
from types import SimpleNamespace

os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-dashboard-honesty-tests')

import bcrypt
import pytest

import src.api.auth as auth
from src.agent import chart_data
from src.agent.nse_order_queue import NseOrderQueue
from src.agent.nse_paper_account import NsePaperAccount
from src.api.api_server import TradingAPI
from src.api.auth import create_token

COSTS = {'nse': {'commission_pct': 0.017, 'min_commission': 0.0, 'slippage_pct': 0.003}}
CONFIG = {'nse_paper_trading': {'starting_capital_kes': 200000}, 'costs': COSTS,
          'data_manager': {'symbols': ['AAPL', 'BTC-USD'], 'crypto_symbols': ['BTC-USD'],
                           'nse_symbols': ['SCOM', 'KCB']}}


@pytest.fixture(autouse=True)
def _a_real_user(tmp_path, monkeypatch):
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'tester': {
        'password': bcrypt.hashpw(b'unused', bcrypt.gensalt()).decode(), 'role': 'operator'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))


@pytest.fixture
def queue(tmp_path):
    q = NseOrderQueue(str(tmp_path / 'escalations.db'))
    yield q
    q.close()


class _NSE:
    def __init__(self, prices):
        self.prices = prices

    def get_quote(self, symbol):
        return {'symbol': symbol, 'price_kes': self.prices.get(symbol), 'source': 'nse_ticker'}


def _client(queue, paper=None, nse=None, journal=None, volatility=None):
    agent = SimpleNamespace(
        components={'nse_order_queue': queue, 'nse_paper_account': paper,
                    'data_manager': SimpleNamespace(connectors={'nse': nse}) if nse else None},
        config=CONFIG, order_journal=journal, volatility=volatility)
    api = TradingAPI(agent, {})
    headers = {'Authorization': f"Bearer {create_token('tester', 'operator')}"}
    return api.app.test_client(), headers


def _buy(paper, queue, symbol='SCOM', price=36.2, notional=50000):
    qty, _ = paper.plan(symbol, 'buy', price, notional)
    tid = queue.create_ticket(symbol, 'buy', qty, suggested_limit_price=price, strategy='ensemble')
    paper.fill(tid, 'buy', price, qty)
    return qty


# ------------------------------------------------------------ paper view

def test_a_new_account_shows_its_capital_and_where_the_curve_starts(queue):
    paper = NsePaperAccount(queue, CONFIG)
    client, h = _client(queue, paper, _NSE({}))
    data = client.get('/api/nse/paper', headers=h).get_json()
    assert data['enabled'] is True
    assert data['account']['cash_kes'] == 200000 and data['account']['holdings'] == []
    assert [p['equity_kes'] for p in data['equity_curve']] == [200000]
    assert data['rules']['stop_loss']['stop_loss_pct'] == 0.08


def test_holdings_carry_their_value_profit_and_stops(queue):
    paper = NsePaperAccount(queue, CONFIG)
    qty = _buy(paper, queue)
    tracker = SimpleNamespace(atr_pct=lambda s: None)
    client, h = _client(queue, paper, _NSE({'SCOM': 40.0}), volatility=tracker)
    data = client.get('/api/nse/paper', headers=h).get_json()
    (hold,) = data['account']['holdings']
    assert hold['symbol'] == 'SCOM' and hold['quantity'] == qty
    assert hold['market_value_kes'] == pytest.approx(qty * 40.0)
    assert hold['unrealised_pnl_pct'] > 0
    assert hold['stops']['stop_loss_kes'] == pytest.approx(hold['avg_cost_kes'] * 0.92, abs=0.01)
    assert hold['stops']['trailing_stop_kes'] == pytest.approx(36.31 * 0.9, abs=0.01)
    (fill,) = data['fills']
    assert (fill['side'], fill['quantity'], fill['strategy']) == ('buy', qty, 'ensemble')


def test_the_equity_curve_keeps_one_reading_a_day(queue):
    paper = NsePaperAccount(queue, CONFIG)
    _buy(paper, queue)
    paper.record_equity({'SCOM': 37.0})
    paper.record_equity({'SCOM': 38.0})
    (today,) = queue.paper_equity_history()
    assert today['holdings_kes'] > 0 and today['equity_kes'] == pytest.approx(
        today['cash_kes'] + today['holdings_kes'], abs=0.01)


def test_without_a_paper_account_the_view_says_so(queue):
    client, h = _client(queue, None)
    assert client.get('/api/nse/paper', headers=h).get_json() == {'enabled': False}


def test_the_paper_view_needs_a_login(queue):
    client, _ = _client(queue, NsePaperAccount(queue, CONFIG))
    assert client.get('/api/nse/paper').status_code == 401


# ----------------------------------------------------------------- charts

def _csv(folder, symbol, rows):
    fields = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'change_pct', 'source']
    with open(folder / f'{symbol}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({'symbol': symbol, **r})


def test_nse_bars_are_real_rows_only_oldest_first(tmp_path):
    _csv(tmp_path, 'SCOM', [
        {'date': '2026-09-23', 'open': 36.6, 'high': 37, 'low': 36, 'close': 36.2, 'volume': 5, 'source': 'nse_pricelist'},
        {'date': '2026-09-19', 'close': 99, 'source': 'synthetic'},
        {'date': '2026-09-22', 'open': 36.5, 'high': 37.0, 'low': 36.4, 'close': 36.6, 'volume': 3, 'source': 'nse_pricelist'},
        {'date': '2026-09-24', 'open': '', 'high': '', 'low': '', 'close': 36.1, 'volume': 1, 'source': 'nse_ticker'},
    ])
    bars = chart_data.nse_bars('SCOM', tmp_path)
    assert [b['time'] for b in bars] == ['2026-09-22', '2026-09-23', '2026-09-24']
    assert bars[-1] == {'time': '2026-09-24', 'open': 36.1, 'high': 36.1, 'low': 36.1,
                        'close': 36.1, 'volume': 1}


def test_markers_say_what_each_fill_did():
    fills = [{'time': '2026-09-01T10:00', 'side': 'buy', 'quantity': 100, 'price': 10},
             {'time': '2026-09-05T10:00', 'side': 'buy', 'quantity': 50, 'price': 11},
             {'time': '2026-09-08T10:00', 'side': 'sell', 'quantity': 75, 'price': 12},
             {'time': '2026-09-10T10:00', 'side': 'sell', 'quantity': 75, 'price': 12.5},
             {'time': '2026-09-11T10:00', 'side': 'buy', 'quantity': 10, 'price': 12},
             {'time': '2026-09-12T10:00', 'side': 'sell', 'quantity': 10, 'price': 11,
              'strategy': 'stop_loss'}]
    assert [m['kind'] for m in chart_data.markers(fills)] == ['buy', 'add', 'trim', 'sell', 'buy', 'stop']
    assert chart_data.markers(fills)[0]['time'] == '2026-09-01'


@pytest.mark.parametrize('symbol, market, tv', [('SCOM', 'nse', 'NSEKE:SCOM'),
                                                 ('BTC-USD', 'crypto', 'COINBASE:BTCUSD'),
                                                 ('AAPL', 'us_equity', 'AAPL')])
def test_tradingview_symbols(symbol, market, tv):
    assert chart_data.tradingview_symbol(symbol, market) == tv


def test_an_nse_chart_has_real_bars_and_the_paper_accounts_trades(queue, tmp_path, monkeypatch):
    import src.connectors.nse_connector as nse_connector
    monkeypatch.setattr(nse_connector, 'NSE_CSV_DIR', tmp_path)
    _csv(tmp_path, 'SCOM', [{'date': '2026-09-23', 'close': 36.2, 'source': 'nse_pricelist'}])
    paper = NsePaperAccount(queue, CONFIG)
    _buy(paper, queue)
    client, h = _client(queue, paper, _NSE({}))
    data = client.get('/api/chart/SCOM', headers=h).get_json()
    assert (data['currency'], data['tradingview_symbol']) == ('KES', 'NSEKE:SCOM')
    assert [b['close'] for b in data['bars']] == [36.2]
    assert [m['kind'] for m in data['markers']] == ['buy']


def test_a_us_chart_uses_the_live_feeds_bars_and_the_journal(queue, monkeypatch):
    monkeypatch.setattr(chart_data, 'yfinance_bars', lambda s, period='2y': [
        {'time': '2026-09-23', 'open': 1, 'high': 2, 'low': 1, 'close': 2, 'volume': 9}])
    journal = SimpleNamespace(orders_for_symbol=lambda s, limit=500: [
        {'status': 'filled', 'filled_quantity': 2, 'filled_avg_price': 1.5, 'side': 'buy',
         'created_at': '2026-09-23T14:00', 'strategy': 'ensemble'},
        {'status': 'submitted', 'filled_quantity': None, 'side': 'sell', 'created_at': 'x'}])
    client, h = _client(queue, None, journal=journal)
    data = client.get('/api/chart/AAPL', headers=h).get_json()
    assert data['currency'] == 'USD' and len(data['bars']) == 1
    assert [(m['kind'], m['price']) for m in data['markers']] == [('buy', 1.5)]


def test_an_untracked_symbol_has_no_chart(queue):
    client, h = _client(queue, None)
    assert client.get('/api/chart/ZZZZ', headers=h).status_code == 404


# ------------------------------------------------------------- drill-down

def test_the_drilldown_prices_an_nse_stock_from_the_nse_feed(queue):
    """KCB on the US feed is a different company; the drill-down asked it anyway."""
    us_feed = SimpleNamespace(get_real_time_data=lambda s: {'price': 999.0})
    agent = SimpleNamespace(
        components={'nse_order_queue': queue, 'nse_paper_account': None,
                    'data_manager': SimpleNamespace(connectors={'nse': _NSE({'KCB': 41.5}),
                                                                'real_data': us_feed})},
        config=CONFIG, order_journal=None, volatility=None, decision_journal=None)
    client = TradingAPI(agent, {}).app.test_client()
    h = {'Authorization': f"Bearer {create_token('tester', 'operator')}"}
    assert client.get('/api/symbol/KCB', headers=h).get_json()['current_price'] == 41.5
    assert client.get('/api/symbol/AAPL', headers=h).get_json()['current_price'] == 999.0
