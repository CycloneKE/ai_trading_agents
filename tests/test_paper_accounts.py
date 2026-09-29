"""The paper accounts page (paper_accounts.py, /api/paper/<kind>): what the
US and crypto account and the forex book are worth against their start, what
they hold and where each holding would be stopped out, what they traded and
made, and what is still working. Each account shows only its own markets."""
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-dashboard-honesty-tests')

import bcrypt
import pytest

from src.agent import paper_accounts as pa
from src.agent.order_journal import OrderJournal
from src.connectors.base_broker import AccountInfo, Position
from src.utils.config_validator import load_config

CONFIG = load_config('config/config.json')
NOW = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)


def position(symbol, qty, avg, last):
    value, cost = qty * last, qty * avg
    return Position(symbol=symbol, quantity=qty, avg_entry_price=avg, current_price=last,
                    market_value=value, unrealized_pl=value - cost,
                    unrealized_pl_percent=(value - cost) / cost, cost_basis=cost)


class Book:
    def __init__(self, positions, cash, equity, name='alpaca', paper=True, connected=True):
        self.is_connected, self.broker_name, self.is_paper_trading = connected, name, paper
        self._positions, self.cash, self.equity = positions, cash, equity

    def get_account_info(self):
        return AccountInfo(account_id='x', cash=self.cash, equity=self.equity, buying_power=self.cash,
                           initial_margin=0, maintenance_margin=0, day_trade_count=0,
                           last_updated=NOW, broker_name=self.broker_name)

    def get_positions(self):
        return self._positions


def fill(journal, coid, symbol, side, qty, price, strategy='ensemble'):
    assert journal.record_intent(coid, symbol, side, qty, 'market', strategy=strategy)
    journal.mark_final(coid, 'filled', qty, price)


@pytest.fixture
def journal(tmp_path):
    j = OrderJournal(db_path=str(tmp_path / 'orders.db'))
    yield j
    j.close()


def stops(symbol):
    return {'stop_loss_pct': 0.05, 'trailing_stop_pct': 0.03, 'source': 'fixed'}


# ---- which account trades what

def test_each_market_belongs_to_one_account():
    assert pa.book_of('AAPL', CONFIG) == 'us' and pa.book_of('BTC-USD', CONFIG) == 'us'
    assert pa.book_of('EUR_USD', CONFIG) == 'forex'
    assert pa.book_of('SCOM', CONFIG) is None                                    # the NSE account has its own page


# ---- the daily history

def test_the_log_keeps_the_last_reading_of_each_day(tmp_path):
    log = pa.PaperEquityLog(str(tmp_path / 'eq.db'))
    day1 = datetime(2026, 9, 29, 9, 0, tzinfo=timezone.utc)
    log.record('forex', 10000, 10000, now=day1)
    log.record('forex', 9000, 10040, now=day1.replace(hour=20))
    log.record('forex', 9000, 10100, now=datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc))
    log.record('other', 1, 1, now=day1)
    assert [(r['day'], r['equity']) for r in log.history('forex')] == [('2026-09-29', 10040.0), ('2026-09-30', 10100.0)]
    reopened = pa.PaperEquityLog(str(tmp_path / 'eq.db'))                         # a redeploy keeps it
    assert len(reopened.history('forex')) == 2
    log.close(), reopened.close()


def test_a_daily_curve_takes_the_last_value_and_skips_junk():
    points = [{'timestamp': datetime(2026, 9, 28, 10), 'value': 100.0},
              {'timestamp': datetime(2026, 9, 28, 16), 'value': 101.0},
              {'timestamp': '2026-09-29T10:00:00', 'value': 102.0},
              {'timestamp': 'x', 'value': 'bad'}, {'value': 5}, {'timestamp': '2026-09-30', 'value': 0}]
    assert pa.daily_curve(points) == [{'day': '2026-09-28', 'equity': 101.0}, {'day': '2026-09-29', 'equity': 102.0}]


# ---- the page

def test_the_us_page_shows_its_holdings_stops_trades_and_results(journal):
    fill(journal, 'b1', 'NVDA', 'buy', 10, 200.0)
    fill(journal, 'b2', 'NVDA', 'sell', 10, 220.0, strategy='trailing_stop')      # a closed win
    fill(journal, 'b3', 'TSLA', 'buy', 10, 377.0)
    fill(journal, 'b4', 'EUR_USD', 'buy', 1000, 1.08)                             # forex: not on this page
    fill(journal, 'b5', 'SCOM', 'buy', 100, 30.0)                                 # NSE: not on this page
    journal.record_intent('w1', 'AAPL', 'buy', 5, 'market')                       # still working
    book = Book([position('TSLA', 10, 377.0, 355.0), position('VOO', 10, 700.0, 703.0),
                 position('EUR_USD', 1000, 1.08, 1.09)], cash=20000, equity=99000)
    view = pa.build_view('us', book, journal, CONFIG, stops_for=stops, peak_for=lambda s: 380.0,
                         core_symbols={'VOO'}, starting_capital=100000, now=NOW,
                         curve=[{'day': '2026-09-29', 'equity': 99900.0}])

    assert view['enabled'] and view['currency'] == 'USD' and view['broker'] == {'name': 'alpaca', 'paper': True}
    acct = view['account']
    assert acct['equity'] == 99000 and acct['cash'] == 20000 and acct['starting_capital'] == 100000
    assert acct['pnl'] == -1000 and acct['return_pct'] == pytest.approx(-1.0)

    assert [h['symbol'] for h in view['holdings']] == ['VOO', 'TSLA']             # by size; no forex here
    tsla = next(h for h in view['holdings'] if h['symbol'] == 'TSLA')
    assert tsla['unrealised_pnl'] == pytest.approx(-220.0) and tsla['unrealised_pnl_pct'] == pytest.approx(-5.84, abs=0.01)
    assert tsla['stops']['stop_loss_price'] == pytest.approx(377.0 * 0.95)
    assert tsla['stops']['trailing_stop_price'] == pytest.approx(380.0 * 0.97)
    voo = next(h for h in view['holdings'] if h['symbol'] == 'VOO')
    assert voo['core'] is True and voo['stops'] is None                           # the core has no stops

    assert [f['symbol'] for f in view['fills']] == ['TSLA', 'NVDA', 'NVDA']       # newest first
    assert all(f['symbol'] not in ('EUR_USD', 'SCOM') for f in view['fills'])
    stats = view['stats']
    assert stats['closed_trades'] == 1 and stats['win_rate'] == 1.0 and stats['realised_pnl'] > 190
    assert stats['modelled_costs'] > 0                                            # Alpaca's paper account charges none
    assert [w['symbol'] for w in view['working_orders']] == ['AAPL']
    assert view['costs']['crypto']['verified'] is not None and 'us_equity' in view['costs']
    assert view['rules']['core']['symbols'] == CONFIG['core_satellite']['core_symbols']

    # The curve ends with today's live value even before the day's reading is logged.
    assert view['equity_curve'][-1] == {'day': '2026-09-30', 'equity': 99000}
    assert view['equity_curve'][0]['day'] == '2026-09-29'


def test_the_forex_page_shows_only_currency_pairs(journal):
    fill(journal, 'f1', 'EUR_USD', 'buy', 1000, 1.0800)
    fill(journal, 'a1', 'AAPL', 'buy', 5, 300.0)
    book = Book([position('EUR_USD', 1000, 1.08, 1.0850), position('AAPL', 5, 300, 310)],
                cash=8900, equity=9985, name='paper')
    book.initial_cash = 10000
    view = pa.build_view('forex', book, journal, CONFIG, stops_for=stops, now=NOW,
                         session={'open': True, 'hours': 'x'})
    assert [h['symbol'] for h in view['holdings']] == ['EUR_USD']
    assert view['holdings'][0]['region'] == 'Forex'
    assert view['account']['starting_capital'] == 10000                           # the book's own start
    assert view['account']['return_pct'] == pytest.approx(-0.15)
    assert [f['symbol'] for f in view['fills']] == ['EUR_USD']
    assert view['stats']['closed_trades'] == 0 and view['stats']['win_rate'] is None
    assert view['costs']['forex']['verified'] in (False, None) and view['session']['open'] is True
    assert view['rules']['core'] is None


def test_an_account_that_is_off_says_so_instead_of_showing_zeros(journal):
    assert pa.build_view('forex', None, journal, CONFIG)['enabled'] is False
    down = Book([], 0, 0, connected=False)
    assert 'not connected' in pa.build_view('us', down, journal, CONFIG)['reason']
    silent = Book([], 0, None)
    assert pa.build_view('us', silent, journal, CONFIG)['enabled'] is False


def test_a_live_account_is_flagged_as_not_paper(journal):
    view = pa.build_view('us', Book([], 100, 100, paper=False), journal, CONFIG, starting_capital=100)
    assert view['broker']['paper'] is False


# ---- through the API

def _client(tmp_path, monkeypatch, us, fx, journal, analytics=None, log=None):
    import src.api.auth as auth
    from src.api.api_server import TradingAPI
    from src.api.auth import create_token
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'tester': {
        'password': bcrypt.hashpw(b'unused', bcrypt.gensalt()).decode(), 'role': 'operator'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))

    class Books:
        def get_broker(self, name=None):
            return fx if name == 'forex_paper' else us
    agent = SimpleNamespace(
        components={'broker_manager': Books(), 'nse_paper_account': None, 'data_manager': None},
        config=CONFIG, order_journal=journal, performance_analytics=analytics, paper_equity_log=log,
        risk_manager=None)
    api = TradingAPI(agent, {})
    return api.app.test_client(), {'Authorization': f"Bearer {create_token('tester', 'operator')}"}


def test_the_api_serves_each_account_and_a_summary(tmp_path, monkeypatch, journal):
    fill(journal, 'f1', 'EUR_USD', 'buy', 1000, 1.08)
    us = Book([position('MSFT', 3, 500, 508)], cash=90000, equity=91524, name='alpaca')
    fx = Book([position('EUR_USD', 1000, 1.08, 1.09)], cash=8920, equity=10010, name='paper')
    fx.initial_cash = 10000
    log = pa.PaperEquityLog(str(tmp_path / 'eq.db'))
    log.record('forex', 10000, 10000, now=datetime(2026, 9, 29, 12, tzinfo=timezone.utc))
    analytics = SimpleNamespace(portfolio_values=[
        {'timestamp': datetime(2026, 9, 28, 12), 'value': 100000.0},
        {'timestamp': datetime(2026, 9, 29, 12), 'value': 99900.0}])
    client, headers = _client(tmp_path, monkeypatch, us, fx, journal, analytics, log)

    body = client.get('/api/paper/us', headers=headers).get_json()
    assert body['enabled'] and [h['symbol'] for h in body['holdings']] == ['MSFT']
    assert [p['day'] for p in body['equity_curve']][:2] == ['2026-09-28', '2026-09-29']
    assert body['holdings'][0]['stops']['stop_loss_pct'] == 0.05                  # the fixed fallback
    body = client.get('/api/paper/forex', headers=headers).get_json()
    assert [h['symbol'] for h in body['holdings']] == ['EUR_USD'] and body['equity_curve'][0]['day'] == '2026-09-29'
    assert body['session']['hours'].startswith('Sunday 17:00')
    assert client.get('/api/paper/europe', headers=headers).status_code == 404

    summary = client.get('/api/paper-accounts', headers=headers).get_json()['accounts']
    assert [(a['id'], a['equity']) for a in summary] == [('us', 91524), ('forex', 10010)]
    assert summary[1]['starting_capital'] == 10000 and summary[1]['return_pct'] == pytest.approx(0.1)
    log.close()


def test_the_api_needs_a_login(tmp_path, monkeypatch, journal):
    client, _ = _client(tmp_path, monkeypatch, None, None, journal)
    assert client.get('/api/paper/us').status_code in (401, 403)
    assert client.get('/api/paper-accounts').status_code in (401, 403)
