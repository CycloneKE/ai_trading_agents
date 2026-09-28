"""Forex paper trading: currency pairs (EUR_USD) on a paper book of their
own (config brokers.forex_paper), priced live, costed by the spread, traded
only while the currency market is open, never routed to Alpaca, and with
stop-losses like every other book."""
import types
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.agent.core_portfolio import fx_session_open
from src.agent.cost_model import classify, costs_for, round_trip_pct
from src.agent.main import FOREX_BOOK, TradingAgent
from src.agent.order_journal import OrderJournal
from src.connectors.base_broker import Position
from src.utils.config_validator import load_config
from src.utils.real_price_feed import yahoo_symbol
from tests.test_live_execution_rules import FillingBroker, _NoOp

CONFIG = load_config('config/config.json')


def ny(y, m, d, hh, mm=0):
    from zoneinfo import ZoneInfo
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo('America/New_York'))


def test_a_currency_pair_is_its_own_market_with_spread_costs():
    assert classify('EUR_USD', CONFIG) == 'forex' and classify('NZD_USD', {}) == 'forex'
    c = costs_for('GBP_USD', CONFIG)
    assert c['commission_pct'] == 0.0 and c['slippage_pct'] == 0.0001
    assert round_trip_pct('EUR_USD', CONFIG) == pytest.approx(0.0002)
    assert classify('BTC-USD', CONFIG) == 'crypto' and classify('AAPL', CONFIG) == 'us_equity'


def test_the_shipped_config_trades_three_usd_pairs_on_a_paper_book():
    dm = CONFIG['data_manager']
    assert set(dm['forex_symbols']) == {'EUR_USD', 'GBP_USD', 'AUD_USD'} <= set(dm['symbols'])
    book = CONFIG['brokers'][FOREX_BOOK]
    assert book['type'] == 'paper' and book['state_file'] == 'forex_paper_state.json'
    assert not book.get('primary') and CONFIG['brokers']['alpaca_broker']['primary']


def test_the_currency_market_is_open_sunday_evening_to_friday_evening():
    assert not fx_session_open(ny(2026, 9, 26, 12))                       # Saturday
    assert not fx_session_open(ny(2026, 9, 27, 16, 59))                   # Sunday afternoon
    assert fx_session_open(ny(2026, 9, 27, 17, 0))                        # Sunday 17:00 New York
    assert fx_session_open(ny(2026, 9, 30, 3, 0))                         # Wednesday, any hour
    assert fx_session_open(ny(2026, 10, 2, 16, 59)) and not fx_session_open(ny(2026, 10, 2, 17, 0))


def test_prices_come_from_yahoo_under_its_own_name():
    assert yahoo_symbol('EUR_USD') == 'EURUSD=X' and yahoo_symbol('eur_usd') == 'EURUSD=X'
    assert yahoo_symbol('SCOM') == 'SCOM' and yahoo_symbol('BTC-USD') == 'BTC-USD'


def test_finnhub_and_alpha_vantage_are_not_asked_for_a_pair(monkeypatch):
    from src.connectors.real_data_connector import RealDataConnector
    conn = RealDataConnector({})
    conn.finnhub_key = conn.alpha_vantage_key = 'k'
    asked = []
    monkeypatch.setattr(conn, '_get_finnhub_data', lambda s: asked.append(s))
    monkeypatch.setattr(conn, '_get_alpha_vantage_data', lambda s: asked.append(s))
    assert conn.get_real_time_data(['EUR_USD']) == {} and asked == []


def test_each_paper_book_keeps_its_own_file_and_fills_only_on_a_live_price(tmp_path, monkeypatch):
    from src.agent import paper_trading
    from src.connectors.base_broker import OrderRequest
    monkeypatch.setattr(paper_trading, 'DATA_DIR', tmp_path)
    live = {'EUR_USD': None}
    monkeypatch.setattr(paper_trading.price_feed, 'get_price', lambda s: live.get(s))
    book = paper_trading.PaperTradingBroker({'initial_cash': 10000, 'state_file': 'forex_paper_state.json'})
    book.is_connected = True
    book.market_prices['EUR_USD'] = 1.05                                 # an old price
    order = book.place_order(OrderRequest(symbol='EUR_USD', quantity=1000, side='buy', order_type='market', time_in_force='gtc'))
    book._process_order(order)
    assert order.status == 'new' and book.cash == 10000                   # waits for a live price
    live['EUR_USD'] = 1.08
    book._process_order(order)
    assert order.status == 'filled' and book.cash == pytest.approx(10000 - 1080)
    book._save_state()
    assert (tmp_path / 'forex_paper_state.json').exists()
    assert not (tmp_path / 'paper_trading_state.json').exists()


class _Books:
    def __init__(self, primary, forex=None):
        self.primary, self.forex = primary, forex

    def get_broker(self, name=None):
        return self.forex if name == FOREX_BOOK else self.primary


def _agent(tmp_path, books):
    agent = SimpleNamespace(
        config=load_config('config/config.json'),
        components={'broker_manager': books},
        order_journal=OrderJournal(db_path=str(tmp_path / 'orders.db')), monitoring_service=None,
        risk_manager=_NoOp(), event_calendar=SimpleNamespace(risk_multiplier=lambda: 1.0),
        trading_halted=False, _cycle_decisions={'EUR_USD': {'symbol': 'EUR_USD'}},
        _pdt_blocks_order=lambda *a: False)
    agent._position_gate = types.MethodType(TradingAgent._position_gate, agent)
    return agent


def _buy(agent, price=1.08):
    TradingAgent._execute_trades(agent, {'EUR_USD': {
        'action': 'buy', 'confidence': 0.9, 'position_size': 0.05, 'price': price,
        'strategy': 'ensemble', 'per_strategy': {}}}, {})
    return agent._cycle_decisions['EUR_USD']


def test_a_forex_buy_goes_to_the_forex_book_sized_from_its_own_balance(tmp_path):
    alpaca, fx = FillingBroker(price=500.0), FillingBroker(price=1.08)
    fx.cash = 10000.0
    agent = _agent(tmp_path, _Books(alpaca, fx))
    _buy(agent)
    assert alpaca.orders == [] and len(fx.orders) == 1
    spent = float(fx.orders[0].quantity) * 1.08
    assert 0 < spent <= 10000 * CONFIG['risk_limits']['max_position_size'] + 1   # its own balance's cap


def test_without_the_forex_book_nothing_is_sent_to_alpaca(tmp_path):
    alpaca = FillingBroker(price=500.0)
    agent = _agent(tmp_path, _Books(alpaca, None))
    assert _buy(agent).get('skip_reason') == 'no_broker'
    assert alpaca.orders == []


def test_forex_signals_wait_while_the_market_is_shut(tmp_path):
    fx = FillingBroker(price=1.08)
    agent = _agent(tmp_path, _Books(FillingBroker(price=500.0), fx))
    saturday = ny(2026, 9, 26, 12).astimezone(timezone.utc)
    assert TradingAgent._held_back_before_review(agent, 'EUR_USD', 'buy', 1.08, now=saturday) == 'market_closed'
    wednesday = ny(2026, 9, 30, 12).astimezone(timezone.utc)
    assert TradingAgent._held_back_before_review(agent, 'EUR_USD', 'sell', 1.08, now=wednesday) == 'no_position'
    agent.components['broker_manager'] = _Books(FillingBroker(price=500.0), None)
    assert TradingAgent._held_back_before_review(agent, 'EUR_USD', 'buy', 1.08, now=wednesday) == 'no_broker'


def test_stop_losses_cover_the_forex_book_too(tmp_path):
    losing = Position(symbol='EUR_USD', quantity=1000, avg_entry_price=1.10, current_price=1.00,
                      market_value=1000.0, unrealized_pl=-100.0, unrealized_pl_percent=-0.09,
                      cost_basis=1100.0)
    fx = SimpleNamespace(is_connected=True, orders=[], get_positions=lambda: [losing],
                         get_orders=lambda *a: [])
    fx.place_order = lambda o: fx.orders.append(o)
    alpaca = SimpleNamespace(is_connected=True, get_positions=lambda: [], orders=[])
    agent = _agent(tmp_path, _Books(alpaca, fx))
    agent._core_symbols = lambda: set()
    agent._stop_distances_for = lambda *a: {'stop_loss_pct': 0.05, 'trailing_stop_pct': 0.03, 'source': 'test'}
    agent._has_open_close_order = lambda *a: False
    agent.order_journal = None
    agent.risk_manager = None
    TradingAgent._enforce_stop_losses(agent, 0.05, 0.03)
    assert [(o.symbol, o.side) for o in fx.orders] == [('EUR_USD', 'sell')]
    assert alpaca.orders == []


def test_the_portfolio_page_shows_the_forex_book(tmp_path, monkeypatch):
    import json
    import os
    os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-dashboard-honesty-tests')
    import bcrypt
    import src.api.auth as auth
    from src.api.api_server import TradingAPI
    from src.api.auth import create_token
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'tester': {
        'password': bcrypt.hashpw(b'unused', bcrypt.gensalt()).decode(), 'role': 'operator'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    held = Position(symbol='EUR_USD', quantity=1000, avg_entry_price=1.07, current_price=1.08,
                    market_value=1080.0, unrealized_pl=10.0, unrealized_pl_percent=0.0093,
                    cost_basis=1070.0)
    fx = SimpleNamespace(is_connected=True, initial_cash=10000, get_positions=lambda: [held],
                         get_account_info=lambda: SimpleNamespace(cash=8930.0, equity=10010.0))
    alpaca = SimpleNamespace(is_connected=True, get_positions=lambda: [],
                             get_account_info=lambda: SimpleNamespace(cash=100000.0, equity=100000.0))
    agent = SimpleNamespace(components={'broker_manager': _Books(alpaca, fx)},
                            config={'data_manager': {'symbols': [], 'nse_symbols': []}})
    client = TradingAPI(agent, {}).app.test_client()
    body = client.get('/api/portfolio', headers={
        'Authorization': f"Bearer {create_token('tester', 'operator')}"}).get_json()
    assert body['forex_account'] == {'cash': 8930.0, 'equity': 10010.0, 'initial': 10000.0, 'currency': 'USD'}
    [pos] = [p for p in body['positions'] if p['region'] == 'Forex']
    assert pos['symbol'] == 'EUR_USD' and pos['market_value'] == 1080.0 and pos['current_price'] == 1.08
    assert body['summary']['regional_allocation_usd']['Forex'] == 1080.0
