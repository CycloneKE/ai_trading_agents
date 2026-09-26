"""The dashboard must show what the system knows, and nothing it invented.

Two endpoints used to fabricate data. /api/portfolio-allocation returned a
35/30/20/15 split of cash, Safaricom, Equity Group and KCB whenever the
account held nothing, and defaulted cash to 100000.0 when the broker could
not be read. /api/nse-market listed eighteen NSE symbols of which nine had
no price source, and gave the dashboard no market phase, so the dashboard
recomputed one and called every morning's pre-open auction a holiday.

These drive the real TradingAPI through Flask's test client, with a stand-in
agent, so they test the endpoint code itself.
"""
import os
from types import SimpleNamespace

os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-dashboard-honesty-tests')

import pytest

import json

import bcrypt

import src.api.auth as auth
from src.api.api_server import TradingAPI
from src.api.auth import create_token


@pytest.fixture(autouse=True)
def _a_real_user(tmp_path, monkeypatch):
    """token_required checks the user exists, so give it one. The auth
    layer rejecting an unknown user is correct and not what is under test."""
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'tester': {
        'password': bcrypt.hashpw(b'unused', bcrypt.gensalt()).decode(),
        'role': 'operator'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))


class _Broker:
    def __init__(self, cash=None, positions=None, account_raises=False):
        self._cash, self._positions, self._raises = cash, positions or [], account_raises

    def get_account_info(self):
        if self._raises:
            raise RuntimeError('broker unreachable')
        return None if self._cash is None else SimpleNamespace(cash=self._cash)

    def get_positions(self):
        return self._positions


class _BrokerManager:
    def __init__(self, broker):
        self._broker = broker

    def get_broker(self):
        return self._broker


class _NSE:
    """Stand-in connector: fixed quotes per symbol with a chosen source."""

    SOURCES = {'SCOM': 'afx_kwayisi', 'EQTY': 'synthetic', 'KCB': 'nse_website',
               'BAMB': 'none', 'NMG': 'none'}

    def __init__(self, phase='preopen'):
        self.phase = phase
        self.asked_for = None

    def get_all_quotes(self, symbols=None):
        self.asked_for = symbols
        wanted = symbols or list(self.SOURCES)
        return [{'symbol': s, 'price_kes': 0.0 if self.SOURCES.get(s) == 'none' else 10.0,
                 'change_pct': 0.0, 'source': self.SOURCES.get(s, 'none')} for s in wanted]

    def get_top_movers(self, symbols=None):
        return {'gainers': [], 'losers': []}

    def get_sector_performance(self, symbols=None):
        return []

    def get_status(self):
        return {}

    def get_kes_usd_rate(self):
        return 0.0077

    def is_market_open(self):
        return self.phase == 'open'

    def market_phase(self):
        return self.phase


def _client(broker=None, nse=None, nse_symbols=None):
    agent = SimpleNamespace(
        components={'broker_manager': _BrokerManager(broker) if broker else None,
                    'data_manager': SimpleNamespace(connectors={'nse': nse}) if nse else None},
        config={'data_manager': {'nse_symbols': nse_symbols or []}},
    )
    api = TradingAPI(agent, {})
    headers = {'Authorization': f"Bearer {create_token('tester', 'operator')}"}
    return api.app.test_client(), headers


# ------------------------------------------------------ allocation donut

def test_an_empty_account_is_shown_as_all_cash_not_invented_holdings():
    client, h = _client(broker=_Broker(cash=100000.0))
    data = client.get('/api/portfolio-allocation', headers=h).get_json()
    assert data == [{'name': 'Cash', 'value': 100000.0, 'color': '#64748b'}]
    names = {slice_['name'] for slice_ in data}
    assert not names & {'Safaricom (SCOM)', 'Equity Group (EQTY)', 'KCB Group (KCB)'}


def test_an_unreadable_account_shows_nothing_rather_than_a_default_balance():
    client, h = _client(broker=_Broker(account_raises=True))
    assert client.get('/api/portfolio-allocation', headers=h).get_json() == []


def test_positions_are_shown_even_when_the_balance_is_unavailable():
    pos = [SimpleNamespace(symbol='AAPL', quantity=2, current_price=190.0)]
    client, h = _client(broker=_Broker(cash=None, positions=pos))
    data = client.get('/api/portfolio-allocation', headers=h).get_json()
    assert [d['name'] for d in data] == ['AAPL']


# ------------------------------------------------------------ NSE market

def test_nse_market_reports_the_backend_phase():
    """The dashboard displays this instead of recomputing trading hours."""
    client, h = _client(nse=_NSE(phase='preopen'), nse_symbols=['SCOM'])
    data = client.get('/api/nse-market', headers=h).get_json()
    assert data['market_phase'] == 'preopen'
    assert data['market_open'] is False


def test_nse_market_lists_every_stock_with_real_data_and_the_traded_ones(tmp_path, monkeypatch):
    # Step B: the whole exchange, not only the traded list. A stock with a
    # real bar is listed; one with only synthetic seed data, which nothing
    # trades, is not shown as a row of made-up prices.
    import src.connectors.nse_connector as nse_connector
    monkeypatch.setattr(nse_connector, 'NSE_CSV_DIR', tmp_path)
    (tmp_path / 'JUB.csv').write_text('date,close,source\n2026-09-25,180.0,nse_ticker\n')
    (tmp_path / 'SMER.csv').write_text('date,close,source\n2026-09-25,3.0,synthetic\n')
    nse = _NSE()
    client, h = _client(nse=nse, nse_symbols=['scom', 'eqty', 'kcb'])
    data = client.get('/api/nse-market', headers=h).get_json()
    assert nse.asked_for == ['EQTY', 'JUB', 'KCB', 'SCOM']
    assert {q['symbol']: q['traded'] for q in data['quotes']} == {
        'EQTY': True, 'JUB': False, 'KCB': True, 'SCOM': True}


def test_nse_provenance_counts_synthetic_prices_as_synthetic():
    nse = _NSE()
    client, h = _client(nse=nse, nse_symbols=['SCOM', 'EQTY', 'KCB', 'BAMB'])
    prov = client.get('/api/nse-market', headers=h).get_json()['provenance']
    assert prov == {'real': 2, 'synthetic': 1, 'missing': 1, 'unverified': 0}


def test_an_unlabelled_price_is_unverified_not_real():
    """Strict: only a named scraped source counts as a real price."""
    nse = _NSE()
    nse.SOURCES = dict(_NSE.SOURCES, SCOM='csv')
    client, h = _client(nse=nse, nse_symbols=['SCOM'])
    prov = client.get('/api/nse-market', headers=h).get_json()['provenance']
    assert prov['real'] == 0 and prov['unverified'] == 1
