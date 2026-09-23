"""Whether an NSE price is real must survive to every place that uses it.

The scraper seeds 730 days of synthetic history on first run and overlays
real bars only when a live source answers. Three things hid which was which:
the connector relabelled every CSV price as "csv"; the scraper's status said
"running" whether or not it had ever fetched a real price; and the NSE
decision path, unlike the US and crypto path, never checked. So an invented
price could reach the dashboard as a quote and the strategies as a signal,
and from there an order ticket the operator places with real money.
"""
import csv
import logging
from types import SimpleNamespace

import pytest

import src.connectors.nse_connector as nse_connector
import src.connectors.nse_scraper as nse_scraper
from src.agent.main import TradingAgent
from src.connectors.nse_connector import NSEConnector


def _write_csv(folder, symbol, source, close=28.5):
    fields = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'change_pct', 'source']
    with open(folder / f'{symbol}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({'date': '2026-09-22', 'symbol': symbol, 'open': close, 'high': close,
                    'low': close, 'close': close, 'volume': 1000, 'change_pct': 0.5,
                    'source': source})


# ------------------------------------------------------------- connector

@pytest.mark.parametrize('source', ['synthetic', 'afx_kwayisi', 'nse_website'])
def test_a_csv_quote_keeps_the_bars_own_source(tmp_path, monkeypatch, source):
    monkeypatch.setattr(nse_connector, 'NSE_CSV_DIR', tmp_path)
    _write_csv(tmp_path, 'SCOM', source)
    assert NSEConnector({}).get_quote('SCOM')['source'] == source


def test_a_csv_row_without_a_source_column_is_still_readable(tmp_path, monkeypatch):
    monkeypatch.setattr(nse_connector, 'NSE_CSV_DIR', tmp_path)
    (tmp_path / 'SCOM.csv').write_text('date,close\n2026-09-22,28.5\n')
    q = NSEConnector({}).get_quote('SCOM')
    assert q['price_kes'] == 28.5 and q['source'] == 'csv'


def test_get_all_quotes_can_be_limited_to_the_watched_universe(tmp_path, monkeypatch):
    monkeypatch.setattr(nse_connector, 'NSE_CSV_DIR', tmp_path)
    got = NSEConnector({}).get_all_quotes(['SCOM', 'KCB'])
    assert [q['symbol'] for q in got] == ['SCOM', 'KCB']


# -------------------------------------------------------- decision path

class _Recorder:
    def __init__(self):
        self.records = []

    def record(self, dec):
        self.records.append(dec)


class _Strategies:
    def __init__(self):
        self.calls = 0

    def generate_signals(self, symbol_data):
        self.calls += 1
        return {'action': 'hold', 'confidence': 0.0}


def _agent_with_quote(source):
    quote = {'symbol': 'SCOM', 'price_kes': 28.5, 'open_kes': 28.0,
             'volume': 1000, 'source': source}
    nse = SimpleNamespace(is_market_open=lambda: True, get_quote=lambda s: dict(quote))
    strategies = _Strategies()
    agent = SimpleNamespace(
        components={'nse_order_queue': SimpleNamespace(expire_stale=lambda h: None),
                    'data_manager': SimpleNamespace(connectors={'nse': nse}),
                    'strategy_manager': strategies, 'llm_orchestrator': None},
        config={'data_manager': {'nse_symbols': ['SCOM'], 'nse_eval_interval': 1800}},
        decision_journal=_Recorder(), order_journal=None, _nse_last_price={},
    )
    return agent, strategies


def test_the_nse_path_refuses_a_synthetic_price():
    agent, strategies = _agent_with_quote('synthetic')
    TradingAgent._evaluate_nse_symbols(agent)
    assert strategies.calls == 0, 'an invented price reached the strategies'
    assert agent._nse_last_price == {}, 'an invented price entered the freshness buffer'
    assert [d['skip_reason'] for d in agent.decision_journal.records] == ['fallback_price']


def test_the_nse_path_still_evaluates_a_real_price():
    """Control: the guard blocks synthetic prices, not everything."""
    agent, strategies = _agent_with_quote('afx_kwayisi')
    TradingAgent._evaluate_nse_symbols(agent)
    assert strategies.calls == 1
    assert agent._nse_last_price == {'SCOM': 28.5}


# ---------------------------------------------------------------- scraper

def _scraper(monkeypatch, fetched):
    monkeypatch.setattr(nse_scraper, 'scrape_nse_website', lambda syms: dict(fetched))
    monkeypatch.setattr(nse_scraper, 'scrape_afx_kwayisi', lambda syms: {})
    monkeypatch.setattr(nse_scraper, 'save_bars_csv', lambda sym, bars: None)
    monkeypatch.setattr(nse_scraper, 'save_bars_db', lambda bars, db: None)
    return nse_scraper.NSEPeriodicScraper(symbols=['SCOM', 'EQTY', 'KCB'])


def test_status_reports_a_scraper_that_fetched_nothing(monkeypatch, caplog):
    """'running: true' is equally true of a scraper serving only seed data."""
    s = _scraper(monkeypatch, {})
    with caplog.at_level(logging.WARNING):
        s.run_once()
    st = s.get_status()
    assert st['last_cycle_real_prices'] == 0
    assert st['last_cycle_missing'] == ['EQTY', 'KCB', 'SCOM']
    assert st['last_real_price_at'] is None
    assert any('NO real prices' in r.getMessage() for r in caplog.records)


def test_status_reports_partial_success(monkeypatch):
    bar = nse_scraper.DailyBar('2026-09-23', 'SCOM', 28.0, 29.0, 27.5, 28.5, 1000, 0.5, 'nse_website')
    s = _scraper(monkeypatch, {'SCOM': bar})
    s.run_once()
    st = s.get_status()
    assert st['last_cycle_real_prices'] == 1
    assert st['last_cycle_missing'] == ['EQTY', 'KCB']
    assert st['last_real_price_at'] is not None


def test_status_before_the_first_cycle_says_so():
    st = nse_scraper.NSEPeriodicScraper(symbols=['SCOM']).get_status()
    assert st['last_cycle_real_prices'] is None
