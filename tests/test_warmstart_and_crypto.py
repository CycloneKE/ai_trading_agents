"""The history warm-start, Alpaca's crypto format, and the gate that checks both."""
import csv
import json
import logging
import types
from datetime import date
from types import SimpleNamespace

import pytest

from src.agent import history_warmstart as hw
from src.agent.run_readiness import BLOCK, WARN, Readiness, check_history, history_needed
from src.connectors.alpaca_broker import (AlpacaAPIError, AlpacaBroker, from_alpaca_symbol,
                                          is_crypto_symbol, to_alpaca_symbol)
from src.connectors.base_broker import OrderRequest

TODAY = date(2026, 9, 24)


# ------------------------------------------------------- US / crypto history

def _rows(n, end_day=23):
    return [(date(2026, 9, end_day) - __import__('datetime').timedelta(days=n - 1 - i),
             101.0 + i, 99.0 + i, 100.0 + i) for i in range(n)]


def test_history_keeps_completed_bars_oldest_first_and_drops_today():
    rows = _rows(70) + [(TODAY, 999.0, 1.0, 500.0)]          # today's partial bar
    got = hw.fetch_daily_history(['SPY'], bars=60, today=TODAY, fetch=lambda s: list(reversed(rows)))
    closes = got['SPY']['close']
    assert len(closes) == 60 and 500.0 not in closes
    assert closes == sorted(closes)                            # oldest first
    assert len(got['SPY']['high']) == len(got['SPY']['low']) == 60


def test_one_failing_symbol_does_not_stop_the_rest():
    def fetch(sym):
        if sym == 'BAD':
            raise ConnectionError('403')
        return _rows(60)
    got = hw.fetch_daily_history(['BAD', 'SPY'], today=TODAY, fetch=fetch)
    assert set(got) == {'SPY'}


# --------------------------------------------------------------- NSE history

def _write(tmp_path, sym, rows):
    with open(tmp_path / f'{sym}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['date', 'close', 'high', 'low', 'source'])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_nse_history_uses_only_real_rows(tmp_path):
    """Seeding from synthetic rows compared today's real price with an
    invented moving average, and NSE signals become real-money tickets."""
    rows = [{'date': f'2026-08-{d:02d}', 'close': 10 + d, 'high': '', 'low': '', 'source': 'synthetic'}
            for d in range(1, 29)]
    rows += [{'date': '2026-09-21', 'close': 28.0, 'high': 28.5, 'low': 27.5, 'source': 'afx_history'},
             {'date': '2026-09-22', 'close': 28.2, 'high': '', 'low': '', 'source': 'afx_kwayisi'},
             {'date': '2026-09-22', 'close': 28.3, 'high': '', 'low': '', 'source': 'nse_website'},
             {'date': '2026-09-23', 'close': 99.0, 'high': '', 'low': '', 'source': ''}]
    _write(tmp_path, 'SCOM', rows)
    got = hw.read_nse_history(['SCOM', 'MISSING'], tmp_path, today=TODAY)
    assert got['SCOM']['close'] == [28.0, 28.3]      # real only, one per date, no unlabelled row
    assert 'MISSING' not in got


# ------------------------------------------------ the agent's warm-start method

def test_agent_warm_start_seeds_what_it_can_and_logs_the_rest(monkeypatch, tmp_path, caplog):
    from src.agent.main import TradingAgent
    from src.agent.strategy_manager import StrategyManager
    from src.agent.volatility import VolatilityTracker
    from src.utils.config_validator import load_config
    import src.connectors.nse_scraper as scraper
    import src.connectors.nse_connector as nse_conn

    cfg = load_config('config/config.json')
    cfg['data_manager']['symbols'] = ['SPY', 'NVDA']
    cfg['data_manager']['nse_symbols'] = ['SCOM']
    monkeypatch.setattr(hw, 'fetch_daily_history',
                        lambda syms: {s: {'close': [100.0 + i for i in range(60)],
                                          'high': [101.0 + i for i in range(60)],
                                          'low': [99.0 + i for i in range(60)]}
                                      for s in syms if s == 'SPY'})
    monkeypatch.setattr(scraper, 'backfill_afx_history', lambda syms: {})
    monkeypatch.setattr(nse_conn, 'NSE_CSV_DIR', tmp_path)

    agent = SimpleNamespace(config=cfg, components={'strategy_manager': StrategyManager(cfg)},
                            volatility=VolatilityTracker(length=14),
                            _history_seeded=set(), _last_history_attempt=0.0)
    agent._history_needed = types.MethodType(TradingAgent._history_needed, agent)
    with caplog.at_level(logging.ERROR):
        TradingAgent._warm_start_history(agent)
    assert agent._history_seeded == {'SPY'}
    assert agent.volatility.bars('SPY') == 60
    assert any('cannot signal yet' in r.getMessage() and 'NVDA' in r.getMessage()
               for r in caplog.records)


def test_warm_start_asks_afx_only_for_nse_symbols_short_of_history(monkeypatch, tmp_path):
    """afx was asked for every NSE symbol on every start. Unreachable from
    the server, it held start-up for twelve minutes and the heartbeat
    watchdog set the kill switch."""
    from src.agent.main import TradingAgent
    from src.agent.strategy_manager import StrategyManager
    from src.agent.volatility import VolatilityTracker
    from src.utils.config_validator import load_config
    import src.connectors.nse_scraper as scraper
    import src.connectors.nse_connector as nse_conn

    start = date(2026, 6, 1)
    _write(tmp_path, 'SCOM', [{'date': (start + __import__('datetime').timedelta(days=i)).isoformat(),
                               'close': 25 + i * 0.01, 'high': '', 'low': '', 'source': 'nse_pricelist'}
                              for i in range(70)])
    _write(tmp_path, 'KCB', [{'date': f'2026-09-0{d}', 'close': 40.0, 'high': '', 'low': '',
                              'source': 'nse_ticker'} for d in range(1, 6)])
    cfg = load_config('config/config.json')
    cfg['data_manager']['symbols'] = []
    cfg['data_manager']['nse_symbols'] = ['SCOM', 'KCB']
    asked = []
    monkeypatch.setattr(scraper, 'backfill_afx_history', lambda syms: asked.append(list(syms)) or {})
    monkeypatch.setattr(nse_conn, 'NSE_CSV_DIR', tmp_path)

    agent = SimpleNamespace(config=cfg, components={'strategy_manager': StrategyManager(cfg)},
                            volatility=VolatilityTracker(length=14),
                            _history_seeded=set(), _last_history_attempt=0.0)
    agent._history_needed = types.MethodType(TradingAgent._history_needed, agent)
    TradingAgent._warm_start_history(agent)
    assert asked == [['KCB']]
    assert agent._history_seeded == {'SCOM'}


def test_afx_backfill_stops_at_the_first_sign_the_site_is_unreachable(monkeypatch):
    """Each symbol cost about 80 seconds of connection timeouts."""
    import requests
    import src.connectors.nse_scraper as scraper
    calls = []

    def unreachable(url, **kwargs):
        calls.append(url)
        raise requests.ConnectionError('Network is unreachable')

    monkeypatch.setattr(scraper.requests, 'get', unreachable)
    assert scraper.backfill_afx_history(['SCOM', 'EQTY', 'KCB']) == {}
    assert len(calls) == 1


# --------------------------------------------------------- Alpaca crypto format

def test_symbol_translation_both_ways():
    assert to_alpaca_symbol('BTC-USD') == 'BTC/USD'
    assert to_alpaca_symbol('AAPL') == 'AAPL' and to_alpaca_symbol('BRK-B') == 'BRK-B'
    assert from_alpaca_symbol('BTC/USD') == 'BTC-USD'
    assert from_alpaca_symbol('BTCUSD', 'crypto') == 'BTC-USD'
    assert from_alpaca_symbol('SOLUSDT', 'crypto') == 'SOL-USDT'
    assert from_alpaca_symbol('AAPL', 'us_equity') == 'AAPL'
    assert is_crypto_symbol('ETH-USD') and not is_crypto_symbol('SPY')


class _Resp:
    def __init__(self, code=200, payload=None):
        self.status_code, self._p = code, payload
        self.ok = 200 <= code < 300
        self.text = json.dumps(payload or {})
        self.content = self.text.encode()

    def json(self):
        return self._p


class _Session:
    def __init__(self, *responses):
        self.responses, self.calls, self.headers = list(responses), [], {}

    def request(self, method, url, timeout=None, **kw):
        self.calls.append({'method': method, 'url': url, **kw})
        return self.responses.pop(0)


def _broker(session):
    b = AlpacaBroker({'api_key': 'k', 'api_secret': 's'})
    b.session, b.is_connected = session, True
    return b


def test_a_crypto_order_uses_alpacas_symbol_and_a_valid_time_in_force():
    s = _Session(_Resp(payload={'id': '1', 'symbol': 'BTC/USD', 'qty': '0.05', 'status': 'accepted'}))
    resp = _broker(s).place_order(OrderRequest(symbol='BTC-USD', quantity=0.05, side='buy',
                                               order_type='market', time_in_force='day'))
    payload = s.calls[0]['json']
    assert payload['symbol'] == 'BTC/USD'
    assert payload['time_in_force'] == 'gtc'        # Alpaca rejects 'day' for crypto
    assert 'extended_hours' not in payload
    assert resp.symbol == 'BTC-USD'                  # answers in our symbols


def test_a_stock_order_is_unchanged():
    s = _Session(_Resp(payload={'id': '1', 'symbol': 'AAPL', 'qty': '1', 'status': 'accepted'}))
    _broker(s).place_order(OrderRequest(symbol='AAPL', quantity=1, side='buy',
                                        order_type='market', time_in_force='day'))
    assert s.calls[0]['json']['symbol'] == 'AAPL' and s.calls[0]['json']['time_in_force'] == 'day'


def test_a_held_crypto_position_matches_its_own_symbol():
    """Alpaca reports 'BTCUSD'; without mapping, the agent never saw it held."""
    s = _Session(_Resp(payload=[{'symbol': 'BTCUSD', 'asset_class': 'crypto', 'qty': '0.05'}]))
    assert [p.symbol for p in _broker(s).fetch_positions()] == ['BTC-USD']


def test_strict_fetches_raise_while_lenient_getters_do_not():
    assert _broker(_Session(_Resp(500, {'message': 'boom'}))).get_positions() == []
    with pytest.raises(AlpacaAPIError):
        _broker(_Session(_Resp(500, {'message': 'boom'}))).fetch_positions()
    with pytest.raises(AlpacaAPIError):
        _broker(_Session(_Resp(500, {'message': 'boom'}))).fetch_open_orders('SPY')


def test_open_orders_are_filtered_to_the_symbol():
    s = _Session(_Resp(payload=[{'id': '1', 'symbol': 'BTC/USD', 'status': 'new'},
                                {'id': '2', 'symbol': 'AAPL', 'status': 'new'}]))
    assert [o.symbol for o in _broker(s).fetch_open_orders('BTC-USD')] == ['BTC-USD']


def test_order_lookup_tells_not_found_from_could_not_ask():
    assert _broker(_Session(_Resp(404, {'message': 'order not found'}))) \
        .get_order_by_client_order_id('x') is None
    with pytest.raises(AlpacaAPIError):
        _broker(_Session(_Resp(503, {'message': 'unavailable'}))).get_order_by_client_order_id('x')
    s = _Session(_Resp(payload={'id': '9', 'client_order_id': 'x', 'symbol': 'AAPL',
                                'status': 'filled', 'filled_qty': '2'}))
    o = _broker(s).get_order_by_client_order_id('x')
    assert o.status == 'filled' and o.filled_quantity == 2.0
    assert s.calls[0]['url'].endswith('/v2/orders:by_client_order_id')
    assert s.calls[0]['params'] == {'client_order_id': 'x'}


# ------------------------------------------------------------ readiness gate

def _gate_cfg():
    return {'data_manager': {'symbols': ['SPY', 'BTC-USD'], 'nse_symbols': ['SCOM']},
            'strategies': {'momentum': {'enabled': True, 'type': 'technical', 'lookback_period': 50},
                           'mean_reversion': {'enabled': True, 'type': 'technical', 'lookback_period': 20},
                           'rsi_strategy': {'enabled': True, 'type': 'technical'}}}


def _named(r, name):
    return next(c for c in r.checks if c.name == name)


def test_gate_needs_the_longest_lookback():
    assert history_needed(_gate_cfg()) == 50


def test_gate_warns_when_history_was_not_probed():
    r = Readiness(); check_history(_gate_cfg(), r, None)
    c = _named(r, 'daily history for warm-start')
    assert not c.ok and c.level == WARN


def test_gate_blocks_when_no_symbol_could_signal():
    """What production actually had, and the gate never noticed."""
    r = Readiness(); check_history(_gate_cfg(), r, {'SPY': 0, 'BTC-USD': 3})
    c = _named(r, 'daily history for warm-start')
    assert not c.ok and c.level == BLOCK


def test_gate_warns_on_a_partial_shortfall_and_passes_a_full_one():
    r = Readiness(); check_history(_gate_cfg(), r, {'SPY': 60, 'BTC-USD': 10, 'SCOM': 2})
    assert _named(r, 'daily history for warm-start').level == WARN
    assert not _named(r, 'NSE real daily history').ok
    r = Readiness(); check_history(_gate_cfg(), r, {'SPY': 60, 'BTC-USD': 60, 'SCOM': 60})
    assert _named(r, 'daily history for warm-start').ok and _named(r, 'NSE real daily history').ok
