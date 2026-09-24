"""The live KES/USD rate (src/connectors/fx_rate.py)."""
import json

import pytest

from src.connectors import fx_rate as fx
from src.connectors.nse_connector import NSEConnector


def _q(rate, source='yahoo'):
    return lambda: {'kes_per_usd': rate, 'source': source, 'label': source, 'as_of': '2026-09-24T10:00:00+00:00'}


def _fx(tmp_path, *sources, clock=lambda: 1000.0):
    return fx.FxRate(tmp_path / 'fx.json', sources=list(sources), clock=clock, background=False)


def test_it_was_a_constant_and_now_follows_the_market(tmp_path):
    rate = _fx(tmp_path, _q(128.9))
    snap = rate.snapshot()
    assert snap['kes_per_usd'] == 128.9 and snap['source'] == 'yahoo' and snap['live'] is True
    assert snap['usd_per_kes'] == pytest.approx(1 / 128.9)


def test_the_daily_rate_steps_in_when_the_market_quote_fails(tmp_path):
    def broken():
        raise ConnectionError('no route')
    snap = _fx(tmp_path, broken, _q(129.3, 'exchangerate_api')).snapshot()
    assert (snap['kes_per_usd'], snap['source']) == (129.3, 'exchangerate_api')


def test_a_restart_keeps_the_last_good_rate_not_the_constant(tmp_path):
    _fx(tmp_path, _q(127.5)).snapshot()
    reborn = _fx(tmp_path)  # every source down
    snap = reborn.snapshot()
    assert snap['kes_per_usd'] == 127.5 and snap['source'] == 'yahoo'


def test_with_nothing_ever_fetched_it_says_it_is_the_fixed_fallback(tmp_path):
    snap = _fx(tmp_path).snapshot()
    assert snap['kes_per_usd'] == fx.FALLBACK_KES_PER_USD
    assert snap['source'] == 'fixed' and snap['live'] is False


@pytest.mark.parametrize('bad', [0.0, 1.3, 13000.0, None])
def test_implausible_quotes_are_rejected(tmp_path, bad):
    snap = _fx(tmp_path, _q(bad), _q(129.0, 'exchangerate_api')).snapshot()
    assert snap['kes_per_usd'] == 129.0


def test_a_big_jump_needs_a_second_source_to_agree(tmp_path):
    clock = {'t': 1000.0}
    first = _fx(tmp_path, _q(129.0), clock=lambda: clock['t'])
    first.snapshot()
    # A 20% jump from one source alone is kept out...
    lone = _fx(tmp_path, _q(155.0), clock=lambda: clock['t'])
    assert lone.snapshot()['kes_per_usd'] == 129.0
    # ...and believed when the second source confirms it.
    confirmed = _fx(tmp_path, _q(155.0), _q(154.0, 'exchangerate_api'), clock=lambda: clock['t'])
    assert confirmed.snapshot()['kes_per_usd'] == 155.0


def test_it_refreshes_every_five_minutes_not_every_request(tmp_path):
    calls, clock = [], {'t': 1000.0}

    def counted():
        calls.append(clock['t'])
        return {'kes_per_usd': 129.0, 'source': 'yahoo', 'label': 'y', 'as_of': None}

    rate = _fx(tmp_path, counted, clock=lambda: clock['t'])
    rate.snapshot(); rate.snapshot()
    clock['t'] += 299
    rate.snapshot()
    clock['t'] += 2
    rate.snapshot()
    assert len(calls) == 2


def test_a_rate_goes_stale_when_refreshes_keep_failing(tmp_path):
    clock = {'t': 1000.0}
    ok = {'on': True}

    def flaky():
        if not ok['on']:
            raise ConnectionError('down')
        return {'kes_per_usd': 129.0, 'source': 'yahoo', 'label': 'y', 'as_of': None}

    rate = _fx(tmp_path, flaky, clock=lambda: clock['t'])
    assert rate.snapshot()['live'] is True
    ok['on'] = False
    clock['t'] += 3 * fx.REFRESH_SECONDS + 1
    snap = rate.snapshot()
    assert snap['kes_per_usd'] == 129.0 and snap['live'] is False


def test_the_saved_file_is_plain_json(tmp_path):
    _fx(tmp_path, _q(128.0)).snapshot()
    assert json.loads((tmp_path / 'fx.json').read_text())['kes_per_usd'] == 128.0


def test_a_background_refresh_never_blocks_the_caller(tmp_path):
    import threading
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return {'kes_per_usd': 129.0, 'source': 'yahoo', 'label': 'y', 'as_of': None}

    rate = fx.FxRate(tmp_path / 'fx.json', sources=[slow], background=True)
    assert rate.snapshot()['source'] == 'fixed'   # returned at once
    gate.set()


def test_nse_quotes_and_the_connector_use_the_live_rate(tmp_path):
    conn = NSEConnector({}, fx=_fx(tmp_path, _q(125.0)))
    assert conn.get_kes_usd_rate() == pytest.approx(1 / 125.0)
    assert conn.get_fx()['source'] == 'yahoo'
    assert conn.get_status()['kes_usd_rate'] == pytest.approx(1 / 125.0)


def test_yahoos_reply_is_read_as_shillings_per_dollar(monkeypatch):
    import sys
    import types
    import pandas as pd
    idx = pd.DatetimeIndex(['2026-09-24 09:55', '2026-09-24 10:00'], tz='UTC')
    frame = pd.DataFrame({'Close': [129.10, 129.15]}, index=idx)
    fake = types.SimpleNamespace(Ticker=lambda sym: types.SimpleNamespace(
        history=lambda **kw: frame if sym == 'KES=X' else pd.DataFrame()))
    monkeypatch.setitem(sys.modules, 'yfinance', fake)
    q = fx.yahoo_rate()
    assert q['kes_per_usd'] == 129.15 and q['as_of'].startswith('2026-09-24T10:00')


def test_exchangerate_apis_reply_is_read_with_its_attribution(monkeypatch):
    reply = {'result': 'success', 'time_last_update_unix': 1790208001,
             'rates': {'USD': 1, 'KES': 129.2}}
    monkeypatch.setattr(fx.requests, 'get', lambda url, timeout: type('R', (), {'json': lambda self: reply})())
    q = fx.er_api_rate()
    assert q['kes_per_usd'] == 129.2 and q['attribution_url'].startswith('https://')


def test_the_nse_page_and_the_portfolio_show_the_rate_and_its_source(tmp_path, monkeypatch):
    import bcrypt
    from types import SimpleNamespace
    import src.api.auth as auth
    from src.api.api_server import TradingAPI
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'t': {'password': bcrypt.hashpw(b'x', bcrypt.gensalt()).decode(),
                                       'role': 'viewer'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    conn = NSEConnector({}, fx=_fx(tmp_path, _q(126.0)))
    monkeypatch.setattr(conn, 'get_all_quotes', lambda watched=None: [])
    agent = SimpleNamespace(components={'data_manager': SimpleNamespace(connectors={'nse': conn})},
                            config={'data_manager': {'nse_symbols': ['SCOM']}})
    client = TradingAPI(agent, {}).app.test_client()
    h = {'Authorization': f"Bearer {auth.create_token('t', 'viewer')}"}
    market = client.get('/api/nse-market', headers=h).get_json()
    assert market['fx']['kes_per_usd'] == 126.0 and market['fx']['source'] == 'yahoo'
    assert client.get('/api/portfolio', headers=h).get_json()['fx']['kes_per_usd'] == 126.0
