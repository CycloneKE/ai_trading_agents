"""Forex that can actually trade, and a forex pair that can be opened.

A major currency pair moves about 0.5% a day against a stock's 1.5%, so the
stock-sized 2% thresholds almost never crossed on forex (about one buy day a
year per pair in a simulation). Forex now carries its own thresholds
(`market_overrides`) and its own trend strategy, `forex_trend`. Separately,
the chart asked Yahoo for "EUR_USD" (its name is EURUSD=X), so a pair's panel
had no chart, and EURUSD or EUR/USD, the way a person types a pair, found
nothing.
"""
import os
from types import SimpleNamespace

os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-dashboard-honesty-tests')

import pytest

from src.agent import chart_data
from src.agent.strategy_manager import StrategyManager
from src.agent.technical_strategy import TechnicalStrategy
from src.utils.config_validator import load_config

CONFIG = load_config('config/config.json')
STRATEGIES = CONFIG['strategies']


def _ramp(start, total_move, n=10, flat=50):
    """`flat` bars at `start`, then n bars rising by `total_move` (a fraction)."""
    return [start] * flat + [start * (1 + total_move * (i + 1) / n) for i in range(n)]


def _signal(name, symbol, closes):
    s = TechnicalStrategy(name, STRATEGIES[name])
    s.seed_history(symbol, closes[:-1])
    return s.generate_signals({'symbol': symbol, 'price': closes[-1]})


# ------------------------------------------------------- forex thresholds

def test_a_currency_sized_rise_is_a_momentum_buy_on_forex_but_not_on_a_stock():
    up = _ramp(1.10, 0.008)                       # 0.8% in ten days: a strong move for a pair
    assert _signal('momentum', 'EUR_USD', up)['action'] == 'buy'
    assert _signal('momentum', 'AAPL', [p * 100 for p in up])['action'] == 'hold'


def test_a_currency_sized_dip_is_a_mean_reversion_buy_on_forex_but_not_on_a_stock():
    dip = [1.10] * 30 + [1.10 * 0.992]            # 0.8% under its average
    assert _signal('mean_reversion', 'GBP_USD', dip)['action'] == 'buy'
    assert _signal('mean_reversion', 'MSFT', [p * 100 for p in dip])['action'] == 'hold'


def test_the_stock_thresholds_are_untouched():
    """Only forex has overrides: a stock still needs a 2% move."""
    assert STRATEGIES['momentum']['threshold'] == 0.02
    assert set(STRATEGIES['momentum']['market_overrides']) - {'_comment'} == {'forex'}
    assert _signal('momentum', 'AAPL', _ramp(100.0, 0.03))['action'] == 'buy'


def test_a_market_without_an_override_keeps_the_strategys_own_settings():
    s = TechnicalStrategy('momentum', {'threshold': 0.02, 'market_overrides': {'forex': {'threshold': 0.006}}})
    assert s._params('EUR_USD')['threshold'] == 0.006
    assert s._params('AAPL')['threshold'] == 0.02
    assert s._params('BTC-USD')['threshold'] == 0.02


# --------------------------------------------------------- forex_trend

@pytest.fixture(scope='module')
def manager():
    return StrategyManager(CONFIG)


def test_forex_trend_votes_on_forex_only_and_the_other_markets_are_unchanged(manager):
    voters = lambda sym: {n for n in manager.strategies if manager.in_scope(n, sym)}
    assert voters('EUR_USD') == {'momentum', 'mean_reversion', 'rsi_strategy', 'forex_trend'}
    assert voters('AAPL') == {'momentum', 'mean_reversion', 'rsi_strategy', 'us_earnings_drift'}
    assert voters('BTC-USD') == {'crypto_trend'}


def test_forex_trend_abstains_until_it_has_fifty_bars_then_buys_above_its_band(manager):
    trend = manager.strategies['forex_trend']
    assert trend.sma_period == 50
    trend.seed_history('AUD_USD', [0.66] * 48)        # this call's price is the 49th bar
    assert trend.generate_signals({'symbol': 'AUD_USD', 'price': 0.66})['abstain'] is True
    trend.seed_history('AUD_USD', [0.66] * 50)
    up = trend.generate_signals({'symbol': 'AUD_USD', 'price': 0.66 * 1.008})
    assert up['action'] == 'buy'
    down = trend.generate_signals({'symbol': 'AUD_USD', 'price': 0.66 * 0.99})
    assert down['action'] == 'sell'


def test_a_currency_sized_uptrend_reaches_a_buy_in_the_full_ensemble(manager):
    """End to end through the regime filter and the confidence gate: the same
    percentage move that was silent before is a buy on a pair, and still
    silent on a stock."""
    up = _ramp(1.10, 0.008)
    fresh = StrategyManager(CONFIG)
    fresh.warm_start({'EUR_USD': up[:-1], 'AAPL': [p * 100 for p in up[:-1]]})
    fx = fresh.generate_signals({'symbol': 'EUR_USD', 'price': up[-1], 'bar_date': 'd1'})
    stock = fresh.generate_signals({'symbol': 'AAPL', 'price': up[-1] * 100, 'bar_date': 'd1'})
    assert fx['action'] == 'buy' and fx['confidence'] >= CONFIG.get('min_trade_confidence', 0.5)
    assert stock['action'] == 'hold'


# ------------------------------------------------- charts and the drill-down

def test_the_chart_asks_yahoo_for_a_pair_by_yahoos_name(monkeypatch):
    import pandas as pd
    import yfinance
    asked = []

    class Ticker:
        def __init__(self, name):
            asked.append(name)

        def history(self, **kw):
            return pd.DataFrame({'Open': [1.1], 'High': [1.11], 'Low': [1.09], 'Close': [1.10], 'Volume': [0]},
                                index=pd.to_datetime(['2026-09-29']))

    monkeypatch.setattr(yfinance, 'Ticker', Ticker)
    bars = chart_data.yfinance_bars('EUR_USD')
    assert asked == ['EURUSD=X'] and bars[0]['close'] == 1.1
    chart_data.yfinance_bars('AAPL')
    assert asked[-1] == 'AAPL'


def test_a_pair_links_to_its_tradingview_chart():
    assert chart_data.tradingview_symbol('EUR_USD', 'forex') == 'FX:EURUSD'
    assert chart_data.tradingview_symbol('AAPL', 'us_equity') == 'AAPL'


@pytest.fixture
def client(tmp_path, monkeypatch):
    import json
    import bcrypt
    import src.api.auth as auth
    from src.api.api_server import TradingAPI
    from src.api.auth import create_token
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'tester': {
        'password': bcrypt.hashpw(b'unused', bcrypt.gensalt()).decode(), 'role': 'operator'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    config = {'data_manager': {'symbols': ['AAPL', 'BTC-USD', 'EUR_USD'], 'crypto_symbols': ['BTC-USD'],
                               'forex_symbols': ['EUR_USD'], 'nse_symbols': []}}
    agent = SimpleNamespace(
        components={'data_manager': SimpleNamespace(connectors={})}, config=config,
        order_journal=None, volatility=None, decision_journal=None)
    headers = {'Authorization': f"Bearer {create_token('tester', 'operator')}"}
    return TradingAPI(agent, {}).app.test_client(), headers


def test_a_pair_opens_however_it_is_typed(client, monkeypatch):
    c, h = client
    monkeypatch.setattr('src.utils.real_price_feed.price_feed.get_price', lambda s: 1.0812)
    for typed in ('EUR_USD', 'eur_usd', 'EURUSD', 'eurusd'):
        r = c.get(f'/api/symbol/{typed}', headers=h)
        assert r.status_code == 200, typed
        assert r.get_json()['symbol'] == 'EUR_USD'
        assert r.get_json()['current_price'] == 1.0812


def test_a_pairs_chart_opens_and_says_it_is_forex(client, monkeypatch):
    c, h = client
    monkeypatch.setattr(chart_data, 'yfinance_bars', lambda s, period='2y': [
        {'time': '2026-09-29', 'open': 1.1, 'high': 1.11, 'low': 1.09, 'close': 1.1, 'volume': 0}])
    data = c.get('/api/chart/EURUSD', headers=h).get_json()
    assert (data['symbol'], data['market'], data['tradingview_symbol']) == ('EUR_USD', 'forex', 'FX:EURUSD')


def test_other_symbols_are_unchanged_and_unknown_ones_are_still_refused(client):
    c, h = client
    assert c.get('/api/symbol/BTC-USD', headers=h).get_json()['symbol'] == 'BTC-USD'
    assert c.get('/api/symbol/ZZZZ', headers=h).status_code == 404
    assert c.get('/api/symbol/JPYCHF', headers=h).status_code == 404
