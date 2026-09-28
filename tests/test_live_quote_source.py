"""Which quote a symbol trades on (TradingAgent._extract_symbol_data).

The data manager keeps every source's last batch, including an old batch
from the synthetic fallback generator. Crypto is not covered by the Finnhub /
Alpha Vantage quote path, so BTC-USD and ETH-USD only ever found the
fallback and every crypto signal was set aside as fallback_price (the
Notifications page: "price feed is serving synthetic fallback data")."""
from types import SimpleNamespace

import pytest

from src.agent.main import TradingAgent
from src.utils import real_price_feed

AGENT = SimpleNamespace(_live_quote=TradingAgent._live_quote)
MARKET = {'market_data': {
    'fallback': {'timestamp': 't0', 'data': {'AAPL': {'close': 1.0}, 'BTC-USD': {'close': 5.0}}},
    'real_data': {'timestamp': 't1', 'data': {'AAPL': {'close': 227.5, 'price': 227.5}}},
}}


@pytest.fixture
def live(monkeypatch):
    prices, asked = {}, []

    def get_price(symbol):
        asked.append(symbol)
        return prices.get(symbol)
    monkeypatch.setattr(real_price_feed.price_feed, 'get_price', get_price)
    return prices, asked


def quote(symbol):
    return TradingAgent._extract_symbol_data(AGENT, MARKET, symbol)


def test_a_real_source_wins_over_an_older_fallback_batch(live):
    q = quote('AAPL')
    assert q['source'] == 'real_data' and q['price'] == 227.5
    assert live[1] == []                                                  # no extra lookup needed


def test_crypto_gets_a_real_price_from_the_live_feed(live):
    live[0]['BTC-USD'] = 65432.1
    q = quote('BTC-USD')
    assert q['source'] == 'yfinance' and q['price'] == q['close'] == 65432.1


def test_without_a_real_price_the_fallback_is_returned_and_refused_downstream(live):
    assert quote('BTC-USD')['source'] == 'fallback'
    live[0]['ETH-USD'] = 0                                                # a zero is no price
    assert quote('ETH-USD') is None
    assert live[1] == ['BTC-USD', 'ETH-USD']
