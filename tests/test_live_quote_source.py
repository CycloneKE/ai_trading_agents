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


class _Broker:
    is_connected = True

    def __init__(self, prices):
        self.prices, self.asked = prices, []

    def get_market_data(self, symbol):
        self.asked.append(symbol)
        p = self.prices.get(symbol)
        return {'symbol': symbol, 'price': p, 'close': p, 'source': 'alpaca'} if p else None


def _agent_with(broker):
    return SimpleNamespace(_live_quote=TradingAgent._live_quote,
                           components={'broker_manager': SimpleNamespace(get_broker=lambda *a: broker)})


def test_the_broker_that_fills_crypto_prices_it_first(live):
    broker = _Broker({'BTC-USD': 65000.0})
    q = TradingAgent._extract_symbol_data(_agent_with(broker), MARKET, 'BTC-USD')
    assert q['source'] == 'alpaca' and q['price'] == 65000.0
    assert live[1] == []                                                  # Yahoo not needed


def test_when_the_broker_has_no_price_yahoo_is_asked(live):
    live[0]['BTC-USD'] = 64990.0
    q = TradingAgent._extract_symbol_data(_agent_with(_Broker({})), MARKET, 'BTC-USD')
    assert q['source'] == 'yfinance' and q['price'] == 64990.0


def test_currency_pairs_skip_the_broker(live):
    broker = _Broker({'EUR_USD': 9.99})
    live[0]['EUR_USD'] = 1.08
    q = TradingAgent._extract_symbol_data(_agent_with(broker), MARKET, 'EUR_USD')
    assert q['source'] == 'yfinance' and broker.asked == []


def test_alpaca_quotes_crypto_and_stocks_from_its_market_data(monkeypatch):
    from src.connectors.alpaca_broker import AlpacaBroker, DATA_URL
    calls = []

    class _Resp:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    def get(url, params=None, timeout=None):
        calls.append((url, params))
        if 'crypto' in url:
            return _Resp({'trades': {'BTC/USD': {'p': 65432.1, 't': '2026-09-28T10:15:02.123456789Z'}}})
        return _Resp({'symbol': 'AAPL', 'trade': {'p': 227.5, 't': '2026-09-28T14:30:00Z'}})
    broker = AlpacaBroker({'api_key': 'k', 'api_secret': 's'})
    broker.session = SimpleNamespace(get=get)
    btc = broker.get_market_data('BTC-USD')
    assert btc['price'] == 65432.1 and btc['source'] == 'alpaca' and btc['timestamp'].startswith('2026-09-28T10:15:02')
    assert calls[0] == (f"{DATA_URL}/v1beta3/crypto/us/latest/trades", {'symbols': 'BTC/USD'})
    assert broker.get_market_data('AAPL')['price'] == 227.5
    assert calls[1] == (f"{DATA_URL}/v2/stocks/AAPL/trades/latest", {'feed': 'iex'})

    def broken(url, params=None, timeout=None):
        raise ConnectionError('down')
    broker.session = SimpleNamespace(get=broken)
    assert broker.get_market_data('BTC-USD') is None
