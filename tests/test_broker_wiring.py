"""Which venue the orders actually reach.

config/config.json marks Alpaca `primary`. Its connector imported
`alpaca_trade_api`, which was never installed in the runtime image, so the
guarded import in broker_manager failed, the 'alpaca' type was never
registered, the broker was dropped with a single warning, and every order
went to the internal paper simulator instead. The readiness gate read the
same config file, saw Alpaca marked paper, and reported READY.

Nothing in that chain was a crash. Each test below is one link of it.
"""
import json
import pytest

from src.agent.broker_manager import (BrokerConfigurationError, BrokerManager,
                                      available_broker_types, broker_is_enabled)
from src.connectors.alpaca_broker import (LIVE_BASE_URL, PAPER_BASE_URL,
                                          AlpacaAPIError, AlpacaBroker)
from src.connectors.base_broker import OrderRequest


# ------------------------------------------------------- the broker manager

def test_the_alpaca_type_registers_without_any_extra_dependency():
    """The root cause. This is what was false in production."""
    assert 'alpaca' in available_broker_types()


def test_the_connector_does_not_import_the_alpaca_sdk():
    """Reintroducing it would pin urllib3 < 1.25 under our credentials."""
    src = open('src/connectors/alpaca_broker.py').read()
    assert 'import alpaca_trade_api' not in src


def test_a_disabled_broker_is_not_created():
    """`enabled: false` used to mean nothing: the loop never read it."""
    mgr = BrokerManager({'brokers': {
        'paper_broker': {'type': 'paper'},
        'coinbase_broker': {'type': 'coinbase', 'enabled': False},
    }})
    assert 'coinbase_broker' not in mgr.brokers
    assert 'coinbase_broker' in mgr.disabled_brokers


def test_an_absent_enabled_flag_still_means_enabled():
    assert broker_is_enabled({'type': 'paper'}) is True
    assert broker_is_enabled({'type': 'paper', 'enabled': False}) is False


def test_the_broker_marked_primary_wins_over_the_first_one_built():
    mgr = BrokerManager({'brokers': {
        'paper_broker': {'type': 'paper'},
        'alpaca_broker': {'type': 'alpaca', 'paper': True, 'primary': True},
    }})
    assert mgr.primary_broker == 'alpaca_broker'
    assert isinstance(mgr.get_broker(), AlpacaBroker)


def test_an_unavailable_primary_broker_refuses_to_start():
    """The production bug, as the code would now behave."""
    with pytest.raises(BrokerConfigurationError) as e:
        BrokerManager({'brokers': {
            'paper_broker': {'type': 'paper'},
            'ibkr_broker': {'type': 'interactive_brokers', 'primary': True},
        }})
    assert 'ibkr_broker' in str(e.value)
    assert 'venue other than the configured one' in str(e.value)


def test_a_disabled_primary_broker_refuses_to_start():
    with pytest.raises(BrokerConfigurationError) as e:
        BrokerManager({'brokers': {
            'paper_broker': {'type': 'paper'},
            'alpaca_broker': {'type': 'alpaca', 'enabled': False, 'primary': True},
        }})
    assert 'alpaca_broker' in str(e.value)


def test_all_brokers_unavailable_refuses_rather_than_substituting_paper():
    """Silently inventing a paper broker is how the venue changed unnoticed."""
    with pytest.raises(BrokerConfigurationError) as e:
        BrokerManager({'brokers': {'ibkr': {'type': 'interactive_brokers'}}})
    assert 'none created' in str(e.value)


def test_an_unavailable_secondary_is_recorded_not_swallowed():
    mgr = BrokerManager({'brokers': {
        'paper_broker': {'type': 'paper', 'primary': True},
        'ibkr_broker': {'type': 'interactive_brokers'},
    }})
    assert mgr.unavailable_brokers == {'ibkr_broker': 'interactive_brokers'}
    assert mgr.primary_broker == 'paper_broker'


def test_no_brokers_configured_still_gets_the_default_paper_broker():
    mgr = BrokerManager({})
    assert mgr.primary_broker == 'default_paper'


def test_primary_broker_is_a_name_and_get_broker_is_the_object():
    """scripts/start_paper_run.py returned the name and asked a str whether
    it was connected, which blocked runs whose broker was fine."""
    mgr = BrokerManager({'brokers': {'paper_broker': {'type': 'paper'}}})
    assert isinstance(mgr.primary_broker, str)
    assert mgr.get_broker() is mgr.brokers['paper_broker']


def test_the_shipped_config_routes_to_alpaca_paper():
    """The end state the operator asked for, read from the real config."""
    cfg = json.load(open('config/config.json'))
    brokers = cfg['brokers']
    primary = [n for n, b in brokers.items() if b.get('primary')]
    assert primary == ['alpaca_broker']
    assert brokers['alpaca_broker']['type'] == 'alpaca'
    assert brokers['alpaca_broker']['paper'] is True
    assert brokers['alpaca_broker']['type'] in available_broker_types()


# ------------------------------------------------------- the Alpaca connector

class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=''):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})
        self.content = self.text.encode()

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._payload is None:
            raise ValueError('no json')
        return self._payload


class _FakeSession:
    """Records every call so the tests can assert on the wire format."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []
        self.headers = {}

    def request(self, method, url, timeout=None, **kwargs):
        self.calls.append({'method': method, 'url': url,
                           'timeout': timeout, **kwargs})
        return self._responses.pop(0) if self._responses else _FakeResponse(payload={})

    def close(self):
        pass


def _broker(session=None, **cfg):
    b = AlpacaBroker({'api_key': 'k', 'api_secret': 's', **cfg})
    if session is not None:
        b.session = session
        b.is_connected = True
    return b


ORDER_JSON = {
    'id': 'abc-123', 'client_order_id': 'cid-1', 'symbol': 'AAPL',
    'qty': '3', 'filled_qty': '3', 'side': 'buy', 'type': 'market',
    'status': 'filled', 'created_at': '2026-09-22T13:30:00.123456Z',
    'updated_at': '2026-09-22T13:30:01Z', 'limit_price': None,
    'stop_price': None, 'filled_avg_price': '190.25',
}


def test_paper_and_live_base_urls_are_distinct():
    assert _broker(paper=True).base_url == PAPER_BASE_URL
    assert _broker(paper=False).base_url == LIVE_BASE_URL
    assert _broker().base_url == PAPER_BASE_URL  # default must be paper


def test_place_order_posts_the_documented_payload():
    s = _FakeSession(_FakeResponse(payload=ORDER_JSON))
    b = _broker(s)
    b.place_order(OrderRequest(symbol='AAPL', quantity=3, side='buy',
                               order_type='market', time_in_force='gtc'))
    call = s.calls[0]
    assert call['method'] == 'POST'
    assert call['url'] == f'{PAPER_BASE_URL}/v2/orders'
    assert call['json'] == {'symbol': 'AAPL', 'qty': '3', 'side': 'buy',
                            'type': 'market', 'time_in_force': 'gtc',
                            'extended_hours': False}


def test_optional_prices_are_omitted_rather_than_sent_as_null():
    s = _FakeSession(_FakeResponse(payload=ORDER_JSON))
    b = _broker(s)
    b.place_order(OrderRequest(symbol='AAPL', quantity=1, side='buy',
                               order_type='limit', time_in_force='day',
                               limit_price=190.5))
    payload = s.calls[0]['json']
    assert payload['limit_price'] == '190.5'
    assert 'stop_price' not in payload


def test_a_fractional_quantity_forces_day_time_in_force():
    """Alpaca rejects GTC on fractional shares."""
    s = _FakeSession(_FakeResponse(payload=ORDER_JSON))
    b = _broker(s)
    b.place_order(OrderRequest(symbol='AAPL', quantity=1.5, side='buy',
                               order_type='market', time_in_force='gtc'))
    assert s.calls[0]['json']['time_in_force'] == 'day'


def test_the_order_response_is_parsed_out_of_the_json():
    s = _FakeSession(_FakeResponse(payload=ORDER_JSON))
    resp = _broker(s).place_order(
        OrderRequest(symbol='AAPL', quantity=3, side='buy',
                     order_type='market', time_in_force='day'))
    assert resp.order_id == 'abc-123'
    assert resp.filled_quantity == 3.0
    assert resp.filled_avg_price == 190.25
    assert resp.limit_price is None           # null stays None, not 0.0
    assert resp.created_at.year == 2026
    assert resp.broker_name == 'alpaca'


def test_an_api_error_surfaces_alpacas_own_message():
    s = _FakeSession(_FakeResponse(403, {'message': 'insufficient buying power'}))
    with pytest.raises(AlpacaAPIError) as e:
        _broker(s)._request('GET', 'account')
    assert 'insufficient buying power' in str(e.value)
    assert '403' in str(e.value)


def test_place_order_returns_none_on_an_api_error_rather_than_raising():
    """The trading loop treats None as 'no fill' and carries on."""
    s = _FakeSession(_FakeResponse(422, {'message': 'asset not tradable'}))
    assert _broker(s).place_order(
        OrderRequest(symbol='XXXX', quantity=1, side='buy',
                     order_type='market', time_in_force='day')) is None


def test_every_request_carries_a_timeout():
    """Plain requests has no default. A hung socket would stall the loop."""
    s = _FakeSession(_FakeResponse(payload={}))
    b = _broker(s)
    b.get_positions()
    assert s.calls[0]['timeout'] == 15.0


def test_the_account_and_positions_parse():
    s = _FakeSession(
        _FakeResponse(payload={'id': 'acct-1', 'cash': '9000.5',
                               'equity': '10000', 'buying_power': '20000',
                               'daytrade_count': 2}),
        _FakeResponse(payload=[{'symbol': 'AAPL', 'qty': '3',
                                'avg_entry_price': '180', 'current_price': '190',
                                'market_value': '570', 'unrealized_pl': '30',
                                'unrealized_plpc': '0.0555', 'cost_basis': '540'}]),
    )
    b = _broker(s)
    acct = b.get_account_info()
    assert acct.cash == 9000.5 and acct.day_trade_count == 2
    assert acct.initial_margin == 0.0       # absent field, not a crash
    pos = b.get_positions()
    assert pos[0].symbol == 'AAPL' and pos[0].unrealized_pl == 30.0


def test_cancel_order_uses_delete_and_tolerates_an_empty_body():
    s = _FakeSession(_FakeResponse(204, payload=None, text=''))
    b = _broker(s)
    assert b.cancel_order('abc-123') is True
    assert s.calls[0]['method'] == 'DELETE'
    assert s.calls[0]['url'].endswith('/v2/orders/abc-123')


def test_connect_without_credentials_fails_without_calling_the_api():
    b = AlpacaBroker({'api_key': '', 'api_secret': ''})
    s = _FakeSession()
    b.session = s
    assert b.connect() is False
    assert b.is_connected is False
    assert s.calls == []


def test_connect_verifies_the_credentials_against_the_account_endpoint():
    s = _FakeSession(_FakeResponse(payload={'id': 'acct-1', 'status': 'ACTIVE'}))
    b = AlpacaBroker({'api_key': 'k', 'api_secret': 's'})
    b._build_session = lambda: s
    assert b.connect() is True
    assert b.is_connected is True
    assert s.calls[0]['url'] == f'{PAPER_BASE_URL}/v2/account'
