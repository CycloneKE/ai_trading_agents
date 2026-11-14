"""Buys go in as price-capped (limit) orders; exits stay market orders
(src/agent/main.py, TradingAgent._entry_order_terms).

Before: every order was a market order, so a buy in a thin or fast market
could fill well above the price the decision was made on, and the
execution settings that claimed to control this (max_slippage, TWAP, order
slicing) were read by a module the agent built and never used.
"""
from types import SimpleNamespace

import pytest

from src.agent.main import TradingAgent


def _terms(action, asset_type, price, tif='day', **execution):
    agent = SimpleNamespace(config={'execution': {'max_slippage': 0.005, 'limit_entries': True, **execution}})
    return TradingAgent._entry_order_terms(agent, action, asset_type, price, tif)


def test_a_buy_is_capped_half_a_percent_above_the_decision_price():
    assert _terms('buy', 'stock', 200.0) == ('limit', 201.0, 'day')


def test_a_sell_always_goes_at_market_so_an_exit_is_never_left_unfilled():
    assert _terms('sell', 'stock', 200.0) == ('market', None, 'day')


def test_a_crypto_buy_is_capped_and_cancels_what_does_not_fill_at_once():
    order_type, limit, tif = _terms('buy', 'crypto', 60000.0, tif='gtc')
    assert (order_type, tif) == ('limit', 'ioc') and limit == pytest.approx(60300.0)


def test_cheap_prices_keep_four_decimals():
    assert _terms('buy', 'crypto', 0.5)[1] == 0.5025


def test_forex_and_a_switched_off_cap_stay_market():
    assert _terms('buy', 'forex', 1.1)[0] == 'market'
    assert _terms('buy', 'stock', 200.0, limit_entries=False)[0] == 'market'
    assert _terms('buy', 'stock', 200.0, max_slippage=0)[0] == 'market'
    assert _terms('buy', 'stock', 0)[0] == 'market'                 # no price, no cap to set


def test_the_shipped_config_has_only_settings_that_are_read():
    import json
    with open('config/config.json') as f:
        cfg = json.load(f)
    assert {k for k in cfg['execution'] if not k.startswith('_')} == {'max_slippage', 'limit_entries'}
    assert 'nlp' not in cfg and 'automl' not in cfg


def test_the_paper_simulator_fills_a_capped_buy_at_the_market_not_the_cap():
    """It filled a buy limit at the limit itself, which would have charged
    every price-capped paper buy the whole 0.5% cap."""
    from src.agent.paper_trading import PaperTradingBroker
    from src.connectors.base_broker import OrderResponse
    fills = []
    broker = SimpleNamespace(slippage_percent=0.1,
                             _get_current_price=lambda symbol, live_only=False: 200.0,
                             _execute_order=lambda order, price: fills.append(price))

    def order(side, limit):
        from datetime import datetime
        now = datetime(2026, 11, 13, 15, 0)
        return OrderResponse(order_id='o', client_order_id='c', symbol='AAPL', quantity=1,
                             filled_quantity=0, side=side, order_type='limit', status='pending',
                             created_at=now, updated_at=now, limit_price=limit)
    PaperTradingBroker._process_order(broker, order('buy', 201.0))
    PaperTradingBroker._process_order(broker, order('sell', 199.0))
    PaperTradingBroker._process_order(broker, order('buy', 200.1))     # simulated slippage stops at the cap
    PaperTradingBroker._process_order(broker, order('buy', 199.0))     # below the market: waits
    assert fills == pytest.approx([200.2, 199.8, 200.1])
