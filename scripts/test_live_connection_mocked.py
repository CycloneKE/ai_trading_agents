#!/usr/bin/env python3
"""Unit-style mocked tests for broker connections.

This file fakes a `LiveCoinbaseBroker` to validate how higher-level code handles broker methods
without making network calls.
"""

def make_fake_broker():
    class FakeBroker:
        def __init__(self):
            self.connected = True

        def connect(self):
            self.connected = True
            return True

        def get_accounts(self):
            return [{"currency": "USD", "balance": "1000", "available": "1000"}]

        def get_ticker(self, symbol):
            return {"price": "100.0"}

        def get_orders(self):
            return []

        @property
        def is_connected(self):
            return self.connected

    return FakeBroker()


def test_fake_broker_basic():
    broker = make_fake_broker()
    assert broker.connect() is True
    accounts = broker.get_accounts()
    assert isinstance(accounts, list)
    ticker = broker.get_ticker('BTC-USD')
    assert 'price' in ticker
    orders = broker.get_orders()
    assert orders == []
