"""Unit tests for the write-ahead order journal (idempotency layer)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.order_journal import OrderJournal, make_client_order_id


def make_journal():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    return OrderJournal(db_path=path), path


def test_client_order_id_deterministic_and_safe():
    a = make_client_order_id('momentum', 'AAPL', 'buy', 12345)
    b = make_client_order_id('momentum', 'AAPL', 'buy', 12345)
    assert a == b
    assert make_client_order_id('momentum', 'AAPL', 'buy', 12346) != a
    # characters that would break broker ids get stripped
    assert ' ' not in make_client_order_id('mean reversion!', 'BRK.B', 'buy', 1)


def test_duplicate_intent_blocked():
    journal, path = make_journal()
    try:
        coid = make_client_order_id('momentum', 'AAPL', 'buy', 1)
        assert journal.record_intent(coid, 'AAPL', 'buy', 5, 'market') is True
        # Same decision again (retry/restart) must be refused
        assert journal.record_intent(coid, 'AAPL', 'buy', 5, 'market') is False
        # Even after failure the id stays burned
        journal.mark_failed(coid, 'timeout')
        assert journal.record_intent(coid, 'AAPL', 'buy', 5, 'market') is False
    finally:
        journal.close()
        os.unlink(path)


def test_lifecycle_and_unresolved():
    journal, path = make_journal()
    try:
        c1 = make_client_order_id('s', 'AAPL', 'buy', 1)
        c2 = make_client_order_id('s', 'MSFT', 'buy', 1)
        c3 = make_client_order_id('s', 'SPY', 'buy', 1)
        for coid, sym in [(c1, 'AAPL'), (c2, 'MSFT'), (c3, 'SPY')]:
            journal.record_intent(coid, sym, 'buy', 1, 'market')

        journal.mark_submitted(c1, 'broker-1')
        journal.mark_final(c1, 'filled', 1, 100.0)
        journal.mark_submitted(c2, 'broker-2')
        # c3 stays at intent (simulated crash before submit)

        unresolved = {r['client_order_id'] for r in journal.unresolved()}
        assert unresolved == {c2, c3}
        assert journal.get(c1)['status'] == 'filled'
        assert journal.get(c1)['filled_avg_price'] == 100.0
    finally:
        journal.close()
        os.unlink(path)


class FakeOrder:
    def __init__(self, coid, status, filled_qty=0, avg=None, oid='b-1'):
        self.client_order_id = coid
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = avg
        self.id = oid


class FakeAlpacaApi:
    def __init__(self, orders):
        self._orders = {o.client_order_id: o for o in orders}

    def get_order_by_client_order_id(self, coid):
        if coid not in self._orders:
            raise Exception('order not found')
        return self._orders[coid]


class FakeBroker:
    def __init__(self, orders):
        self.api = FakeAlpacaApi(orders)


def test_reconcile_resolves_crash_states():
    journal, path = make_journal()
    try:
        filled = make_client_order_id('s', 'AAPL', 'buy', 1)   # filled at broker
        open_ = make_client_order_id('s', 'MSFT', 'buy', 1)    # still open at broker
        lost = make_client_order_id('s', 'SPY', 'buy', 1)      # never reached broker

        for coid, sym in [(filled, 'AAPL'), (open_, 'MSFT'), (lost, 'SPY')]:
            journal.record_intent(coid, sym, 'buy', 1, 'market')
        # crash happened: none were marked submitted

        broker = FakeBroker([
            FakeOrder(filled, 'filled', filled_qty=1, avg=101.5),
            FakeOrder(open_, 'new'),
        ])
        summary = journal.reconcile(broker)

        assert summary == {'checked': 3, 'resolved': 1, 'still_open': 1, 'aborted': 1}
        assert journal.get(filled)['status'] == 'filled'
        assert journal.get(filled)['filled_avg_price'] == 101.5
        assert journal.get(open_)['status'] == 'submitted'
        assert journal.get(lost)['status'] == 'aborted'
        # nothing left unresolved except the genuinely open order
        assert {r['client_order_id'] for r in journal.unresolved()} == {open_}
    finally:
        journal.close()
        os.unlink(path)
