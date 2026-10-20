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


# ---------------------------------------------------------------------------
# Lock durability
#
# Regression for 2026-07-24: `Error enforcing stop-loss for TSLA: database is
# locked` x10 while the position was 21% underwater. The journal is the
# gate every order passes through, so a contended write must wait, not fail.
# ---------------------------------------------------------------------------

def test_journal_connection_carries_busy_timeout():
    from src.utils.sqlite_util import BUSY_TIMEOUT_MS
    journal, path = make_journal()
    try:
        got = journal._conn.execute('PRAGMA busy_timeout').fetchone()[0]
        assert got == BUSY_TIMEOUT_MS
    finally:
        journal.close()
        os.unlink(path)


class _FlakyConn:
    """Wraps a real connection and injects N lock errors on statements
    starting with `verb`, so the retry path can be exercised deterministically
    (sqlite3.Connection.execute itself is read-only and cannot be patched)."""

    def __init__(self, conn, verb, failures):
        self._conn = conn
        self._verb = verb.upper()
        self.left = failures

    def execute(self, sql, *args, **kwargs):
        import sqlite3
        if sql.lstrip().upper().startswith(self._verb) and self.left > 0:
            self.left -= 1
            raise sqlite3.OperationalError('database is locked')
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_record_intent_retries_past_a_lock_instead_of_propagating():
    """sqlite's own busy_timeout covers short contention; this covers what
    happens when it is exhausted. Before the fix the OperationalError
    escaped into _enforce_stop_losses and the stop-loss was skipped for that
    cycle — which is exactly what the 2026-07-24 log shows."""
    journal, path = make_journal()
    try:
        real = journal._conn
        flaky = _FlakyConn(real, 'INSERT', 2)
        journal._conn = flaky
        coid = make_client_order_id('momentum', 'AAPL', 'buy', 99)
        assert journal.record_intent(coid, 'AAPL', 'buy', 5, 'market') is True
        assert flaky.left == 0                # both lock errors were absorbed
        journal._conn = real
        assert journal.get(coid)['status'] == 'intent'
    finally:
        journal.close()
        os.unlink(path)


def test_mark_final_retries_past_a_lock():
    """Fill resolution matters as much as submission — an unretried lock here
    leaves the journal claiming an order is still in flight."""
    journal, path = make_journal()
    try:
        coid = make_client_order_id('momentum', 'AAPL', 'buy', 3)
        journal.record_intent(coid, 'AAPL', 'buy', 5, 'market')

        real = journal._conn
        flaky = _FlakyConn(real, 'UPDATE', 1)
        journal._conn = flaky
        journal.mark_final(coid, 'filled', 5, 100.0)
        journal._conn = real
        assert journal.get(coid)['status'] == 'filled'
    finally:
        journal.close()
        os.unlink(path)


def test_duplicate_is_still_refused_under_contention():
    """The retry must not turn an IntegrityError into a retry-then-succeed:
    a duplicate decision has to stay refused, contention or not."""
    journal, path = make_journal()
    try:
        coid = make_client_order_id('momentum', 'AAPL', 'buy', 7)
        assert journal.record_intent(coid, 'AAPL', 'buy', 5, 'market') is True
        assert journal.record_intent(coid, 'AAPL', 'buy', 5, 'market') is False
    finally:
        journal.close()
        os.unlink(path)
