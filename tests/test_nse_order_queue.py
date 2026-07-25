# tests/test_nse_order_queue.py
import os
import sqlite3
import tempfile

import pytest

from src.agent.nse_order_queue import NseOrderQueue


@pytest.fixture
def queue():
    path = os.path.join(tempfile.mkdtemp(), 'nse_test.db')
    q = NseOrderQueue(path)
    yield q
    q.close()


def test_create_and_get_pending(queue):
    tid = queue.create_ticket('SCOM', 'buy', 500, suggested_limit_price=15.2,
                              rationale='r', ensemble_confidence=0.7)
    assert tid is not None
    pending = queue.get_pending()
    assert len(pending) == 1
    assert pending[0]['symbol'] == 'SCOM'
    assert pending[0]['status'] == 'pending'


def test_duplicate_pending_symbol_side_is_deduped(queue):
    first = queue.create_ticket('SCOM', 'buy', 500, suggested_limit_price=15.2)
    dup = queue.create_ticket('SCOM', 'buy', 100, suggested_limit_price=15.3)
    assert first is not None
    assert dup is None  # same symbol+side already pending
    # opposite side is allowed
    other = queue.create_ticket('SCOM', 'sell', 100, suggested_limit_price=15.3)
    assert other is not None


def test_invalid_inputs_rejected(queue):
    assert queue.create_ticket('SCOM', 'hodl', 10) is None       # bad side
    assert queue.create_ticket('SCOM', 'buy', 0) is None         # non-positive qty


def test_place_then_fill_round_trip(queue):
    tid = queue.create_ticket('EQTY', 'buy', 200, suggested_limit_price=40.0)
    assert queue.mark_placed(tid) is True
    ok, ticket = queue.mark_filled(tid, 41.0, 200)
    assert ok is True
    assert ticket['status'] == 'filled'
    assert ticket['fill_price'] == 41.0
    assert queue.get_pending() == []
    assert len(queue.recent_fills()) == 1


def test_fill_with_bad_values_rejected(queue):
    tid = queue.create_ticket('KCB', 'sell', 300, suggested_limit_price=30.0)
    ok, _ = queue.mark_filled(tid, 0, 300)       # zero price
    assert ok is False
    ok, _ = queue.mark_filled(tid, 30.0, 0)      # zero qty
    assert ok is False
    # ticket is still pending after failed fills
    assert len(queue.get_pending()) == 1


def test_cancel(queue):
    tid = queue.create_ticket('COOP', 'buy', 100, suggested_limit_price=12.0)
    assert queue.cancel(tid) is True
    assert queue.get_pending() == []
    # cannot fill a cancelled ticket
    ok, _ = queue.mark_filled(tid, 12.0, 100)
    assert ok is False


def test_positions_net_from_fills(queue):
    b = queue.create_ticket('SCOM', 'buy', 1000, suggested_limit_price=15.0)
    queue.mark_filled(b, 15.0, 1000)
    s = queue.create_ticket('SCOM', 'sell', 400, suggested_limit_price=16.0)
    queue.mark_filled(s, 16.0, 400)
    pos = queue.positions()
    assert pos['SCOM']['quantity'] == 600            # 1000 bought - 400 sold
    assert pos['SCOM']['avg_entry_price_kes'] == 15.0  # cost basis from the buy


def test_mark_filled_writes_order_journal(queue):
    class FakeJournal:
        def __init__(self):
            self.intents = []
            self.finals = []

        def record_intent(self, coid, symbol, side, qty, order_type, strategy, limit_price):
            self.intents.append((coid, symbol, side, qty, strategy))
            return True

        def mark_final(self, coid, status, filled_quantity=None, filled_avg_price=None):
            self.finals.append((coid, status, filled_quantity, filled_avg_price))

    j = FakeJournal()
    tid = queue.create_ticket('EABL', 'buy', 50, suggested_limit_price=150.0)
    ok, _ = queue.mark_filled(tid, 152.0, 50, order_journal=j)
    assert ok is True
    assert len(j.intents) == 1 and j.intents[0][1] == 'EABL'
    assert j.finals[0][1] == 'filled' and j.finals[0][3] == 152.0


def test_default_book_is_trading(queue):
    queue.create_ticket('SCOM', 'buy', 100, suggested_limit_price=15.0)
    pending = queue.get_pending()
    assert pending[0]['book'] == 'trading'


def test_long_term_book_does_not_collide_with_trading_dup_check(queue):
    trading_id = queue.create_ticket('SCOM', 'buy', 100, suggested_limit_price=15.0, book='trading')
    sleeve_id = queue.create_ticket('SCOM', 'buy', 50, suggested_limit_price=15.0, book='long_term')
    assert trading_id is not None
    assert sleeve_id is not None
    books = {t['book'] for t in queue.get_pending()}
    assert books == {'trading', 'long_term'}


def test_positions_filtered_by_book(queue):
    trading_id = queue.create_ticket('SCOM', 'buy', 1000, book='trading')
    queue.mark_filled(trading_id, 15.0, 1000)
    sleeve_id = queue.create_ticket('SCOM', 'buy', 200, book='long_term')
    queue.mark_filled(sleeve_id, 16.0, 200)

    trading_only = queue.positions(book='trading')
    sleeve_only = queue.positions(book='long_term')
    everything = queue.positions()

    assert trading_only['SCOM']['quantity'] == 1000
    assert sleeve_only['SCOM']['quantity'] == 200
    assert everything['SCOM']['quantity'] == 1200


def test_migrates_existing_db_without_book_column(tmp_path):
    path = str(tmp_path / 'old_schema.db')
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE nse_order_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            symbol TEXT NOT NULL, side TEXT NOT NULL, quantity INTEGER NOT NULL,
            suggested_limit_price REAL, rationale TEXT, ensemble_confidence REAL,
            llm_reasoning TEXT, status TEXT DEFAULT 'pending', fill_price REAL,
            fill_quantity INTEGER, fill_at TEXT, operator_notes TEXT, resolved_by TEXT
        );
    """)
    conn.execute(
        "INSERT INTO nse_order_tickets (created_at, symbol, side, quantity, status)"
        " VALUES ('2026-01-01T00:00:00', 'OLD', 'buy', 10, 'pending')")
    conn.commit()
    conn.close()

    q = NseOrderQueue(path)
    pending = q.get_pending()
    assert len(pending) == 1
    assert pending[0]['book'] == 'trading'  # DEFAULT backfilled onto the pre-existing row
    q.close()


def test_expire_stale_spares_long_term_book(queue):
    tid_trading = queue.create_ticket('SCOM', 'buy', 100, book='trading')
    tid_sleeve = queue.create_ticket('EQTY', 'buy', 50, book='long_term')
    # age both tickets far past the cutoff
    with queue._lock:
        queue._conn.execute("UPDATE nse_order_tickets SET created_at = '2020-01-01T00:00:00'")
        queue._conn.commit()
    expired = queue.expire_stale(max_age_hours=24)
    assert expired == 1  # only the trading ticket
    pending = queue.get_pending()
    assert [t['symbol'] for t in pending] == ['EQTY']
    assert pending[0]['book'] == 'long_term'
