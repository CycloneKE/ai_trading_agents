"""Durability layer for the SQLite-backed journals.

The live agent lost a stop-loss to `database is locked` (10 occurrences on
2026-07-24) because a second process held the write lock while the risk
control tried to journal its intent. A risk control that dies on lock
contention is not a risk control, so every write goes through a bounded
retry and every connection carries a busy_timeout.
"""

import sqlite3
import threading
import time

import pytest

from src.utils.sqlite_util import connect, write_retry, BUSY_TIMEOUT_MS


def test_connect_sets_wal_and_busy_timeout(tmp_path):
    conn = connect(str(tmp_path / 'x.db'))
    try:
        assert conn.execute('PRAGMA journal_mode').fetchone()[0].lower() == 'wal'
        assert conn.execute('PRAGMA busy_timeout').fetchone()[0] == BUSY_TIMEOUT_MS
    finally:
        conn.close()


def test_write_retry_returns_value_on_first_success():
    assert write_retry(lambda: 'ok') == 'ok'


def test_write_retry_recovers_from_transient_lock():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise sqlite3.OperationalError('database is locked')
        return 'ok'

    assert write_retry(flaky, attempts=5, base_delay=0.001) == 'ok'
    assert len(calls) == 3


def test_write_retry_gives_up_and_raises_after_attempts():
    def always_locked():
        raise sqlite3.OperationalError('database is locked')

    with pytest.raises(sqlite3.OperationalError):
        write_retry(always_locked, attempts=3, base_delay=0.001)


def test_write_retry_does_not_swallow_real_errors():
    """A schema/constraint error must surface immediately, not be retried —
    retrying an IntegrityError would silently burn the idempotency check."""
    calls = []

    def bad_sql():
        calls.append(1)
        raise sqlite3.IntegrityError('UNIQUE constraint failed')

    with pytest.raises(sqlite3.IntegrityError):
        write_retry(bad_sql, attempts=3, base_delay=0.001)
    assert len(calls) == 1


def test_write_survives_a_real_concurrent_writer(tmp_path):
    """End-to-end: a second connection holds the write lock, and the retry
    still lands the row once the lock clears."""
    db = str(tmp_path / 'contended.db')
    a = connect(db)
    b = connect(db)
    a.execute('CREATE TABLE t (v INTEGER)')
    a.commit()

    b.execute('BEGIN IMMEDIATE')
    b.execute('INSERT INTO t VALUES (1)')

    released = threading.Event()

    def release_soon():
        time.sleep(0.25)
        b.commit()
        released.set()

    threading.Thread(target=release_soon, daemon=True).start()

    def do_write():
        a.execute('INSERT INTO t VALUES (2)')
        a.commit()

    write_retry(do_write, attempts=10, base_delay=0.05)
    assert released.is_set()
    assert a.execute('SELECT count(*) FROM t').fetchone()[0] == 2
    a.close()
    b.close()
