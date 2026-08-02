"""Shared SQLite durability helpers for the agent's journals.

Every journal in this system is a safety mechanism — the order journal is
the idempotency guarantee, the exposure guard is a risk limit. On
2026-07-24 a stop-loss was lost ten times to a bare ``database is locked``
because a second process (the API server) held the write lock. A control
that dies on lock contention is not a control.

Two rules, applied everywhere:

1. Every connection carries ``busy_timeout``, so SQLite itself waits for a
   contended lock instead of failing instantly.
2. Every write is wrapped in :func:`write_retry`, which retries ONLY the
   lock/busy family of errors. Logic errors (IntegrityError and friends)
   surface immediately — retrying an IntegrityError would silently defeat
   the duplicate-order check that idempotency rests on.
"""

import logging
import sqlite3
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')

BUSY_TIMEOUT_MS = 15000     # SQLite's own wait before it reports SQLITE_BUSY
DEFAULT_ATTEMPTS = 5
DEFAULT_BASE_DELAY = 0.05   # seconds; doubled per attempt

# Substrings that identify a *contention* failure, i.e. one that is worth
# retrying because nothing about the statement itself is wrong.
_TRANSIENT = ('database is locked', 'database table is locked', 'database is busy')


def connect(db_path: str, timeout: float = 30.0) -> sqlite3.Connection:
    """Open a WAL-mode connection with a busy timeout already applied."""
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=timeout)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute(f'PRAGMA busy_timeout={BUSY_TIMEOUT_MS}')
    return conn


def is_transient(err: BaseException) -> bool:
    """True for lock/busy contention, False for real logic errors."""
    if not isinstance(err, sqlite3.OperationalError):
        return False
    return any(marker in str(err).lower() for marker in _TRANSIENT)


def write_retry(fn: Callable[[], T],
                attempts: int = DEFAULT_ATTEMPTS,
                base_delay: float = DEFAULT_BASE_DELAY) -> T:
    """Run ``fn``, retrying with exponential backoff on lock contention.

    Re-raises the final error once ``attempts`` is exhausted: a write that
    cannot land must be visible to the caller, never silently dropped.
    """
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except sqlite3.Error as e:
            if not is_transient(e) or attempt == attempts:
                raise
            logger.warning(
                f"SQLite contention ({e}); retry {attempt}/{attempts - 1} in {delay:.3f}s")
            time.sleep(delay)
            delay *= 2
    raise AssertionError('unreachable')  # pragma: no cover
