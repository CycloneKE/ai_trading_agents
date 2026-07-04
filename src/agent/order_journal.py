"""Write-ahead order journal for idempotent order submission.

Every order gets a deterministic client_order_id and is journaled BEFORE it
is sent to the broker:

    intent -> submitted -> (filled | canceled | rejected | expired)
           -> failed   (broker call raised / returned nothing)
           -> aborted  (intent never reached the broker; set by reconcile)

Idempotency rests on two rules:
1. ``record_intent`` refuses a client_order_id it has already seen, so the
   same logical decision (same strategy/symbol/side/cycle) cannot be
   submitted twice — even across a crash and restart.
2. The client_order_id is forwarded to the broker (Alpaca deduplicates on
   it server-side), so a retry after a lost response cannot double-fill.

``reconcile`` runs at startup: every non-final row is checked against the
broker, so the agent never starts trading with unknown in-flight state.
"""

import logging
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FINAL_STATUSES = {'filled', 'canceled', 'cancelled', 'rejected', 'expired', 'failed', 'aborted'}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    broker_order_id TEXT,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    quantity        REAL NOT NULL,
    order_type      TEXT NOT NULL,
    limit_price     REAL,
    strategy        TEXT,
    status          TEXT NOT NULL DEFAULT 'intent',
    detail          TEXT,
    filled_quantity REAL,
    filled_avg_price REAL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
"""


def make_client_order_id(strategy: str, symbol: str, side: str, cycle: int) -> str:
    """Deterministic id for one logical trading decision.

    The same (strategy, symbol, side, cycle) always maps to the same id, so
    retries and restarts within the same decision cycle collapse into one
    order. Alpaca allows up to 128 chars; keep ids short and log-friendly.
    """
    safe = lambda s: ''.join(c for c in str(s) if c.isalnum() or c in '-_')[:24]
    return f"aegis-{safe(strategy)}-{safe(symbol)}-{safe(side)}-{cycle}"


class OrderJournal:
    """SQLite-backed write-ahead journal. Thread-safe via a module lock
    (the trading loop, stop-loss enforcement and reconcile can run from
    different threads)."""

    def __init__(self, db_path: str = os.path.join('data', 'order_journal.db')):
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record_intent(self, client_order_id: str, symbol: str, side: str,
                      quantity: float, order_type: str,
                      strategy: Optional[str] = None,
                      limit_price: Optional[float] = None) -> bool:
        """Journal the intent to place an order.

        Returns True if the caller OWNS this order and may submit it.
        Returns False if the id was already journaled (duplicate decision, a
        retry after restart, or an order already in flight) — the caller
        must NOT submit.
        """
        now = datetime.utcnow().isoformat()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO orders (client_order_id, symbol, side, quantity,"
                    " order_type, limit_price, strategy, status, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, 'intent', ?, ?)",
                    (client_order_id, symbol, side, quantity, order_type,
                     limit_price, strategy, now, now))
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                logger.warning(
                    f"Duplicate order intent blocked: {client_order_id} "
                    f"({side} {quantity} {symbol})")
                return False

    def mark_submitted(self, client_order_id: str, broker_order_id: str,
                       status: str = 'submitted'):
        self._update(client_order_id, status='submitted',
                     broker_order_id=broker_order_id,
                     detail=f'broker_status={status}')

    def mark_failed(self, client_order_id: str, reason: str):
        """Broker call failed or returned nothing. The order may or may not
        have reached the broker — reconcile() resolves it on next startup;
        the id stays burned so this decision is never retried blindly."""
        self._update(client_order_id, status='failed', detail=reason)

    def mark_final(self, client_order_id: str, status: str,
                   filled_quantity: Optional[float] = None,
                   filled_avg_price: Optional[float] = None):
        self._update(client_order_id, status=status,
                     filled_quantity=filled_quantity,
                     filled_avg_price=filled_avg_price)

    def _update(self, client_order_id: str, **fields):
        fields['updated_at'] = datetime.utcnow().isoformat()
        cols = ', '.join(f'{k} = ?' for k in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE orders SET {cols} WHERE client_order_id = ?",
                (*fields.values(), client_order_id))
            self._conn.commit()

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,))
            cur.row_factory = sqlite3.Row
            row = cur.fetchone()
        return dict(row) if row else None

    def unresolved(self) -> List[Dict[str, Any]]:
        """Rows whose outcome is unknown (intent or submitted, plus failed
        rows that may have reached the broker)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM orders WHERE status IN ('intent', 'submitted', 'failed')"
                " ORDER BY created_at")
            cur.row_factory = sqlite3.Row
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def filled_orders(self) -> List[Dict[str, Any]]:
        """All filled orders, oldest first — the input to P&L attribution."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM orders WHERE status = 'filled'"
                " AND filled_quantity > 0 ORDER BY created_at")
            cur.row_factory = sqlite3.Row
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def orders_for_symbol(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Every journaled order for a symbol, newest first — the drill-down's
        order/fill history (decision price, fill price, status, strategy)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM orders WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                (symbol, limit))
            cur.row_factory = sqlite3.Row
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def sync_fills(self, broker) -> int:
        """Resolve 'submitted' rows against the broker (fills, cancels).

        Safe to run every loop: it only touches rows already at the broker,
        never 'intent' rows that another thread may be mid-submission on.
        Returns the number of rows that reached a final status.
        """
        updated = 0
        api = getattr(broker, 'api', None)
        with self._lock:
            cur = self._conn.execute(
                "SELECT client_order_id FROM orders WHERE status = 'submitted'")
            submitted = [r[0] for r in cur.fetchall()]
        for coid in submitted:
            try:
                if api is not None and hasattr(api, 'get_order_by_client_order_id'):
                    o = api.get_order_by_client_order_id(coid)
                else:
                    continue
                status = str(getattr(o, 'status', ''))
                if status in FINAL_STATUSES:
                    qty = float(getattr(o, 'filled_qty', 0) or 0)
                    avg_raw = getattr(o, 'filled_avg_price', None)
                    self.mark_final(coid, status, qty,
                                    float(avg_raw) if avg_raw else None)
                    updated += 1
            except Exception as e:
                logger.debug(f"sync_fills: could not resolve {coid}: {e}")
        return updated

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,))
            cur.row_factory = sqlite3.Row
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------

    def reconcile(self, broker) -> Dict[str, int]:
        """Resolve every unresolved journal row against the broker.

        Must run after broker.connect() and before the trading loop starts.
        Uses Alpaca's lookup-by-client-order-id when available; otherwise
        falls back to matching the broker's open orders.
        """
        summary = {'checked': 0, 'resolved': 0, 'still_open': 0, 'aborted': 0}
        open_orders = None  # lazy fallback

        for row in self.unresolved():
            summary['checked'] += 1
            coid = row['client_order_id']
            broker_order = None

            api = getattr(broker, 'api', None)
            if api is not None and hasattr(api, 'get_order_by_client_order_id'):
                try:
                    broker_order = api.get_order_by_client_order_id(coid)
                except Exception:
                    broker_order = None  # broker has no record of this id
            else:
                if open_orders is None:
                    try:
                        open_orders = broker.get_orders() or []
                    except Exception:
                        open_orders = []
                broker_order = next(
                    (o for o in open_orders if getattr(o, 'client_order_id', None) == coid),
                    None)

            if broker_order is None:
                # Never reached the broker: safe to close out as aborted.
                self._update(coid, status='aborted',
                             detail='no broker record at reconcile')
                summary['aborted'] += 1
                logger.info(f"Reconcile: {coid} aborted (never reached broker)")
                continue

            status = str(getattr(broker_order, 'status', 'unknown'))
            filled_qty = float(getattr(broker_order, 'filled_qty', 0) or
                               getattr(broker_order, 'filled_quantity', 0) or 0)
            avg_raw = (getattr(broker_order, 'filled_avg_price', None) or
                       getattr(broker_order, 'avg_fill_price', None))
            avg = float(avg_raw) if avg_raw else None

            if status in FINAL_STATUSES:
                self.mark_final(coid, status, filled_qty, avg)
                summary['resolved'] += 1
                logger.info(f"Reconcile: {coid} -> {status} (filled {filled_qty})")
            else:
                self.mark_submitted(coid, str(getattr(broker_order, 'id', '') or
                                              getattr(broker_order, 'order_id', '')),
                                    status)
                summary['still_open'] += 1
                logger.info(f"Reconcile: {coid} still open at broker ({status})")

        if summary['checked']:
            logger.warning(f"Order journal reconciliation: {summary}")
        else:
            logger.info("Order journal reconciliation: no unresolved orders")
        return summary

    def close(self):
        with self._lock:
            self._conn.close()
