"""NSE order-ticket queue.

The Nairobi Securities Exchange has no broker API the agent can call (the
user's broker, AIB-AXYS, is portal-only). So instead of routing an NSE trade
decision to a broker connector, the agent writes an *order ticket* here for a
human to key into the portal and then record the actual fill.

This mirrors EscalationManager's SQLite/WAL/lock style and shares the same
escalations.db file. It is the swap-in seam for a future real NSE broker
connector: such a connector would satisfy the same "place this order"
contract (create_ticket) and skip the manual review step.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nse_order_tickets (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT NOT NULL,
    symbol                TEXT NOT NULL,
    side                  TEXT NOT NULL,          -- 'buy' | 'sell'
    quantity              INTEGER NOT NULL,       -- NSE trades in whole shares
    suggested_limit_price REAL,                   -- KES
    rationale             TEXT,
    ensemble_confidence   REAL,
    llm_reasoning         TEXT,
    status                TEXT DEFAULT 'pending', -- pending|placed|filled|cancelled|expired
    fill_price            REAL,                   -- KES, operator-entered
    fill_quantity         INTEGER,
    fill_at               TEXT,
    operator_notes        TEXT,
    resolved_by           TEXT,
    book                  TEXT DEFAULT 'trading', -- 'trading' | 'long_term'
    fees_kes              REAL DEFAULT 0,         -- commission charged on the fill
    strategy              TEXT,                   -- what decided it: 'ensemble', 'stop_loss', ...
    strategy_weights      TEXT                    -- JSON {strategy: share of the credit}
);
CREATE INDEX IF NOT EXISTS idx_nse_tickets_status ON nse_order_tickets(status, created_at);
CREATE INDEX IF NOT EXISTS idx_nse_tickets_symbol ON nse_order_tickets(symbol);

-- When the agent's NSE paper account opened. One row; fills before it are
-- not the account's (see nse_paper_account.py).
CREATE TABLE IF NOT EXISTS nse_paper_account (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    started_at  TEXT NOT NULL
);

-- Highest price seen since each paper position opened, for trailing stops.
CREATE TABLE IF NOT EXISTS nse_paper_highs (
    symbol      TEXT PRIMARY KEY,
    opened_at   TEXT NOT NULL,
    high_kes    REAL NOT NULL
);
"""


class NseOrderQueue:
    def __init__(self, db_path: str = str(DATA_DIR / 'escalations.db')):
        self._lock = threading.Lock()
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate_add_book_column()
        logger.info(f"NseOrderQueue initialized at {db_path}")

    def _migrate_add_book_column(self) -> None:
        """CREATE TABLE IF NOT EXISTS doesn't add columns to a table that
        already existed before `book` (and later `fees_kes`) was introduced —
        do that explicitly so pre-existing escalations.db files keep working."""
        cols = [row[1] for row in self._conn.execute(
            "PRAGMA table_info(nse_order_tickets)").fetchall()]
        if 'book' not in cols:
            self._conn.execute(
                "ALTER TABLE nse_order_tickets ADD COLUMN book TEXT DEFAULT 'trading'")
            self._conn.commit()
        for col, ddl in (('fees_kes', 'REAL DEFAULT 0'), ('strategy', 'TEXT'),
                         ('strategy_weights', 'TEXT')):
            if col not in cols:
                self._conn.execute(f"ALTER TABLE nse_order_tickets ADD COLUMN {col} {ddl}")
                self._conn.commit()

    def create_ticket(self, symbol: str, side: str, quantity: int,
                      suggested_limit_price: Optional[float] = None,
                      rationale: str = '', ensemble_confidence: Optional[float] = None,
                      llm_reasoning: str = '', book: str = 'trading',
                      strategy: Optional[str] = None,
                      strategy_weights: Optional[Dict[str, float]] = None) -> Optional[int]:
        """Create a pending order ticket. Returns the ticket id, or None if an
        identical pending ticket (same symbol+side+book) already exists — the
        agent re-proposes the same trade every scrape/cycle while the signal
        persists, and we don't want a pile of duplicates waiting for the
        operator. `book` scopes the dedupe so a trading and a long_term
        ticket for the same symbol/side never collide."""
        symbol = symbol.upper()
        side = side.lower()
        if side not in ('buy', 'sell') or quantity <= 0:
            return None
        with self._lock:
            dup = self._conn.execute(
                "SELECT id FROM nse_order_tickets WHERE symbol = ? AND side = ?"
                " AND status = 'pending' AND book = ? LIMIT 1", (symbol, side, book)).fetchone()
            if dup:
                return None
            cur = self._conn.execute(
                "INSERT INTO nse_order_tickets (created_at, symbol, side, quantity,"
                " suggested_limit_price, rationale, ensemble_confidence, llm_reasoning,"
                " status, book, strategy, strategy_weights)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
                (datetime.utcnow().isoformat(), symbol, side, int(quantity),
                 suggested_limit_price, rationale, ensemble_confidence, llm_reasoning, book,
                 strategy, json.dumps(strategy_weights) if strategy_weights else None))
            self._conn.commit()
            return cur.lastrowid

    def get_pending(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM nse_order_tickets WHERE status IN ('pending', 'placed')"
                " ORDER BY created_at DESC LIMIT ?", (limit,))
            cur.row_factory = sqlite3.Row
            return [dict(r) for r in cur.fetchall()]

    def recent_fills(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM nse_order_tickets WHERE status = 'filled'"
                " ORDER BY fill_at DESC LIMIT ?", (limit,))
            cur.row_factory = sqlite3.Row
            return [dict(r) for r in cur.fetchall()]

    def _get(self, ticket_id: int) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM nse_order_tickets WHERE id = ?", (ticket_id,))
        cur.row_factory = sqlite3.Row
        row = cur.fetchone()
        return dict(row) if row else None

    def mark_placed(self, ticket_id: int, resolved_by: str = 'operator') -> bool:
        """Operator has keyed the order into the broker portal."""
        with self._lock:
            t = self._get(ticket_id)
            if not t or t['status'] != 'pending':
                return False
            self._conn.execute(
                "UPDATE nse_order_tickets SET status = 'placed', resolved_by = ?"
                " WHERE id = ?", (resolved_by, ticket_id))
            self._conn.commit()
            return True

    def mark_filled(self, ticket_id: int, fill_price: float, fill_quantity: int,
                    resolved_by: str = 'operator', notes: str = '',
                    order_journal=None,
                    fees_kes: float = 0.0) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Record the actual fill. Optionally mirror it into the order journal
        so there is one execution record across US and NSE trades. `fees_kes`
        is the commission charged, so a paper account can be kept net of it."""
        if not fill_price or fill_price <= 0 or not fill_quantity or fill_quantity <= 0:
            return False, None
        with self._lock:
            t = self._get(ticket_id)
            if not t or t['status'] not in ('pending', 'placed'):
                return False, None
            self._conn.execute(
                "UPDATE nse_order_tickets SET status = 'filled', fill_price = ?,"
                " fill_quantity = ?, fill_at = ?, resolved_by = ?, operator_notes = ?,"
                " fees_kes = ? WHERE id = ?",
                (float(fill_price), int(fill_quantity), datetime.utcnow().isoformat(),
                 resolved_by, notes, float(fees_kes or 0.0), ticket_id))
            self._conn.commit()
            filled = self._get(ticket_id)

        # Journal outside the lock (order_journal has its own lock).
        if order_journal is not None:
            try:
                coid = f"nse-{t['symbol']}-{t['side']}-{ticket_id}"
                owns = order_journal.record_intent(
                    coid, t['symbol'], t['side'], int(fill_quantity),
                    order_type='limit', strategy=t.get('strategy') or 'nse_manual',
                    strategy_weights=(json.loads(t['strategy_weights'])
                                      if t.get('strategy_weights') else None),
                    limit_price=float(fill_price))
                if owns:
                    order_journal.mark_final(coid, 'filled',
                                             filled_quantity=int(fill_quantity),
                                             filled_avg_price=float(fill_price))
            except Exception as e:
                logger.warning(f"Could not journal NSE fill for ticket {ticket_id}: {e}")
        return True, filled

    def cancel(self, ticket_id: int, resolved_by: str = 'operator', notes: str = '') -> bool:
        with self._lock:
            t = self._get(ticket_id)
            if not t or t['status'] not in ('pending', 'placed'):
                return False
            self._conn.execute(
                "UPDATE nse_order_tickets SET status = 'cancelled', resolved_by = ?,"
                " operator_notes = ? WHERE id = ?", (resolved_by, notes, ticket_id))
            self._conn.commit()
            return True

    def expire_stale(self, max_age_hours: int = 24) -> int:
        """Expire pending tickets older than max_age_hours so the operator is
        never shown a day-old signal to key in at a stale price.

        Scoped to book='trading' only. Sleeve (book='long_term') tickets are
        monthly, not intraday: the trading loop calls this at every NSE eval
        (every ~24h), so a sleeve ticket created e.g. over a weekend would
        otherwise get expired before the operator can act on it — and the
        SleeveManager's own month-gate then blocks re-issue until next month,
        silently losing that month's dividend sweep. Long_term tickets are
        deduped by the sleeve manager itself (create_ticket dedupes on
        symbol+side+book) and the operator can always cancel one manually if
        it's genuinely gone stale, so excluding them here is safe."""
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE nse_order_tickets SET status = 'expired'"
                " WHERE status = 'pending' AND created_at < ? AND book = 'trading'",
                (cutoff,))
            self._conn.commit()
            return cur.rowcount

    def fills(self, book: Optional[str] = None,
              since: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filled tickets, oldest first: symbol, side, quantity, price, fees.

        `since` (an ISO timestamp) keeps only fills at or after it, which is
        how a paper account ignores fills from before it opened.
        """
        query = ("SELECT symbol, side, fill_quantity, fill_price, fees_kes, fill_at"
                 " FROM nse_order_tickets WHERE status = 'filled' AND fill_quantity > 0")
        params: List[Any] = []
        if book is not None:
            query += " AND book = ?"
            params.append(book)
        if since is not None:
            query += " AND fill_at >= ?"
            params.append(since)
        query += " ORDER BY fill_at ASC, id ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [{'symbol': r[0], 'side': r[1], 'quantity': int(r[2]), 'price': float(r[3]),
                 'fees_kes': float(r[4] or 0.0), 'fill_at': r[5]} for r in rows]

    def paper_account_started_at(self) -> str:
        """When the NSE paper account opened, recording now on first call."""
        with self._lock:
            row = self._conn.execute(
                "SELECT started_at FROM nse_paper_account WHERE id = 1").fetchone()
            if row:
                return row[0]
            started = datetime.utcnow().isoformat()
            self._conn.execute(
                "INSERT INTO nse_paper_account (id, started_at) VALUES (1, ?)", (started,))
            self._conn.commit()
            return started

    def paper_high(self, symbol: str, opened_at: str, price: float) -> float:
        """The highest price since the position opened at `opened_at`,
        including `price`. A different opened_at is a new position, so the
        mark starts again from `price`."""
        with self._lock:
            row = self._conn.execute(
                "SELECT opened_at, high_kes FROM nse_paper_highs WHERE symbol = ?",
                (symbol,)).fetchone()
            high = max(row[1], price) if row and row[0] == opened_at else price
            self._conn.execute(
                "INSERT INTO nse_paper_highs (symbol, opened_at, high_kes) VALUES (?, ?, ?)"
                " ON CONFLICT(symbol) DO UPDATE SET opened_at = excluded.opened_at,"
                " high_kes = excluded.high_kes", (symbol, opened_at, float(high)))
            self._conn.commit()
            return high

    def positions(self, book: Optional[str] = None,
                  since: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Net position per symbol from filled tickets: signed quantity and
        the volume-weighted average entry price (KES). Pass `book` to scope
        to just 'trading' or 'long_term' fills; omit for the blended view.
        `since` keeps only fills at or after that ISO timestamp."""
        rows = [(f['symbol'], f['side'], f['quantity'], f['price'])
                for f in self.fills(book=book, since=since)]
        book_map: Dict[str, Dict[str, Any]] = {}
        for symbol, side, qty, price in rows:
            b = book_map.setdefault(symbol, {'quantity': 0, 'buy_qty': 0, 'buy_cost': 0.0})
            b['quantity'] += qty if side == 'buy' else -qty
            if side == 'buy':
                b['buy_qty'] += qty
                b['buy_cost'] += qty * price
        out = {}
        for symbol, b in book_map.items():
            avg = round(b['buy_cost'] / b['buy_qty'], 2) if b['buy_qty'] > 0 else 0.0
            out[symbol] = {'quantity': b['quantity'], 'avg_entry_price_kes': avg}
        return out

    def close(self):
        with self._lock:
            self._conn.close()
