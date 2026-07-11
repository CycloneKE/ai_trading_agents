"""Tracks declared/received dividends per sleeve holding and the resulting
sleeve cash available for the next accumulation cycle (manual DRIP — the
cash is just more capital for SleeveManager to allocate, no separate
reinvestment code path).
"""
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sleeve_dividends (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at   TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    declared_date TEXT,
    amount_kes    REAL NOT NULL,
    status        TEXT DEFAULT 'received',  -- 'declared' | 'received' | 'swept'
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_sleeve_dividends_symbol ON sleeve_dividends(symbol, status);
"""


class DividendLedger:
    def __init__(self, db_path: str = str(DATA_DIR / 'sleeve.db')):
        self._lock = threading.Lock()
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info(f"DividendLedger initialized at {db_path}")

    def record_dividend(self, symbol: str, amount_kes: float,
                        declared_date: Optional[str] = None,
                        status: str = 'received', notes: str = '') -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sleeve_dividends (recorded_at, symbol, declared_date,"
                " amount_kes, status, notes) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), symbol.upper(), declared_date,
                 float(amount_kes), status, notes))
            self._conn.commit()
            return cur.lastrowid

    def unswept_cash_kes(self) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(amount_kes), 0) FROM sleeve_dividends"
                " WHERE status = 'received'").fetchone()
            return float(row[0])

    def mark_swept(self) -> int:
        """Called once SleeveManager has folded unswept cash into a cycle's
        capital, so it isn't counted again next cycle."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sleeve_dividends SET status = 'swept' WHERE status = 'received'")
            self._conn.commit()
            return cur.rowcount

    def history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            if symbol:
                cur = self._conn.execute(
                    "SELECT * FROM sleeve_dividends WHERE symbol = ?"
                    " ORDER BY recorded_at DESC LIMIT ?", (symbol.upper(), limit))
            else:
                cur = self._conn.execute(
                    "SELECT * FROM sleeve_dividends ORDER BY recorded_at DESC LIMIT ?", (limit,))
            cur.row_factory = sqlite3.Row
            return [dict(r) for r in cur.fetchall()]

    def close(self):
        with self._lock:
            self._conn.close()
