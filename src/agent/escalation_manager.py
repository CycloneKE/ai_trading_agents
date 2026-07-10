"""
Escalation Manager: Handles the SQLite database logic for broker research uploads,
extracted research signals, operator escalation approval flows, and watchlist management.
"""
import os
import sqlite3
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_uploads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    source          TEXT DEFAULT 'aib_axys',
    uploaded_at     TEXT NOT NULL,
    processed_at    TEXT,
    signals_count   INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'processing'  -- 'processing' | 'completed' | 'failed'
);

CREATE TABLE IF NOT EXISTS research_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id       INTEGER REFERENCES research_uploads(id),
    symbol          TEXT NOT NULL,
    market          TEXT,
    current_price   REAL,
    target_price    REAL,
    recommendation  TEXT,
    rationale       TEXT,
    risk_factors    TEXT,  -- JSON array stored as text
    time_horizon    TEXT,
    confidence      REAL,
    extracted_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS escalations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       INTEGER REFERENCES research_signals(id),
    symbol          TEXT NOT NULL,
    action          TEXT NOT NULL,     -- 'follow' | 'unfollow' | 'override_stop_loss'
    reason          TEXT,
    risk_level      TEXT DEFAULT 'medium',
    status          TEXT DEFAULT 'pending',  -- 'pending' | 'approved' | 'rejected' | 'expired'
    operator_notes  TEXT,
    created_at      TEXT NOT NULL,
    resolved_at     TEXT,
    resolved_by     TEXT
);

CREATE TABLE IF NOT EXISTS position_watchlist (
    symbol          TEXT PRIMARY KEY,
    market          TEXT,
    source          TEXT,              -- 'config' | 'research_upload' | 'manual'
    recommendation  TEXT,
    target_price    REAL,
    rationale       TEXT,
    status          TEXT DEFAULT 'active',  -- 'active' | 'paused' | 'removed'
    added_at        TEXT NOT NULL,
    last_updated    TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_history (
    timestamp       TEXT PRIMARY KEY,
    value           REAL NOT NULL
);
"""


class EscalationManager:
    def __init__(self, db_path: str = os.path.join('data', 'escalations.db')):
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._lock = threading.Lock()
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info(f"EscalationManager database initialized at {db_path}")

    def record_upload(self, filename: str, source: str = 'aib_axys') -> int:
        """Record a new uploaded research document."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO research_uploads (filename, source, uploaded_at) VALUES (?, ?, ?)",
                (filename, source, datetime.utcnow().isoformat())
            )
            self._conn.commit()
            return cur.lastrowid

    def update_upload_status(self, upload_id: int, status: str, signals_count: int = 0) -> None:
        """Update processing status of a research document."""
        with self._lock:
            self._conn.execute(
                "UPDATE research_uploads SET status = ?, signals_count = ?, processed_at = ? WHERE id = ?",
                (status, signals_count, datetime.utcnow().isoformat(), upload_id)
            )
            self._conn.commit()

    def record_signal(self, upload_id: int, signal: Dict[str, Any]) -> int:
        """Record an extracted research signal."""
        import json
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO research_signals (upload_id, symbol, market, current_price, target_price, "
                "recommendation, rationale, risk_factors, time_horizon, confidence, extracted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    upload_id,
                    signal.get("symbol", "").upper(),
                    signal.get("market", "kenyan"),
                    signal.get("current_price"),
                    signal.get("target_price"),
                    signal.get("recommendation", "HOLD").upper(),
                    signal.get("rationale"),
                    json.dumps(signal.get("risk_factors", [])),
                    signal.get("time_horizon", "medium_term"),
                    signal.get("confidence", 1.0),
                    datetime.utcnow().isoformat()
                )
            )
            self._conn.commit()
            return cur.lastrowid

    def create_escalation(self, signal_id: Optional[int], symbol: str, action: str, reason: str, risk_level: str = 'medium') -> int:
        """Create a new operator escalation record."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO escalations (signal_id, symbol, action, reason, risk_level, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (
                    signal_id,
                    symbol.upper(),
                    action,
                    reason,
                    risk_level,
                    datetime.utcnow().isoformat()
                )
            )
            self._conn.commit()
            return cur.lastrowid

    def get_pending_escalations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all pending operator escalations along with their associated research signals."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT e.*, s.market, s.current_price, s.target_price, s.recommendation, s.rationale, "
                "s.risk_factors, s.time_horizon, s.confidence "
                "FROM escalations e "
                "LEFT JOIN research_signals s ON e.signal_id = s.id "
                "WHERE e.status = 'pending' "
                "ORDER BY e.created_at DESC LIMIT ?",
                (limit,)
            )
            cur.row_factory = sqlite3.Row
            rows = [dict(r) for r in cur.fetchall()]
            
            import json
            for r in rows:
                if r.get('risk_factors'):
                    try:
                        r['risk_factors'] = json.loads(r['risk_factors'])
                    except Exception:
                        r['risk_factors'] = []
                else:
                    r['risk_factors'] = []
            return rows

    def resolve_escalation(self, escalation_id: int, status: str, operator_notes: str = '', resolved_by: str = 'operator') -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Resolve an escalation (status should be 'approved' or 'rejected').
        Returns (success, escalation_details).
        """
        if status not in ('approved', 'rejected', 'expired'):
            raise ValueError(f"Invalid escalation resolution status: {status}")

        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM escalations WHERE id = ?", (escalation_id,)
            )
            cur.row_factory = sqlite3.Row
            escalation = cur.fetchone()
            if not escalation:
                return False, None
            
            # If already resolved, return failure
            if escalation['status'] != 'pending':
                return False, dict(escalation)

            self._conn.execute(
                "UPDATE escalations SET status = ?, operator_notes = ?, resolved_at = ?, resolved_by = ? "
                "WHERE id = ?",
                (status, operator_notes, datetime.utcnow().isoformat(), resolved_by, escalation_id)
            )
            self._conn.commit()

            # Retrieve updated record
            cur = self._conn.execute(
                "SELECT e.*, s.market, s.current_price, s.target_price, s.recommendation, s.rationale "
                "FROM escalations e "
                "LEFT JOIN research_signals s ON e.signal_id = s.id "
                "WHERE e.id = ?", (escalation_id,)
            )
            cur.row_factory = sqlite3.Row
            updated_escalation = cur.fetchone()
            return True, dict(updated_escalation)

    def add_to_watchlist(self, symbol: str, market: str, source: str, recommendation: str = 'HOLD', 
                         target_price: float = 0.0, rationale: str = '') -> None:
        """Add or update a symbol in the position watchlist."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO position_watchlist (symbol, market, source, recommendation, "
                "target_price, rationale, status, added_at, last_updated) "
                "VALUES (?, ?, ?, ?, ?, ?, 'active', "
                "COALESCE((SELECT added_at FROM position_watchlist WHERE symbol = ?), ?), ?)",
                (
                    symbol.upper(),
                    market.lower(),
                    source,
                    recommendation.upper(),
                    target_price,
                    rationale,
                    symbol.upper(),
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat()
                )
            )
            self._conn.commit()

    def pause_watchlist_symbol(self, symbol: str) -> bool:
        """Pause tracking of a watchlist symbol."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE position_watchlist SET status = 'paused', last_updated = ? "
                "WHERE symbol = ? AND status = 'active'",
                (datetime.utcnow().isoformat(), symbol.upper())
            )
            self._conn.commit()
            return cur.rowcount > 0

    def resume_watchlist_symbol(self, symbol: str) -> bool:
        """Resume tracking of a paused watchlist symbol."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE position_watchlist SET status = 'active', last_updated = ? "
                "WHERE symbol = ? AND status = 'paused'",
                (datetime.utcnow().isoformat(), symbol.upper())
            )
            self._conn.commit()
            return cur.rowcount > 0

    def remove_from_watchlist(self, symbol: str) -> bool:
        """Mark a symbol as removed in the watchlist."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE position_watchlist SET status = 'removed', last_updated = ? "
                "WHERE symbol = ?",
                (datetime.utcnow().isoformat(), symbol.upper())
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_active_watchlist(self) -> List[Dict[str, Any]]:
        """Get list of active symbols currently tracked in the watchlist."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM position_watchlist WHERE status = 'active' ORDER BY symbol ASC"
            )
            cur.row_factory = sqlite3.Row
            return [dict(r) for r in cur.fetchall()]

    def get_upload_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get history of research uploads."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM research_uploads ORDER BY uploaded_at DESC LIMIT ?",
                (limit,)
            )
            cur.row_factory = sqlite3.Row
            return [dict(r) for r in cur.fetchall()]

    def get_signals_for_upload(self, upload_id: int) -> List[Dict[str, Any]]:
        """Get all signals extracted from a specific upload."""
        import json
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM research_signals WHERE upload_id = ?",
                (upload_id,)
            )
            cur.row_factory = sqlite3.Row
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                if r.get('risk_factors'):
                    try:
                        r['risk_factors'] = json.loads(r['risk_factors'])
                    except Exception:
                        r['risk_factors'] = []
            return rows

    def save_portfolio_value(self, timestamp: str, value: float) -> None:
        """Insert or replace a portfolio value point in the database."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO portfolio_history (timestamp, value) VALUES (?, ?)",
                (timestamp, value)
            )
            self._conn.commit()

    def get_portfolio_history(self) -> List[Tuple[str, float]]:
        """Get the full history of recorded portfolio values, sorted by timestamp ascending."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT timestamp, value FROM portfolio_history ORDER BY timestamp ASC"
            )
            return [(r[0], r[1]) for r in cur.fetchall()]

    def close(self):
        """Close connection."""
        with self._lock:
            self._conn.close()
