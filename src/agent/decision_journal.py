"""Decision journal: one row per symbol per cycle capturing WHY the agent
did or did not act — the data the drill-down view reads.

The trading loop generates rich per-cycle context (per-strategy signals,
LLM verdict, the reason a trade was skipped) and previously discarded it
every cycle. This persists it, with change-detection so an idle symbol
holding at confidence 0.1 all day produces a handful of rows, not 390.
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

# Canonical skip reasons (null skip_reason + executed=1 means a trade went out)
SKIP_REASONS = {
    'hold',               # strategies produced no directional signal
    'below_confidence',   # under the execution confidence floor
    'bias_downgrade',     # bias detector pushed the signal to hold
    'llm_veto',           # LLM validation changed action to hold
    'fallback_price',     # price was synthetic fallback, not tradeable
    'halted',             # kill switch engaged
    'pdt_guard',          # pattern-day-trader block
    'min_notional',       # sized below the minimum order value
    'no_account_info',    # account info unavailable, couldn't size
    'dissent',            # parallel mode: no strategy agreed with direction
    'duplicate',          # journal blocked a duplicate decision
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT NOT NULL,
    cycle               INTEGER,
    symbol              TEXT NOT NULL,
    action              TEXT,           -- buy | sell | hold
    executed            INTEGER DEFAULT 0,
    skip_reason         TEXT,           -- null when executed; else a SKIP_REASONS value
    ensemble_confidence REAL,
    per_strategy_json   TEXT,           -- {name: {action, confidence}}
    llm_verdict_json    TEXT,           -- {action, confidence, reasoning}
    price               REAL,
    target_value        REAL,
    client_order_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
"""


class DecisionJournal:
    """SQLite-backed, thread-safe (the loop, and later the API, touch it from
    different threads). Change-detection lives here so callers can flush every
    symbol every cycle without flooding the table."""

    def __init__(self, db_path: str = str(DATA_DIR / 'order_journal.db'),
                 heartbeat_cycles: int = 30):
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.db_path = db_path
        # Per-symbol last recorded state + cycle, for change-detection.
        self._last_state: Dict[str, tuple] = {}
        self._last_cycle: Dict[str, int] = {}
        self.heartbeat_cycles = heartbeat_cycles

    def _state_key(self, d: Dict[str, Any]) -> tuple:
        # A decision is "the same" if action, skip reason and coarse confidence
        # match — confidence bucketed to 0.1 so tiny wiggles don't spam rows.
        conf = d.get('ensemble_confidence') or 0.0
        return (d.get('action'), d.get('skip_reason'), bool(d.get('executed')),
                round(conf, 1))

    def record(self, decision: Dict[str, Any]) -> bool:
        """Persist a decision if its state changed since the symbol's last
        recorded row, or if the heartbeat interval elapsed. Returns True if a
        row was written. An executed order ALWAYS writes (never suppressed)."""
        symbol = decision.get('symbol')
        if not symbol:
            return False
        cycle = decision.get('cycle') or 0
        key = self._state_key(decision)

        with self._lock:
            changed = self._last_state.get(symbol) != key
            stale = cycle - self._last_cycle.get(symbol, -10 ** 9) >= self.heartbeat_cycles
            if not (changed or stale or decision.get('executed')):
                return False

            self._conn.execute(
                "INSERT INTO decisions (ts, cycle, symbol, action, executed,"
                " skip_reason, ensemble_confidence, per_strategy_json,"
                " llm_verdict_json, price, target_value, client_order_id)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    datetime.utcnow().isoformat(), cycle, symbol,
                    decision.get('action'),
                    1 if decision.get('executed') else 0,
                    decision.get('skip_reason'),
                    decision.get('ensemble_confidence'),
                    json.dumps(decision.get('per_strategy') or {}),
                    json.dumps(decision.get('llm_verdict') or {}),
                    decision.get('price'),
                    decision.get('target_value'),
                    decision.get('client_order_id'),
                ))
            self._conn.commit()
            self._last_state[symbol] = key
            self._last_cycle[symbol] = cycle
        return True

    def recent(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            if symbol:
                cur = self._conn.execute(
                    "SELECT * FROM decisions WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                    (symbol, limit))
            else:
                cur = self._conn.execute(
                    "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))
            cur.row_factory = sqlite3.Row
            rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r['per_strategy'] = json.loads(r.pop('per_strategy_json') or '{}')
            r['llm_verdict'] = json.loads(r.pop('llm_verdict_json') or '{}')
            r['executed'] = bool(r['executed'])
        return rows

    def close(self):
        with self._lock:
            self._conn.close()
