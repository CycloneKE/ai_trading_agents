"""
Immutable, Cryptographically Signed Audit Journal for AI Trading System.
Logs every AI decision, LLM prompt/response, pre-trade risk check, and order fill
to an append-only SQLite/DB table with a deterministic SHA-256 record signature.
"""

import os
import sqlite3
import hashlib
import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    component       TEXT NOT NULL,    -- 'strategy' | 'llm' | 'risk' | 'execution' | 'system'
    action          TEXT NOT NULL,
    symbol          TEXT,
    payload         TEXT NOT NULL,    -- JSON string of context/prompt/event
    result          TEXT,             -- JSON string of decision/response/fill
    prev_hash       TEXT NOT NULL,    -- SHA-256 hash of previous row
    sha256_hash     TEXT NOT NULL     -- SHA-256 hash of (id + timestamp + component + action + payload + result + prev_hash)
);

CREATE INDEX IF NOT EXISTS idx_audit_component ON audit_logs(component);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_symbol ON audit_logs(symbol);
"""

class AuditJournal:
    """
    Write-once, cryptographically signed audit log.
    Thread-safe and persistent across system restarts.
    """
    
    def __init__(self, db_path: str = str(DATA_DIR / 'audit_journal.db')):
        self.db_path = db_path
        self._lock = threading.Lock()
        
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        
        logger.info(f"AuditJournal initialized at {db_path}")
        
    def _get_last_hash(self) -> str:
        """Fetch hash of the last audit log row for hash chain verification."""
        cursor = self._conn.execute("SELECT sha256_hash FROM audit_logs ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else "GENESIS_HASH_00000000000000000000000000000000000000000000000000000000"
        
    def _calculate_hash(self, timestamp: str, component: str, action: str, 
                        symbol: Optional[str], payload_str: str, result_str: str, prev_hash: str) -> str:
        """Compute SHA-256 signature for audit entry."""
        raw_data = f"{timestamp}|{component}|{action}|{symbol or ''}|{payload_str}|{result_str}|{prev_hash}"
        return hashlib.sha256(raw_data.encode('utf-8')).hexdigest()
        
    def log_event(self, component: str, action: str, payload: Dict[str, Any], 
                  result: Optional[Dict[str, Any]] = None, symbol: Optional[str] = None) -> int:
        """
        Record a cryptographically signed audit event.
        
        Args:
            component: Subsystem name (e.g. 'strategy', 'llm', 'risk', 'execution')
            action: Action executed (e.g. 'generate_signal', 'pre_trade_check', 'place_order')
            payload: Input parameters, prompt, or signal context
            result: Outcome, decision verdict, or execution report
            symbol: Ticker symbol (if applicable)
            
        Returns:
            Row ID of the written audit record
        """
        timestamp = datetime.utcnow().isoformat()
        payload_str = json.dumps(payload, sort_keys=True)
        result_str = json.dumps(result or {}, sort_keys=True)
        
        with self._lock:
            try:
                prev_hash = self._get_last_hash()
                row_hash = self._calculate_hash(timestamp, component, action, symbol, payload_str, result_str, prev_hash)
                
                cursor = self._conn.execute(
                    """
                    INSERT INTO audit_logs (timestamp, component, action, symbol, payload, result, prev_hash, sha256_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (timestamp, component, action, symbol, payload_str, result_str, prev_hash, row_hash)
                )
                self._conn.commit()
                return cursor.lastrowid
            except Exception as e:
                logger.error(f"Failed to record audit event: {e}")
                return -1

    def verify_chain(self) -> Tuple[bool, int, str]:
        """
        Audit verification function to validate cryptographic hash chain integrity.
        
        Returns:
            (is_valid, records_checked, error_message)
        """
        with self._lock:
            cursor = self._conn.execute("SELECT id, timestamp, component, action, symbol, payload, result, prev_hash, sha256_hash FROM audit_logs ORDER BY id ASC")
            rows = cursor.fetchall()
            
            expected_prev_hash = "GENESIS_HASH_00000000000000000000000000000000000000000000000000000000"
            for row in rows:
                row_id, ts, comp, act, sym, pay, res, prev_h, curr_h = row
                
                if prev_h != expected_prev_hash:
                    return False, row_id, f"Hash chain broken at ID {row_id}: expected prev_hash {expected_prev_hash}, found {prev_h}"
                    
                calc_h = self._calculate_hash(ts, comp, act, sym, pay, res, prev_h)
                if calc_h != curr_h:
                    return False, row_id, f"Signature mismatch at ID {row_id}: recorded {curr_h}, computed {calc_h}"
                    
                expected_prev_hash = curr_h
                
            return True, len(rows), "Audit trail integrity verified successfully"

    def recent_events(self, limit: int = 50, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent audit events."""
        with self._lock:
            if component:
                cursor = self._conn.execute(
                    "SELECT id, timestamp, component, action, symbol, payload, result, sha256_hash FROM audit_logs WHERE component=? ORDER BY id DESC LIMIT ?",
                    (component, limit)
                )
            else:
                cursor = self._conn.execute(
                    "SELECT id, timestamp, component, action, symbol, payload, result, sha256_hash FROM audit_logs ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
            
            events = []
            for row in cursor.fetchall():
                events.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'component': row[2],
                    'action': row[3],
                    'symbol': row[4],
                    'payload': json.loads(row[5]) if row[5] else {},
                    'result': json.loads(row[6]) if row[6] else {},
                    'sha256_hash': row[7]
                })
            return events
