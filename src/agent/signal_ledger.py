"""What every signal did next: evidence the agent can learn from without
waiting for a trade to close.

The agent's other ways of learning need closed round trips: a strategy's
weight moves only after it has closed five trades, the AI review waits for
fifteen. It is long-only and selective, so trades close slowly; after two days
of the paper run there were none. Meanwhile the decision journal already holds,
every few minutes, what each strategy said about each symbol, whether or not
the agent traded on it.

This turns that into evidence. For every buy or sell signal, by the ensemble
and by each strategy on its own, it records what the price did 5 and 20
trading days later, in the signal's direction, and how often that symbol
simply went up anyway (so a rising market does not make every buy look
skilful). Outcomes are recorded once they are known and never change, in
data/signal_outcomes.db, so a redeploy or a Yahoo outage cannot lose them.

From that, per strategy and per market, it measures skill: the hit rate,
how far it sits above chance, and how many standard errors that is (z).

By default this only measures, and shows the result (the scorecard's Learning
section). With `learning.signal_skill_tilt.enabled` it also scales each
strategy's vote in each market by that evidence, inside the same guardrails as
every other adjustment (guardrails.signal_tilt): nothing under 50 signals,
never more than 0.5x to 1.5x, never switching a strategy off.

What it does not do: it measures a strategy's raw signal, before the regime
filter and the ensemble vote, and it ignores costs. It is evidence of
direction, not of profit.
"""
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import date, datetime, timezone
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

HORIZONS = (5, 20)
ENSEMBLE = 'ensemble'

DEFAULTS: Dict[str, Any] = {
    'enabled': False,          # tilt weights from this evidence; off: measure and show only
    'horizon': 5,              # trading days ahead the tilt is judged on
    'min_signals': 50,         # signals with a known outcome before a strategy is tilted
    'refresh_hours': 6,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_outcomes (
    symbol      TEXT NOT NULL,
    day         TEXT NOT NULL,
    source      TEXT NOT NULL,      -- 'ensemble' or a strategy's name
    action      TEXT NOT NULL,      -- buy | sell
    horizon     INTEGER NOT NULL,   -- trading days ahead
    market      TEXT,
    ret         REAL NOT NULL,      -- move in the signal's direction
    chance      REAL NOT NULL,      -- how often that direction happened anyway
    version     TEXT,               -- code version that made the signal
    computed_at TEXT NOT NULL,
    PRIMARY KEY (symbol, day, source, action, horizon)
);
CREATE INDEX IF NOT EXISTS idx_outcomes_day ON signal_outcomes(day);
"""


def settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = (((config or {}).get('learning') or {}).get('signal_skill_tilt')) or {}
    return {**DEFAULTS, **{k: v for k, v in cfg.items() if not str(k).startswith('_')}}


# ------------------------------------------------------------------ signals

def votes(decision: Dict[str, Any]) -> List[Tuple[str, str]]:
    """The (source, action) directional votes in one journal row: the
    ensemble's, and each strategy's own, whatever the ensemble decided."""
    out: List[Tuple[str, str]] = []
    action = str(decision.get('action') or '').lower()
    if action in ('buy', 'sell'):
        out.append((ENSEMBLE, action))
    per = decision.get('per_strategy') or {}
    if isinstance(per, dict):
        for name, v in per.items():
            if not isinstance(v, dict) or v.get('abstain'):
                continue
            act = str(v.get('action') or '').lower()
            try:
                conf = float(v.get('confidence') or 0)
            except (TypeError, ValueError):
                conf = 0.0
            if act in ('buy', 'sell') and conf > 0:
                out.append((str(name), act))
    return out


def signals_from(decisions: Iterable[Dict[str, Any]], config: Optional[Dict[str, Any]] = None
                 ) -> List[Dict[str, Any]]:
    """Every distinct signal: once per symbol, day, source and direction (a
    signal that persists is journaled again and again)."""
    from src.agent.ai_scorecard import _day
    from src.agent.cost_model import classify
    from src.agent.scorecard import _market_tz
    seen, out = set(), []
    for d in decisions:
        symbol = d.get('symbol')
        if not symbol or not d.get('price'):
            continue
        day = _day(d.get('ts'), _market_tz(symbol, config))
        if day is None:
            continue
        for source, action in votes(d):
            key = (symbol, day, source, action)
            if key in seen:
                continue
            seen.add(key)
            out.append({'symbol': symbol, 'day': day, 'source': source, 'action': action,
                        'market': classify(symbol, config), 'version': d.get('code_version')})
    return out


def outcomes_for(sigs: Iterable[Dict[str, Any]], closes_for: Callable[[str], Dict[date, float]],
                 horizons: Iterable[int] = HORIZONS) -> List[Dict[str, Any]]:
    """What each signal did over each horizon, for those whose outcome is known."""
    from src.agent.ai_scorecard import forward_return
    from src.agent.scorecard import _base_rate
    cache: Dict[str, Tuple[Dict[date, float], List[date]]] = {}
    base: Dict[Tuple[str, int], Optional[float]] = {}
    rows: List[Dict[str, Any]] = []
    for s in sigs:
        sym = s['symbol']
        if sym not in cache:
            try:
                closes = closes_for(sym) or {}
            except Exception:
                closes = {}
            cache[sym] = (closes, sorted(closes))
        closes, ordered = cache[sym]
        for h in horizons:
            r = forward_return(closes, s['day'], int(h), ordered)
            if r is None:
                continue
            if (sym, h) not in base:
                base[(sym, h)] = _base_rate(closes, ordered, int(h))
            up = base[(sym, h)]
            if up is None:
                continue
            buy = s['action'] == 'buy'
            rows.append({**s, 'horizon': int(h), 'ret': r if buy else -r, 'chance': up if buy else 1 - up})
    return rows


# ------------------------------------------------------------------- ledger

class SignalLedger:
    """The recorded outcomes, in a small SQLite file in the data volume."""

    def __init__(self, path: Optional[str] = None):
        self.path = str(path or DATA_DIR / 'signal_outcomes.db')
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(self, rows: Iterable[Dict[str, Any]], now: Optional[datetime] = None) -> int:
        """Store outcomes not already held; a recorded outcome is final. Returns
        how many were new."""
        stamp = (now or datetime.now(timezone.utc)).isoformat(timespec='seconds')
        data = [(r['symbol'], str(r['day']), r['source'], r['action'], int(r['horizon']), r.get('market'),
                 float(r['ret']), float(r['chance']), r.get('version'), stamp) for r in rows]
        with self._lock:
            before = self._conn.total_changes
            self._conn.executemany(
                "INSERT OR IGNORE INTO signal_outcomes (symbol, day, source, action, horizon, market, ret,"
                " chance, version, computed_at) VALUES (?,?,?,?,?,?,?,?,?,?)", data)
            self._conn.commit()
            return self._conn.total_changes - before

    def rows(self, since_day: Optional[str] = None) -> List[Dict[str, Any]]:
        return read_ledger(self.path, since_day)

    def update(self, decisions: Iterable[Dict[str, Any]], closes_for: Callable[[str], Dict[date, float]],
               config: Optional[Dict[str, Any]] = None, horizons: Iterable[int] = HORIZONS,
               now: Optional[datetime] = None) -> int:
        """Record the outcomes now known for the signals in `decisions`."""
        return self.record(outcomes_for(signals_from(decisions, config), closes_for, horizons), now)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def read_ledger(path: str, since_day: Optional[str] = None) -> List[Dict[str, Any]]:
    """The recorded outcomes, read-only (safe against a live agent); [] when
    there is no ledger yet."""
    if not os.path.exists(path):
        return []
    try:
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=10)
    except sqlite3.Error:
        return []
    try:
        conn.row_factory = sqlite3.Row
        sql = "SELECT symbol, day, source, action, horizon, market, ret, chance, version FROM signal_outcomes"
        rows = conn.execute(sql + (" WHERE day >= ?" if since_day else "") + " ORDER BY day",
                            (since_day,) if since_day else ()).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def read_decisions(db_path: str, since: Optional[str] = None) -> List[Dict[str, Any]]:
    """The journal rows that carry a directional vote, read-only. Holds are most
    of the journal; only rows where the ensemble or a strategy said buy or sell
    are needed."""
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=10)
    except sqlite3.Error:
        return []
    try:
        conn.row_factory = sqlite3.Row
        query = ("SELECT ts, symbol, action, price, per_strategy_json, code_version FROM decisions WHERE"
                 " (action IN ('buy', 'sell') OR per_strategy_json LIKE '%\"buy\"%' OR per_strategy_json LIKE '%\"sell\"%')")
        if since:
            query += " AND ts >= ?"
        out = []
        for r in conn.execute(query + " ORDER BY ts", (since,) if since else ()).fetchall():
            d = dict(r)
            try:
                d['per_strategy'] = json.loads(d.pop('per_strategy_json') or '{}')
            except ValueError:
                d['per_strategy'] = {}
            out.append(d)
        return out
    except sqlite3.Error:
        return []
    finally:
        conn.close()


# ------------------------------------------------------------------- skill

def skill(rows: Iterable[Dict[str, Any]], horizon: int = 5, min_signals: int = 30
          ) -> List[Dict[str, Any]]:
    """Per source (a strategy, or the ensemble) and market: how often its
    signals moved the right way, against how often chance would have, at one
    horizon. `z` is how many standard errors the hit rate is above chance."""
    from src.agent.scorecard import wilson
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if int(r['horizon']) == int(horizon):
            groups.setdefault((r['source'], r.get('market') or '?'), []).append(r)
    out = []
    for (source, market), g in sorted(groups.items()):
        n = len(g)
        hits = sum(1 for r in g if r['ret'] > 0)
        rate, chance = hits / n, mean(r['chance'] for r in g)
        se = (chance * (1 - chance) / n) ** 0.5 if 0 < chance < 1 else 0.0
        lo, hi = wilson(hits, n)
        out.append({'source': source, 'market': market, 'n': n, 'hit_rate': round(rate, 3),
                    'chance': round(chance, 3), 'lift': round(rate - chance, 3),
                    'z': round((rate - chance) / se, 2) if se else 0.0,
                    'low': round(lo, 3), 'high': round(hi, 3),
                    'avg_move_pct': round(mean(r['ret'] for r in g) * 100, 2),
                    'enough': n >= int(min_signals)})
    return out


def tilts(skill_rows: Iterable[Dict[str, Any]], min_signals: int = 50) -> Dict[Tuple[str, str], float]:
    """The weight scale each strategy would get in each market from its skill
    (never the ensemble's: it is not a voter)."""
    from src.agent.guardrails import signal_tilt
    return {(r['source'], r['market']): signal_tilt(r['n'], r['z'], min_signals)
            for r in skill_rows if r['source'] != ENSEMBLE}


# --------------------------------------------------------------- price data

def closes_provider(config: Optional[Dict[str, Any]] = None, ttl: float = 3600.0,
                    clock: Callable[[], float] = time.time) -> Callable[[str], Dict[date, float]]:
    """Daily closes by date for any symbol: a Kenyan stock from the NSE's own
    price files (its ticker can name a different company on the US feed),
    everything else from Yahoo Finance. A failure is remembered for the ttl, so
    a throttled Yahoo is not asked again every minute."""
    from src.agent import benchmarks as bm
    from src.agent import chart_data
    from src.agent.cost_model import classify
    cache: Dict[str, Tuple[float, Dict[date, float]]] = {}

    def closes_for(symbol: str) -> Dict[date, float]:
        hit = cache.get(symbol)
        if hit and clock() - hit[0] < ttl:
            return hit[1]
        try:
            if classify(symbol, config) == 'nse':
                from src.connectors.nse_connector import NSE_CSV_DIR
                closes = bm.nse_closes(symbol, NSE_CSV_DIR)
            else:
                from src.agent.scorecard import bars_to_closes
                closes = bars_to_closes(chart_data.yfinance_bars(symbol, period='2y'))
        except Exception as e:
            logger.debug(f"No daily closes for {symbol}: {e}")
            closes = {}
        cache[symbol] = (clock(), closes)
        return closes

    return closes_for


# ---------------------------------------------------------------- the job

class SignalLearning:
    """Keeps the ledger up to date in the background and, when switched on,
    hands the strategy manager the skill-based weight scales."""

    def __init__(self, config: Dict[str, Any], ledger: SignalLedger, strategy_manager=None,
                 decisions_path: Optional[str] = None, closes_for=None,
                 clock: Callable[[], float] = time.time):
        self.config = config
        self.rule = settings(config)
        self.ledger = ledger
        self.sm = strategy_manager
        self.decisions_path = str(decisions_path or DATA_DIR / 'decision_journal.db')
        self.closes_for = closes_for or closes_provider(config)
        self._clock = clock
        self._last = 0.0
        self._thread: Optional[threading.Thread] = None
        self.last_result: Dict[str, Any] = {}

    def due(self) -> bool:
        return self._clock() - self._last >= float(self.rule['refresh_hours']) * 3600

    def maybe_update(self) -> bool:
        """Start an update in the background if one is due. Never blocks the loop."""
        if (self._thread is not None and self._thread.is_alive()) or not self.due():
            return False
        self._last = self._clock()
        self._thread = threading.Thread(target=self.run, daemon=True, name='signal-learning')
        self._thread.start()
        return True

    def run(self) -> Dict[str, Any]:
        try:
            decisions = read_decisions(self.decisions_path)
            new = self.ledger.update(decisions, self.closes_for, self.config)
            rows = self.ledger.rows()
            table = skill(rows, self.rule['horizon'], self.rule['min_signals'])
            scale = tilts(table, self.rule['min_signals'])
            applied = bool(self.rule['enabled'])
            if self.sm is not None and hasattr(self.sm, 'set_signal_skill'):
                self.sm.set_signal_skill(scale if applied else {})
            self.last_result = {'at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                'new_outcomes': new, 'outcomes': len(rows), 'applied': applied,
                                'tilted': sorted(f"{s}/{m}={t:.2f}" for (s, m), t in scale.items() if t != 1.0)}
            if new:
                logger.info(f"Signal learning: {new} new outcomes ({len(rows)} in all); "
                            f"tilt {'ON' if applied else 'off (measuring only)'}")
        except Exception as e:
            logger.error(f"Signal learning failed: {e}")
        return self.last_result
