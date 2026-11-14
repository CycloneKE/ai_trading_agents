"""How each broker's calls turned out: a scoreboard of research sources.

Why this exists: a rating such as "BUY, target KES 45" was stored and shown
to the AI's trade review, but nothing recorded whether that broker's calls
were any good. The agent already scores its own strategies' signals this way
(signal_ledger.py); a broker's rating is a signal of the same kind.

Every directional rating the agent has (BUY, ACCUMULATE, OUTPERFORM... as
buys; SELL, REDUCE, UNDERPERFORM... as sells; HOLD and NEUTRAL call no
direction and are left out) is taken from:

- the ratings read from uploaded analyst notes (escalations.db,
  research_signals, with the upload's detected publisher), dated the day the
  agent read them, since that is when it could have acted;
- the daily recommendation sheets (daily_whispers.py), dated by the sheet.

For each it measures what the price did 20, 60 and 120 trading days later,
in the called direction, against how often the stock simply moved that way
anyway, with signal_ledger.outcomes_for and signal_ledger.skill: hit rate,
the rate chance would give, how many standard errors apart (z), and the
average move after round-trip costs.

What it is used for: the Research page (/api/research/broker-scores) and,
once a publisher has at least `min_calls` scored calls, one line in the AI's
trade review saying how that publisher's calls have done. Nothing is weighted
or traded on it: the record has to be long before it means anything.
"""
import logging
import sqlite3
import threading
import time
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

HORIZONS = (20, 60, 120)
RECORD_HORIZON = 60
DEFAULTS = {'min_calls': 20, 'refresh_hours': 12}

BUY_WORDS = {'BUY', 'STRONG BUY', 'ACCUMULATE', 'OUTPERFORM', 'OVERWEIGHT', 'ADD', 'SPECULATIVE BUY'}
SELL_WORDS = {'SELL', 'STRONG SELL', 'REDUCE', 'UNDERPERFORM', 'UNDERWEIGHT', 'LIGHTEN'}


def direction(recommendation: Any) -> Optional[str]:
    """'buy', 'sell', or None for a rating that calls no direction."""
    word = ' '.join(str(recommendation or '').upper().replace('-', ' ').split())
    if word in BUY_WORDS:
        return 'buy'
    if word in SELL_WORDS:
        return 'sell'
    return None


def _day(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def note_calls(db_path: str) -> List[Dict[str, Any]]:
    """Ratings read from uploaded notes, with their publisher. [] if none."""
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=10)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            "SELECT s.symbol, s.recommendation, s.extracted_at, u.source "
            "FROM research_signals s LEFT JOIN research_uploads u ON u.id = s.upload_id").fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        conn.close()
    return [{'symbol': r[0], 'recommendation': r[1], 'day': r[2], 'source': r[3] or 'Unknown publisher'}
            for r in rows]


def sheet_calls() -> List[Dict[str, Any]]:
    """Ratings from the daily recommendation sheets."""
    from src.agent import daily_whispers, market_pulse
    store = market_pulse._read(daily_whispers.STORE, {}) or {}
    out = []
    for symbol, history in store.items():
        for h in history or []:
            out.append({'symbol': symbol, 'recommendation': h.get('recommendation'),
                        'day': h.get('date'), 'source': market_pulse.DAILY_SHEET_SOURCE})
    return out


def to_signals(calls: Iterable[Dict[str, Any]], config: Optional[Dict[str, Any]] = None
               ) -> List[Dict[str, Any]]:
    """Directional calls in the signal ledger's shape, one per publisher,
    symbol, day and direction (a sheet repeats a rating day after day)."""
    from src.agent.cost_model import classify
    seen, out = set(), []
    for c in calls:
        action, day, symbol = direction(c.get('recommendation')), _day(c.get('day')), c.get('symbol')
        if not action or not day or not symbol:
            continue
        symbol = str(symbol).upper()
        key = (c['source'], symbol, day, action)
        if key in seen:
            continue
        seen.add(key)
        try:
            market = classify(symbol, config)
        except Exception:
            market = '?'
        out.append({'source': c['source'], 'symbol': symbol, 'day': day, 'action': action,
                    'market': market})
    return out


def scoreboard(signals: List[Dict[str, Any]], closes_for: Callable[[str], Dict[date, float]],
               config: Optional[Dict[str, Any]] = None, horizons=HORIZONS,
               min_calls: int = DEFAULTS['min_calls']) -> Dict[str, Any]:
    """Each publisher's record at each horizon (signal_ledger.skill rows)."""
    from src.agent.signal_ledger import outcomes_for, skill
    rows = outcomes_for(signals, closes_for, horizons=horizons, config=config)
    scored = {(r['source'], r['symbol'], r['day'], r['action']) for r in rows}
    return {'calls': len(signals), 'scored': len(scored), 'waiting': len(signals) - len(scored),
            'min_calls': int(min_calls),
            'horizons': {str(h): skill(rows, horizon=h, min_signals=min_calls) for h in horizons},
            'computed_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}


def record_line(board: Optional[Dict[str, Any]], source: str,
                horizon: int = RECORD_HORIZON) -> Optional[str]:
    """One sentence on a publisher's record, or None until it is long enough."""
    if not board or not source:
        return None
    rows = [r for r in (board.get('horizons') or {}).get(str(horizon), [])
            if str(r.get('source', '')).lower() == str(source).lower() and r.get('enough')]
    if not rows:
        return None
    r = max(rows, key=lambda x: x['n'])
    net = f", {r['net_move_pct']:+.1f}% on average after costs" if r.get('net_move_pct') is not None else ''
    return (f"of {r['n']} past {source} calls, {r['hit_rate']:.0%} moved the called way over "
            f"{horizon} trading days, against {r['chance']:.0%} by chance (z {r['z']:+.1f}){net}.")


class BrokerCallScores:
    """Keeps the scoreboard fresh in the background and answers record_for."""

    def __init__(self, config: Dict[str, Any], db_path: str,
                 closes_for: Optional[Callable[[str], Dict[date, float]]] = None,
                 calls: Optional[Callable[[], List[Dict[str, Any]]]] = None,
                 clock: Callable[[], float] = time.time):
        from src.agent.position_rules import rule
        self.config = config
        self.rule = rule(DEFAULTS, (config or {}).get('broker_scores'))
        self.db_path = db_path
        self._closes_for = closes_for
        self._calls = calls or (lambda: note_calls(self.db_path) + sheet_calls())
        self.clock = clock
        self.latest: Optional[Dict[str, Any]] = None
        self._last = 0.0
        self._thread: Optional[threading.Thread] = None

    def refresh(self) -> Dict[str, Any]:
        from src.agent.signal_ledger import closes_provider
        closes_for = self._closes_for or closes_provider(self.config)
        board = scoreboard(to_signals(self._calls(), self.config), closes_for, self.config,
                           min_calls=self.rule['min_calls'])
        self.latest = board
        return board

    def maybe_refresh(self) -> bool:
        """Start a refresh in the background when one is due. Never blocks."""
        if self._thread is not None and self._thread.is_alive():
            return False
        if self.latest is not None and self.clock() - self._last < self.rule['refresh_hours'] * 3600:
            return False
        self._last = self.clock()

        def run():
            try:
                self.refresh()
            except Exception as e:
                logger.warning(f"Broker call scores could not be refreshed: {e}")

        self._thread = threading.Thread(target=run, daemon=True, name='broker-scores')
        self._thread.start()
        return True

    def record_for(self, source: str) -> Optional[str]:
        return record_line(self.latest, source)
