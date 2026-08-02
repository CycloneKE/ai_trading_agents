"""Portfolio-level position gate: aggregate exposure cap + post-exit cooldown.

`max_position_size` used to be applied to each order in isolation, which is
not a position limit at all — it is a per-ticket limit. On 2026-07-24 the
live agent bought TSLA twenty times on the same repeating signal and built a
position worth roughly 29% of equity against a 5% cap, then did the mirror
image on the short side with XLF and XLV.

This module answers one question for the execution path:

    given what I already hold, how much of this order may I actually send?

Two rules:

* ADDS are clamped to the headroom left under `max_position_size * equity`.
* REDUCES are never blocked. An agent that cannot exit is strictly more
  dangerous than a concentrated one, so a closing order always passes — and
  an order that crosses through flat only has its *new* side capped.

Plus a cooldown: after the risk layer force-closes a symbol (stop-loss,
trailing stop, kill switch), entries in that symbol are refused for a
window. Without it the entry signal — which is still firing, that is why the
position existed — simply rebuys, which is exactly what the journal recorded
ten seconds after the TSLA stop filled. The window is persisted because the
agent restarts often enough that in-memory state would not survive.
"""

import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional

from src.utils.paths import DATA_DIR
from src.utils.sqlite_util import connect as _connect, write_retry
from src.utils.timeutil import utc_now_iso

logger = logging.getLogger(__name__)

# Applied when the operator's config predates this feature. Deliberately
# conservative: a missing key must not mean "unlimited".
DEFAULT_MAX_POSITION_PCT = 0.05
DEFAULT_COOLDOWN_SECONDS = 900          # 15 minutes; 0 disables
DEFAULT_MIN_NOTIONAL = 5.0              # mirrors position_sizing
DEFAULT_MAX_NET_EXPOSURE = 0.30         # |long - short| / equity

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forced_exits (
    symbol      TEXT PRIMARY KEY,
    exited_at   REAL NOT NULL,
    reason      TEXT,
    recorded_at TEXT NOT NULL
);
"""


def _finite(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Coerce to a finite float, or None if that is impossible.

    Returns `default` for None/empty input so callers can distinguish "not
    supplied" (fine, use a default) from "supplied but garbage" (refuse).
    NaN and +/-Infinity are garbage: they silently pass every subsequent
    comparison rather than failing loudly.
    """
    if value is None or value == '':
        return default
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _is_reduce(side: str, existing_qty: float) -> bool:
    """True when the order shrinks the absolute size of the open position."""
    if not existing_qty:
        return False
    return (existing_qty > 0 and side == 'sell') or (existing_qty < 0 and side == 'buy')


class ExposureGuard:
    def __init__(self, config: Dict[str, Any],
                 db_path: Optional[str] = None):
        limits = (config or {}).get('risk_limits', {}) or {}
        self.max_position_pct = float(
            limits.get('max_position_size', DEFAULT_MAX_POSITION_PCT))
        self.cooldown_seconds = float(
            limits.get('reentry_cooldown_seconds', DEFAULT_COOLDOWN_SECONDS))
        # Bounds the directional bet across the whole book. The per-symbol cap
        # alone does not: eleven separate 5% longs is a 55% long book that
        # simply tracks the index.
        self.max_net_exposure = float(
            limits.get('max_net_exposure', DEFAULT_MAX_NET_EXPOSURE))
        # Headroom smaller than this is not usable: the caller would journal
        # an intent (burning a client_order_id) only for size_order to reject
        # it. Kept in sync with position_sizing.DEFAULT_MIN_NOTIONAL.
        self.min_notional = float(
            ((config or {}).get('trading', {}) or {}).get(
                'min_notional', DEFAULT_MIN_NOTIONAL))

        self.db_path = db_path or str(DATA_DIR / 'exposure.db')
        self._lock = threading.Lock()
        self._conn = _connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Aggregate exposure
    # ------------------------------------------------------------------

    def permitted_notional(self, symbol: str, side: str,
                           existing_qty: Optional[float],
                           price: Optional[float],
                           equity: Optional[float],
                           target_value: Optional[float],
                           net_notional: Optional[float] = None) -> float:
        """Dollar notional this order may actually use (0.0 = send nothing).

        `existing_qty` is signed: positive long, negative short.
        """
        side = (side or '').lower()

        # Reject non-finite inputs at the boundary, once, before any branch
        # can use them. Every comparison against NaN is False, so a NaN that
        # gets past here silently defeats EVERY guard below: `nan <= 0` is
        # False so a NaN target passes the sanity check, and `headroom < min`
        # is False so the at-the-cap refusal never fires — reopening the
        # exact pyramiding bug this module exists to prevent, via nothing
        # worse than one bad vendor tick.
        #
        # price/existing_qty/target_value are refused unconditionally: they
        # make the arithmetic itself meaningless, so there is no safe answer
        # for an add OR an exit. `equity` is deliberately NOT checked here —
        # it is only needed to size an add, and blocking an exit because the
        # broker returned a bad equity would trap the agent in a position it
        # is trying to close. `_cap` handles that case.
        existing_qty = _finite(existing_qty, default=0.0)
        price = _finite(price)
        target_value = _finite(target_value)
        if existing_qty is None or price is None or target_value is None:
            logger.warning(
                f"Exposure gate: refusing {side} {symbol} — non-finite input "
                f"(qty={existing_qty!r} price={price!r} target={target_value!r})")
            return 0.0

        if target_value <= 0:
            return 0.0
        if price <= 0:
            return 0.0

        held_notional = abs(existing_qty) * price

        # An exit is always allowed, and is checked before equity so that a
        # broker glitch returning no equity can never trap the agent in a
        # position it is trying to close.
        if _is_reduce(side, existing_qty):
            if target_value <= held_notional:
                return float(target_value)
            # Crosses through flat: the held_notional portion closes the
            # existing side and always passes -- that is an absolute
            # invariant. The remainder opens a fresh position on the
            # opposite side, which is an ADD and so is subject to both the
            # per-symbol cap and the net-exposure cap, but is never trimmed
            # back below held_notional.
            cap = self._cap(equity)
            if cap is None:
                allowed = float(held_notional)
            else:
                allowed = held_notional + min(target_value - held_notional, cap)
            allowed = self._apply_net_cap(symbol, side, allowed, equity,
                                          net_notional, floor=held_notional)
            return float(allowed)

        # From here on the order ADDS exposure.
        cap = self._cap(equity)
        if cap is None:
            logger.warning(
                f"Exposure gate: refusing {side} {symbol} — unusable equity {equity!r}")
            return 0.0

        if self.in_cooldown(symbol):
            logger.warning(
                f"Exposure gate: refusing {side} {symbol} — "
                f"{self.cooldown_remaining(symbol):.0f}s left in post-exit cooldown")
            return 0.0

        headroom = cap - held_notional
        if headroom < self.min_notional:
            logger.warning(
                f"Exposure gate: refusing {side} {symbol} — already holding "
                f"${held_notional:,.0f} against a ${cap:,.0f} cap")
            return 0.0

        allowed = min(float(target_value), headroom)
        if allowed < target_value:
            logger.info(
                f"Exposure gate: trimming {side} {symbol} from ${target_value:,.0f} "
                f"to ${allowed:,.0f} (holding ${held_notional:,.0f} of ${cap:,.0f})")

        allowed = self._apply_net_cap(symbol, side, allowed, equity, net_notional)
        return allowed

    def _cap(self, equity: Optional[float]) -> Optional[float]:
        """Per-symbol notional ceiling, or None when equity is unusable."""
        if equity is None:
            return None
        equity = float(equity)
        if not math.isfinite(equity) or equity <= 0:
            return None
        return equity * self.max_position_pct

    def _apply_net_cap(self, symbol: str, side: str, allowed: float,
                       equity: Optional[float],
                       net_notional: Optional[float],
                       floor: float = 0.0) -> float:
        """Trim an ADD so the book's net directional exposure stays bounded.

        `net_notional` is signed: long positive, short negative. Passing None
        skips the check entirely, so existing six-argument callers are
        unaffected. An order that moves the book toward neutral always
        passes — capping a de-risking trade would be perverse.

        `floor` is the portion of `allowed` that closes an existing position
        (0.0 for a pure add, `held_notional` for an order that crosses
        through flat). That portion is never trimmed away — exits are an
        absolute invariant regardless of net exposure or garbage inputs.
        """
        if net_notional is None or allowed <= floor:
            return allowed
        if not math.isfinite(net_notional):
            logger.warning(
                f"Exposure gate: refusing {side} {symbol} add — "
                f"non-finite net_notional {net_notional!r}")
            return floor
        cap = self._cap_net(equity)
        if cap is None:
            return allowed

        signed = allowed if side == 'buy' else -allowed
        prospective = net_notional + signed
        if abs(prospective) <= abs(net_notional) or abs(prospective) <= cap:
            return allowed

        # Trim to whatever keeps |net| exactly at the cap, but never below
        # the closing portion of a crossing order.
        headroom = cap - abs(net_notional)
        trimmed = max(headroom, floor)
        if floor <= 0.0 and trimmed < self.min_notional:
            logger.warning(
                f"Exposure gate: refusing {side} {symbol} — net exposure "
                f"${abs(net_notional):,.0f} against a ${cap:,.0f} cap")
            return 0.0
        if trimmed != allowed:
            logger.info(
                f"Exposure gate: trimming {side} {symbol} to ${trimmed:,.0f} "
                f"on net-exposure cap")
        return trimmed

    def _cap_net(self, equity: Optional[float]) -> Optional[float]:
        if equity is None:
            return None
        equity = float(equity)
        if not math.isfinite(equity) or equity <= 0:
            return None
        return equity * self.max_net_exposure

    # ------------------------------------------------------------------
    # Post-exit cooldown
    # ------------------------------------------------------------------

    def record_forced_exit(self, symbol: str, reason: str,
                           now: Optional[float] = None) -> None:
        """Start (or restart) the re-entry cooldown for a symbol."""
        ts = time.time() if now is None else float(now)

        def _write():
            self._conn.execute(
                "INSERT INTO forced_exits (symbol, exited_at, reason, recorded_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(symbol) DO UPDATE SET"
                "   exited_at=excluded.exited_at,"
                "   reason=excluded.reason,"
                "   recorded_at=excluded.recorded_at",
                (symbol, ts, reason, utc_now_iso()))
            self._conn.commit()

        with self._lock:
            write_retry(_write)
        logger.warning(
            f"Exposure gate: {symbol} entered {self.cooldown_seconds:.0f}s "
            f"re-entry cooldown after {reason}")

    def cooldown_remaining(self, symbol: str, now: Optional[float] = None) -> float:
        if self.cooldown_seconds <= 0:
            return 0.0
        with self._lock:
            row = self._conn.execute(
                "SELECT exited_at FROM forced_exits WHERE symbol = ?", (symbol,)
            ).fetchone()
        if not row:
            return 0.0
        ts = time.time() if now is None else float(now)
        return max(0.0, (row[0] + self.cooldown_seconds) - ts)

    def in_cooldown(self, symbol: str, now: Optional[float] = None) -> bool:
        return self.cooldown_remaining(symbol, now) > 0

    def clear_cooldown(self, symbol: str) -> None:
        """Operator override — lets a human re-enable a symbol immediately."""
        def _write():
            self._conn.execute("DELETE FROM forced_exits WHERE symbol = ?", (symbol,))
            self._conn.commit()

        with self._lock:
            write_retry(_write)

    def active_cooldowns(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Symbols currently blocked, with why and for how much longer.

        Read this to report blocked symbols: a cooled-down symbol produces no
        orders and so is indistinguishable on a dashboard from one with no
        signal. Sorted soonest-to-expire first.
        """
        if self.cooldown_seconds <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT symbol, exited_at, reason FROM forced_exits").fetchall()
        ts = time.time() if now is None else float(now)
        out = []
        for symbol, exited_at, reason in rows:
            left = (exited_at + self.cooldown_seconds) - ts
            if left > 0:
                out.append({'symbol': symbol, 'reason': reason,
                            'seconds_remaining': left})
        return sorted(out, key=lambda e: e['seconds_remaining'])

    def close(self):
        with self._lock:
            self._conn.close()
