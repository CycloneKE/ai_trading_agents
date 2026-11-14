"""Per-symbol volatility tracking and volatility-relative stop distances.

Why this exists: the shipped risk limits used a flat 3% trailing stop and a
flat 5% hard stop for every instrument. Measured over 2,148 daily GOOG bars,
ATR(14) averages 2.64% of price, so a 3% trailing stop sits barely above one
average day's true range. Simulated against that series, noise alone trips it
57% of the time within ten days and 83% within twenty. A stop tighter than
the instrument's own volatility does not manage risk, it guarantees the
position is closed before the thesis resolves.

Stops here are expressed as ATR multiples and converted to a percentage of
price per symbol, so the same configuration behaves sensibly on a 1%-a-day
NSE bank and a 6%-a-day crypto pair.
"""

import logging
from collections import deque
from typing import Any, Dict, Optional

from src.agent.daily_bars import REPLACE, SKIP, classify
from src.agent.indicators import atr, atr_from_closes

logger = logging.getLogger(__name__)

DEFAULT_LENGTH = 14
# Enough history for the ATR seed plus room for the recursive average to settle.
_MAXLEN = 400


class VolatilityTracker:
    """Rolling ATR per symbol, fed whatever the data feed provides.

    Accepts full OHLC when available and degrades to close-only bars when not.
    `atr_pct` returns None until a symbol has more than `length` bars, which
    callers must treat as "fall back to the fixed percentage" rather than
    "no volatility".
    """

    def __init__(self, length: int = DEFAULT_LENGTH):
        self.length = max(2, int(length))
        self._close: Dict[str, deque] = {}
        self._high: Dict[str, deque] = {}
        self._low: Dict[str, deque] = {}
        self._have_hl: Dict[str, bool] = {}
        self._bar_dates: Dict[str, str] = {}

    def _bufs(self, symbol: str):
        if symbol not in self._close:
            self._close[symbol] = deque(maxlen=_MAXLEN)
            self._high[symbol] = deque(maxlen=_MAXLEN)
            self._low[symbol] = deque(maxlen=_MAXLEN)
            self._have_hl[symbol] = True
        return self._close[symbol], self._high[symbol], self._low[symbol]

    def update(self, symbol: str, close: float,
               high: Optional[float] = None, low: Optional[float] = None,
               bar_date: Optional[str] = None) -> None:
        """Add one observation. Ignores non-positive or unusable prices.

        With bar_date, observations build daily bars (see daily_bars.py):
        the forming bar's close is replaced and its range widened. Without,
        every call is a bar. Fed once a minute, ATR measured minute-to-minute
        volatility: about 0.06% for Bitcoin instead of about 2% a day, which
        pinned every stop to the 2% floor.
        """
        try:
            close = float(close)
        except (TypeError, ValueError):
            return
        if close <= 0:
            return
        c, h, l = self._bufs(symbol)
        # A single bar without high/low permanently downgrades the symbol to
        # close-only: mixing the two would silently corrupt the true range.
        if high is None or low is None:
            self._have_hl[symbol] = False
            high = low = close
        else:
            try:
                high, low = float(high), float(low)
            except (TypeError, ValueError):
                self._have_hl[symbol] = False
                high = low = close
            if high < low or high <= 0 or low <= 0:
                self._have_hl[symbol] = False
                high = low = close
        step = classify(self._bar_dates.get(symbol), bar_date,
                        c[-1] if c else None, close)
        if step == SKIP:
            return
        if step == REPLACE:
            c[-1] = close
            h[-1] = max(h[-1], high)
            l[-1] = min(l[-1], low)
            return
        c.append(close)
        h.append(high)
        l.append(low)
        if bar_date is not None:
            self._bar_dates[symbol] = bar_date

    def warm_start(self, symbol: str, closes, highs=None, lows=None) -> int:
        """Replace the symbol's history with these bars. Returns the number
        accepted. Replace, not append: a retried warm-start must not land
        after live bars."""
        for store in (self._close, self._high, self._low, self._have_hl):
            store.pop(symbol, None)
        closes = list(closes or [])
        highs = list(highs or []) if highs is not None else None
        lows = list(lows or []) if lows is not None else None
        use_hl = highs is not None and lows is not None \
            and len(highs) == len(closes) and len(lows) == len(closes)
        n = 0
        for i, cl in enumerate(closes):
            self.update(symbol, cl,
                        highs[i] if use_hl else None,
                        lows[i] if use_hl else None)
            n += 1
        self._bar_dates.pop(symbol, None)
        return n

    def rescale(self, symbol: str, factor: float) -> None:
        """Multiply the symbol's held bars by `factor` (an NSE corporate
        action taking effect while running). ATR as a fraction of price is
        unchanged by it; what changes is that the next true range no longer
        spans the event's gap."""
        if not factor or factor == 1.0 or symbol not in self._close:
            return
        for store in (self._close, self._high, self._low):
            buf = store[symbol]
            store[symbol] = deque((p * factor for p in buf), maxlen=buf.maxlen)

    def bars(self, symbol: str) -> int:
        return len(self._close.get(symbol, ()))

    def forget(self, symbol: str) -> None:
        """Drop a symbol the agent no longer trades."""
        for store in (self._close, self._high, self._low, self._have_hl, self._bar_dates):
            store.pop(symbol, None)

    def atr(self, symbol: str) -> Optional[float]:
        """Latest ATR in price units, or None without enough history."""
        c = self._close.get(symbol)
        if not c or len(c) <= self.length:
            return None
        if self._have_hl.get(symbol):
            series = atr(list(self._high[symbol]), list(self._low[symbol]), list(c), self.length)
        else:
            series = atr_from_closes(list(c), self.length)
        if series is None or series.empty:
            return None
        value = series.iloc[-1]
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return None if value != value or value <= 0 else value  # NaN-safe

    def atr_pct(self, symbol: str) -> Optional[float]:
        """Latest ATR as a fraction of the latest close, or None."""
        a = self.atr(symbol)
        if a is None:
            return None
        last = self._close[symbol][-1]
        return (a / last) if last > 0 else None


# Keys a market may override in risk_limits.market_overrides.<market>.
MARKET_STOP_KEYS = ('stop_loss_pct', 'trailing_stop_pct', 'stop_loss_atr_mult',
                    'trailing_stop_atr_mult', 'min_stop_pct', 'max_stop_pct',
                    'trailing_stop')


def risk_limits_for(cfg: Dict[str, Any], market: Optional[str] = None) -> Dict[str, Any]:
    """risk_limits with `market`'s overrides applied, when it has any.

    A trend strategy that holds through pullbacks (crypto's 200-day trend)
    cannot live with the same trailing stop as a strategy that trades
    bounces: the stop sells the dip, the still-valid trend signal buys it
    back a day later, and every cycle locks in the drawdown and pays the fees.
    """
    risk = cfg.get('risk_limits', cfg) if isinstance(cfg, dict) else {}
    overrides = ((risk.get('market_overrides') or {}).get(market) or {}) if market else {}
    if not overrides:
        return risk
    return {**risk, **{k: v for k, v in overrides.items() if k in MARKET_STOP_KEYS}}


def stop_distances(cfg: Dict[str, Any], atr_pct: Optional[float] = None,
                   market: Optional[str] = None) -> Dict[str, float]:
    """Resolve (hard stop, trailing stop) as fractions of price.

    With an ATR reading, each is `<n> * ATR%` clamped into [min_stop_pct,
    max_stop_pct]. Without one, falls back to the fixed percentages so a
    cold start or a thin feed never leaves a position unprotected.

    The clamp matters in both directions: a very quiet instrument would
    otherwise get a stop so tight that the spread alone trips it, and a very
    volatile one a stop so wide it is no longer a stop.

    `market` applies risk_limits.market_overrides for that market. With
    `trailing_stop: false` there the trailing distance is 0.0, which every
    caller reads as "no trailing stop"; the hard stop always remains.
    """
    risk = risk_limits_for(cfg, market)
    trailing_on = risk.get('trailing_stop', True) is not False
    fixed_stop = float(risk.get('stop_loss_pct', 0.05))
    fixed_trail = float(risk.get('trailing_stop_pct', 0.03)) if trailing_on else 0.0
    if atr_pct is None or atr_pct <= 0:
        return {'stop_loss_pct': fixed_stop, 'trailing_stop_pct': fixed_trail,
                'source': 'fixed'}

    stop_mult = float(risk.get('stop_loss_atr_mult', 2.5))
    trail_mult = float(risk.get('trailing_stop_atr_mult', 3.0))
    lo = float(risk.get('min_stop_pct', 0.02))
    hi = float(risk.get('max_stop_pct', 0.25))
    if lo > hi:
        lo, hi = hi, lo

    def clamp(x):
        return max(lo, min(hi, x))

    return {'stop_loss_pct': clamp(stop_mult * atr_pct),
            'trailing_stop_pct': clamp(trail_mult * atr_pct) if trailing_on else 0.0,
            'source': 'atr'}
