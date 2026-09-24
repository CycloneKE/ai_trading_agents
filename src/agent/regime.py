"""Market regime classification, per symbol.

Momentum buys strength and mean reversion buys weakness. Blending them into
one weighted score means they cancel on entries while agreeing on exits.
Measured over 2,148 daily GOOG bars, the three configured strategies cast
1,372 buy votes between them; the ensemble turned those into 9 buys against
121 sells, because two strategies agreed on sell 298 times and on buy only 77.
The system could not accumulate a position in a stock that rose sevenfold.

The fix is not to reweight the blend but to stop asking opposed strategies to
vote at the same time. This classifies each symbol's regime so the caller can
let momentum act in trends and mean reversion act in ranges, and have the
other abstain rather than dilute.
"""

import logging
from collections import deque
from typing import Any, Dict, Optional

from src.agent.daily_bars import APPEND, REPLACE, classify

logger = logging.getLogger(__name__)

TRENDING_UP = 'trending_up'
TRENDING_DOWN = 'trending_down'
RANGING = 'ranging'
UNKNOWN = 'unknown'

# Which strategy families are allowed to act in each regime. A strategy whose
# name matches none of these keys is never suppressed.
FAMILY_REGIMES = {
    'momentum': {TRENDING_UP, TRENDING_DOWN},
    'reversion': {RANGING},
}


class RegimeDetector:
    """Classifies a symbol as trending or ranging from its own closes.

    Two simple moving averages. Price above both, with the fast above the
    slow, is an uptrend; the mirror is a downtrend; anything else is a range.
    Returns `unknown` until `slow_period` bars exist, which callers must treat
    as "apply no filter" rather than "no trades".
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50,
                 maxlen: int = 400):
        self.fast_period = max(2, int(fast_period))
        self.slow_period = max(self.fast_period + 1, int(slow_period))
        self._closes: Dict[str, deque] = {}
        self._bar_dates: Dict[str, str] = {}
        self._maxlen = max(maxlen, self.slow_period + 1)

    def update(self, symbol: str, price: float, bar_date: Optional[str] = None) -> None:
        """Add a price. With bar_date, builds daily bars (see daily_bars.py);
        without, every call is a bar. The 20/50-period averages were being
        computed over 20/50 minutes of live quotes."""
        try:
            price = float(price)
        except (TypeError, ValueError):
            return
        if price <= 0:
            return
        buf = self._closes.setdefault(symbol, deque(maxlen=self._maxlen))
        step = classify(self._bar_dates.get(symbol), bar_date,
                        buf[-1] if buf else None, price)
        if step == APPEND:
            buf.append(price)
            if bar_date is not None:
                self._bar_dates[symbol] = bar_date
        elif step == REPLACE:
            buf[-1] = price

    def warm_start(self, symbol: str, closes) -> int:
        """Replace the symbol's history with these closes. Replace, not
        append: a retried warm-start must not land after live bars."""
        self._closes.pop(symbol, None)
        n = 0
        for c in (closes or []):
            self.update(symbol, c)
            n += 1
        self._bar_dates.pop(symbol, None)
        return n

    def bars(self, symbol: str) -> int:
        return len(self._closes.get(symbol, ()))

    def regime(self, symbol: str) -> str:
        buf = self._closes.get(symbol)
        if not buf or len(buf) < self.slow_period:
            return UNKNOWN
        vals = list(buf)
        price = vals[-1]
        fast = sum(vals[-self.fast_period:]) / self.fast_period
        slow = sum(vals[-self.slow_period:]) / self.slow_period
        if price > slow and fast > slow:
            return TRENDING_UP
        if price < slow and fast < slow:
            return TRENDING_DOWN
        return RANGING


def family_of(strategy_name: str) -> Optional[str]:
    """Map a configured strategy name to a family key, or None.

    Matches on substring so `momentum`, `fast_momentum` and `momentum_v2` all
    resolve to the same family, mirroring how TechnicalStrategy dispatches on
    its own name.
    """
    name = (strategy_name or '').lower()
    for family in FAMILY_REGIMES:
        if family in name:
            return family
    return None


def is_active(strategy_name: str, regime: str) -> bool:
    """Whether this strategy should vote in this regime.

    Unknown regimes and unrecognised strategy names are permissive: the
    filter narrows who votes, and must never be the reason nothing trades.
    """
    if regime == UNKNOWN:
        return True
    family = family_of(strategy_name)
    if family is None:
        return True
    return regime in FAMILY_REGIMES[family]


def filter_signals(strategy_signals: Dict[str, Dict[str, Any]],
                   regime: str) -> Dict[str, Dict[str, Any]]:
    """Drop the votes of strategies that are out of regime.

    Dropping rather than down-weighting is deliberate. A suppressed vote that
    still counts toward the denominator keeps the opposing strategy below the
    confidence gate, which is the failure this exists to remove.

    Never returns empty: if the filter would silence everyone, the unfiltered
    set is returned so the ensemble decides as it did before.
    """
    if not strategy_signals:
        return strategy_signals
    kept = {n: s for n, s in strategy_signals.items() if is_active(n, regime)}
    return kept or strategy_signals
