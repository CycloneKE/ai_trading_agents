"""One rule for turning a stream of live prices into daily bars.

The strategies, the regime detector and the ATR tracker were all written
and backtested on daily bars, but the live loop calls them once a minute
with the latest quote, and each call appended a new "bar". So "the last 50
bars" meant the last 50 minutes. Thresholds calibrated for days (a 2% move
over 10 bars, a close 2% from its 20-bar mean) are almost unreachable
minute to minute, and in production the agent produced no trade signal in
36 hours. Overnight it was worse: US quotes are frozen outside the session,
so the same closing price was appended hundreds of times, flattening every
indicator until the next session flushed them out.

With a `bar_date`, live prices now build daily bars instead:

- same date as the last bar  -> replace it (the forming bar moves intraday)
- a new date, price moved     -> append a new bar
- a new date, price unchanged -> skip; the market is closed and a frozen
  quote is not a trading day. Stocks therefore get no weekend or holiday
  bars, while crypto, whose price always moves, rolls over at 00:00 UTC.

Without a `bar_date` every call is its own bar, exactly as before. Backtests
and the readiness check feed one bar per call and rely on that.

UTC dates are used everywhere. The US session (13:30-21:00 UTC) and the NSE
session (06:00-12:00 UTC) each fall within one UTC date, and daily crypto
bars conventionally close at 00:00 UTC.
"""
from datetime import datetime, timezone
from typing import Optional

APPEND = 'append'
REPLACE = 'replace'
SKIP = 'skip'


def utc_bar_date(now: Optional[datetime] = None) -> str:
    """The daily-bar date for a live observation, as 'YYYY-MM-DD' in UTC."""
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()


def classify(last_date: Optional[str], bar_date: Optional[str],
             last_close: Optional[float], price: float) -> str:
    """How a new observation updates a daily series: APPEND, REPLACE or SKIP."""
    if bar_date is None:
        return APPEND
    if last_date is not None and bar_date == last_date:
        return REPLACE
    if last_close is not None and price == last_close:
        return SKIP
    return APPEND
