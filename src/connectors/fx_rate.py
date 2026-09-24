"""The Kenya shilling's exchange rate against the US dollar.

It used to be a constant, 130 shillings to the dollar, shown on the
dashboard as the "Central Bank rate", and the Portfolio page used another
constant, 129.64. Both only converted figures for display and for the
consolidated USD total, but a stale rate misstates every KES holding in
dollars.

The rate now comes from the market, in this order:

1. Yahoo Finance's USD/KES quote (KES=X), via yfinance, which the agent
   already uses for US prices. It updates through the trading day, though
   the shilling is thinly traded, so the quote can sit still for a while.
2. ExchangeRate-API's open endpoint: a daily reference rate, no key needed.
   Its terms ask for an attribution link wherever the rate is shown; the
   dashboard shows one when this source is in use.
3. The last good rate, saved to disk, so a restart or an outage never falls
   back to the constant.
4. The constant, labelled as such, only if nothing has ever answered.

Refreshes happen every five minutes in a background thread, so a slow
source never holds up an API request or the trading loop.
"""
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

FALLBACK_KES_PER_USD = 130.0
REFRESH_SECONDS = 300
# A quote outside this band is a bad print, not the shilling.
SANE_RANGE = (60.0, 400.0)
# A move this large against the last accepted rate needs a second source to
# agree (within CONFIRM_TOLERANCE) before it is believed.
MAX_JUMP = 0.10
CONFIRM_TOLERANCE = 0.02

ER_API_URL = 'https://open.er-api.com/v6/latest/USD'
ER_API_ATTRIBUTION = 'https://www.exchangerate-api.com'

Quote = Dict[str, Any]


def yahoo_rate() -> Optional[Quote]:
    """USD/KES from Yahoo Finance, the latest 5-minute close."""
    import yfinance as yf
    df = yf.Ticker('KES=X').history(period='5d', interval='5m')
    if df is None or df.empty:
        return None
    ts = df.index[-1].to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return {'kes_per_usd': float(df['Close'].iloc[-1]), 'source': 'yahoo',
            'label': 'Yahoo Finance market rate', 'as_of': ts.astimezone(timezone.utc).isoformat()}


def er_api_rate() -> Optional[Quote]:
    """The daily reference rate from ExchangeRate-API's open endpoint."""
    resp = requests.get(ER_API_URL, timeout=(5, 15))
    data = resp.json()
    if data.get('result') != 'success':
        return None
    kes = (data.get('rates') or {}).get('KES')
    if not kes:
        return None
    as_of = datetime.fromtimestamp(int(data.get('time_last_update_unix') or 0), timezone.utc)
    return {'kes_per_usd': float(kes), 'source': 'exchangerate_api',
            'label': 'ExchangeRate-API daily rate', 'as_of': as_of.isoformat(),
            'attribution_url': ER_API_ATTRIBUTION}


def _sane(q: Optional[Quote]) -> bool:
    try:
        return bool(q) and SANE_RANGE[0] < float(q['kes_per_usd']) < SANE_RANGE[1]
    except (TypeError, ValueError, KeyError):
        return False


class FxRate:
    """KES per USD, refreshed from the market in the background."""

    MARKET_SOURCES = ('yahoo', 'exchangerate_api')

    def __init__(self, path: Path, sources: Optional[List[Callable[[], Optional[Quote]]]] = None,
                 clock: Callable[[], float] = time.time, refresh_seconds: int = REFRESH_SECONDS,
                 background: bool = True):
        self.path = Path(path)
        self.sources = sources if sources is not None else [yahoo_rate, er_api_rate]
        self.clock = clock
        self.refresh_seconds = refresh_seconds
        self.background = background
        self._lock = threading.Lock()
        self._refreshing = False
        self._last_attempt = float('-inf')
        self._current: Quote = self._load() or {
            'kes_per_usd': FALLBACK_KES_PER_USD, 'source': 'fixed',
            'label': 'Fixed fallback rate (no live source has answered yet)',
            'as_of': None, 'fetched_at': None}

    # ---------------------------------------------------------------- reads

    def snapshot(self) -> Quote:
        """The current rate and where it came from; starts a refresh if due."""
        self._maybe_refresh()
        q = dict(self._current)
        q['usd_per_kes'] = 1.0 / q['kes_per_usd']
        fetched = q.get('fetched_at')
        q['live'] = bool(q.get('source') in self.MARKET_SOURCES and fetched
                         and self.clock() - fetched < 3 * self.refresh_seconds)
        return q

    def kes_per_usd(self) -> float:
        return self.snapshot()['kes_per_usd']

    def usd_per_kes(self) -> float:
        return 1.0 / self.kes_per_usd()

    # ------------------------------------------------------------- refresh

    def _maybe_refresh(self) -> None:
        with self._lock:
            if self._refreshing or self.clock() - self._last_attempt < self.refresh_seconds:
                return
            self._refreshing = True
            self._last_attempt = self.clock()
        if self.background:
            threading.Thread(target=self._refresh_guarded, daemon=True, name='fx-rate').start()
        else:
            self._refresh_guarded()

    def _refresh_guarded(self) -> None:
        try:
            self.refresh()
        except Exception as e:  # never let a source take the caller down
            logger.warning(f"KES/USD refresh failed: {e}")
        finally:
            with self._lock:
                self._refreshing = False

    def _fetch_all(self) -> List[Quote]:
        quotes = []
        for source in self.sources:
            try:
                q = source()
            except Exception as e:
                logger.info(f"KES/USD source {getattr(source, '__name__', source)} failed: {e}")
                continue
            if _sane(q):
                quotes.append(q)
            elif q:
                logger.warning(f"KES/USD quote rejected as implausible: {q}")
        return quotes

    def refresh(self) -> bool:
        """Take the first plausible quote; True if the rate was updated."""
        quotes = self._fetch_all()
        if not quotes:
            return False
        chosen = quotes[0]
        prev = self._current
        if prev.get('source') in self.MARKET_SOURCES:
            jump = abs(chosen['kes_per_usd'] / prev['kes_per_usd'] - 1)
            if jump > MAX_JUMP:
                agree = [q for q in quotes[1:]
                         if abs(q['kes_per_usd'] / chosen['kes_per_usd'] - 1) <= CONFIRM_TOLERANCE]
                if not agree:
                    logger.warning(
                        f"KES/USD {chosen['kes_per_usd']:.2f} from {chosen['source']} is "
                        f"{jump:.0%} away from {prev['kes_per_usd']:.2f} and no second source "
                        f"confirms it; keeping the previous rate")
                    return False
        chosen = {**chosen, 'fetched_at': self.clock()}
        self._current = chosen
        self._save(chosen)
        return True

    # --------------------------------------------------------- persistence

    def _load(self) -> Optional[Quote]:
        try:
            q = json.loads(self.path.read_text())
            return q if _sane(q) else None
        except (OSError, ValueError):
            return None

    def _save(self, q: Quote) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(q))
            tmp.replace(self.path)
        except OSError as e:
            logger.debug(f"Could not save the KES/USD rate: {e}")


_shared: Optional[FxRate] = None
_shared_lock = threading.Lock()


def shared_fx() -> FxRate:
    """The process-wide rate, saved under the data directory."""
    global _shared
    with _shared_lock:
        if _shared is None:
            from src.utils.paths import DATA_DIR
            _shared = FxRate(DATA_DIR / 'fx_rate.json')
        return _shared
