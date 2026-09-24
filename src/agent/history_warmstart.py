"""Daily price history for warm-starting the live indicators.

The strategies need `lookback_period` daily bars (50) before they can
signal. Without a warm-start that is fifty trading days of silence after
every deploy. The warm-start that existed called `primary.api.get_bars`,
an Alpaca SDK object: the SDK was never installed, and the REST connector
that replaced it has no `.api`, so the lookup returned None and the whole
warm-start was skipped without a log line. It has never run in production.

US and crypto history now comes from yfinance, the same source the live
price feed already uses, so history and live quotes share one symbol format
and one price basis (split- and dividend-adjusted closes, which line up with
today's raw quote).

NSE history comes from the scraper's CSVs, keeping only rows a real source
produced. Those files are seeded with 730 days of synthetic prices on first
run; seeding the strategies from them compared today's real price against
an invented moving average.
"""
import csv
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from src.connectors.nse_scraper import REAL_NSE_SOURCES

logger = logging.getLogger(__name__)

Row = Tuple[date, float, float, float]  # (date, high, low, close)
History = Dict[str, Dict[str, List[float]]]  # symbol -> {'close','high','low'}


def _yfinance_daily(symbol: str, period: str = '6mo') -> List[Row]:
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval='1d', auto_adjust=True)
    if df is None or df.empty:
        return []
    return [(idx.date(), float(r['High']), float(r['Low']), float(r['Close']))
            for idx, r in df.iterrows()]


def _to_history(rows: Iterable[Row], today: date, bars: int) -> Optional[Dict[str, List[float]]]:
    """Completed bars only (dated before today, UTC), oldest first, last `bars`.

    Today's partial bar is dropped so the first live price starts today's
    bar instead of the history already containing half of it.
    """
    kept = sorted((r for r in rows if r[0] < today and r[3] and r[3] > 0),
                  key=lambda r: r[0])[-bars:]
    if not kept:
        return None
    return {'close': [r[3] for r in kept],
            'high': [r[1] for r in kept],
            'low': [r[2] for r in kept]}


def fetch_daily_history(symbols: Iterable[str], bars: int = 60,
                        today: Optional[date] = None,
                        fetch: Callable[[str], List[Row]] = _yfinance_daily) -> History:
    """Completed daily bars per symbol from a vendor (yfinance by default)."""
    today = today or datetime.now(timezone.utc).date()
    out: History = {}
    for sym in symbols:
        try:
            hist = _to_history(fetch(sym), today, bars)
        except Exception as e:
            logger.warning(f"Daily history fetch failed for {sym}: {e}")
            continue
        if hist:
            out[sym] = hist
    return out


def read_nse_history(symbols: Iterable[str], csv_dir: Path, bars: int = 60,
                     today: Optional[date] = None) -> History:
    """Real daily bars per NSE symbol from the scraper's CSVs.

    Synthetic rows are dropped, as are rows with no recorded source, since a
    price of unknown origin must not set a moving average that decides a
    real-money order ticket.
    """
    today = today or datetime.now(timezone.utc).date()
    out: History = {}
    for sym in symbols:
        path = Path(csv_dir) / f"{sym}.csv"
        if not path.exists():
            continue
        try:
            rows: List[Row] = []
            with open(path, newline='', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    if r.get('source') not in REAL_NSE_SOURCES:
                        continue
                    try:
                        d = date.fromisoformat((r.get('date') or '')[:10])
                        close = float(r.get('close') or 0)
                        high = float(r.get('high') or close)
                        low = float(r.get('low') or close)
                    except ValueError:
                        continue
                    rows.append((d, high, low, close))
            # One row per date: the scraper can append the same day twice.
            by_date = {r[0]: r for r in rows}
            hist = _to_history(by_date.values(), today, bars)
        except Exception as e:
            logger.warning(f"NSE history read failed for {sym}: {e}")
            continue
        if hist:
            out[sym] = hist
    return out
