"""Loading historical bars from CSV files on disk.

Shared by the backtest harness and the universe tooling so they agree on
what counts as usable history. Rejecting a symbol here is cheap; discovering
mid-backtest that a series has gaps or zero prices is not.
"""

import logging
import os
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

MIN_BARS = 60
_DATE_COLS = ('date', 'timestamp', 'time', 'datetime')


def load_price_csv(path: str, min_bars: int = MIN_BARS) -> Optional[pd.DataFrame]:
    """One <SYMBOL>.csv into a date-indexed frame, or None if unusable.

    Drops non-positive and unparseable closes, and de-duplicates repeated
    timestamps keeping the last, which is what a re-scrape of the same day
    should mean.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.warning(f"{path}: unreadable ({e})")
        return None

    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next((c for c in _DATE_COLS if c in df.columns), None)
    if date_col is None or 'close' not in df.columns:
        logger.warning(f"{path}: needs a date column and a close column")
        return None

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    df = df[df['close'] > 0]

    if len(df) < min_bars:
        logger.warning(f"{path}: only {len(df)} usable bars, need {min_bars}")
        return None
    return df


def load_price_dir(path: str, symbols: Optional[Set[str]] = None,
                   min_bars: int = MIN_BARS) -> Dict[str, pd.DataFrame]:
    """Every usable <SYMBOL>.csv in a directory, keyed by upper-case symbol."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Not a directory: {path}")
    out = {}
    for fn in sorted(os.listdir(path)):
        if not fn.lower().endswith('.csv'):
            continue
        sym = os.path.splitext(fn)[0].upper()
        if symbols and sym not in symbols:
            continue
        df = load_price_csv(os.path.join(path, fn), min_bars)
        if df is not None:
            out[sym] = df
    return out


def data_quality(df: pd.DataFrame) -> Dict[str, object]:
    """Findings a human should see before trusting a series.

    Gaps and repeated prices are the two that quietly corrupt a backtest:
    a stale feed that repeats yesterday's close looks like zero volatility,
    which makes stops too tight and positions too large.
    """
    closes = df['close']
    gaps = df.index.to_series().diff().dt.days.dropna()
    typical = float(gaps.median()) if len(gaps) else 0.0
    long_gaps = int((gaps > max(typical * 3, typical + 5)).sum()) if typical else 0
    repeats = int((closes.diff() == 0).sum())

    issues: List[str] = []
    if long_gaps:
        issues.append(f"{long_gaps} unusually long gaps between bars")
    if repeats > 0.2 * len(closes):
        issues.append(f"{repeats} of {len(closes)} bars repeat the previous close, "
                      f"which understates volatility")
    if len(closes) and closes.iloc[-1] <= 0:
        issues.append("final close is not positive")

    return {
        'bars': int(len(closes)),
        'start': df.index.min(),
        'end': df.index.max(),
        'typical_gap_days': typical,
        'long_gaps': long_gaps,
        'repeated_closes': repeats,
        'issues': issues,
    }
