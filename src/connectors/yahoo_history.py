"""Free deep-history source: Yahoo Finance chart API (keyless).

~500 daily bars per symbol on the free endpoint — vs 60-150 from the
Alpaca IEX feed — which materially improves the nightly practice loop's
train/test windows and strategy warm-starts. No API key; be polite
(sequential requests, generous timeout, browser UA).
"""

import logging
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_daily_closes(symbol: str, range_: str = "2y") -> List[float]:
    """Daily closes for a symbol, oldest first. Empty list on any failure."""
    try:
        resp = requests.get(_URL.format(symbol=symbol),
                            params={"range": range_, "interval": "1d"},
                            headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        result = resp.json()["chart"]["result"][0]
        closes = result["indicators"]["quote"][0]["close"]
        return [float(c) for c in closes if c]
    except Exception as e:
        logger.warning(f"Yahoo history failed for {symbol}: {e}")
        return []


def fetch_many(symbols: List[str], range_: str = "2y") -> Dict[str, List[float]]:
    out = {}
    for sym in symbols:
        closes = fetch_daily_closes(sym, range_)
        if closes:
            out[sym] = closes
    return out
