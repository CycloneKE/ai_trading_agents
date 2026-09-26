"""
NSE Kenya Market Data Scraper
Scrapes end-of-day and intraday prices from public web sources,
stores them in Supabase and local CSV, and runs periodically
as a background thread inside the trading agent.

Public data sources used (no API key required):
  1. The NSE's own price ticker feed (the strip on nse.co.ke's home page)
  2. afx.kwayisi.org (public price tables), for symbols the ticker lacks
  3. Synthetic generation from known fundamentals (seed/fallback)

Usage:
    # As CLI (one-shot scrape)
    python -m src.connectors.nse_scraper

    # Programmatic (periodic, inside the agent)
    from src.connectors.nse_scraper import NSEPeriodicScraper
    scraper = NSEPeriodicScraper(database_manager=db, interval_minutes=30)
    scraper.start()  # non-blocking background thread
"""

import os
import sys
import csv
import time
import json
import re
import logging
import random
import threading
import requests
from pathlib import Path
from html.parser import HTMLParser
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

EAT = timezone(timedelta(hours=3))


def _eat_now() -> datetime:
    return datetime.now(EAT)

# Fallback only, for the CLI entry point and any caller that passes no
# symbols. The running agent passes data_manager.nse_symbols instead; see
# NSEPeriodicScraper.__init__. Keep this list broad rather than in sync with
# config: narrowing it would silently shrink what a bare `python -m
# src.connectors.nse_scraper` backfills.
DEFAULT_SYMBOLS = [
    "SCOM", "EQTY", "KCB", "COOP", "SCBK", "SBIC", "ABSA",
    "BAT", "EABL", "KEGN", "KNRE", "BAMB",
    "TOTL", "CTUM", "NMG", "NCBA", "BRIT", "CIC",
]

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "nse_historical"

# The source tags a bar carries when a live source published its price.
# Everything else (synthetic seed data, 'csv' of unknown origin, 'none') is
# not a market price. The API, the warm-start and the dashboard all judge
# NSE prices by this one list. nse_website stays although that scraper is
# gone: bars it wrote before the NSE site was rebuilt are still real.
# nse_pricelist is the checked OCR backfill (src/connectors/nse_pricelist.py).
REAL_NSE_SOURCES = frozenset({"nse_ticker", "nse_pricelist", "nse_website",
                              "afx_kwayisi", "afx_history"})


@dataclass
class DailyBar:
    date: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    change_pct: float
    source: str


# ======================================================================
# Public web scrapers (no API key needed)
# ======================================================================

class _NSETableParser(HTMLParser):
    """Parse HTML tables, tolerating implicit tag closes.

    Real-world sources (afx.kwayisi.org) serve minified HTML5 where </td>
    and </tr> are legally omitted: ``<td>12,043<td>339.00<tr><td>...``.
    html.parser only fires handle_endtag for explicit closes, so a new
    <td>/<tr> start tag must flush the still-open cell/row first — the
    same implicit-close rule browsers apply.
    """
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_row: List[str] = []
        self.rows: List[List[str]] = []
        self.cell_data = ""

    def _flush_cell(self):
        if self.in_cell:
            self.current_row.append(self.cell_data.strip())
            self.in_cell = False
            self.cell_data = ""

    def _flush_row(self):
        self._flush_cell()
        if self.in_row and self.current_row:
            self.rows.append(self.current_row)
        self.in_row = False
        self.current_row = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
        elif tag == "tr" and self.in_table:
            self._flush_row()  # implicit </tr> (and </td>) of previous row
            self.in_row = True
        elif tag in ("td", "th") and self.in_row:
            self._flush_cell()  # implicit </td> of previous cell
            self.in_cell = True
            self.cell_data = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._flush_cell()
        elif tag == "tr":
            self._flush_row()
        elif tag == "table":
            self._flush_row()  # close any dangling row at table end
            self.in_table = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell_data += data



def _warn_if_unrecognised(source: str, url: str, rows: List[List[str]], symbols) -> None:
    """Say so when a page loads but none of its rows name a watched symbol.

    That is what a changed page layout looks like from here: HTTP 200, no
    error, and nothing extracted. Without this it was indistinguishable
    from the source simply having no prices.
    """
    if not any(row and any(s in row[0].upper() for s in symbols) for row in rows):
        logger.warning(
            f"{source} {url} loaded but no table row names a watched symbol "
            f"({len(rows)} rows parsed); the page layout may have changed")


# The NSE's own price ticker, the strip across the top of nse.co.ke. Its
# script posts {"nopage": "true", "isinno": <account>} to this feed, where the
# account is the data-account attribute on the ticker element of the home
# page. The feed rejects any other value with "Invalid account", so the
# account is read from the home page the way a browser gets it, never
# hardcoded: if the NSE changes it, the next read picks up the new one.
#
# This replaced the scraper of www.nse.co.ke/market-statistics/*.html, which
# have returned 404 since the site was rebuilt.
NSE_HOME_URL = "https://www.nse.co.ke/"
NSE_TICKER_URL = "https://nsenairobi.nse.co.ke/nseticker/api/v1/ticker"
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_ACCOUNT_RE = re.compile(r"data-account\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
_ACCOUNT_TTL = 24 * 3600
_ticker_account: Dict[str, Any] = {}

# The NSE limits how far a share can move in one session (10% either way for
# most equities). A price further than this from the previous close is a
# mis-keyed or misread field, not a market move, and must not become a bar.
MAX_SESSION_MOVE = 0.15


def _nse_ticker_account(refresh: bool = False) -> Optional[str]:
    """The account the NSE home page gives its ticker, cached for a day."""
    cached = _ticker_account.get("value")
    if cached and not refresh and time.time() - _ticker_account.get("at", 0) < _ACCOUNT_TTL:
        return cached
    try:
        resp = requests.get(NSE_HOME_URL, headers={"User-Agent": _BROWSER_UA}, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"NSE home page {NSE_HOME_URL} returned HTTP {resp.status_code}")
            return cached
        m = _ACCOUNT_RE.search(resp.text)
        if not m:
            logger.warning("NSE home page loaded but has no ticker data-account; "
                           "the page layout may have changed")
            return cached
        _ticker_account.update(value=m.group(1).strip(), at=time.time())
        return _ticker_account["value"]
    except Exception as e:
        logger.warning(f"NSE home page fetch error: {e}")
        return cached


def _post_ticker(account: str):
    return requests.post(
        NSE_TICKER_URL,
        headers={"User-Agent": _BROWSER_UA, "Content-Type": "application/json",
                 "Accept": "application/json", "Referer": NSE_HOME_URL,
                 "Origin": NSE_HOME_URL.rstrip("/")},
        data=json.dumps({"nopage": "true", "isinno": account}), timeout=20)


def scrape_nse_ticker(symbols: Optional[List[str]] = None) -> Dict[str, DailyBar]:
    """Current prices from the NSE's own ticker feed, as DailyBars: for
    `symbols`, or for every listed issuer when `symbols` is None."""
    account = _nse_ticker_account()
    if not account:
        return {}
    try:
        resp = _post_ticker(account)
        if resp.status_code == 400 and "account" in resp.text.lower():
            # The NSE changed the account; read the new one and ask again.
            fresh = _nse_ticker_account(refresh=True)
            if fresh and fresh != account:
                resp = _post_ticker(fresh)
        if resp.status_code != 200:
            logger.warning(f"NSE ticker feed {NSE_TICKER_URL} returned HTTP "
                           f"{resp.status_code}: {resp.text[:120]}")
            return {}
        payload = resp.json()
    except Exception as e:
        logger.warning(f"NSE ticker feed error: {e}")
        return {}
    try:
        found = ticker_isins(payload)
        if found:
            from src.connectors import nse_isin
            nse_isin.remember(found, "nse_ticker feed")
    except Exception as e:
        logger.debug(f"NSE ticker ISINs not recorded: {e}")
    return parse_ticker_reply(payload, symbols, _eat_now().date())


def _snapshot_rows(payload: Any) -> Optional[List[Any]]:
    msg = payload.get("message") if isinstance(payload, dict) else None
    parts = [m for m in msg if isinstance(m, dict)] if isinstance(msg, list) else []
    return next((p["snapshot"] for p in parts if isinstance(p.get("snapshot"), list)), None)


def ticker_isins(payload: Any) -> Dict[str, str]:
    """ISIN -> ticker for feed rows that carry an ISIN beside the issuer."""
    from src.connectors.nse_isin import ISIN_RE
    out: Dict[str, str] = {}
    for item in _snapshot_rows(payload) or []:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("issuer", "")).strip().upper()
        isins = {str(v).strip().upper() for k, v in item.items()
                 if k != "issuer" and isinstance(v, str) and ISIN_RE.match(v.strip().upper())}
        if sym and len(isins) == 1:
            out[isins.pop()] = sym
    return out


# Longer than any NSE closure (Easter is four days) plus a weekend. A feed
# date older than this is a stuck feed, and storing under it would write
# today's prices into a past session's bar.
MAX_FEED_AGE_DAYS = 7


def _ticker_date(raw: Any, today) -> Any:
    """The session date the feed reports (dd/mm/yyyy), never later than today."""
    try:
        d = datetime.strptime(str(raw).strip(), "%d/%m/%Y").date()
        return min(d, today)
    except ValueError:
        return today


def _plausible(item: Dict[str, Any], field: str, ref: float) -> Optional[float]:
    """A price field, if positive and within a session's reach of ref."""
    v = _parse_num(str(item.get(field)))
    return v if v > 0 and abs(v / ref - 1) <= MAX_SESSION_MOVE else None


def parse_ticker_reply(payload: Any, symbols: Optional[List[str]], today) -> Dict[str, DailyBar]:
    """Turn a ticker feed reply into one DailyBar per watched symbol, or per
    listed issuer when `symbols` is None.

    The reply is {"message": [{"snapshot": [...]}, {"updated_at": {...}}]},
    each snapshot row carrying issuer, price, prev_price, today_open/high/low
    and volume. The feed's own open/high/low are not always consistent with
    its price (an open above the high has been seen), so each is kept only
    if it lies within a session's reach of the previous close, and the bar's
    high and low are widened to contain its open and close.
    """
    msg = payload.get("message") if isinstance(payload, dict) else None
    parts = [m for m in msg if isinstance(m, dict)] if isinstance(msg, list) else []
    snapshot = _snapshot_rows(payload)
    if snapshot is None:
        logger.warning("NSE ticker feed answered without a price snapshot; "
                       "the feed format may have changed")
        return {}
    updated = next((p["updated_at"] for p in parts if isinstance(p.get("updated_at"), dict)), {})
    session = _ticker_date(updated.get("date"), today)
    if session.weekday() >= 5:
        # No NSE session on a weekend; a bar dated one would be a flat copy
        # of Friday that the indicators would count as a trading day.
        logger.info(f"NSE ticker feed dated {session} (weekend); nothing stored")
        return {}
    if (today - session).days > MAX_FEED_AGE_DAYS:
        logger.warning(f"NSE ticker feed is dated {session}, {(today - session).days} days ago; "
                       f"the feed may be stuck, nothing stored")
        return {}

    wanted = {s.upper() for s in symbols} if symbols is not None else None
    results: Dict[str, DailyBar] = {}
    for item in snapshot:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("issuer", "")).strip().upper()
        if not sym or (wanted is not None and sym not in wanted):
            continue
        price = _parse_num(str(item.get("price"))) or _parse_num(str(item.get("ltp")))
        prev = _parse_num(str(item.get("prev_price")))
        if price <= 0:
            continue
        if prev > 0 and abs(price / prev - 1) > MAX_SESSION_MOVE:
            logger.warning(f"NSE ticker {sym} price {price} is {price / prev - 1:+.0%} from "
                           f"the previous close {prev}; not stored")
            continue
        ref = prev if prev > 0 else price
        open_ = _plausible(item, "today_open", ref) or ref
        high = max(v for v in (_plausible(item, "today_high", ref), open_, price) if v)
        low = min(v for v in (_plausible(item, "today_low", ref), open_, price) if v)
        results[sym] = DailyBar(
            date=session.isoformat(), symbol=sym,
            open=round(open_, 2), high=round(high, 2), low=round(low, 2),
            close=round(price, 2), volume=int(_parse_num(str(item.get("volume")))),
            change_pct=round((price - prev) / prev * 100, 2) if prev > 0 else 0.0,
            source="nse_ticker",
        )
    if wanted and snapshot and not results:
        logger.warning(f"NSE ticker feed listed {len(snapshot)} rows but none of the "
                       f"watched symbols; the feed format may have changed")
    return results


def scrape_afx_kwayisi(symbols: List[str]) -> Dict[str, DailyBar]:
    """
    Scrape from afx.kwayisi.org/nse — a public aggregator for NSE data.
    """
    results: Dict[str, DailyBar] = {}
    today = datetime.now(EAT).strftime("%Y-%m-%d")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    try:
        resp = requests.get("https://afx.kwayisi.org/nse/", headers=headers, timeout=20)
        if resp.status_code != 200:
            logger.warning(f"AFX Kwayisi https://afx.kwayisi.org/nse/ returned HTTP {resp.status_code}")
            return results
        parser = _NSETableParser()
        parser.feed(resp.text)

        symbol_set = set(s.upper() for s in symbols)
        _warn_if_unrecognised('AFX Kwayisi', 'https://afx.kwayisi.org/nse/', parser.rows, symbol_set)
        for row in parser.rows:
            if len(row) < 4:
                continue
            ticker_cell = row[0].upper().strip()
            matched = None
            for sym in symbol_set:
                if sym in ticker_cell:
                    matched = sym
                    break
            if not matched:
                continue
            try:
                # Live columns: Ticker | Name | Volume | Price | Change(abs)
                # e.g. ['SCOM', 'Safaricom Plc', '2,785,507', '34.20', '+0.15']
                if len(row) < 5:
                    continue
                price = _parse_num(row[3])
                if price <= 0:
                    continue
                vol = int(_parse_num(row[2]))
                chg_abs = _parse_num(row[4])
                prev = price - chg_abs
                chg_pct = (chg_abs / prev * 100) if prev > 0 else 0.0
                results[matched] = DailyBar(
                    date=today, symbol=matched,
                    open=round(prev, 2), high=max(price, prev), low=min(price, prev),
                    close=round(price, 2),
                    volume=vol, change_pct=round(chg_pct, 2), source="afx_kwayisi",
                )
            except (ValueError, IndexError):
                continue
    except Exception as e:
        logger.warning(f"AFX Kwayisi scrape error: {e}")
    return results


_COMPANY_PAGE_FIELDS = {
    'Earnings Per Share': 'eps',
    'Price/Earning Ratio': 'pe_ratio',
    'Dividend Per Share': 'dividend_per_share',
    'Dividend Yield': 'dividend_yield_pct',
}


def scrape_afx_company_page(symbol: str) -> Optional[Dict[str, Any]]:
    """Scrape EPS / P-E / dividend fields from a company's afx.kwayisi.org
    page. Returns None on any network error, non-200, or unrecognized page
    layout — callers must treat that as "no fundamentals available", never
    guess a value."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    url = f"https://afx.kwayisi.org/nse/{symbol.lower()}.html"
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return None
        html_text = resp.text
    except Exception as e:
        logger.warning(f"AFX company page scrape error ({symbol}): {e}")
        return None
    return _parse_company_page(html_text, symbol)


def _parse_company_page(html_text: str, symbol: str) -> Optional[Dict[str, Any]]:
    """Pure parser for the "Growth & Valuation" table afx renders as
    <tr><td>Label</td><td>Value</td></tr> rows. No network, safe to unit test
    directly against a captured HTML fixture."""
    marker = html_text.find('Growth &amp; Valuation')
    if marker == -1:
        marker = html_text.find('Growth & Valuation')
    if marker == -1:
        return None
    section = html_text[marker:marker + 2000]
    row_re = re.compile(r'<td>([^<]+)</td>\s*<td[^>]*>([^<]*)</td>', re.IGNORECASE)
    values: Dict[str, str] = {}
    for label, value in row_re.findall(section):
        label = label.strip()
        if label in _COMPANY_PAGE_FIELDS:
            values[_COMPANY_PAGE_FIELDS[label]] = value.strip()
    if 'eps' not in values or 'dividend_yield_pct' not in values:
        return None
    return {
        'symbol': symbol.upper(),
        'eps': _parse_num(values.get('eps', '0')),
        'pe_ratio': _parse_num(values.get('pe_ratio', '0')),
        'dividend_per_share': _parse_num(values.get('dividend_per_share', '0')),
        'dividend_yield_pct': _parse_num(values.get('dividend_yield_pct', '0')),
    }


def _parse_num(s: str) -> float:
    """Parse a number, removing commas and % signs."""
    try:
        return float(s.replace(",", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return 0.0


# ======================================================================
# Synthetic data generation (deterministic seed / fallback)
# ======================================================================

_PROFILES = {
    "SCOM": {"base": 28.0,  "vol": 0.015, "trend": 0.0001},
    "EQTY": {"base": 42.0,  "vol": 0.018, "trend": 0.0003},
    "KCB":  {"base": 35.0,  "vol": 0.020, "trend": 0.0002},
    "COOP": {"base": 13.0,  "vol": 0.016, "trend": 0.0001},
    "SCBK": {"base": 175.0, "vol": 0.012, "trend": 0.0001},
    "SBIC": {"base": 115.0, "vol": 0.015, "trend": 0.0003},
    "ABSA": {"base": 14.0,  "vol": 0.017, "trend": 0.0002},
    "BAT":  {"base": 350.0, "vol": 0.010, "trend": -0.0001},
    "EABL": {"base": 160.0, "vol": 0.013, "trend": 0.0001},
    "KEGN": {"base": 5.5,   "vol": 0.020, "trend": 0.0002},
    "KNRE": {"base": 2.5,   "vol": 0.022, "trend": 0.0001},
    "BAMB": {"base": 30.0,  "vol": 0.018, "trend": 0.0000},
    "TOTL": {"base": 22.0,  "vol": 0.015, "trend": 0.0001},
    "CTUM": {"base": 12.0,  "vol": 0.025, "trend": 0.0003},
    "NMG":  {"base": 18.0,  "vol": 0.020, "trend": -0.0001},
    "NCBA": {"base": 40.0,  "vol": 0.017, "trend": 0.0002},
    "BRIT": {"base": 6.0,   "vol": 0.023, "trend": 0.0002},
    "CIC":  {"base": 2.0,   "vol": 0.030, "trend": 0.0003},
}


def generate_synthetic(symbol: str, days: int = 730) -> List[DailyBar]:
    """Generate realistic synthetic history for backtesting."""
    rng = random.Random(hash(symbol) + 42)
    profile = _PROFILES.get(symbol, {"base": 20.0, "vol": 0.020, "trend": 0.0001})
    base, vol, trend = profile["base"], profile["vol"], profile["trend"]

    bars: List[DailyBar] = []
    price = base * rng.uniform(0.8, 1.0)
    current = datetime.now() - timedelta(days=days)

    while current <= datetime.now():
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue
        mr = (base - price) / base * 0.05
        ret = rng.gauss(trend + mr, vol)
        price = max(price * (1 + ret), 0.1)
        o = price * rng.uniform(0.995, 1.005)
        h = max(price, o) * rng.uniform(1.0, 1.0 + vol)
        l = min(price, o) * rng.uniform(1.0 - vol, 1.0)
        v = int(rng.lognormvariate(12, 1.5))
        bars.append(DailyBar(
            date=current.strftime("%Y-%m-%d"), symbol=symbol,
            open=round(o, 2), high=round(h, 2), low=round(l, 2),
            close=round(price, 2), volume=v,
            change_pct=round(ret * 100, 2), source="synthetic",
        ))
        current += timedelta(days=1)
    return bars


def backfill_afx_history(symbols: List[str]) -> Dict[str, int]:
    """Backfill recent daily bars from afx per-stock pages
    (afx.kwayisi.org/nse/<sym>.html carries ~2 weeks of dated rows:
    Date | Volume | Close | Change | Change%). Merged into the CSVs by
    date, so re-runs and overlaps are harmless."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    added: Dict[str, int] = {}
    symbols = list(symbols)
    for i, sym in enumerate(symbols):
        try:
            # A short connect timeout: afx resolves to several addresses and
            # each is tried in turn, so 20 seconds each cost 80 per symbol
            # when the site was unreachable from the server.
            resp = requests.get(f"https://afx.kwayisi.org/nse/{sym.lower()}.html",
                                headers=headers, timeout=(5, 20))
            if resp.status_code != 200:
                logger.warning(f"AFX history page for {sym} returned HTTP {resp.status_code}")
                continue
            parser = _NSETableParser()
            parser.feed(resp.text)
            bars = []
            for row in parser.rows:
                if len(row) < 3 or not row[0][:4].isdigit():
                    continue
                try:
                    close = _parse_num(row[2])
                    if close <= 0:
                        continue
                    chg = _parse_num(row[3]) if len(row) > 3 and row[3] else 0.0
                    prev = close - chg
                    pct = (chg / prev * 100) if prev > 0 else 0.0
                    bars.append(DailyBar(
                        date=row[0], symbol=sym,
                        open=round(prev, 2), high=max(close, prev),
                        low=min(close, prev), close=round(close, 2),
                        volume=int(_parse_num(row[1])),
                        change_pct=round(pct, 2), source="afx_history",
                    ))
                except (ValueError, IndexError):
                    continue
            if bars:
                save_bars_csv(sym, bars)
                added[sym] = len(bars)
                logger.info(f"  {sym}: backfilled {len(bars)} historical bars")
        except (requests.ConnectionError, requests.Timeout) as e:
            # The site, not the symbol: the rest would fail the same way.
            logger.warning(f"afx.kwayisi.org unreachable ({e.__class__.__name__}); "
                           f"skipping its history backfill for {len(symbols) - i} symbol(s)")
            break
        except Exception as e:
            logger.warning(f"Backfill failed for {sym}: {e}")
    return added


# ======================================================================
# Storage helpers
# ======================================================================

def save_bars_csv(symbol: str, bars: List[DailyBar]):
    """Save/merge bars into a CSV file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"{symbol}.csv"
    existing: Dict[str, dict] = {}
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["date"]] = row
    for bar in bars:
        existing[bar.date] = asdict(bar)
    fields = ["date", "symbol", "open", "high", "low", "close", "volume", "change_pct", "source"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in sorted(existing.keys()):
            w.writerow(existing[d])


def save_bars_db(bars: List[DailyBar], db):
    """Upsert bars into the nse_daily_prices Supabase table."""
    if not db:
        return
    try:
        client = getattr(db, 'supabase_client', None)
        if not client:
            return
        for bar in bars:
            payload = {
                "symbol": bar.symbol,
                "date": bar.date,
                "open_price": bar.open,
                "high_price": bar.high,
                "low_price": bar.low,
                "close_price": bar.close,
                "volume": bar.volume,
                "change_pct": bar.change_pct,
                "prev_close": bar.open,  # approximation
                "source": bar.source,
                "updated_at": datetime.now(EAT).isoformat(),
            }
            try:
                client.table("nse_daily_prices").upsert(
                    payload, on_conflict="symbol,date"
                ).execute()
            except Exception as e:
                logger.debug(f"DB upsert {bar.symbol}/{bar.date}: {e}")
    except Exception as e:
        logger.warning(f"save_bars_db error: {e}")


def load_csv(symbol: str) -> List[Dict[str, Any]]:
    csv_path = DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ======================================================================
# Periodic scraper (runs as a background thread)
# ======================================================================

class NSEPeriodicScraper:
    """
    Background scraper that periodically fetches NSE prices from public
    web sources, stores them in the database AND local CSV, and keeps
    the NSEConnector cache up to date.

    Interval logic:
      - During NSE hours (09:00-15:00 EAT, Mon-Fri): runs every
        `interval_minutes` (default 30 min).
      - Outside hours / weekends: runs once every 6 hours to check for
        end-of-day updates.
    """

    def __init__(
        self,
        database_manager=None,
        nse_connector=None,
        interval_minutes: int = 30,
        symbols: Optional[List[str]] = None,
        record_all_listed: bool = True,
    ):
        self.db = database_manager
        # Store a bar for every stock the ticker feed lists, not only the
        # watched ones, so every NSE stock builds real history and Market
        # Watch and the screener can cover the whole exchange. The feed is
        # one request either way.
        self.record_all_listed = record_all_listed
        self._last_listed_count: Optional[int] = None
        self.nse_connector = nse_connector
        self.interval = interval_minutes * 60  # to seconds
        self.off_hours_interval = 6 * 3600     # 6 hours
        self.symbols = symbols or DEFAULT_SYMBOLS
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_run: Optional[float] = None
        # What the last cycle actually got. get_status() used to report only
        # that the thread was running, which is equally true of a scraper
        # that has never fetched one real price and is serving synthetic
        # seed data. These make "is it working?" answerable from outside.
        self._last_real_count: Optional[int] = None
        self._last_missing: List[str] = []
        self._last_real_success: Optional[float] = None

    def start(self):
        """Start the periodic scraper in a background daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="nse-scraper")
        self._thread.start()
        logger.info(f"NSE periodic scraper started (interval={self.interval // 60}min, symbols={len(self.symbols)})")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("NSE periodic scraper stopped")

    def run_once(self) -> Dict[str, int]:
        """Run a single scrape cycle. Returns {symbol: bar_count}."""
        logger.info("NSE scraper: starting cycle...")
        results: Dict[str, int] = {}

        # 1. The NSE's own ticker feed
        web_bars = scrape_nse_ticker(None if self.record_all_listed else self.symbols)
        logger.info(f"  NSE ticker: {len(web_bars)} symbols")
        extra = {s: b for s, b in web_bars.items() if s not in self.symbols}
        for sym, bar in extra.items():
            save_bars_csv(sym, [bar])
            save_bars_db([bar], self.db)
        if self.record_all_listed:
            self._last_listed_count = len(web_bars)
            from src.connectors.nse_universe import register
            register(web_bars)

        # 2. Try AFX Kwayisi aggregator for symbols we didn't get
        missing = [s for s in self.symbols if s not in web_bars]
        if missing:
            afx_bars = scrape_afx_kwayisi(missing)
            logger.info(f"  AFX Kwayisi: {len(afx_bars)} symbols")
            web_bars.update(afx_bars)

        # 3. For symbols still missing, use last CSV data (don't regenerate synthetic for daily runs)
        for sym in self.symbols:
            if sym in web_bars:
                bar = web_bars[sym]
                save_bars_csv(sym, [bar])
                save_bars_db([bar], self.db)
                results[sym] = 1
            else:
                results[sym] = 0

        # 4. Refresh connector cache
        if self.nse_connector:
            self.nse_connector.refresh_cache()

        scraped = sum(1 for v in results.values() if v > 0)
        self._last_real_count = scraped
        self._last_missing = sorted(s for s, v in results.items() if v == 0)
        if scraped:
            self._last_real_success = time.time()
        if scraped == 0 and _eat_now().weekday() >= 5:
            # The ticker stores nothing dated a weekend, so an empty weekend
            # cycle is the calendar, not a failed source.
            logger.info("NSE scraper cycle: weekend, no session to record")
        elif scraped == 0:
            # Every live source failed. The connector will keep serving
            # whatever is newest on disk, which after first-run seeding is
            # synthetic. Say so at WARNING, not buried in an INFO count.
            logger.warning(
                f"NSE scraper cycle got NO real prices (0/{len(self.symbols)}); "
                f"NSE quotes are stale or synthetic until a live source answers")
        else:
            logger.info(f"NSE scraper cycle complete: {scraped}/{len(self.symbols)} symbols updated")
        self._last_run = time.time()
        return results

    def seed_historical(self, days: int = 730):
        """
        Seed historical data for all symbols. Uses synthetic data as
        the baseline, then overlays any real scraped data on top.
        Call this once on first setup.
        """
        logger.info(f"Seeding NSE historical data ({days} days, {len(self.symbols)} symbols)...")
        for sym in self.symbols:
            csv_path = DATA_DIR / f"{sym}.csv"
            if csv_path.exists():
                existing = load_csv(sym)
                # Never bury real scraped bars under synthetic seed data.
                if any((b.get('source') if isinstance(b, dict) else getattr(b, 'source', '')) != 'synthetic'
                       for b in existing):
                    logger.info(f"  {sym}: has real bars, skipping synthetic seed")
                    continue
                if len(existing) >= days // 2:
                    logger.info(f"  {sym}: already has {len(existing)} bars, skipping seed")
                    continue
            bars = generate_synthetic(sym, days)
            save_bars_csv(sym, bars)
            save_bars_db(bars, self.db)
            logger.info(f"  {sym}: seeded {len(bars)} bars")
        logger.info("Historical seed complete")

    def _loop(self):
        """Main background loop."""
        # Initial run on startup
        try:
            self.run_once()
        except Exception as e:
            logger.error(f"NSE scraper initial run failed: {e}")

        while self._running:
            try:
                now = datetime.now(EAT)
                from src.connectors.nse_connector import NSE_PREOPEN_START, NSE_CLOSE
                is_market_hours = (
                    now.weekday() < 5
                    and NSE_PREOPEN_START <= now.timetz().replace(tzinfo=None) < NSE_CLOSE
                )
                wait_time = self.interval if is_market_hours else self.off_hours_interval

                # Sleep in small increments so we can stop quickly
                slept = 0
                while slept < wait_time and self._running:
                    time.sleep(min(30, wait_time - slept))
                    slept += 30

                if not self._running:
                    break

                self.run_once()

            except Exception as e:
                logger.error(f"NSE scraper loop error: {e}")
                time.sleep(60)  # back off on error

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "last_run": datetime.fromtimestamp(self._last_run, tz=EAT).isoformat() if self._last_run else None,
            "interval_minutes": self.interval // 60,
            "symbols_count": len(self.symbols),
            # Every stock the feed listed last cycle (all of them get a bar).
            "last_cycle_listed": self._last_listed_count,
            # None until the first cycle finishes.
            "last_cycle_real_prices": self._last_real_count,
            "last_cycle_missing": self._last_missing,
            "last_real_price_at": (datetime.fromtimestamp(self._last_real_success, tz=EAT).isoformat()
                                   if self._last_real_success else None),
        }


# ======================================================================
# Table creation for Supabase
# ======================================================================

NSE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS nse_daily_prices (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(12, 2),
    high_price DECIMAL(12, 2),
    low_price DECIMAL(12, 2),
    close_price DECIMAL(12, 2),
    volume BIGINT DEFAULT 0,
    change_pct DECIMAL(8, 2) DEFAULT 0,
    prev_close DECIMAL(12, 2),
    source VARCHAR(30) DEFAULT 'scraper',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_nse_symbol_date ON nse_daily_prices(symbol, date DESC);
"""


def create_nse_table(db) -> bool:
    """Create the nse_daily_prices table in Postgres/Supabase."""
    if not db:
        return False

    # Try via Supabase RPC or direct Postgres
    try:
        if hasattr(db, 'get_connection') and db.connection_pool:
            conn = db.get_connection()
            if conn:
                try:
                    cur = conn.cursor()
                    cur.execute(NSE_TABLE_SQL)
                    conn.commit()
                    cur.close()
                    logger.info("Created nse_daily_prices table (Postgres)")
                    return True
                finally:
                    db.return_connection(conn)
    except Exception as e:
        logger.warning(f"Postgres table creation failed: {e}")

    # For Supabase, the table needs to be created via the dashboard or migration.
    # Log instructions.
    logger.info(
        "To create the NSE table in Supabase, run this SQL in the SQL Editor:\n"
        + NSE_TABLE_SQL
    )
    return False


# ======================================================================
# CLI
# ======================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    scraper = NSEPeriodicScraper()

    if len(sys.argv) >= 2 and sys.argv[1] == "--seed":
        days = int(sys.argv[2]) if len(sys.argv) >= 3 else 730
        scraper.seed_historical(days)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--scrape":
        results = scraper.run_once()
        print(f"\nScraped {sum(1 for v in results.values() if v > 0)}/{len(results)} symbols")
    elif len(sys.argv) >= 2 and sys.argv[1] == "--backfill":
        added = backfill_afx_history(DEFAULT_SYMBOLS)
        print(f"\nBackfilled {sum(added.values())} bars across {len(added)} symbols")
    else:
        print("=" * 55)
        print("  NSE Kenya Market Data Scraper")
        print("=" * 55)
        print("\nUsage:")
        print("  --seed [days]   Seed historical data (default: 730 days)")
        print("  --scrape        Run a single scrape cycle")
        print("  --backfill      Backfill recent history from afx per-stock pages")
        print("\nSeeding historical data now...")
        scraper.seed_historical()


if __name__ == "__main__":
    main()
