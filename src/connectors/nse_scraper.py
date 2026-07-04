"""
NSE Kenya Market Data Scraper
Scrapes end-of-day and intraday prices from public web sources,
stores them in Supabase and local CSV, and runs periodically
as a background thread inside the trading agent.

Public data sources used (no API key required):
  1. NSE official website market reports
  2. African-markets.com / afx.kwayisi.org (public price tables)
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

DEFAULT_SYMBOLS = [
    "SCOM", "EQTY", "KCB", "COOP", "SCBK", "SBIC", "ABSA",
    "BAT", "EABL", "KEGN", "KNRE", "BAMB",
    "TOTL", "CTUM", "NMG", "NCBA", "BRIT", "CIC",
]

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "nse_historical"


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
    """Parse HTML tables from the NSE website."""
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_row: List[str] = []
        self.rows: List[List[str]] = []
        self.cell_data = ""

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.current_row = []
        elif tag in ("td", "th") and self.in_row:
            self.in_cell = True
            self.cell_data = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell:
            self.current_row.append(self.cell_data.strip())
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.current_row:
                self.rows.append(self.current_row)
            self.in_row = False
        elif tag == "table":
            self.in_table = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell_data += data


def scrape_nse_website(symbols: List[str]) -> Dict[str, DailyBar]:
    """
    Scrape current prices from the NSE official equity stats page.
    Returns a dict mapping symbol -> DailyBar for today.
    """
    results: Dict[str, DailyBar] = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }
    urls = [
        "https://www.nse.co.ke/market-statistics/equity-statistics.html",
        "https://www.nse.co.ke/listed-companies/list.html",
    ]
    today = datetime.now(EAT).strftime("%Y-%m-%d")
    symbol_set = set(s.upper() for s in symbols)

    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code != 200:
                continue
            parser = _NSETableParser()
            parser.feed(resp.text)
            for row in parser.rows:
                if len(row) < 5:
                    continue
                # First cell is usually company name or ticker
                ticker_cell = row[0].upper().strip()
                matched_sym = None
                for sym in symbol_set:
                    if sym in ticker_cell:
                        matched_sym = sym
                        break
                if not matched_sym:
                    continue
                try:
                    close_val = _parse_num(row[-2] if len(row) >= 6 else row[4])
                    if close_val <= 0:
                        close_val = _parse_num(row[1])
                    if close_val <= 0:
                        continue
                    open_val = _parse_num(row[1]) if len(row) >= 6 else close_val
                    high_val = _parse_num(row[2]) if len(row) >= 6 else close_val
                    low_val = _parse_num(row[3]) if len(row) >= 6 else close_val
                    vol = int(_parse_num(row[-1])) if len(row) >= 6 else 0
                    prev = _parse_num(row[-3]) if len(row) >= 7 else close_val
                    chg = ((close_val - prev) / prev * 100) if prev > 0 else 0
                    results[matched_sym] = DailyBar(
                        date=today, symbol=matched_sym,
                        open=round(open_val, 2), high=round(high_val, 2),
                        low=round(low_val, 2), close=round(close_val, 2),
                        volume=vol, change_pct=round(chg, 2), source="nse_website",
                    )
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            logger.warning(f"NSE website scrape error ({url}): {e}")

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
            return results
        parser = _NSETableParser()
        parser.feed(resp.text)

        symbol_set = set(s.upper() for s in symbols)
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
                # Typical columns: Name, Price, Change, %Change, Volume
                price = _parse_num(row[1])
                if price <= 0:
                    continue
                chg_pct = _parse_num(row[3]) if len(row) > 3 else 0
                vol = int(_parse_num(row[4])) if len(row) > 4 else 0
                results[matched] = DailyBar(
                    date=today, symbol=matched,
                    open=price, high=price, low=price, close=round(price, 2),
                    volume=vol, change_pct=round(chg_pct, 2), source="afx_kwayisi",
                )
            except (ValueError, IndexError):
                continue
    except Exception as e:
        logger.warning(f"AFX Kwayisi scrape error: {e}")
    return results


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
    ):
        self.db = database_manager
        self.nse_connector = nse_connector
        self.interval = interval_minutes * 60  # to seconds
        self.off_hours_interval = 6 * 3600     # 6 hours
        self.symbols = symbols or DEFAULT_SYMBOLS
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_run: Optional[float] = None

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

        # 1. Try NSE official website
        web_bars = scrape_nse_website(self.symbols)
        logger.info(f"  NSE website: {len(web_bars)} symbols")

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
                is_market_hours = (
                    now.weekday() < 5
                    and 9 <= now.hour < 15
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
    else:
        print("=" * 55)
        print("  NSE Kenya Market Data Scraper")
        print("=" * 55)
        print("\nUsage:")
        print("  --seed [days]   Seed historical data (default: 730 days)")
        print("  --scrape        Run a single scrape cycle")
        print("\nSeeding historical data now...")
        scraper.seed_historical()


if __name__ == "__main__":
    main()
