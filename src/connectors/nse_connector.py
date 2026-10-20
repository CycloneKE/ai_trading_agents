"""
Nairobi Securities Exchange (NSE) Data Connector
Serves Kenya stock data from the database and local CSV cache.
No external API keys required - data is populated by the periodic scraper.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# East Africa Time offset (UTC+3)
EAT_OFFSET = timezone(timedelta(hours=3))

# NSE equities session (Nairobi time): pre-open auction 09:00-09:30,
# continuous trading 09:30-15:00. Orders may only be submitted while 'open'.
from datetime import time as dtime
NSE_PREOPEN_START = dtime(9, 0)
NSE_OPEN = dtime(9, 30)
NSE_CLOSE = dtime(15, 0)

# Kenya's fixed-date public holidays, 2026 (NSE closes on all gazetted public
# holidays). Easter-linked (Good Friday, Easter Monday) and Eid al-Fitr/
# al-Adha move every year and are deliberately NOT included — add them (and
# refresh this list annually) via config.data_manager.nse_holidays.
_DEFAULT_NSE_HOLIDAYS_2026 = {
    "2026-01-01",  # New Year's Day
    "2026-05-01",  # Labour Day
    "2026-06-01",  # Madaraka Day
    "2026-10-10",  # Utamaduni Day
    "2026-10-20",  # Mashujaa Day
    "2026-12-12",  # Jamhuri Day
    "2026-12-25",  # Christmas Day
    "2026-12-26",  # Boxing Day
}

# -------------------------------------------------------------------
# Target NSE Symbols
# -------------------------------------------------------------------
NSE_TIER1_SYMBOLS = [
    "SCOM",   # Safaricom PLC (Telecom / M-Pesa)
    "EQTY",   # Equity Group Holdings (Banking)
    "KCB",    # KCB Group (Banking)
    "COOP",   # Co-operative Bank (Banking)
    "SCBK",   # Standard Chartered Kenya (Banking)
    "SBIC",   # Stanbic Holdings (Banking)
    "ABSA",   # ABSA Bank Kenya (Banking)
    "BAT",    # BAT Kenya (Consumer)
    "EABL",   # East African Breweries (Consumer)
    "KEGN",   # KenGen (Energy)
    "KNRE",   # Kenya Re (Insurance)
    "BAMB",   # Bamburi Cement (Industrials)
]

NSE_TIER2_SYMBOLS = [
    "TOTL",   # TotalEnergies Marketing (Energy)
    "CTUM",   # Centum Investment (Diversified)
    "NMG",    # Nation Media Group (Media)
    "NCBA",   # NCBA Group (Banking)
    "BRIT",   # Britam Holdings (Insurance)
    "CIC",    # CIC Insurance Group (Insurance)
]

ALL_NSE_SYMBOLS = NSE_TIER1_SYMBOLS + NSE_TIER2_SYMBOLS

# Sector classification for dashboard heatmap
NSE_SECTORS = {
    "Banking":     ["EQTY", "KCB", "COOP", "SCBK", "SBIC", "ABSA", "NCBA"],
    "Telecom":     ["SCOM"],
    "Consumer":    ["BAT", "EABL"],
    "Energy":      ["KEGN", "TOTL"],
    "Insurance":   ["KNRE", "BRIT", "CIC"],
    "Industrials": ["BAMB"],
    "Diversified": ["CTUM"],
    "Media":       ["NMG"],
}

# KES/USD fallback (updated periodically; 1 KES in USD)
_DEFAULT_KES_USD = 1 / 130.0


class NSEConnector:
    """
    Data connector for Nairobi Securities Exchange equities.

    Primary data source is the Supabase database (populated by the periodic
    scraper).  Falls back to local CSV files under data/nse_historical/ when
    the database is unavailable.
    """

    def __init__(self, config: Dict[str, Any], database_manager=None):
        self.config = config
        self.db = database_manager

        # In-memory latest-quote cache: symbol -> dict
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: Dict[str, float] = {}
        self._cache_ttl = config.get("cache_ttl", 1800)  # 30 min

        # KES/USD
        self._kes_usd_rate: float = _DEFAULT_KES_USD

        # Gazetted NSE closure dates (ISO 'YYYY-MM-DD'), config-overridable.
        # The default covers Kenya's fixed-date public holidays for the
        # current year; Easter-linked (Good Friday/Easter Monday) and Eid
        # dates move every year and are NOT auto-computed here — update
        # config.data_manager.nse_holidays annually / when the government
        # gazettes moveable dates.
        self._holidays = set(config.get("nse_holidays", _DEFAULT_NSE_HOLIDAYS_2026))

        logger.info(
            f"NSEConnector initialized - DB: {'yes' if self.db else 'no'}, "
            f"tracking {len(ALL_NSE_SYMBOLS)} symbols"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "connector": "nse_kenya",
            "database": bool(self.db),
            "cached_symbols": len(self._cache),
            "kes_usd_rate": self._kes_usd_rate,
            "market_open": self.is_market_open(),
            "market_phase": self.market_phase(),
            "timestamp": datetime.now(EAT_OFFSET).isoformat(),
            "status": "ok",
        }

    def market_phase(self, now=None) -> str:
        now = now or datetime.now(EAT_OFFSET)
        if now.weekday() > 4:
            return 'closed'
        if now.date().isoformat() in getattr(self, '_holidays', ()):
            return 'closed'
        t = now.timetz().replace(tzinfo=None)
        if NSE_PREOPEN_START <= t < NSE_OPEN:
            return 'preopen'
        if NSE_OPEN <= t < NSE_CLOSE:
            return 'open'
        return 'closed'

    def is_market_open(self, now=None) -> bool:
        return self.market_phase(now) == 'open'

    def get_all_quotes(self) -> List[Dict[str, Any]]:
        results = []
        for symbol in ALL_NSE_SYMBOLS:
            q = self.get_quote(symbol)
            if q:
                results.append(q)
        return results

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        # 1. Check in-memory cache
        if symbol in self._cache:
            age = time.time() - self._cache_ts.get(symbol, 0)
            if age < self._cache_ttl:
                return self._cache[symbol]

        # 2. Try database
        quote = self._read_from_db(symbol)
        if quote:
            self._cache[symbol] = quote
            self._cache_ts[symbol] = time.time()
            return quote

        # 3. Fall back to CSV
        quote = self._read_from_csv(symbol)
        if quote:
            self._cache[symbol] = quote
            self._cache_ts[symbol] = time.time()
            return quote

        return self._empty_quote(symbol)

    def get_top_movers(self) -> Dict[str, List[Dict[str, Any]]]:
        quotes = self.get_all_quotes()
        valid = [q for q in quotes if q.get("change_pct") is not None]
        s = sorted(valid, key=lambda x: x.get("change_pct", 0), reverse=True)
        return {
            "gainers": s[:5],
            "losers": s[-5:][::-1] if len(s) >= 5 else [],
        }

    def get_sector_performance(self) -> List[Dict[str, Any]]:
        quotes = {q["symbol"]: q for q in self.get_all_quotes()}
        perf = []
        for sector, symbols in NSE_SECTORS.items():
            changes = [quotes[s]["change_pct"] for s in symbols if s in quotes and quotes[s].get("change_pct") is not None]
            perf.append({
                "sector": sector,
                "change": round(sum(changes) / len(changes), 2) if changes else 0,
                "symbols_tracked": len(symbols),
                "symbols_reporting": len(changes),
            })
        return perf

    def get_kes_usd_rate(self) -> float:
        return self._kes_usd_rate

    def refresh_cache(self):
        """Force-refresh the in-memory cache from DB/CSV."""
        self._cache.clear()
        self._cache_ts.clear()
        self.get_all_quotes()

    # ------------------------------------------------------------------
    # Database reads
    # ------------------------------------------------------------------

    def _read_from_db(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.db:
            return None
        try:
            # Try Supabase REST client first
            if hasattr(self.db, 'supabase_client') and self.db.supabase_client:
                resp = (
                    self.db.supabase_client
                    .table('nse_daily_prices')
                    .select('*')
                    .eq('symbol', symbol)
                    .order('date', desc=True)
                    .limit(1)
                    .execute()
                )
                if resp.data and len(resp.data) > 0:
                    row = resp.data[0]
                    return self._row_to_quote(row)

            # Postgres fallback
            if hasattr(self.db, 'get_connection'):
                conn = self.db.get_connection()
                if conn:
                    try:
                        cur = conn.cursor()
                        cur.execute(
                            "SELECT * FROM nse_daily_prices WHERE symbol = %s ORDER BY date DESC LIMIT 1",
                            (symbol,)
                        )
                        row_raw = cur.fetchone()
                        if row_raw:
                            cols = [d[0] for d in cur.description]
                            row = dict(zip(cols, row_raw))
                            return self._row_to_quote(row)
                    finally:
                        if conn:
                            self.db.return_connection(conn)
        except Exception as e:
            logger.debug(f"DB read for {symbol}: {e}")
        return None

    def _row_to_quote(self, row: Dict[str, Any]) -> Dict[str, Any]:
        price_kes = float(row.get('close_price') or row.get('close', 0))
        kes_usd = self._kes_usd_rate
        return {
            "symbol": row.get('symbol', ''),
            "market": "NSE",
            "currency": "KES",
            "price_kes": round(price_kes, 2),
            "price_usd": round(price_kes * kes_usd, 4),
            "open_kes": float(row.get('open_price') or row.get('open', price_kes)),
            "high_kes": float(row.get('high_price') or row.get('high', price_kes)),
            "low_kes": float(row.get('low_price') or row.get('low', price_kes)),
            "volume": int(row.get('volume', 0)),
            "change_pct": float(row.get('change_pct', 0)),
            "prev_close_kes": float(row.get('prev_close', price_kes)),
            "kes_usd_rate": kes_usd,
            "timestamp": str(row.get('date', '')),
            "source": row.get('source', 'database'),
            "tier": "tier1" if row.get('symbol', '') in NSE_TIER1_SYMBOLS else "tier2",
            "sector": self._get_sector(row.get('symbol', '')),
        }

    # ------------------------------------------------------------------
    # CSV fallback
    # ------------------------------------------------------------------

    def _read_from_csv(self, symbol: str) -> Optional[Dict[str, Any]]:
        import csv
        from pathlib import Path
        csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "nse_historical" / f"{symbol}.csv"
        if not csv_path.exists():
            return None
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                if not reader:
                    return None
                last = reader[-1]  # most recent bar
                price_kes = float(last.get('close', 0))
                kes_usd = self._kes_usd_rate
                return {
                    "symbol": symbol,
                    "market": "NSE",
                    "currency": "KES",
                    "price_kes": round(price_kes, 2),
                    "price_usd": round(price_kes * kes_usd, 4),
                    "open_kes": float(last.get('open', price_kes)),
                    "high_kes": float(last.get('high', price_kes)),
                    "low_kes": float(last.get('low', price_kes)),
                    "volume": int(last.get('volume', 0)),
                    "change_pct": float(last.get('change_pct', 0)),
                    "prev_close_kes": price_kes,
                    "kes_usd_rate": kes_usd,
                    "timestamp": last.get('date', ''),
                    "source": "csv",
                    "tier": "tier1" if symbol in NSE_TIER1_SYMBOLS else "tier2",
                    "sector": self._get_sector(symbol),
                }
        except Exception as e:
            logger.warning(f"CSV read for {symbol}: {e}")
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_sector(self, symbol: str) -> str:
        for sector, symbols in NSE_SECTORS.items():
            if symbol in symbols:
                return sector
        return "Other"

    def _empty_quote(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol, "market": "NSE", "currency": "KES",
            "price_kes": 0, "price_usd": 0, "open_kes": 0, "high_kes": 0,
            "low_kes": 0, "volume": 0, "change_pct": 0, "prev_close_kes": 0,
            "kes_usd_rate": self._kes_usd_rate,
            "timestamp": datetime.now(EAT_OFFSET).isoformat(),
            "source": "none", "tier": "tier1" if symbol in NSE_TIER1_SYMBOLS else "tier2",
            "sector": self._get_sector(symbol), "_stale": True,
        }
