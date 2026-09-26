"""The NSE's listed equities: names and sectors, and which ones have data.

Which stocks the dashboard shows is decided by the data, not by this list:
the NSE ticker feed lists every issuer, the scraper stores a daily bar for
each, and a stock appears in Market Watch once it has a real bar. This file
only puts a company name and a sector beside each ticker; a ticker it does
not know is shown under its own code as "Unclassified" rather than dropped.

Sectors follow the NSE's own classification. The screener caps how many
names it takes from one sector, so the banking cluster that dominates the
exchange cannot fill the whole shortlist.
"""
import csv
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

UNCLASSIFIED = 'Unclassified'

# What an NSE equity ticker looks like. The feed also lists bonds under
# codes such as "IFB1/2023/17"; those, and anything that could name a path
# outside the data folder, are not stocks and are never stored.
TICKER_RE = re.compile(r'^[A-Z0-9][A-Z0-9.&-]{0,15}$')


def valid_ticker(symbol: str) -> bool:
    return bool(TICKER_RE.match(symbol or '')) and '..' not in symbol

# ticker: (company, sector). Names are for display only.
LISTED: Dict[str, Tuple[str, str]] = {
    # Agricultural
    'EGAD': ('Eaagads', 'Agricultural'),
    'KUKZ': ('Kakuzi', 'Agricultural'),
    'KAPC': ('Kapchorua Tea', 'Agricultural'),
    'LIMT': ('Limuru Tea', 'Agricultural'),
    'SASN': ('Sasini', 'Agricultural'),
    'WTK': ('Williamson Tea Kenya', 'Agricultural'),
    # Automobiles and accessories
    'CGEN': ('Car & General', 'Automobiles'),
    # Banking
    'ABSA': ('Absa Bank Kenya', 'Banking'),
    'BKG': ('BK Group', 'Banking'),
    'COOP': ('Co-operative Bank', 'Banking'),
    'DTK': ('Diamond Trust Bank', 'Banking'),
    'EQTY': ('Equity Group', 'Banking'),
    'HFCK': ('HF Group', 'Banking'),
    'IMH': ('I&M Group', 'Banking'),
    'KCB': ('KCB Group', 'Banking'),
    'NCBA': ('NCBA Group', 'Banking'),
    'SBIC': ('Stanbic Holdings', 'Banking'),
    'SCBK': ('Standard Chartered Kenya', 'Banking'),
    # Commercial and services
    'XPRS': ('Express Kenya', 'Commercial & Services'),
    'HBER': ('Homeboyz Entertainment', 'Commercial & Services'),
    'KQ': ('Kenya Airways', 'Commercial & Services'),
    'LKL': ('Longhorn Publishers', 'Commercial & Services'),
    'NBV': ('Nairobi Business Ventures', 'Commercial & Services'),
    'NMG': ('Nation Media Group', 'Commercial & Services'),
    'SMER': ('Sameer Africa', 'Commercial & Services'),
    'SGL': ('Standard Group', 'Commercial & Services'),
    'TPSE': ('TPS Eastern Africa (Serena)', 'Commercial & Services'),
    'UCHM': ('Uchumi Supermarkets', 'Commercial & Services'),
    'SCAN': ('WPP Scangroup', 'Commercial & Services'),
    # Construction and allied
    'BAMB': ('Bamburi Cement', 'Construction'),
    'CRWN': ('Crown Paints', 'Construction'),
    'CABL': ('East African Cables', 'Construction'),
    'PORT': ('East African Portland Cement', 'Construction'),
    # Energy and petroleum
    'KEGN': ('KenGen', 'Energy'),
    'KPLC': ('Kenya Power', 'Energy'),
    'TOTL': ('TotalEnergies Marketing Kenya', 'Energy'),
    'UMME': ('Umeme', 'Energy'),
    # Insurance
    'BRIT': ('Britam Holdings', 'Insurance'),
    'CIC': ('CIC Insurance Group', 'Insurance'),
    'JUB': ('Jubilee Holdings', 'Insurance'),
    'KNRE': ('Kenya Re', 'Insurance'),
    'LBTY': ('Liberty Kenya', 'Insurance'),
    'SLAM': ('Sanlam Kenya', 'Insurance'),
    # Investment
    'CTUM': ('Centum Investment', 'Investment'),
    'HAFR': ('Home Afrika', 'Investment'),
    'KURV': ('Kurwitu Ventures', 'Investment'),
    'OCH': ('Olympia Capital', 'Investment'),
    'TCL': ('TransCentury', 'Investment'),
    # Investment services
    'NSE': ('Nairobi Securities Exchange', 'Investment Services'),
    # Manufacturing and allied
    'AMAC': ('Africa Mega Agricorp', 'Manufacturing'),
    'BOC': ('BOC Kenya', 'Manufacturing'),
    'BAT': ('BAT Kenya', 'Manufacturing'),
    'CARB': ('Carbacid Investments', 'Manufacturing'),
    'EABL': ('East African Breweries', 'Manufacturing'),
    'EVRD': ('Eveready East Africa', 'Manufacturing'),
    'FTGH': ('Flame Tree Group', 'Manufacturing'),
    'ORCH': ('Kenya Orchards', 'Manufacturing'),
    'SKL': ('Shri Krishana Overseas', 'Manufacturing'),
    'UNGA': ('Unga Group', 'Manufacturing'),
    # Telecommunication
    'SCOM': ('Safaricom', 'Telecommunication'),
}

# Tickers seen at runtime (the ticker feed, the screener's shortlist) that
# the list above does not name. They are NSE stocks all the same.
_seen: Set[str] = set()


def register(symbols: Iterable[str]) -> None:
    """Record tickers known, from the data, to be NSE stocks."""
    _seen.update(s.upper() for s in symbols if s and valid_ticker(s.upper()))


def is_nse_symbol(symbol: str) -> bool:
    """A listed stock, one the feed has reported, or one with NSE history on
    disk. The last check keeps the answer the same right after a restart,
    before the scraper has reported anything."""
    s = (symbol or '').upper()
    if s in LISTED or s in _seen:
        return True
    if not valid_ticker(s):
        return False
    try:
        from src.connectors.nse_connector import NSE_CSV_DIR
        if (Path(NSE_CSV_DIR) / f"{s}.csv").exists():
            _seen.add(s)
            return True
    except Exception as e:  # never let a lookup break costing
        logger.debug(f"NSE universe: history lookup for {s} failed: {e}")
    return False


def name_of(symbol: str) -> str:
    return LISTED.get((symbol or '').upper(), (symbol, None))[0]


def sector_of(symbol: str) -> str:
    return LISTED.get((symbol or '').upper(), (None, UNCLASSIFIED))[1]


def symbols_with_real_data(csv_dir: Path) -> List[str]:
    """Every ticker whose history holds at least one real market bar.

    Reads only the last row of each file: the scraper appends in date
    order and a stock with a live price has a live last bar.
    """
    from src.connectors.nse_scraper import REAL_NSE_SOURCES
    out = []
    try:
        paths = sorted(Path(csv_dir).glob('*.csv'))
    except OSError:
        return out
    for path in paths:
        try:
            last = _last_row(path)
            if last and last.get('source') in REAL_NSE_SOURCES:
                out.append(path.stem.upper())
        except (OSError, ValueError, csv.Error) as e:
            logger.debug(f"NSE universe: cannot read {path.name}: {e}")
    register(out)
    return out


def _last_row(path: Path, block: int = 4096) -> Optional[dict]:
    """The file's last data row, read from its end rather than in full."""
    with open(path, 'rb') as f:
        header = f.readline().decode('utf-8').strip()
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(size - block, 0))
        lines = [l for l in f.read().decode('utf-8', 'replace').splitlines() if l.strip()]
    if not header or not lines or lines[-1].strip() == header:
        return None
    return next(csv.DictReader([header, lines[-1]]), None)


def market_symbols(csv_dir: Path, watched: Iterable[str] = ()) -> List[str]:
    """What Market Watch lists: every stock with real data, plus the ones
    the agent trades even before their first real bar."""
    return sorted(set(symbols_with_real_data(csv_dir)) | {s.upper() for s in watched})
