"""End-of-day NSE prices from a licensed data feed, delivered as files.

Why this exists: every NSE price the agent has comes from web pages and
uploaded reports (the NSE ticker strip, afx.kwayisi.org, the daily price
list, AIB-AXYS's Market Pulse). Each works, but a site redesign can leave a
gap, and none of them is a licensed source. The NSE's own Data Services and
data vendors sell end-of-day prices, usually as a daily file (by email, SFTP
or a download link) rather than a live API.

This reads such files from a folder, so a feed can be switched on without
code: copy each day's file into the folder (by hand or with the vendor's own
download tool) and the agent imports it at the start of its next NSE cycle.
Bars are stored as source `licensed_eod`, which outranks every scraped
source for the days it covers (nse_scraper.SOURCE_RANK); the scrapers keep
running beside it as a cross-check, and a close that differs from theirs by
more than 3% is reported in the scraper's status.

Configuration (config.json, `nse_licensed_feed`):

    enabled      false until a feed is bought
    folder       where the files arrive (default data/nse_feed)
    columns      the vendor's column names for symbol, date, open, high,
                 low, close and volume (close, symbol and date are needed)
    date_format  how the vendor writes dates, e.g. "%d/%m/%Y"
    delimiter    "," by default

Each file is imported once (the folder's .imported.json lists them) and a
file that changes is imported again. A row that cannot be read is skipped
and counted, never guessed at. Corporate actions are not imported: a bonus
issue or split changes share counts, so the vendor's notice is added to
config/nse_corporate_actions.json by hand after checking it.
"""
import csv
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SOURCE = 'licensed_eod'

DEFAULT_COLUMNS = {'symbol': 'Symbol', 'date': 'Date', 'open': 'Open', 'high': 'High',
                   'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
DEFAULTS: Dict[str, Any] = {'enabled': False, 'folder': None, 'columns': DEFAULT_COLUMNS,
                            'date_format': '%Y-%m-%d', 'delimiter': ','}


def settings(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = {k: v for k, v in (cfg or {}).items() if not str(k).startswith('_')}
    out = {**DEFAULTS, **cfg}
    out['columns'] = {**DEFAULT_COLUMNS, **(cfg.get('columns') or {})}
    if not out['folder']:
        from src.utils.paths import DATA_DIR
        out['folder'] = str(DATA_DIR / 'nse_feed')
    return out


def _number(text: Any) -> Optional[float]:
    """'1,234.50' -> 1234.5; blank, '-' or text -> None."""
    try:
        v = float(str(text).replace(',', '').strip())
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _plausible(value: Optional[float], close: float) -> float:
    """An open, high or low within half of the close either way, else the
    close: a misplaced decimal point (41,00 read as 4100) is not believed."""
    return value if value and 0.5 * close <= value <= 1.5 * close else close


def parse_file(path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The bars in one vendor file, and how many rows could not be read."""
    from src.connectors.nse_scraper import DailyBar
    from src.connectors.nse_universe import valid_ticker
    cols = cfg['columns']
    bars: List[Any] = []
    skipped = 0
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, delimiter=cfg.get('delimiter') or ','):
            symbol = str(row.get(cols['symbol']) or '').strip().upper()
            close = _number(row.get(cols['close']))
            try:
                day = datetime.strptime(str(row.get(cols['date']) or '').strip(), cfg['date_format']).date()
            except ValueError:
                day = None
            if not symbol or not valid_ticker(symbol) or day is None or not close or close <= 0:
                skipped += 1
                continue
            o = _plausible(_number(row.get(cols.get('open'))), close)
            h = _plausible(_number(row.get(cols.get('high'))), close)
            low = _plausible(_number(row.get(cols.get('low'))), close)
            vol = _number(row.get(cols.get('volume'))) or 0
            bars.append(DailyBar(date=day.isoformat(), symbol=symbol, open=o, high=max(h, o, close),
                                 low=min(low, o, close), close=close, volume=int(vol),
                                 change_pct=round((close / o - 1) * 100, 2) if o else 0.0, source=SOURCE))
    return {'bars': bars, 'skipped': skipped}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def import_folder(cfg: Optional[Dict[str, Any]], save=None) -> Dict[str, Any]:
    """Import every new or changed file in the folder. `save(symbol, bars)`
    stores bars (nse_scraper.save_bars_csv by default) and returns its result."""
    cfg = settings(cfg)
    folder = Path(cfg['folder'])
    summary: Dict[str, Any] = {'files': 0, 'bars': 0, 'kept_scraped': 0, 'skipped_rows': 0,
                               'disagreements': [], 'at': datetime.utcnow().isoformat(timespec='seconds')}
    if not folder.is_dir():
        return summary
    if save is None:
        from src.connectors.nse_scraper import save_bars_csv as save
    ledger_path = folder / '.imported.json'
    try:
        done = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    except (OSError, ValueError):
        done = {}
    for path in sorted(folder.glob('*.csv')):
        digest = _digest(path)
        if done.get(path.name) == digest:
            continue
        try:
            parsed = parse_file(path, cfg)
        except (OSError, csv.Error, UnicodeDecodeError) as e:
            logger.warning(f"NSE licensed feed: {path.name} could not be read: {e}")
            continue
        by_symbol: Dict[str, List[Any]] = {}
        for bar in parsed['bars']:
            by_symbol.setdefault(bar.symbol, []).append(bar)
        for symbol, bars in by_symbol.items():
            result = save(symbol, bars) or {}
            summary['bars'] += len(result.get('written', bars))
            summary['kept_scraped'] += len(result.get('kept', []))
            summary['disagreements'] += list(result.get('disagreements', []))
        summary['files'] += 1
        summary['skipped_rows'] += parsed['skipped']
        if parsed['skipped']:
            logger.warning(f"NSE licensed feed: {parsed['skipped']} row(s) of {path.name} could not be read "
                           f"(check nse_licensed_feed.columns and date_format)")
        done[path.name] = digest
    if summary['files']:
        ledger_path.write_text(json.dumps(done, indent=2))
        logger.info(f"NSE licensed feed: {summary['bars']} bars from {summary['files']} file(s)")
    return summary
