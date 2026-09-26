"""AIB-AXYS Market Pulse: the broker's daily market report, read exactly.

The research upload path was built for analyst notes (a BUY or SELL with a
target price) and asks an AI to find recommendations. The Market Pulse has
none: it is market data. Read that way, it produced a "HOLD" for every
company it named, with "prices" lifted from percentages beside the name.

What the report does carry is worth more than a rating, and it is laid out
as tables, so it is read with rules rather than an AI:

- the Market Scorecard (page 3): for every listed stock, the closing price,
  day and year-to-date change, volume, book value, market cap, trailing
  earnings per share and dividend per share. P/E, P/B, dividend yield and
  payout are recomputed from those rather than read, because the printed
  ratio columns are sometimes shifted or missing on a row;
- Treasury bill rates (91, 182, 364 days) and the interbank rate;
- the KES exchange rates;
- the day's company announcements (results, AGMs, board changes, listings).

What the agent does with it:

- fundamentals feed the dividend sleeve's ranking (no hand-kept file
  needed) and the AI's review of NSE trades;
- closing prices become real daily bars (source aib_market_pulse) for
  dates the NSE ticker feed did not record, so more stocks reach the
  history the screener requires;
- the T-bill rate updates the benchmark and the interest on idle cash;
- announcements go to the trade review and the short list's AI review.

Company names become tickers through nse_universe's names and the aliases
below; a name still unknown is matched against the prices and volumes the
ticker feed recorded that day, and learned only if exactly one stock fits.
"""
import csv
import io
import json
import logging
import re
import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SOURCE = 'aib_market_pulse'
_lock = threading.Lock()

# Printed names that do not simply match nse_universe's names.
_ALIASES = {
    'the co-operative bank of kenya': 'COOP', 'co-operative bank': 'COOP',
    'hfcb group': 'HFCK', 'hf group': 'HFCK', 'diamond trust bank kenya': 'DTK',
    'diamond trust bank': 'DTK', 'i&m group': 'IMH', 'bk group': 'BKG',
    'stanbic holdings': 'SBIC', 'standard chartered bank kenya': 'SCBK',
    'standard chartered bank': 'SCBK', 'absa bank kenya': 'ABSA', 'absa bank': 'ABSA',
    'equity group holdings': 'EQTY', 'equity group': 'EQTY', 'kcb group': 'KCB',
    'ncba group': 'NCBA', 'the limuru tea co': 'LIMT', 'kapchorua tea kenya': 'KAPC',
    'williamson tea kenya': 'WTK', 'eaagads': 'EGAD', 'kakuzi': 'KUKZ', 'sasini': 'SASN',
    'car & general (k)': 'CGEN', 'eveready east africa': 'EVRD', 'express kenya': 'XPRS',
    'homeboyz entertainment': 'HBER', 'kenya airways': 'KQ', 'longhorn publishers': 'LKL',
    'nairobi business ventures': 'NBV', 'nation media group': 'NMG', 'sameer africa': 'SMER',
    'standard group': 'SGL', 'tps eastern africa (serena)': 'TPSE', 'tps eastern africa': 'TPSE',
    'uchumi supermarket': 'UCHM', 'uchumi supermarkets': 'UCHM', 'wpp scangroup': 'SCAN',
    'bamburi cement': 'BAMB', 'crown paints kenya': 'CRWN', 'ea cables': 'CABL',
    'ea portland cement co': 'PORT', 'kengen co': 'KEGN', 'kengen': 'KEGN',
    'kenya power & lighting co': 'KPLC', 'kenya power and lighting company': 'KPLC',
    'kenya power': 'KPLC', 'totalenergies marketing kenya': 'TOTL', 'umeme': 'UMME',
    'britam holdings': 'BRIT', 'cic insurance group': 'CIC', 'jubilee holdings': 'JUB',
    'kenya re- insurance corporation': 'KNRE', 'kenya re-insurance corporation': 'KNRE',
    'liberty kenya holdings': 'LBTY', 'sanlam allianz holdings (kenya)': 'SLAM',
    'centum investment co': 'CTUM', 'centum investment company': 'CTUM',
    'home afrika': 'HAFR', 'kurwitu ventures': 'KURV', 'olympia capital holdings': 'OCH',
    'trans-century': 'TCL', 'nairobi securities exchange': 'NSE', 'boc kenya': 'BOC',
    'british american tobacco kenya': 'BAT', 'carbacid investments': 'CARB',
    'east african breweries': 'EABL', 'flame tree group holdings': 'FTGH',
    'africa mega agricop': 'AMAC', 'africa mega agricorp': 'AMAC', 'unga group': 'UNGA',
    'shri krishana overseas': 'SKL', 'safaricom': 'SCOM',
}
_DROP_WORDS = {'plc', 'ltd', 'limited', 'the'}


def normalise(name: str) -> str:
    """A company name reduced for matching: lower case, no dots or commas,
    no 'Plc'/'Ltd'/'Limited'/'The'."""
    s = re.sub(r'[.,]', '', (name or '').lower())
    return ' '.join(w for w in s.split() if w not in _DROP_WORDS)


def _alias_table(learned: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    from src.connectors.nse_universe import LISTED
    table = {normalise(n): t for t, (n, _s) in LISTED.items()}
    table.update({normalise(k): v for k, v in _ALIASES.items()})
    table.update({normalise(k): v for k, v in (learned or {}).items()})
    return table


# ------------------------------------------------------------------ text

def page_texts(path: str) -> List[str]:
    """Each page's text. pypdf keeps the scorecard's numbers whole;
    pdfplumber splits some ("1 .57"), which _clean repairs."""
    try:
        from pypdf import PdfReader
        return [p.extract_text() or '' for p in PdfReader(path).pages]
    except ImportError:
        from src.utils import pdf_parser
        return [p.get('text') or '' for p in pdf_parser.extract_all(path).get('pages', [])]


def announcement_links(path: str) -> List[str]:
    """The first page's hyperlinks, top to bottom: the "(here)" beside each
    announcement links to the filing. Empty when they cannot be read."""
    try:
        from pypdf import PdfReader
        page = PdfReader(path).get_page(0)
        found = []
        for annot in page.annotations or []:
            obj = annot.get_object()
            uri = (obj.get('/A') or {}).get('/URI')
            if uri and obj.get('/Rect'):
                found.append((-float(obj['/Rect'][1]), float(obj['/Rect'][0]), str(uri)))
        return [u for _y, _x, u in sorted(found)]
    except Exception:
        return []


def _clean(line: str) -> str:
    line = re.sub(r'(\d) \.(\d)', r'\1.\2', line)        # "1 .57" -> "1.57"
    line = re.sub(r'\( (\d)', r'(\1', line)               # "( 3.55)" -> "(3.55)"
    line = re.sub(r'\b(\d) (\d[\d,]*\.\dx)', r'\1\2', line)  # "1 2.5x" -> "12.5x"
    return re.sub(r'\s+', ' ', line).strip()


def is_market_pulse(texts: List[str]) -> bool:
    joined = '\n'.join(texts)
    return 'MARKET SCORECARD' in joined.upper() and 'PULSE' in joined.upper()


_MONTHS = {m: i + 1 for i, m in enumerate(
    ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}


def report_date(texts: List[str]) -> Optional[date]:
    joined = '\n'.join(texts)
    m = re.search(r'As of:\s*(\d{1,2})/([A-Za-z]{3})[a-z]*/(\d{2,4})', joined)
    if m:
        year = int(m.group(3)) + (2000 if len(m.group(3)) == 2 else 0)
        return date(year, _MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.search(r'Pulse\s*[–-]\s*(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3})[a-z]*\s+(\d{4})', joined)
    if m and m.group(2).lower() in _MONTHS:
        return date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
    return None


# ------------------------------------------------------------ scorecard

@dataclass
class ScoreRow:
    name: str
    sector: str
    price: float
    change_pct: Optional[float] = None
    ytd_pct: Optional[float] = None
    volume: Optional[int] = None
    book_value_per_share: Optional[float] = None
    market_cap_mn: Optional[float] = None
    eps: Optional[float] = None
    dps: Optional[float] = None
    roe_pct: Optional[float] = None
    roa_pct: Optional[float] = None
    symbol: Optional[str] = None

    @property
    def pe(self) -> Optional[float]:
        return round(self.price / self.eps, 2) if self.eps and self.eps > 0 else None

    @property
    def pb(self) -> Optional[float]:
        bv = self.book_value_per_share
        return round(self.price / bv, 2) if bv and bv > 0 else None

    @property
    def dividend_yield_pct(self) -> float:
        return round((self.dps or 0.0) / self.price * 100, 2) if self.price else 0.0

    @property
    def payout_ratio(self) -> Optional[float]:
        return round(self.dps / self.eps, 4) if self.dps and self.eps and self.eps > 0 else None

    @property
    def earnings_yield_pct(self) -> Optional[float]:
        return round(self.eps / self.price * 100, 2) if self.eps is not None and self.price else None


_PRICE = r'\d{1,3}(?:,\d{3})*\.\d{2}'
_ROW = re.compile(rf'^(?P<name>[A-Za-z&(][^\d]*?)\s+(?P<price>{_PRICE})\s+(?P<rest>.+)$')
_TOKEN = re.compile(r'[▲▼]\s*\(?[\d.,]+%\)?|\(?-?[\d.,]+x?\)?%?|(?<!\S)-(?!\S)')
_SECTION = re.compile(r'^([A-Z][A-Z&, ]+?)\s+(?:Current|YTD)\b')


def _num(tok: Optional[str]) -> Optional[float]:
    """'1,234.5' 1234.5, '(3.55)' -3.55, '▲ 0.5%' 0.5, '▼ (1.1%)' -1.1, '-' None."""
    if tok is None:
        return None
    t = tok.strip()
    if t in ('-', ''):
        return None
    neg = t.startswith('▼') or ('(' in t and ')' in t) or t.startswith('-')
    digits = re.sub(r'[^\d.]', '', t)
    if not digits or digits == '.':
        return None
    try:
        v = float(digits)
    except ValueError:
        return None
    return -v if neg else v


def parse_scorecard(texts: List[str]) -> List[ScoreRow]:
    """Every company row of the Market Scorecard."""
    rows: List[ScoreRow] = []
    started, sector = False, ''
    for text in texts:
        for raw in text.splitlines():
            line = _clean(raw)
            if 'MARKET SCORECARD' in line.upper():
                started = True
                continue
            if not started or not line:
                continue
            sec = _SECTION.match(line)
            if sec and not _ROW.match(line):
                sector = sec.group(1).strip().title()
                continue
            m = _ROW.match(line)
            if not m or m.group('name').strip().lower().startswith(('industry', 'market')):
                continue
            toks = _TOKEN.findall(m.group('rest'))
            if len(toks) < 8:
                continue
            pcts = [t for t in toks[8:] if t.endswith('%') or t.endswith('%)')]
            row = ScoreRow(
                name=m.group('name').strip(), sector=sector, price=_num(m.group('price')),
                change_pct=_num(toks[0]), ytd_pct=_num(toks[1]),
                volume=int(_num(toks[2]) or 0), book_value_per_share=_num(toks[3]),
                market_cap_mn=_num(toks[4]), eps=_num(toks[6]), dps=_num(toks[7]))
            if len(pcts) >= 4:
                row.roe_pct, row.roa_pct = _num(pcts[-2]), _num(pcts[-1])
            if row.price and toks[5].endswith('%'):
                rows.append(row)
    return rows


# ---------------------------------------------------- rates, FX, news

def parse_rates(texts: List[str]) -> Dict[str, float]:
    """Today's T-bill and interbank rates, as fractions (8.78% -> 0.0878)."""
    out: Dict[str, float] = {}
    labels = {'91-day rate': 'tbill_91', '182-day rate': 'tbill_182',
              '364-day rate': 'tbill_364', 'Interbank Rate': 'interbank'}
    for text in texts:
        for line in text.splitlines():
            for label, key in labels.items():
                m = re.search(rf'{re.escape(label)}\s+(\d+(?:\.\d+)?)%', line)
                if m and key not in out:
                    out[key] = round(float(m.group(1)) / 100, 6)
    return out


def parse_fx(texts: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for text in texts:
        for line in text.splitlines():
            m = re.match(r'\s*(US Dollar|Euro|Sterling Pound)\s+(\d+(?:\.\d+)?)\s', line)
            if m and m.group(1) not in out:
                out[m.group(1)] = float(m.group(2))
    return {'usd_kes': out.get('US Dollar'), 'eur_kes': out.get('Euro'),
            'gbp_kes': out.get('Sterling Pound')} if out else {}


_KINDS = [('profit_warning', r'profit warning'), ('results', r'financial results|results for'),
          ('dividend', r'dividend|book closure'), ('agm', r'\bagm\b|general meeting'),
          ('listing', r'intention[_ ]to[_ ]float|listing|\bipo\b|delist'),
          ('suspension', r'suspen'), ('board', r'appointment|resignation|director|ceo')]


def parse_announcements(texts: List[str], table: Dict[str, str]) -> List[Dict[str, Any]]:
    """The 'Capital News Update' bullets, each with its ticker and kind."""
    joined = '\n'.join(texts)
    start = joined.find('Capital News Update')
    if start < 0:
        return []
    items: List[str] = []
    for line in joined[start:].splitlines()[1:]:
        s = line.strip()
        if s.startswith('❖'):
            items.append(s.lstrip('❖ ').strip())
        elif items and s and not re.match(r'^(Top Foreign|Equities Indices|Source|Company)', s):
            items[-1] += ' ' + s
        elif items and s:
            break
    keys = sorted(table, key=len, reverse=True)
    out = []
    for item in items:
        text = re.sub(r'\s*\(here\)\s*$', '', item).strip()
        norm = normalise(text.replace('_', ' ').replace('-', ' - '))
        symbol = next((table[k] for k in keys if k and re.search(rf'(?<![a-z]){re.escape(k)}(?![a-z])', norm)), None)
        kind = next((k for k, pat in _KINDS if re.search(pat, text, re.IGNORECASE)), 'announcement')
        out.append({'symbol': symbol, 'kind': kind, 'text': text})
    return out


# ---------------------------------------------------------------- store

def _dir() -> Path:
    from src.utils.paths import DATA_DIR
    return DATA_DIR / 'market_pulse'


def _read(name: str, default):
    try:
        return json.loads((_dir() / name).read_text())
    except (OSError, ValueError):
        return default


def _write(name: str, data) -> None:
    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f'{name}.tmp'
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(d / name)


def reports() -> List[Dict[str, Any]]:
    """The reports read so far, oldest first."""
    return _read('reports.json', [])


def recent_announcements(n: int = 20) -> List[Dict[str, Any]]:
    """The last `n` announcements stored, newest first."""
    return _read('announcements.json', [])[-n:][::-1]


def learned_names() -> Dict[str, str]:
    return _read('name_map.json', {})


def learn_names(rows: List[ScoreRow], on: date, recorded: Dict[str, Dict[date, Tuple[float, Optional[float]]]],
                volumes: Dict[str, Dict[date, float]], taken: set) -> Dict[str, str]:
    """Printed name -> ticker for rows no name matched: the ticker whose
    recorded close that day is within 0.5% of the row's price and whose
    volume is within 2% of it, when exactly one ticker fits."""
    out: Dict[str, str] = {}
    for r in rows:
        fits = [sym for sym, days in recorded.items() if sym not in taken and on in days
                and abs(days[on][0] / r.price - 1) <= 0.005
                and r.volume and abs((volumes.get(sym, {}).get(on) or 0) / r.volume - 1) <= 0.02]
        if len(fits) == 1:
            out[r.name] = fits[0]
    return out


def _bar_volumes(symbols) -> Dict[str, Dict[date, float]]:
    from src.connectors.nse_scraper import load_csv
    out: Dict[str, Dict[date, float]] = {}
    for sym in symbols:
        days = {}
        for r in load_csv(sym):
            try:
                if r.get('source') == 'nse_ticker':
                    days[date.fromisoformat(str(r['date'])[:10])] = float(r.get('volume') or 0)
            except (ValueError, KeyError):
                continue
        out[sym] = days
    return out


def _merge_bars(rows: List[ScoreRow], on: date) -> Tuple[int, int]:
    """Closing prices as daily bars, never over a bar a live source stored."""
    from src.connectors.nse_scraper import REAL_NSE_SOURCES, DailyBar, load_csv, save_bars_csv
    written = skipped = 0
    for r in rows:
        if not r.symbol:
            continue
        if any(b.get('date') == on.isoformat() and b.get('source') in REAL_NSE_SOURCES
               for b in load_csv(r.symbol)):
            skipped += 1
            continue
        prev = r.price / (1 + r.change_pct / 100) if r.change_pct not in (None, -100) else r.price
        save_bars_csv(r.symbol, [DailyBar(
            date=on.isoformat(), symbol=r.symbol, open=round(prev, 2),
            high=round(max(prev, r.price), 2), low=round(min(prev, r.price), 2),
            close=round(r.price, 2), volume=int(r.volume or 0),
            change_pct=round(r.change_pct or 0.0, 2), source=SOURCE)])
        written += 1
    return written, skipped


def ingest(path: str, texts: Optional[List[str]] = None) -> Dict[str, Any]:
    """Read a Market Pulse and store what it says; returns a summary."""
    texts = texts if texts is not None else page_texts(path)
    on = report_date(texts)
    if on is None:
        raise ValueError('Market Pulse without a readable date')
    learned = learned_names()
    table = _alias_table(learned)
    rows = parse_scorecard(texts)
    for r in rows:
        r.symbol = table.get(normalise(r.name))
    unmapped = [r for r in rows if not r.symbol]
    if unmapped:
        from src.connectors import nse_pricelist as pl
        from src.connectors.nse_connector import NSE_CSV_DIR
        tickers = sorted(p.stem.upper() for p in Path(NSE_CSV_DIR).glob('*.csv'))
        found = learn_names(unmapped, on, pl.recorded_closes(tickers), _bar_volumes(tickers),
                            {r.symbol for r in rows if r.symbol})
        if found:
            learned.update(found)
            _write('name_map.json', learned)
            for r in unmapped:
                r.symbol = found.get(r.name)
            logger.info(f"Market Pulse: learned tickers for {found}")
    mapped = [r for r in rows if r.symbol]
    from src.connectors.nse_universe import register
    register(r.symbol for r in mapped)

    rates, fx = parse_rates(texts), parse_fx(texts)
    news = parse_announcements(texts, table)
    links = announcement_links(path)
    if news and len(links) == len(news):  # one link per announcement, in order
        for a, link in zip(news, links):
            a['link'] = link
    with _lock:
        funds = _read('fundamentals.json', {})
        for r in mapped:
            prev = funds.get(r.symbol) or {}
            if prev.get('as_of') and prev['as_of'] > on.isoformat():
                continue  # an older report never overwrites a newer one
            hist = [h for h in prev.get('eps_history', []) if h[0] != on.isoformat()]
            hist = sorted(hist + [[on.isoformat(), r.eps]])[-12:]
            funds[r.symbol] = {**asdict(r), 'as_of': on.isoformat(), 'pe': r.pe, 'pb': r.pb,
                               'dividend_yield_pct': r.dividend_yield_pct,
                               'payout_ratio': r.payout_ratio,
                               'earnings_yield_pct': r.earnings_yield_pct,
                               'eps_history': hist, 'source': SOURCE}
        _write('fundamentals.json', funds)
        if rates:
            hist = _read('rates.json', {}).get('history', {})
            hist[on.isoformat()] = {**rates, **fx}
            latest = max(hist)
            _write('rates.json', {'as_of': latest, **hist[latest],
                                  'history': dict(sorted(hist.items())[-60:])})
        stored = _read('announcements.json', [])
        seen = {(a['date'], a['text']) for a in stored}
        for a in news:
            if (on.isoformat(), a['text']) not in seen:
                stored.append({**a, 'date': on.isoformat()})
        cutoff = (on - timedelta(days=120)).isoformat()
        _write('announcements.json', sorted((a for a in stored if a['date'] >= cutoff),
                                            key=lambda a: a['date']))
    written, skipped = _merge_bars(mapped, on)
    summary = {
        'document_type': 'market_pulse', 'as_of': on.isoformat(),
        'stocks_read': len(rows), 'stocks_mapped': len(mapped),
        'unmapped': [r.name for r in rows if not r.symbol],
        'price_bars_added': written, 'price_bars_already_recorded': skipped,
        'rates': rates, 'fx': fx,
        'announcements': [a for a in news],
    }
    with _lock:
        reports = [r for r in _read('reports.json', []) if r.get('as_of') != on.isoformat()]
        reports.append({k: summary[k] for k in ('as_of', 'stocks_read', 'stocks_mapped', 'unmapped',
                                                'price_bars_added')})
        _write('reports.json', sorted(reports, key=lambda r: r['as_of'])[-60:])
    logger.info(f"Market Pulse {on}: {len(mapped)}/{len(rows)} stocks, {written} price bars, "
                f"rates {rates}, {len(news)} announcements")
    return summary


# ------------------------------------------------------------ consumers

def fundamentals(symbol: Optional[str] = None, max_age_days: int = 45,
                 today: Optional[date] = None) -> Dict[str, Any]:
    """Stored fundamentals no older than `max_age_days`: one stock's, or all."""
    today = today or datetime.now(timezone.utc).date()
    cutoff = (today - timedelta(days=max_age_days)).isoformat()
    funds = {s: f for s, f in _read('fundamentals.json', {}).items() if f.get('as_of', '') >= cutoff}
    return funds.get(symbol.upper(), {}) if symbol else funds


def eps_trend(entry: Dict[str, Any]) -> str:
    """'positive', 'negative' or 'flat': latest trailing EPS against the
    oldest stored, 5% either way."""
    hist = [e for _d, e in entry.get('eps_history', []) if e is not None]
    if len(hist) < 2 or not hist[0]:
        return 'flat'
    change = (hist[-1] - hist[0]) / abs(hist[0])
    return 'positive' if change > 0.05 else 'negative' if change < -0.05 else 'flat'


def latest_rates(max_age_days: int = 30, today: Optional[date] = None) -> Dict[str, Any]:
    r = _read('rates.json', {})
    today = today or datetime.now(timezone.utc).date()
    if not r.get('as_of') or r['as_of'] < (today - timedelta(days=max_age_days)).isoformat():
        return {}
    return {k: v for k, v in r.items() if k != 'history'}


def announcements(symbol: str, days: int = 30, today: Optional[date] = None) -> List[Dict[str, Any]]:
    today = today or datetime.now(timezone.utc).date()
    cutoff = (today - timedelta(days=days)).isoformat()
    return [a for a in _read('announcements.json', [])
            if a.get('symbol') == symbol.upper() and a.get('date', '') >= cutoff]


def research_context(symbol: str, today: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """What the broker's reports say about a stock, for the AI's trade
    review: the latest fundamentals and the last month's announcements."""
    from src.agent import daily_whispers
    f = fundamentals(symbol, today=today)
    news = announcements(symbol, today=today)
    rated = daily_whispers.latest(symbol, today=today)
    if not f and not news and not rated:
        return None
    parts = []
    if rated:
        parts.append(f"AIB-AXYS rating {rated['date']}: {rated['recommendation']} at "
                     f"{rated['current_price']}, target {rated['target_price']} "
                     f"({rated['upside_pct']:+.1f}%). {rated.get('rationale') or ''}".strip())
    if f:
        bits = [f"P/E {f['pe']}x" if f.get('pe') else 'no positive earnings',
                f"P/B {f['pb']}x" if f.get('pb') else None,
                f"dividend yield {f['dividend_yield_pct']}%",
                f"ROE {f['roe_pct']}%" if f.get('roe_pct') is not None else None,
                f"{f['ytd_pct']:+.1f}% this year" if f.get('ytd_pct') is not None else None]
        parts.append(f"AIB-AXYS Market Pulse {f['as_of']}: " + ', '.join(b for b in bits if b) + '.')
    if news:
        parts.append('Announcements: ' + '; '.join(f"{a['date']} {a['text']}" for a in news[-5:]) + '.')
    return {'recommendation': rated['recommendation'] if rated else None,
            'target_price': rated['target_price'] if rated else None,
            'rationale': ' '.join(parts)}
