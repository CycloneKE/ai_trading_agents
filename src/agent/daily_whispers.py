"""AIB-AXYS's "Daily Whispers": a recommendation sheet sent as a picture.

Each row names a company and gives its year-to-date move, current price,
target price, upside, the analysts' rationale and a rating (BUY, HOLD...).
It arrives as a JPG, so there is no text to read: an AI model that can see
images reads the table, and every row is then checked by rules before the
agent believes it:

- the printed name must match a listed NSE stock (the model's own guess at
  a ticker is never used), or, on a Global Equity sheet, the row must print
  a stock code and a US exchange (NYSE, NASDAQ...), which is then read from
  the picture as printed and checked like any other row;
- the rating must be one of the known ratings;
- target / price - 1 must give the printed upside, to within rounding, so a
  misread digit in any of the three numbers is caught;
- the price must be within 12% of the stock's last real close on record
  (the NSE's daily limit is 10%), when the agent has one; a US stock's,
  within 20% of its current price, when that can be fetched.

A row that fails any check is rejected with the reason, and shown to the
operator. Accepted rows are kept in data/market_pulse/recommendations.json;
the latest rating of the last 30 days goes to the AI's review of the stock's
trades (market_pulse.research_context), and each row goes through the same
follow-or-escalate rules as any analyst note.
"""
import base64
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

IMAGE_TYPES = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
               '.webp': 'image/webp'}
MAX_IMAGE_BYTES = 5 * 1024 * 1024          # the smallest provider limit
RATINGS = {'BUY', 'SELL', 'HOLD', 'ACCUMULATE', 'REDUCE'}
UPSIDE_TOLERANCE_PTS = 0.35                  # printed upside is rounded to 0.1
PRICE_TOLERANCE = 0.12
INTL_PRICE_TOLERANCE = 0.20
US_EXCHANGES = {'NYSE', 'NASDAQ', 'NYSE AMERICAN', 'NYSE ARCA', 'AMEX', 'CBOE', 'BATS'}
KEEP_DAYS = 30
STORE = 'recommendations.json'

_NUM = {'anyOf': [{'type': 'number'}, {'type': 'null'}]}
SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'required': ['report_title', 'report_date', 'rows'],
    'properties': {
        'report_title': {'type': 'string'},
        'report_date': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
        'rows': {'type': 'array', 'items': {
            'type': 'object',
            'additionalProperties': False,
            'required': ['security_name', 'ticker', 'exchange', 'ytd_pct', 'current_price', 'target_price',
                         'upside_pct', 'recommendation', 'rationale'],
            'properties': {
                'security_name': {'type': 'string'},
                'ticker': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
                'exchange': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
                'ytd_pct': _NUM, 'current_price': _NUM, 'target_price': _NUM,
                'upside_pct': _NUM,
                'recommendation': {'type': 'string'},
                'rationale': {'type': 'array', 'items': {'type': 'string'}},
            }}},
    },
}

SYSTEM_PROMPT = (
    "You transcribe a stockbroker's recommendation table from an image. Copy what is printed; "
    "never calculate, correct or guess. Numbers: digits only, no % sign, no currency; a "
    "falling value is negative. A value you cannot read with certainty is null. "
    "security_name exactly as printed (the name only, without the code or exchange). ticker: the stock "
    "code printed in brackets after the name (for example BA), or null when none is printed. exchange: "
    "the exchange printed after 'EX:' (for example NYSE), or null when none is printed. "
    "recommendation exactly as printed, upper case. "
    "rationale: each bullet point as printed. report_date as YYYY-MM-DD, or null if none is "
    "printed. If the image holds no such table, return an empty rows list.\n"
    "Reply with JSON only, in this shape: {\"report_title\": \"...\", \"report_date\": "
    "\"YYYY-MM-DD\" or null, \"rows\": [{\"security_name\": \"...\", \"ticker\": \"BA\" or null, \"exchange\": \"NYSE\" or null, \"ytd_pct\": 0.0, "
    "\"current_price\": 0.0, \"target_price\": 0.0, \"upside_pct\": 0.0, "
    "\"recommendation\": \"BUY\", \"rationale\": [\"...\"]}]}"
)
USER_PROMPT = "Transcribe the recommendation table in this image."


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_TYPES


def _float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def parse_date(raw: Optional[str], today: date) -> Optional[date]:
    try:
        d = date.fromisoformat(str(raw or '')[:10])
    except ValueError:
        return None
    return d if d <= today else None


_TICKER = re.compile(r'^[A-Z]{1,5}(?:[.\-][A-Z])?$')


def _international(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(symbol, exchange, problem) of a row that is not an NSE stock: the code
    and exchange as printed. The code and exchange may sit in the printed name
    ("Boeing Co (BA) EX: NYSE") when the reader did not split them out."""
    name = str(row.get('security_name') or '')
    ticker = str(row.get('ticker') or '').strip().upper()
    exchange = str(row.get('exchange') or '').strip().upper()
    if not ticker:
        m = re.search(r'\(([A-Za-z][A-Za-z.\-]{0,5})\)', name)
        ticker = m.group(1).upper() if m else ''
    if not exchange:
        m = re.search(r'EX:\s*([A-Za-z ]+)', name)
        exchange = m.group(1).strip().upper() if m else ''
    if not ticker or not exchange:
        return None, None, None                   # not a row of that kind: the NSE message applies
    if not _TICKER.match(ticker):
        return None, None, f'stock code "{ticker}" is not one the agent can use'
    if exchange not in US_EXCHANGES:
        return None, None, (f'listed on {exchange}: only US-listed stocks are read from global sheets so far')
    return ticker, exchange, None


def last_intl_price(symbol: str, on: date) -> Optional[float]:
    """A US stock's current price from the live feed, or None."""
    try:
        from src.utils.real_price_feed import price_feed
        price = price_feed.get_price(symbol)
        return float(price) if price and float(price) > 0 else None
    except Exception:
        return None


def verify(reading: Dict[str, Any], on: date,
           last_close: Callable[[str, date], Optional[float]],
           table: Optional[Dict[str, str]] = None,
           intl_price: Optional[Callable[[str, date], Optional[float]]] = None
           ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """(accepted, rejected) rows of one reading; see the module note."""
    from src.agent import market_pulse
    table = table if table is not None else market_pulse._alias_table(market_pulse.learned_names())
    accepted, rejected = [], []
    seen = set()
    for row in reading.get('rows') or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get('security_name') or '').strip()
        reject = lambda why: rejected.append({'name': name or '?', 'reason': why})
        symbol = table.get(market_pulse.normalise(name))
        market, exchange = 'kenyan', None
        if not symbol:
            symbol, exchange, problem = _international(row)
            if problem:
                reject(problem)
                continue
            if symbol:
                market = 'international'
        rating = str(row.get('recommendation') or '').strip().upper()
        price, target = _float(row.get('current_price')), _float(row.get('target_price'))
        upside = _float(row.get('upside_pct'))
        if not symbol:
            reject('name not matched to an NSE stock, and no US stock code and exchange are printed')
            continue
        if symbol in seen:
            reject(f'{symbol} appears twice')
            continue
        if rating not in RATINGS:
            reject(f'rating "{rating or "unreadable"}" not recognised')
            continue
        if not price or not target or price <= 0 or target <= 0 or upside is None:
            reject('price, target or upside unreadable')
            continue
        implied = (target / price - 1) * 100
        if abs(implied - upside) > UPSIDE_TOLERANCE_PTS:
            reject(f'numbers disagree: {target} / {price} is {implied:+.1f}%, printed {upside:+.1f}% '
                   f'(a digit was probably misread)')
            continue
        if market == 'international':
            close = (intl_price or last_intl_price)(symbol, on)
            if close and abs(price / close - 1) > INTL_PRICE_TOLERANCE:
                reject(f'price {price} is far from the current price {close:.2f}')
                continue
        else:
            close = last_close(symbol, on)
            if close and abs(price / close - 1) > PRICE_TOLERANCE:
                reject(f'price {price} is far from the recorded close {close}')
                continue
        seen.add(symbol)
        accepted.append({
            'symbol': symbol, 'name': re.sub(r'\s*\([A-Za-z.\-]+\)\s*(EX:.*)?$', '', name).strip() or name,
            'market': market, 'exchange': exchange, 'date': on.isoformat(), 'recommendation': rating,
            'current_price': price, 'target_price': target, 'upside_pct': round(implied, 1),
            'ytd_pct': _float(row.get('ytd_pct')),
            'rationale': ' '.join(str(b).strip() for b in row.get('rationale') or [] if str(b).strip()),
            'price_checked': bool(close),
        })
    return accepted, rejected


def last_real_close(symbol: str, on: date) -> Optional[float]:
    """The stock's last real close on or before `on`, from its history."""
    from src.agent.nse_screener import real_rows
    from src.connectors.nse_connector import NSE_CSV_DIR
    rows = [r for r in real_rows(symbol, NSE_CSV_DIR) if r['date'] <= on.isoformat()]
    return rows[-1]['close'] if rows else None


def remember(rows: List[Dict[str, Any]], source_file: str) -> int:
    """Keep accepted rows, one per stock per day. Returns the rows new here."""
    from src.agent import market_pulse
    with market_pulse._lock:
        store = market_pulse._read(STORE, {})
        added = 0
        for r in rows:
            history = [h for h in store.get(r['symbol'], []) if h.get('date') != r['date']]
            added += len(history) == len(store.get(r['symbol'], []))
            store[r['symbol']] = sorted(history + [{**r, 'source_file': source_file}],
                                        key=lambda h: h['date'])[-20:]
        market_pulse._write(STORE, store)
    return added


def latest(symbol: str, days: int = KEEP_DAYS, today: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """The newest rating of the last `days` days, or None."""
    from src.agent import market_pulse
    today = today or date.today()
    history = market_pulse._read(STORE, {}).get((symbol or '').upper(), [])
    recent = [h for h in history if h.get('date', '') >= (today - timedelta(days=days)).isoformat()]
    return recent[-1] if recent else None


def recent_ratings(days: int = KEEP_DAYS, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """The newest rating of each stock from the last `days` days, newest
    first (largest upside first within a day)."""
    from src.agent import market_pulse
    today = today or date.today()
    cutoff = (today - timedelta(days=days)).isoformat()
    rows = []
    for symbol, history in market_pulse._read(STORE, {}).items():
        recent = [h for h in history if h.get('date', '') >= cutoff]
        if recent:
            rows.append({**recent[-1], 'symbol': symbol})
    return sorted(rows, key=lambda r: (r.get('date', ''), r.get('upside_pct') or 0), reverse=True)


def read(path: str, llm, today: Optional[date] = None,
         last_close: Optional[Callable[[str, date], Optional[float]]] = None,
         intl_price: Optional[Callable[[str, date], Optional[float]]] = None) -> Dict[str, Any]:
    """Read one recommendation sheet image. Raises ValueError with a message
    for the operator when it cannot be read at all."""
    today = today or date.today()
    media_type = IMAGE_TYPES.get(Path(path).suffix.lower())
    if not media_type:
        raise ValueError('not an image type the reader takes (JPG, PNG or WebP)')
    data = Path(path).read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f'image is {len(data) / 1e6:.1f} MB; the limit is 5 MB')
    if llm is None or not hasattr(llm, 'read_image_json') or not getattr(llm, 'enabled', False):
        raise ValueError('no AI model is set up to read images (needs GROQ_API_KEY, GEMINI_API_KEY '
                         'or ANTHROPIC_API_KEY)')
    reading = llm.read_image_json(SYSTEM_PROMPT, USER_PROMPT, base64.b64encode(data).decode(),
                                  media_type, SCHEMA)
    if not isinstance(reading, dict):
        why = '; '.join(what if who == 'setup' else f"{who}: {what}"
                        for who, what in getattr(llm, 'last_image_errors', None) or [])
        raise ValueError(f"no AI model could read the image ({why})" if why
                         else 'no AI model could read the image; try again later')
    on = parse_date(reading.get('report_date'), today)
    accepted, rejected = verify(reading, on or today, last_close or last_real_close, intl_price=intl_price)
    if not accepted and not rejected:
        raise ValueError('no recommendation table was found in the image')
    return {'document_type': 'recommendation_sheet', 'title': str(reading.get('report_title') or ''),
            'as_of': (on or today).isoformat(), 'date_read': on is not None,
            'accepted': accepted, 'rejected': rejected}
