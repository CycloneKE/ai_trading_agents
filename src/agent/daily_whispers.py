"""AIB-AXYS's "Daily Whispers": a recommendation sheet sent as a picture.

Each row names a company and gives its year-to-date move, current price,
target price, upside, the analysts' rationale and a rating (BUY, HOLD...).
It arrives as a JPG, so there is no text to read: an AI model that can see
images reads the table, and every row is then checked by rules before the
agent believes it:

- the printed name must match a listed NSE stock (the model's own guess at
  a ticker is never used);
- the rating must be one of the known ratings;
- target / price - 1 must give the printed upside, to within rounding, so a
  misread digit in any of the three numbers is caught;
- the price must be within 12% of the stock's last real close on record
  (the NSE's daily limit is 10%), when the agent has one.

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
            'required': ['security_name', 'ytd_pct', 'current_price', 'target_price',
                         'upside_pct', 'recommendation', 'rationale'],
            'properties': {
                'security_name': {'type': 'string'},
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
    "security_name exactly as printed. recommendation exactly as printed, upper case. "
    "rationale: each bullet point as printed. report_date as YYYY-MM-DD, or null if none is "
    "printed. If the image holds no such table, return an empty rows list.\n"
    "Reply with JSON only, in this shape: {\"report_title\": \"...\", \"report_date\": "
    "\"YYYY-MM-DD\" or null, \"rows\": [{\"security_name\": \"...\", \"ytd_pct\": 0.0, "
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


def verify(reading: Dict[str, Any], on: date,
           last_close: Callable[[str, date], Optional[float]],
           table: Optional[Dict[str, str]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
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
        rating = str(row.get('recommendation') or '').strip().upper()
        price, target = _float(row.get('current_price')), _float(row.get('target_price'))
        upside = _float(row.get('upside_pct'))
        if not symbol:
            reject('name not matched to an NSE stock')
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
        close = last_close(symbol, on)
        if close and abs(price / close - 1) > PRICE_TOLERANCE:
            reject(f'price {price} is far from the recorded close {close}')
            continue
        seen.add(symbol)
        accepted.append({
            'symbol': symbol, 'name': name, 'date': on.isoformat(), 'recommendation': rating,
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


def read(path: str, llm, today: Optional[date] = None,
         last_close: Optional[Callable[[str, date], Optional[float]]] = None) -> Dict[str, Any]:
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
        raise ValueError('no AI model is set up to read images (needs GEMINI_API_KEY or ANTHROPIC_API_KEY)')
    reading = llm.read_image_json(SYSTEM_PROMPT, USER_PROMPT, base64.b64encode(data).decode(),
                                  media_type, SCHEMA)
    if not isinstance(reading, dict):
        raise ValueError('no AI model could read the image; try again later')
    on = parse_date(reading.get('report_date'), today)
    accepted, rejected = verify(reading, on or today, last_close or last_real_close)
    if not accepted and not rejected:
        raise ValueError('no recommendation table was found in the image')
    return {'document_type': 'recommendation_sheet', 'title': str(reading.get('report_title') or ''),
            'as_of': (on or today).isoformat(), 'date_read': on is not None,
            'accepted': accepted, 'rejected': rejected}
