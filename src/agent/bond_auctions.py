"""AIB-AXYS's "Primary Bond Auction Note": a broker's note on one Kenyan
treasury bond auction.

It has no stock ratings, so the analyst-note reader found nothing in it. It
is read here instead, exactly and with no AI: the key auction highlights
(the papers on offer, their coupons, tenors and maturities, the size, the
sale period, the minimum bid and the tax), AXYS's recommended bidding range
for each paper, and the few market figures the note states in words
(inflation, the interbank rate, how far borrowing is ahead of target, the
last auction's average accepted rate).

Every figure is checked for plausibility and dropped, with a warning, when it
is not; a figure the note does not give stays empty rather than guessed. The
note is kept for the Research page (the sale period tells the operator
whether the auction is still open). It is information for the operator, who
bids through the broker: the agent trades no bonds, so nothing here goes to
the approval queue or the watchlist.
"""
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

STORE = 'bond_auctions.json'
KEEP = 24
PAPER = r'(?:FXD|IFB|SDB|TB)\s?\d/\d{4}/\d{1,3}'
DASH = r'[–—\-]'
_MONTHS = {m: i + 1 for i, m in enumerate(
    ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}


def is_bond_auction(texts: List[str]) -> bool:
    joined = '\n'.join(texts).upper()
    return 'BOND AUCTION' in joined and ('KEY AUCTION HIGHLIGHTS' in joined or 'BIDDING RANGE' in joined)


def _code(raw: str) -> str:
    return re.sub(r'\s+', '', raw).upper()


def _num(raw: Any) -> Optional[float]:
    try:
        return float(str(raw).replace(',', ''))
    except (TypeError, ValueError):
        return None


def _date(day: str, month: str, year: str) -> Optional[str]:
    m = _MONTHS.get(month[:3].lower())
    try:
        return date(int(year), m, int(day)).isoformat() if m else None
    except ValueError:
        return None


def _section(text: str, start: str, *ends: str) -> str:
    """The text after the label `start`, up to the first of the labels `ends`."""
    m = re.search(start, text, re.IGNORECASE)
    if not m:
        return ''
    rest = text[m.end():]
    cut = len(rest)
    for end in ends:
        e = re.search(end, rest, re.IGNORECASE)
        if e:
            cut = min(cut, e.start())
    return rest[:cut]


def _by_paper(block: str, value: str) -> Dict[str, tuple]:
    """{paper code: the groups of `value`} for each `PAPER - value` in the block."""
    return {_code(m.group(1)): m.groups()[1:]
            for m in re.finditer(rf'({PAPER})\s*{DASH}\s*{value}', block)}


def parse(texts: List[str]) -> Dict[str, Any]:
    """The note's figures, and `warnings` for anything missing or implausible."""
    text = '\n'.join(texts)
    warnings: List[str] = []
    title_m = re.search(r'AXYS[^\n]*Primary Bond Auction Note[^\n]*', text)
    table = _section(text, r'Table 1:\s*Key Auction Highlights', r'Source:')
    out: Dict[str, Any] = {
        'title': re.sub(r'\s+', ' ', title_m.group(0)).strip() if title_m else 'Primary Bond Auction Note',
        'issuer': None, 'total_kes_bn': None, 'purpose': None, 'sale_from': None, 'sale_to': None,
        'min_bid_kes': None, 'tax_pct': None, 'papers': [], 'market': {}, 'warnings': warnings,
    }
    if not table:
        warnings.append('the key auction highlights table was not found')
        return out

    if m := re.search(r'Issuer:\s*([^\n]+)', table):
        out['issuer'] = m.group(1).strip()
    if m := re.search(r'Total Amount:\s*KES\s*([\d.,]+)\s*(billion|bn|b)\b', table, re.IGNORECASE):
        out['total_kes_bn'] = _num(m.group(1))
    if m := re.search(r'Purpose:\s*([^\n]+)', table):
        out['purpose'] = m.group(1).strip()
    if m := re.search(r'Period of sale:\s*(\d{1,2})\w*\s+([A-Za-z]{3})\w*\s+(\d{4})\s+to\s+'
                      r'(\d{1,2})\w*\s+([A-Za-z]{3})\w*\s+(\d{4})', table):
        out['sale_from'] = _date(m.group(1), m.group(2), m.group(3))
        out['sale_to'] = _date(m.group(4), m.group(5), m.group(6))
    if m := re.search(r'Minimum\s*Amount:\s*KES\s*([\d,]+(?:\.\d+)?)', table):
        out['min_bid_kes'] = _num(m.group(1))
    if m := re.search(r'Taxation:\s*([\d.]+)\s*%', table):
        out['tax_pct'] = _num(m.group(1))

    tenors = _by_paper(_section(table, r'Tenor:', r'Coupon Rate:'),
                       r'\(?\s*([\d.]+)\s*Yrs\)?\s*(?:' + DASH + r'\s*)?(Re-?opened|New\s*issue(?:ance)?)?')
    coupons = _by_paper(_section(table, r'Coupon Rate:', r'Price Quote:', r'Period of sale'), r'([\d.]+)\s*%')
    maturities = _by_paper(_section(table, r'Maturity Dates?:', r'Non-competitive'),
                           r'(\d{1,2})\w*\s*-?\s*([A-Za-z]{3})\w*\s*-?\s*(\d{4})')
    ranges = _by_paper(_section(table, r'Recommendation:', r'Source:'), r'([\d.]+)\s*-\s*([\d.]+)\s*%')
    codes = list(dict.fromkeys([*tenors, *coupons, *maturities, *ranges]))
    for code in codes:
        paper: Dict[str, Any] = {'paper': code, 'tenor_years': None, 'reopened': None, 'coupon_pct': None,
                                 'maturity': None, 'bid_low_pct': None, 'bid_high_pct': None}
        if code in tenors:
            paper['tenor_years'] = _num(tenors[code][0])
            kind = (tenors[code][1] or '').lower()
            paper['reopened'] = None if not kind else kind.startswith('re')
        if code in coupons:
            paper['coupon_pct'] = _num(coupons[code][0])
        if code in maturities:
            paper['maturity'] = _date(*maturities[code])
        if code in ranges:
            paper['bid_low_pct'], paper['bid_high_pct'] = _num(ranges[code][0]), _num(ranges[code][1])
        out['papers'].append(paper)

    out['market'] = _market(text)
    _check(out)
    return out


def _market(text: str) -> Dict[str, Any]:
    flat = re.sub(r'\s+', ' ', text)
    found: Dict[str, Any] = {}
    if m := re.search(r'headline inflation[^.]{0,60}?([\d.]+)%\s*y/y in ([A-Za-z]+)\s+(\d{4})', flat, re.IGNORECASE):
        found['inflation_pct'] = _num(m.group(1))
        found['inflation_month'] = f"{m.group(2)} {m.group(3)}"
    if m := re.search(r'KESONIA\)[^%]{0,120}?at\s*([\d.]+)%', flat, re.IGNORECASE):
        found['kesonia_pct'] = _num(m.group(1))
    if m := re.search(r'([\d.]+)%\s*performance rate', flat, re.IGNORECASE):
        found['borrowing_vs_target_pct'] = _num(m.group(1))
    if m := re.search(r'Weighted Average Rate of Accepted Bids was\s*([\d.]+)\s*%', flat, re.IGNORECASE):
        found['last_accepted_rate_pct'] = _num(m.group(1))
    return found


def _check(out: Dict[str, Any]) -> None:
    """Drop what cannot be right, and say so."""
    w = out['warnings']
    if out['sale_from'] and out['sale_to'] and out['sale_to'] < out['sale_from']:
        w.append('the sale period ends before it starts; both dates dropped')
        out['sale_from'] = out['sale_to'] = None
    if out['tax_pct'] is not None and not 0 <= out['tax_pct'] <= 40:
        w.append(f"tax {out['tax_pct']}% is implausible; dropped")
        out['tax_pct'] = None
    if out['total_kes_bn'] is not None and not 0 < out['total_kes_bn'] <= 1000:
        w.append(f"amount KES {out['total_kes_bn']} bn is implausible; dropped")
        out['total_kes_bn'] = None
    for p in out['papers']:
        for key, lo, hi in (('coupon_pct', 0, 30), ('tenor_years', 0, 50), ('bid_low_pct', 0, 40), ('bid_high_pct', 0, 40)):
            if p[key] is not None and not lo < p[key] <= hi:
                w.append(f"{p['paper']}: {key} {p[key]} is implausible; dropped")
                p[key] = None
        if p['bid_low_pct'] is not None and p['bid_high_pct'] is not None and p['bid_low_pct'] > p['bid_high_pct']:
            w.append(f"{p['paper']}: the bidding range is upside down; dropped")
            p['bid_low_pct'] = p['bid_high_pct'] = None
    for key, lo, hi in (('inflation_pct', -5, 100), ('kesonia_pct', 0, 40), ('last_accepted_rate_pct', 0, 40),
                        ('borrowing_vs_target_pct', 0, 2000)):
        v = out['market'].get(key)
        if v is not None and not lo <= v <= hi:
            w.append(f"{key} {v} is implausible; dropped")
            del out['market'][key]
    if not out['papers']:
        w.append('no bond papers were found')
    if out['total_kes_bn'] is None:
        w.append('the total amount was not found')
    if not out['sale_from'] or not out['sale_to']:
        w.append('the sale period was not found')


def usable(note: Dict[str, Any]) -> bool:
    """Enough to be worth keeping: at least one paper and the sale period."""
    return bool(note.get('papers')) and bool(note.get('sale_from')) and bool(note.get('sale_to'))


def status(note: Dict[str, Any], today: Optional[date] = None) -> str:
    """'upcoming', 'open', 'closed' or 'unknown' by the sale period."""
    today = today or date.today()
    try:
        start, end = date.fromisoformat(note['sale_from']), date.fromisoformat(note['sale_to'])
    except (KeyError, TypeError, ValueError):
        return 'unknown'
    return 'upcoming' if today < start else ('closed' if today > end else 'open')


def ingest(path: str, texts: List[str], today: Optional[date] = None) -> Dict[str, Any]:
    """Read and keep one note (the same note again replaces it). Returns the
    note with `stored` saying whether it was kept."""
    from src.agent import market_pulse
    note = parse(texts)
    note['source_file'] = path.replace('\\', '/').rsplit('/', 1)[-1]
    note['read_at'] = datetime.utcnow().isoformat()
    note['stored'] = usable(note)
    if note['stored']:
        with market_pulse._lock:
            kept = [n for n in market_pulse._read(STORE, []) if n.get('title') != note['title']
                    or n.get('sale_from') != note['sale_from']]
            market_pulse._write(STORE, sorted(kept + [note], key=lambda n: n.get('sale_from') or '')[-KEEP:])
    return note


def recent(limit: int = 6, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """The newest notes first, each with its `status`."""
    from src.agent import market_pulse
    notes = sorted(market_pulse._read(STORE, []), key=lambda n: n.get('sale_from') or '', reverse=True)
    return [{**n, 'status': status(n, today)} for n in notes[:limit]]
