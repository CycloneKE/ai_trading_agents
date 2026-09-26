"""Real NSE price history from the exchange's daily price lists.

The ticker feed (nse_scraper.scrape_nse_ticker) gives today's prices only, so
real history accumulates one bar per trading day and the strategies, which
need 50, would stay silent for ten weeks. The NSE publishes a price list for
every session at wp-content/uploads/DD-MON-YY.pdf, and older lists stay
online. They are scanned images with no text layer, so reading them takes
OCR, and OCR misreads: in the first list read on the server, KCB's price came
out as "006" and three of NCBA's cells were unreadable.

So no single reading is trusted on its own. A list prints each stock's VWAP
(the NSE's official daily price) and the previous session's price, which
means every session's price is printed twice: as VWAP in its own list and as
PREVIOUS in the next one. A price is accepted when both readings agree, or
when only one is readable and it lies within that session's own high and
low. Anything else is rejected and reported, never guessed.

This module is pure: it works on OCR boxes, not images, so it is tested
without the OCR engine. scripts/backfill_nse_pricelists.py does the reading.
"""
import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

from src.connectors.nse_scraper import (
    MAX_SESSION_MOVE, REAL_NSE_SOURCES, DailyBar, load_csv, save_bars_csv)

SOURCE = "nse_pricelist"
PRICELIST_URL = "https://www.nse.co.ke/wp-content/uploads/{name}.pdf"
_MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()

# Rows are matched by ISIN, which the OCR read correctly for every watched
# stock, rather than by company name, which it garbles ("Co-openative").
# The nine checked by hand; the rest are learned (nse_isin.py, learn_isins).
from src.connectors.nse_isin import VERIFIED as ISIN_TO_SYMBOL  # noqa: E402

COLUMNS = ("high", "low", "vwap", "previous", "volume")
_HEADINGS = {"HIGH": "high", "LOW": "low", "VWAP": "vwap", "PREVIOUS": "previous",
             "VOLUME": "volume"}
# Where the headings sat, as a fraction of page width, on the list read on
# the server; used only for a page that carries no heading row.
DEFAULT_ANCHORS = {"high": 877 / 1700, "low": 981 / 1700, "vwap": 1081 / 1700,
                   "previous": 1195 / 1700, "volume": 1333 / 1700}

# A price must look exactly like one: "36.20", "5450.00", "1,135.00". The
# garbles seen so far ("006", "0968", "00°06", "00000") all fail this.
_PRICE_RE = re.compile(r"^(?:0|[1-9]\d{0,2}(?:,?\d{3})*)\.\d{2}$")
_VOLUME_RE = re.compile(r"^(?:0|[1-9]\d{0,2}(?:,?\d{3})*)$")
_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")


class Box(NamedTuple):
    x: float       # left edge
    y: float       # vertical centre
    height: float
    text: str


Reading = Dict[str, Optional[float]]  # column -> value, None if unreadable


def pricelist_name(d: date) -> str:
    """The list's file name: zero-padded day, e.g. 01-JUL-26 (1-JUL-26 is 404)."""
    return f"{d.day:02d}-{_MONTHS[d.month - 1]}-{d:%y}"


def boxes_from_ocr(result) -> List[Box]:
    """RapidOCR's [[4 corner points], text, score] items as Boxes."""
    boxes = []
    for points, text, _score in result or []:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        boxes.append(Box(min(xs), (min(ys) + max(ys)) / 2, max(ys) - min(ys), str(text)))
    return boxes


def group_lines(boxes: Iterable[Box]) -> List[List[Box]]:
    """Boxes on the same printed line, top to bottom, each line left to right."""
    boxes = sorted(boxes, key=lambda b: b.y)
    if not boxes:
        return []
    heights = sorted(b.height for b in boxes)
    tol = 0.6 * heights[len(heights) // 2]
    lines: List[List[Box]] = []
    for b in boxes:
        if lines and abs(lines[-1][0].y - b.y) <= tol:
            lines[-1].append(b)
        else:
            lines.append([b])
    return [sorted(line, key=lambda b: b.x) for line in lines]


def _norm(text: str) -> str:
    return text.replace(" ", "").upper()


def find_anchors(lines: List[List[Box]]) -> Optional[Dict[str, float]]:
    """The x of each price column's heading, if the page carries the headings.

    HIGH, LOW, VWAP and PREVIOUS share one heading line; VOLUME sits on the
    line below it. "52WKHIGH" is a different column and does not match.
    """
    for i, line in enumerate(lines):
        found = {_HEADINGS[_norm(b.text)]: b.x for b in line if _norm(b.text) in _HEADINGS}
        if all(k in found for k in ("high", "low", "vwap", "previous")):
            for nearby in lines[i:i + 3]:
                for b in nearby:
                    if _norm(b.text) == "VOLUME" and b.x > found["previous"]:
                        found["volume"] = b.x
            return found
    return None


def _column(x: float, anchors: Dict[str, float], slack: float) -> Optional[str]:
    """The price column a cell starting at x belongs to, None left of HIGH.

    Values are printed right-aligned under their headings, so a value starts
    at or after its heading's left edge and before the next heading's.
    """
    col = None
    for name, ax in sorted(anchors.items(), key=lambda kv: kv[1]):
        if x >= ax - slack:
            col = name
    return col


def _parse(text: str, column: str) -> Optional[float]:
    t = text.strip()
    pattern = _VOLUME_RE if column == "volume" else _PRICE_RE
    return float(t.replace(",", "")) if pattern.match(t) else None


def read_isin_rows(boxes: List[Box], width: float,
                   anchors: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, Reading], Optional[Dict[str, float]]]:
    """Readings for every row on the page that carries exactly one ISIN,
    keyed by the ISIN, known or not. What learn_isins works from."""
    return _read_rows(boxes, width, anchors, lambda isin: isin)


def read_page(boxes: List[Box], width: float,
              anchors: Optional[Dict[str, float]] = None,
              isin_map: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Reading], Optional[Dict[str, float]]]:
    """Readings per mapped symbol on one page, and the anchors used.

    Pass the previous page's anchors for a page without a heading row. A row
    counts only if one cell on it is exactly a mapped ISIN (`isin_map`, the
    hand-checked nine by default); the dividend notes quote ISINs inside
    longer text, so they never match. A column with no cell, two cells, or
    a cell that is not a well-formed number reads as None: unreadable, never
    guessed.
    """
    mapping = isin_map if isin_map is not None else ISIN_TO_SYMBOL
    return _read_rows(boxes, width, anchors, mapping.get)


def _read_rows(boxes, width, anchors, key_of):
    lines = group_lines(boxes)
    anchors = find_anchors(lines) or anchors or {k: v * width for k, v in DEFAULT_ANCHORS.items()}
    slack = 0.005 * width
    out: Dict[str, Reading] = {}
    seen: Dict[str, int] = {}
    for line in lines:
        isins = [b for b in line if _ISIN_RE.match(_norm(b.text))]
        if len(isins) != 1:
            continue
        sym = key_of(_norm(isins[0].text))
        if not sym:
            continue
        seen[sym] = seen.get(sym, 0) + 1
        cells: Dict[str, List[str]] = {c: [] for c in COLUMNS}
        for b in line:
            if b.x > isins[0].x:
                col = _column(b.x, anchors, slack)
                if col:
                    cells[col].append(b.text)
        out[sym] = {c: _parse(v[0], c) if len(v) == 1 else None for c, v in cells.items()}
    for sym, n in seen.items():
        if n > 1:  # two rows claim one stock: trust neither
            out[sym] = {c: None for c in COLUMNS}
    return out, anchors


def learn_isins(readings: Dict[date, Dict[str, Reading]],
                recorded: Dict[str, Dict[date, Tuple[float, Optional[float]]]],
                known: Dict[str, str], min_sessions: int = 3,
                tolerance: float = 0.01) -> Dict[str, str]:
    """ISIN -> ticker for ISINs the lists print but the map does not know.

    `readings` are the lists' rows keyed by ISIN (read_isin_rows), by
    session. `recorded` is each ticker's close and previous close as the NSE
    ticker feed stored them, by session: an independent source, so an OCR
    misreading cannot confirm itself. On each session where an unknown ISIN
    has a readable VWAP, the candidates are the tickers whose recorded close
    is within `tolerance` of it (and whose previous close agrees too, when
    both are readable). A ticker is learned only if it is the one candidate
    left after intersecting every such session, at least `min_sessions`
    sessions support it, and no session contradicts it. Tickers that
    already have an ISIN are never candidates.
    """
    taken = set(known.values())
    free = {sym: days for sym, days in recorded.items() if sym not in taken}
    near = lambda a, b: a is not None and b is not None and b > 0 and abs(a / b - 1) <= tolerance
    learned: Dict[str, str] = {}
    unknown = {isin for day in readings.values() for isin in day if isin not in known}
    for isin in sorted(unknown):
        # Every free ticker starts as a candidate; a session on which a
        # ticker was recorded at a different price rules it out.
        candidates = set(free)
        support: Dict[str, int] = {}
        for d, rows in readings.items():
            row = rows.get(isin)
            if not row or row.get("vwap") is None:
                continue
            for sym in list(candidates):
                if d not in free[sym]:
                    continue
                close, prev = free[sym][d]
                if near(row["vwap"], close) and (row.get("previous") is None or prev is None
                                                 or near(row["previous"], prev)):
                    support[sym] = support.get(sym, 0) + 1
                else:
                    candidates.discard(sym)
        supported = {sym for sym in candidates if support.get(sym, 0) > 0}
        if len(supported) == 1:
            sym = supported.pop()
            if support[sym] >= min_sessions:
                learned[isin] = sym
    # One ISIN per ticker: two ISINs landing on one ticker are both dropped.
    counts: Dict[str, int] = {}
    for sym in learned.values():
        counts[sym] = counts.get(sym, 0) + 1
    return {isin: sym for isin, sym in learned.items() if counts[sym] == 1}


def recorded_closes(symbols: Iterable[str]) -> Dict[str, Dict[date, Tuple[float, Optional[float]]]]:
    """Each ticker's close and previous close as the NSE ticker feed stored
    them (source nse_ticker only), by session, from the history files."""
    out: Dict[str, Dict[date, Tuple[float, Optional[float]]]] = {}
    for sym in symbols:
        days: Dict[date, Tuple[float, Optional[float]]] = {}
        prev_close = None
        for r in load_csv(sym):
            try:
                d = date.fromisoformat(str(r.get("date"))[:10])
                close = float(r.get("close") or 0)
            except ValueError:
                continue
            if close > 0 and r.get("source") == "nse_ticker":
                pct = float(r.get("change_pct") or 0)
                prev = round(close / (1 + pct / 100), 2) if pct else prev_close
                days[d] = (close, prev)
            if close > 0:
                prev_close = close
        if days:
            out[sym.upper()] = days
    return out


@dataclass
class Verdict:
    symbol: str
    date: date
    bar: Optional[DailyBar]
    how: str  # 'confirmed', 'range-checked', or why it was rejected


def _same(a: float, b: float) -> bool:
    return abs(a - b) < 0.005


def validate(readings: Dict[date, Dict[str, Reading]], symbols: Iterable[str]) -> List[Verdict]:
    """Accept or reject each (symbol, session) from the lists that were read.

    `readings` holds every list read, keyed by session date. A session's
    second reading is the PREVIOUS column of the next list read; if a list
    in between is missing, that column names a different session, the two
    disagree, and the session falls back to the range check or is rejected.
    """
    dates = sorted(readings)
    verdicts: List[Verdict] = []
    for sym in symbols:
        for i, d in enumerate(dates):
            row = readings[d].get(sym)
            if row is None:
                verdicts.append(Verdict(sym, d, None, "row not found in the list"))
                continue
            nxt = readings[dates[i + 1]].get(sym) if i + 1 < len(dates) else None
            own, later = row["vwap"], (nxt or {}).get("previous")
            hi, lo, prev = row["high"], row["low"], row["previous"]
            in_range = hi is not None and lo is not None and lo <= hi
            if own is not None and later is not None:
                if not _same(own, later):
                    verdicts.append(Verdict(sym, d, None, f"read as {own} here but {later} "
                                                          f"as the next list's previous price"))
                    continue
                close, how = own, "confirmed"
            elif own is not None or later is not None:
                close = own if own is not None else later
                if not (in_range and lo - 0.005 <= close <= hi + 0.005):
                    verdicts.append(Verdict(sym, d, None, f"only one readable price ({close}) "
                                                          f"and no high/low that confirms it"))
                    continue
                how = "range-checked"
            else:
                verdicts.append(Verdict(sym, d, None, "price unreadable in both lists"))
                continue
            if prev is not None and abs(close / prev - 1) > MAX_SESSION_MOVE:
                verdicts.append(Verdict(sym, d, None, f"{close} is {close / prev - 1:+.0%} from the "
                                                      f"previous price {prev}, beyond a session's limit"))
                continue
            open_ = prev if prev is not None else close
            if in_range and lo - 0.005 <= close <= hi + 0.005 and abs(hi / close - 1) <= MAX_SESSION_MOVE \
                    and abs(lo / close - 1) <= MAX_SESSION_MOVE:
                high, low = max(hi, open_, close), min(lo, open_, close)
            else:  # high/low unreadable or inconsistent: the bar spans open to close
                high, low = max(open_, close), min(open_, close)
            verdicts.append(Verdict(sym, d, DailyBar(
                date=d.isoformat(), symbol=sym, open=round(open_, 2), high=round(high, 2),
                low=round(low, 2), close=round(close, 2), volume=int(row["volume"] or 0),
                change_pct=round((close - prev) / prev * 100, 2) if prev else 0.0,
                source=SOURCE), how))
    return verdicts


def merge_into_csv(symbol: str, bars: List[DailyBar]) -> Tuple[int, int]:
    """Write accepted bars, never over a bar a live source already recorded.

    Returns (written, skipped). Synthetic seed rows on the same date are
    replaced; that is the point of the backfill.
    """
    real_dates = {r.get("date") for r in load_csv(symbol) if r.get("source") in REAL_NSE_SOURCES}
    new = [b for b in bars if b.date not in real_dates]
    if new:
        save_bars_csv(symbol, new)
    return len(new), len(bars) - len(new)
