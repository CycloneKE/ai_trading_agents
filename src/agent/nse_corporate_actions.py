"""NSE corporate actions: bonus issues, share splits and dividends.

Why this exists: NSE prices were used raw. A 1-for-10 bonus issue drops the
share price by about 9% overnight with no loss to the holder, and a
high-yield bank can drop 5 to 7% on its ex-dividend day while the holder is
paid the difference. Read raw, either one could trip the paper account's 8%
stop-loss on a move that is not a loss (at about 3.9% in round-trip costs),
and it distorted the six-month returns the NSE rotation strategy ranks on.

Two sources, both kept by the operator from company announcements:

- config/nse_corporate_actions.json: bonus issues and splits.
      {"symbol": "KCB", "type": "bonus", "ratio": "1:10", "ex_date": "2026-05-20"}
  "1:10" is one new share for every ten held (11 shares for 10). A split
  "2:1" is two shares for every one held.
- config/nse_dividend_events.json (the paper account's file): dividends,
  each with its ex_date, or a book_closure date from which the ex-date is
  taken as three trading days earlier (NSE settles T+3).

Prices before an ex-date are adjusted backwards, the way data vendors adjust
a price series:

- a bonus or split multiplies earlier prices by shares_before / shares_after;
- a dividend multiplies earlier prices by (P - D) / P, where P is the last
  close before the ex-date and D the gross dividend per share.

So the series reads as total return: an event is not a fall.
"""
import csv
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ACTIONS_DEFAULT = 'config/nse_corporate_actions.json'
DIVIDENDS_DEFAULT = 'config/nse_dividend_events.json'

Row = Tuple[date, float, float, float]  # (date, high, low, close)


@dataclass(frozen=True)
class Action:
    symbol: str
    kind: str                 # 'bonus' | 'split' | 'dividend'
    ex_date: date
    share_factor: float = 1.0  # shares after / shares before (bonus, split)
    dividend: float = 0.0      # gross KES per share (dividend)

    @property
    def key(self) -> str:
        """Stable identity, for remembering an action has been applied."""
        amount = self.dividend if self.kind == 'dividend' else self.share_factor
        return f"{self.symbol}|{self.kind}|{self.ex_date.isoformat()}|{amount:g}"

    def price_factor(self, close_before: Optional[float] = None) -> float:
        """What earlier prices are multiplied by. 1.0 when it cannot be worked
        out (a dividend with no close before it, or one larger than the price)."""
        if self.kind in ('bonus', 'split'):
            return 1.0 / self.share_factor if self.share_factor > 0 else 1.0
        if close_before and close_before > self.dividend > 0:
            return (close_before - self.dividend) / close_before
        return 1.0


def _path(p) -> Path:
    path = Path(p)
    if not path.is_absolute():
        from src.utils.paths import PROJECT_ROOT
        path = PROJECT_ROOT / path
    return path


def _read(path) -> List[dict]:
    try:
        data = json.loads(_path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    events = data.get('events', []) if isinstance(data, dict) else data
    return [e for e in events if isinstance(e, dict)]


def parse_ratio(kind: str, ratio: str) -> Optional[float]:
    """shares after / shares before, from "a:b". Bonus "1:10": one new share
    for every ten held, 11/10. Split "2:1": two shares for every one, 2.0."""
    m = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)\s*', str(ratio or ''))
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    if a <= 0 or b <= 0:
        return None
    return (b + a) / b if kind == 'bonus' else a / b


def load_actions(actions_path=ACTIONS_DEFAULT,
                 dividends_path=DIVIDENDS_DEFAULT) -> List[Action]:
    """Every valid action, oldest first. A malformed entry is skipped with a
    warning rather than guessed."""
    out: List[Action] = []
    for e in _read(actions_path):
        kind = str(e.get('type') or '').lower()
        try:
            ex = date.fromisoformat(str(e.get('ex_date'))[:10])
        except ValueError:
            logger.warning(f"NSE corporate action skipped (no valid ex_date): {e}")
            continue
        factor = parse_ratio(kind, e.get('ratio')) if kind in ('bonus', 'split') else None
        if not e.get('symbol') or not factor:
            logger.warning(f"NSE corporate action skipped (needs symbol, type bonus or split, ratio a:b): {e}")
            continue
        out.append(Action(str(e['symbol']).upper(), kind, ex, share_factor=factor))
    for e in _read(dividends_path):
        try:
            amount = float(e.get('dividend_kes') or 0)
            if e.get('ex_date'):
                ex = date.fromisoformat(str(e['ex_date'])[:10])
            elif e.get('book_closure'):
                from src.agent.nse_paper_account import subtract_trading_days
                bc = datetime.fromisoformat(str(e['book_closure'])[:10])
                ex = subtract_trading_days(bc, 3).date()
            else:
                continue
        except (TypeError, ValueError):
            continue
        if e.get('symbol') and amount > 0:
            out.append(Action(str(e['symbol']).upper(), 'dividend', ex, dividend=amount))
    out.sort(key=lambda a: (a.ex_date, a.symbol, a.kind))
    return out


def for_symbol(actions: Iterable[Action], symbol: str) -> List[Action]:
    sym = str(symbol).upper()
    return [a for a in actions if a.symbol == sym]


def adjust_rows(rows: Sequence[Row], actions: Iterable[Action]) -> List[Row]:
    """The rows (oldest first) with prices before each ex-date adjusted.

    Only actions inside the series count: one dated after its last row has
    not happened yet in this data, and one before its first row changes
    nothing in it.
    """
    rows = sorted(rows, key=lambda r: r[0])
    if not rows:
        return []
    factors = [1.0] * len(rows)
    for a in actions:
        before = [i for i, r in enumerate(rows) if r[0] < a.ex_date]
        if not before or before[-1] == len(rows) - 1:
            continue                       # nothing after it, or nothing before it
        f = a.price_factor(rows[before[-1]][3])
        if f == 1.0:
            continue
        for i in before:
            factors[i] *= f
    return [(r[0], r[1] * f, r[2] * f, r[3] * f) for r, f in zip(rows, factors)]


def effective(actions: Iterable[Action], after: Optional[date], through: date) -> List[Action]:
    """Actions with after < ex_date <= through (after None: from the start)."""
    return [a for a in actions if a.ex_date <= through and (after is None or a.ex_date > after)]


def last_close_before(symbol: str, day: date, csv_dir) -> Optional[float]:
    """The last real close before `day` in the scraper's CSV, unadjusted."""
    from src.connectors.nse_scraper import REAL_NSE_SOURCES
    path = Path(csv_dir) / f"{str(symbol).upper()}.csv"
    best: Optional[Tuple[date, float]] = None
    try:
        with open(path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('source') not in REAL_NSE_SOURCES:
                    continue
                try:
                    d = date.fromisoformat((r.get('date') or '')[:10])
                    close = float(r.get('close') or 0)
                except ValueError:
                    continue
                if d < day and close > 0 and (best is None or d >= best[0]):
                    best = (d, close)
    except OSError:
        return None
    return best[1] if best else None


def combined_factor(actions: Iterable[Action],
                    close_before: Callable[[Action], Optional[float]]) -> float:
    """The product of the actions' price factors."""
    total = 1.0
    for a in actions:
        total *= a.price_factor(close_before(a) if a.kind == 'dividend' else None)
    return total
