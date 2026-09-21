"""Screens layered on top of the sleeve's yield + quality score.

Three things the deterministic scorer does not capture on its own:

1. Ex-dividend timing. A share price falls by roughly the dividend on the
   ex-date. Buying the day before captures a payout you have already paid
   for in the price, and in Kenya that payout is subject to withholding tax
   while the price drop is not deductible. Buying shortly after the ex-date
   gets the same business for less.

2. Payout sustainability. A 15% yield usually means the market expects a
   cut. Dividend cover, earnings divided by dividends, says whether the
   payout is funded by profit or by borrowing and reserves. Cover below 1
   means the company is paying out more than it earns.

3. Sector concentration. Kenyan banks dominate the high-yield list, so an
   unconstrained top-N screen turns a "diversified income sleeve" into a
   leveraged bet on one sector's credit cycle.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Ex-date timing classifications.
APPROACHING = 'approaching_ex_date'   # unfavourable: paying for the dividend
JUST_AFTER = 'just_after_ex_date'     # favourable: price has already dropped
NEUTRAL = 'neutral'
UNKNOWN_DATE = 'unknown'


def _as_date(value: Union[str, date, datetime, None]) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except (ValueError, TypeError):
        return None


def ex_date_timing(ex_dividend_date: Union[str, date, None],
                   today: Optional[date] = None,
                   approach_window: int = 10,
                   after_window: int = 15) -> str:
    """Where today sits relative to a known ex-dividend date.

    `approach_window` days before the ex-date is the window to avoid, and
    `after_window` days following it is the window to prefer. Outside both,
    and whenever the date is unknown, timing is not a factor.
    """
    ex = _as_date(ex_dividend_date)
    if ex is None:
        return UNKNOWN_DATE
    today = today or date.today()
    delta = (ex - today).days
    if 0 <= delta <= approach_window:
        return APPROACHING
    if -after_window <= delta < 0:
        return JUST_AFTER
    return NEUTRAL


def dividend_cover(eps: Optional[float],
                   dividend_per_share: Optional[float]) -> Optional[float]:
    """Earnings divided by dividends. None when it cannot be computed.

    Above roughly 1.5 is comfortable, 1.0 to 1.5 is tight, and below 1.0
    means the payout exceeds earnings and is being funded from somewhere
    else. Negative earnings give a negative cover, which is never passable.
    """
    try:
        eps = float(eps)
        dps = float(dividend_per_share)
    except (TypeError, ValueError):
        return None
    if dps <= 0:
        return None
    return eps / dps


@dataclass
class SustainabilityVerdict:
    passed: bool                    # clean on every check
    cover: Optional[float]
    reasons: List[str]
    disqualifying: List[str]        # the subset that should gate a buy outright

    @property
    def summary(self) -> str:
        return "sustainable" if self.passed else "; ".join(self.reasons)

    @property
    def is_disqualified(self) -> bool:
        return bool(self.disqualifying)


def assess_sustainability(eps: Optional[float],
                          dividend_per_share: Optional[float],
                          yield_pct: Optional[float],
                          years_consecutive_paid: int = 0,
                          eps_trend: str = 'flat',
                          min_cover: float = 1.2,
                          yield_trap_pct: float = 12.0,
                          min_years_paid: int = 3) -> SustainabilityVerdict:
    """Whether a payout looks fundable from earnings rather than hope.

    The yield-trap rule is the important one: a yield above `yield_trap_pct`
    combined with thin cover is the market pricing in a cut, and buying it
    for the headline yield means buying the cut.
    """
    reasons: List[str] = []
    disqualifying: List[str] = []
    cover = dividend_cover(eps, dividend_per_share)

    # Disqualifying: the payout is not funded by profit, or the market is
    # already pricing in a cut. These are facts about affordability, not
    # matters of degree, so they gate rather than score.
    if cover is None:
        reasons.append("dividend cover unavailable")
    elif cover < 0:
        msg = f"negative earnings (cover {cover:.2f})"
        reasons.append(msg); disqualifying.append(msg)
    elif cover < min_cover:
        msg = f"cover {cover:.2f} below minimum {min_cover:.2f}"
        reasons.append(msg)
        if cover < 1.0:
            # Paying out more than it earns, funded from reserves or debt.
            disqualifying.append(msg)

    try:
        y = float(yield_pct)
    except (TypeError, ValueError):
        y = 0.0
    if y >= yield_trap_pct and (cover is None or cover < 1.5):
        msg = (f"yield {y:.1f}% at or above {yield_trap_pct:.1f}% with thin cover: "
               f"likely a cut priced in, not a bargain")
        reasons.append(msg); disqualifying.append(msg)

    if years_consecutive_paid < min_years_paid:
        reasons.append(
            f"only {years_consecutive_paid} consecutive years paid, "
            f"needs {min_years_paid}")

    if eps_trend == 'negative':
        reasons.append("earnings trending down")

    return SustainabilityVerdict(passed=not reasons, cover=cover, reasons=reasons,
                                 disqualifying=disqualifying)


SectorLookup = Union[Dict[str, str], Callable[[str], str], None]


def _sector_of(symbol: str, lookup: SectorLookup) -> str:
    if lookup is None:
        return 'unknown'
    try:
        if callable(lookup):
            return lookup(symbol) or 'unknown'
        return lookup.get(symbol, 'unknown')
    except Exception as e:
        logger.debug(f"Sector lookup failed for {symbol}: {e}")
        return 'unknown'


def apply_concentration_limit(candidates: Sequence[Any],
                              sector_lookup: SectorLookup,
                              max_per_sector: int = 2,
                              symbol_attr: str = 'symbol') -> List[Any]:
    """Keep ranked order, but admit at most `max_per_sector` from each sector.

    Candidates are assumed to arrive best-first. A name displaced by the cap
    is dropped rather than demoted, so the next-best name from another sector
    takes the slot. `unknown` sectors are not capped: silently limiting names
    we failed to classify would quietly shrink the sleeve.
    """
    if max_per_sector <= 0:
        return list(candidates)
    seen: Dict[str, int] = {}
    kept = []
    for c in candidates:
        symbol = getattr(c, symbol_attr, None) or (
            c.get(symbol_attr) if isinstance(c, dict) else None)
        sector = _sector_of(symbol, sector_lookup)
        if sector != 'unknown':
            if seen.get(sector, 0) >= max_per_sector:
                logger.debug(f"Dropping {symbol}: {sector} already at "
                             f"{max_per_sector} holdings")
                continue
            seen[sector] = seen.get(sector, 0) + 1
        kept.append(c)
    return kept
