"""What each account would have made doing the simple thing instead.

Profit means little on its own: the NSE paper account has to beat Kenyan
Treasury bills, which paid about 8.8% at the September 2026 auctions with
no risk, and an equal-weight basket of the stocks it watches, bought once
and held. The US account has to beat the S&P 500 (SPY). These curves start
at each account's own starting value, on its own dates, so the dashboard
can draw them on the same chart.
"""
import csv
import time
from bisect import bisect_right
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.connectors.nse_scraper import REAL_NSE_SOURCES

# Kenyan 91-day T-bill: about 8.78% at the 24 September 2026 auction (CBK),
# and resident individuals pay 15% withholding tax on the interest. Both are
# config-overridable (benchmarks.tbill_rate_pct / tbill_withholding_pct).
TBILL_RATE_DEFAULT = 0.0878
TBILL_TAX_DEFAULT = 0.15


_rate_cache: Dict[str, Any] = {}


def tbill_rate(config: Optional[Dict[str, Any]] = None) -> Tuple[float, str]:
    """The 91-day T-bill rate and where it came from.

    The rate in the latest AIB-AXYS Market Pulse (market_pulse.py) when one
    from the last 30 days was uploaded, else benchmarks.tbill_rate_pct from
    the config, else the default. Read at most once a minute.
    """
    now = time.time()
    if _rate_cache.get('at', 0) > now - 60 and 'pulse' in _rate_cache:
        pulse = _rate_cache['pulse']
    else:
        try:
            from src.agent.market_pulse import latest_rates
            pulse = latest_rates()
        except Exception:
            pulse = {}
        _rate_cache.update(at=now, pulse=pulse)
    if pulse.get('tbill_91'):
        return float(pulse['tbill_91']), f"AIB-AXYS Market Pulse {pulse.get('as_of')}"
    cfg = (config or {}).get('benchmarks') or {}
    if cfg.get('tbill_rate_pct') is not None:
        return float(cfg['tbill_rate_pct']), 'config'
    return TBILL_RATE_DEFAULT, 'default'


def _day(value: Any) -> date:
    return date.fromisoformat(str(value)[:10])


def tbill_values(days: Sequence[str], start_value: float, annual_rate: float,
                 withholding: float = TBILL_TAX_DEFAULT) -> List[float]:
    """Start value compounded daily at the after-tax T-bill rate."""
    if not days:
        return []
    first = _day(days[0])
    net = annual_rate * (1 - withholding)
    return [round(start_value * (1 + net) ** ((_day(d) - first).days / 365), 2) for d in days]


def _asof(prices: Dict[date, float], ordered: List[date], d: date) -> Optional[float]:
    """The latest price on or before `d` (`ordered` is `prices`' dates, sorted)."""
    i = bisect_right(ordered, d)
    return prices[ordered[i - 1]] if i else None


def rebased_values(days: Sequence[str], prices: Dict[date, float],
                   start_value: float) -> List[Optional[float]]:
    """`start_value` moved with the price, from the first day with a price."""
    ordered = sorted(prices)
    base = None
    out: List[Optional[float]] = []
    for d in days:
        px = _asof(prices, ordered, _day(d))
        if px and base is None:
            base = px
        out.append(round(start_value * px / base, 2) if px and base else None)
    return out


def nse_closes(symbol: str, csv_dir: Path) -> Dict[date, float]:
    """Real daily closes from the scraper's CSV (no synthetic seed rows)."""
    path = Path(csv_dir) / f"{symbol.upper()}.csv"
    out: Dict[date, float] = {}
    if not path.exists():
        return out
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try:
                close = float(r.get('close') or 0)
                d = _day(r.get('date'))
            except ValueError:
                continue
            if close > 0 and r.get('source') in REAL_NSE_SOURCES:
                out[d] = close
    return out


def basket_values(days: Sequence[str], closes_by_symbol: Dict[str, Dict[date, float]],
                  start_value: float) -> List[Optional[float]]:
    """An equal-weight basket bought on the first day and held, price only.

    Each stock gets an equal share of the start value at its first price on
    or after the first day; one with no price then is left out, and the
    basket is split among the rest.
    """
    if not days:
        return []
    first = _day(days[0])
    legs = []
    for closes in closes_by_symbol.values():
        ordered = sorted(closes)
        entry_day = next((d for d in ordered if d >= first), None)
        if entry_day is None and ordered and ordered[0] < first:
            entry_day = ordered[bisect_right(ordered, first) - 1]
        if entry_day is not None and closes[entry_day]:
            legs.append((closes, ordered, entry_day, closes[entry_day]))
    if not legs:
        return [None] * len(days)
    share = start_value / len(legs)
    out: List[Optional[float]] = []
    for d in days:
        day, total = _day(d), 0.0
        for closes, ordered, entry_day, entry in legs:
            # Until a stock's entry day the basket holds it at its entry
            # price; an earlier close would start the line off the account.
            px = entry if day < entry_day else (_asof(closes, ordered, day) or entry)
            total += share * px / entry
        out.append(round(total, 2))
    return out


def overlay(points: List[Dict[str, Any]], key: str, values: Iterable[Optional[float]]) -> None:
    """Attach a benchmark value to each chart point under `key`."""
    for p, v in zip(points, values):
        if v is not None:
            p[key] = v
