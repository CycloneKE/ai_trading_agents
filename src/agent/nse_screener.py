"""Which NSE stocks the agent trades: a screen of the whole exchange, then a
short list.

The agent used to trade a fixed list of nine stocks. Now every listed
stock is screened from its real daily history, and the agent trades a
short list rebuilt once a week:

1. Everything it holds, always: a holding must keep being evaluated so its
   exits and stops work.
2. The best-ranked eligible stocks, no more than `max_per_sector` from one
   sector, so the banks that dominate the NSE cannot fill the list.
3. One exploration slot: a stock ranked just below the cut, rotating each
   week, so the agent also learns how its strategies do beyond the obvious
   names. Its trades are credited to the strategies like any other.
4. While too few stocks have enough history, the configured list
   (data_manager.nse_symbols) fills the gaps, under the same sector cap.

An AI review may then remove up to two names, and only for a concrete
reason such as a suspension or a takeover offer. Removed names are replaced
from the ranking. Without an AI provider the list stands as screened.

Eligible means at least `min_history_days` real daily bars and an average
traded value of at least `min_avg_value_kes` a day. The default is ten
times the agent's order size, because the paper account caps an order at
10% of a day's volume: a thinner stock could not take even one order.

Ranking, among eligible stocks: half on momentum (the return over about six
months, skipping the latest month, or over the history there is), a quarter
on trend (price above its 50-day average), a quarter on liquidity. Each is a
percentile within the eligible set, so the score reads 0 to 1.
"""
import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULTS = {
    'enabled': False,             # config.json switches it on
    'max_symbols': 9,
    'max_per_sector': 3,
    'exploration_slots': 1,
    'min_history_days': 50,       # what the strategies need to signal at all
    'min_avg_value_kes': None,   # None: ten times the order size
    'refresh_days': 7,
    'llm_review': True,
    'llm_max_removals': 2,
    'stale_after_days': 10,
}


def settings(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **{k: v for k, v in (config.get('nse_screener') or {}).items()
                          if not k.startswith('_')}}
    if not cfg.get('min_avg_value_kes'):
        notional = float((config.get('nse_order_tickets') or {}).get('trade_notional_kes', 50000))
        cfg['min_avg_value_kes'] = 10 * notional
    return cfg


@dataclass
class StockMetrics:
    symbol: str
    name: str
    sector: str
    days: int = 0
    last_date: Optional[str] = None
    price: Optional[float] = None
    change_pct: Optional[float] = None
    return_1m_pct: Optional[float] = None
    momentum_pct: Optional[float] = None
    momentum_window: Optional[str] = None
    above_sma50: Optional[bool] = None
    avg_value_kes: Optional[float] = None
    volatility_pct: Optional[float] = None
    eligible: bool = False
    reason: Optional[str] = None   # why not eligible
    score: Optional[float] = None
    rank: Optional[int] = None


def real_rows(symbol: str, csv_dir: Path) -> List[Dict[str, Any]]:
    """Real daily bars, one per date, oldest first."""
    from src.connectors.nse_scraper import REAL_NSE_SOURCES
    path = Path(csv_dir) / f"{symbol.upper()}.csv"
    by_date: Dict[str, Dict[str, Any]] = {}
    try:
        with open(path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('source') not in REAL_NSE_SOURCES:
                    continue
                try:
                    close = float(r.get('close') or 0)
                except ValueError:
                    continue
                if close > 0 and r.get('date'):
                    by_date[str(r['date'])[:10]] = {
                        'date': str(r['date'])[:10], 'close': close,
                        'volume': float(r.get('volume') or 0),
                        'change_pct': float(r.get('change_pct') or 0)}
    except OSError:
        return []
    return [by_date[d] for d in sorted(by_date)]


def metrics_for(symbol: str, rows: List[Dict[str, Any]], cfg: Dict[str, Any],
                today: Optional[date] = None) -> StockMetrics:
    from src.connectors.nse_universe import name_of, sector_of
    m = StockMetrics(symbol=symbol.upper(), name=name_of(symbol), sector=sector_of(symbol),
                     days=len(rows))
    if not rows:
        m.reason = 'no real prices yet'
        return m
    closes = [r['close'] for r in rows]
    m.last_date, m.price, m.change_pct = rows[-1]['date'], closes[-1], rows[-1]['change_pct']
    recent = rows[-20:]
    m.avg_value_kes = round(sum(r['close'] * r['volume'] for r in recent) / len(recent), 0)
    if len(closes) > 21:
        m.return_1m_pct = round((closes[-1] / closes[-22] - 1) * 100, 2)
    skip = 20 if len(closes) >= 61 else 0
    lookback = min(120, len(closes) - 1 - skip)
    if lookback >= 10:
        m.momentum_pct = round((closes[-1 - skip] / closes[-1 - skip - lookback] - 1) * 100, 2)
        m.momentum_window = (f"{lookback} days to {skip} days ago" if skip
                             else f"last {lookback} days")
    window = closes[-50:] if len(closes) >= 50 else (closes if len(closes) >= 20 else None)
    if window:
        m.above_sma50 = closes[-1] > sum(window) / len(window)
    rets = [math.log(b / a) for a, b in zip(closes[-21:], closes[-20:]) if a > 0 and b > 0]
    if len(rets) >= 10:
        mean = sum(rets) / len(rets)
        m.volatility_pct = round(math.sqrt(sum((x - mean) ** 2 for x in rets) / len(rets))
                                 * math.sqrt(252) * 100, 1)

    today = today or datetime.now(timezone.utc).date()
    age = (today - date.fromisoformat(m.last_date)).days
    if age > cfg['stale_after_days']:
        m.reason = f'no price for {age} days'
    elif m.days < cfg['min_history_days']:
        m.reason = f"{m.days} of {cfg['min_history_days']} days of history"
    elif (m.avg_value_kes or 0) < cfg['min_avg_value_kes']:
        m.reason = (f"trades about KES {m.avg_value_kes:,.0f} a day; "
                    f"needs KES {cfg['min_avg_value_kes']:,.0f}")
    elif m.momentum_pct is None:
        m.reason = 'too little history to measure momentum'
    else:
        m.eligible = True
    return m


def _percentiles(values: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(values, key=lambda s: values[s])
    n = len(ordered)
    return {s: (i / (n - 1) if n > 1 else 1.0) for i, s in enumerate(ordered)}


def screen(symbols: Iterable[str], csv_dir: Path, cfg: Dict[str, Any],
           today: Optional[date] = None) -> List[StockMetrics]:
    """Every symbol's metrics, eligible ones ranked by score first."""
    ms = [metrics_for(s, real_rows(s, csv_dir), cfg, today) for s in sorted(set(symbols))]
    ok = [m for m in ms if m.eligible]
    mom = _percentiles({m.symbol: m.momentum_pct for m in ok})
    liq = _percentiles({m.symbol: m.avg_value_kes for m in ok})
    for m in ok:
        trend = 1.0 if m.above_sma50 else 0.0
        m.score = round(0.5 * mom[m.symbol] + 0.25 * trend + 0.25 * liq[m.symbol], 3)
    ok.sort(key=lambda m: (-m.score, m.symbol))
    for i, m in enumerate(ok):
        m.rank = i + 1
    rest = sorted((m for m in ms if not m.eligible), key=lambda m: (-(m.days or 0), m.symbol))
    return ok + rest


@dataclass
class Shortlist:
    symbols: List[str]
    roles: Dict[str, str]                 # symbol -> holding | top | exploration | fallback
    removed: List[Dict[str, str]] = field(default_factory=list)   # the AI review's removals
    notes: Optional[str] = None
    reviewed_by_ai: bool = False
    built_at: Optional[str] = None


def build(ranked: List[StockMetrics], holdings: Iterable[str], cfg: Dict[str, Any],
          fallback: Iterable[str] = (), week: int = 0,
          exclude: Iterable[str] = ()) -> Shortlist:
    """The short list, before any AI review (see the module note)."""
    from src.connectors.nse_universe import sector_of
    held = [s.upper() for s in holdings]
    excluded = {s.upper() for s in exclude}
    picked: List[str] = []
    roles: Dict[str, str] = {}
    per_sector: Dict[str, int] = {}

    def take(sym: str, role: str, capped: bool = True) -> bool:
        sector = sector_of(sym)
        if sym in roles or (capped and per_sector.get(sector, 0) >= cfg['max_per_sector']):
            return False
        picked.append(sym)
        roles[sym] = role
        per_sector[sector] = per_sector.get(sector, 0) + 1
        return True

    for sym in held:
        take(sym, 'holding', capped=False)
    explore = max(int(cfg['exploration_slots']), 0)
    top_slots = max(int(cfg['max_symbols']) - explore - len(picked), 0)
    eligible = [m for m in ranked if m.eligible and m.symbol not in excluded]
    for m in eligible:
        if top_slots <= 0:
            break
        if take(m.symbol, 'top'):
            top_slots -= 1
    # Exploration: rotate weekly through the ten ranked just below the cut.
    window = [m.symbol for m in eligible if m.symbol not in roles][:10]
    for _ in range(explore):
        if len(picked) >= cfg['max_symbols'] or not window:
            break
        start = week % len(window)
        for sym in window[start:] + window[:start]:
            if take(sym, 'exploration'):
                window.remove(sym)
                break
        else:
            break
    for sym in (s.upper() for s in fallback):
        if len(picked) >= cfg['max_symbols']:
            break
        if sym not in excluded:
            take(sym, 'fallback')
    return Shortlist(symbols=picked, roles=roles)


_REVIEW_SYSTEM = (
    "You review a short list of Nairobi Securities Exchange stocks that a paper-trading agent "
    "will trade for the next week. The list was chosen by a quantitative screen. Remove a stock "
    "only for a specific, concrete reason you are confident of: it is suspended from trading, "
    "under statutory management or receivership, subject to a takeover or delisting offer that "
    "pins its price, or has just issued a profit warning. General uncertainty, valuation views "
    "and sector outlooks are not reasons. Removing nothing is the normal answer.\n"
    'Reply with JSON only: {"remove": [{"symbol": "...", "reason": "..."}], "notes": "..."}'
)


def ai_review(llm, shortlist: Shortlist, ranked: List[StockMetrics],
              max_removals: int) -> Optional[Dict[str, Any]]:
    """The AI's removals, or None when no AI answered."""
    if llm is None or not getattr(llm, 'enabled', False):
        return None
    from src.agent import market_pulse
    by_sym = {m.symbol: m for m in ranked}
    lines = []
    for sym in shortlist.symbols:
        m = by_sym.get(sym)
        line = (f"- {sym} ({m.name if m else sym}, {m.sector if m else '?'}): role "
                f"{shortlist.roles[sym]}, price {m.price if m else '?'} KES, "
                f"momentum {m.momentum_pct if m else '?'}%")
        # The broker's own figures and the company's recent announcements
        # (an AGM, results, a suspension or listing notice) are the concrete
        # facts this review is for.
        try:
            f = market_pulse.fundamentals(sym)
            news = market_pulse.announcements(sym)
        except Exception:
            f, news = {}, []
        if f:
            line += (f", P/E {f.get('pe') or 'n/a'}, dividend yield {f.get('dividend_yield_pct')}%"
                     f" (AIB-AXYS {f.get('as_of')})")
        if news:
            line += '; announcements: ' + '; '.join(f"{a['date']} {a['text']}" for a in news[-3:])
        lines.append(line)
    user = (f"Proposed list ({len(shortlist.symbols)} stocks). Holdings cannot be removed. "
            f"Remove at most {max_removals}.\n" + "\n".join(lines))
    try:
        reply = llm.propose_json(_REVIEW_SYSTEM, user)
    except Exception as e:
        logger.warning(f"NSE shortlist AI review failed: {e}")
        return None
    if not isinstance(reply, dict):
        return None
    removals = []
    for r in reply.get('remove') or []:
        if not isinstance(r, dict):
            continue
        sym = str(r.get('symbol', '')).upper()
        why = str(r.get('reason', '')).strip()
        if sym in shortlist.roles and shortlist.roles[sym] != 'holding' and why:
            removals.append({'symbol': sym, 'reason': why[:300]})
    return {'remove': removals[:max_removals], 'notes': str(reply.get('notes') or '')[:500]}


def refresh(ranked: List[StockMetrics], holdings: Iterable[str], cfg: Dict[str, Any],
            fallback: Iterable[str] = (), llm=None,
            now: Optional[datetime] = None) -> Shortlist:
    """Build the week's short list and, if an AI answers, apply its review."""
    now = now or datetime.now(timezone.utc)
    week = now.isocalendar()[1]
    sl = build(ranked, holdings, cfg, fallback, week)
    review = ai_review(llm, sl, ranked, int(cfg['llm_max_removals'])) if cfg.get('llm_review') else None
    if review is not None:
        removed = review['remove']
        if removed:
            sl = build(ranked, holdings, cfg, fallback, week,
                       exclude=[r['symbol'] for r in removed])
        sl.removed, sl.notes, sl.reviewed_by_ai = removed, review['notes'] or None, True
    sl.built_at = now.isoformat()
    return sl


class ShortlistStore:
    """The current short list, kept in a JSON file across restarts."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> Optional[Shortlist]:
        try:
            data = json.loads(self.path.read_text())
            return Shortlist(**{f.name: data[f.name] for f in fields(Shortlist) if f.name in data})
        except (OSError, ValueError, TypeError):
            return None

    def save(self, sl: Shortlist) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(asdict(sl), indent=2))
        tmp.replace(self.path)

    def due(self, cfg: Dict[str, Any], now: Optional[datetime] = None) -> bool:
        sl = self.load()
        if sl is None or not sl.built_at:
            return True
        now = now or datetime.now(timezone.utc)
        try:
            built = datetime.fromisoformat(sl.built_at)
        except ValueError:
            return True
        if built.tzinfo is None:
            built = built.replace(tzinfo=timezone.utc)
        return (now - built).days >= int(cfg['refresh_days'])


def as_dicts(ranked: List[StockMetrics]) -> List[Dict[str, Any]]:
    return [asdict(m) for m in ranked]
