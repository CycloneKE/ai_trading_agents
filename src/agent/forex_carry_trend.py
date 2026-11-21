"""Forex carry and trend: what the forex book should hold this week.

Why this exists: the agent's fast technical rules (momentum, mean reversion,
RSI) traded forex long only, about ten round trips a year, and lost money
after costs over ten years (profit factor 0.88). scripts/forex_factor_study.py
tested the two approaches professionals run on currencies, on five major
pairs from 2017 to 2026, at their published settings:

- trend: hold a currency whose excess return against the dollar over the
  last twelve months was positive, sell one whose was negative
  (time-series momentum; Moskowitz, Ooi and Pedersen, 2012). As in the
  paper, the excess return is what a currency forward earned: the price
  move PLUS the interest-rate gap over the year (Hurst, Ooi and Pedersen,
  "A Century of Evidence on Trend-Following Investing", 2017, build their
  currency returns from forwards in the same way);
- carry: hold the currencies paying the higher interest rates, sell those
  paying the lower (Koijen, Moskowitz, Pedersen and Vrugt, 2018).

Their blend, long AND short, had a Sharpe ratio of 0.55 with the trend
measured on excess return (0.47 measured on the price move alone), positive
in both halves (0.51 and 0.60) and 0.47 or better when acted on one to five
days late; every long-only version lost money. This module is that blend,
unchanged:

    signal = (carry + trend) / 2          one of -1, -0.5, 0, +0.5, +1

Each pair is then sized so that, alone, it would move about `vol_target` a
year (inverse volatility, the way trend-following funds size), divided by the
number of pairs, and the whole book is scaled down if its gross exposure
would pass `max_leverage` times equity. "Long" always means long the foreign
currency against the dollar: for USD_JPY that is a SHORT position in the pair.

Rebalanced once a week, with a buffer so a small drift is not traded (Carver,
Systematic Trading): each change is a cost, and a weekly signal does not need
daily trading. Pure functions only; orders are placed by the agent
(TradingAgent._run_fx_carry_cycle) on the forex margin book
(src/connectors/forex_margin_broker.py).
"""
import json
import logging
import math
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

STRATEGY = 'forex_carry_trend'
TRADING_DAYS = 252

# The FRED (OECD) 3-month interbank rate series for each currency.
FRED_SERIES = {'USD': 'US', 'EUR': 'EZ', 'GBP': 'GB', 'AUD': 'AU', 'NZD': 'NZ',
               'JPY': 'JP', 'CAD': 'CA', 'CHF': 'CH'}
FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=IR3TIB01{code}M156N'

DEFAULT_PAIRS = ('EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD', 'USD_JPY')


@dataclass
class FxConfig:
    enabled: bool = False
    pairs: Tuple[str, ...] = DEFAULT_PAIRS
    vol_target: float = 0.10          # yearly volatility each pair is sized to, alone
    max_leverage: float = 3.0         # gross exposure cap, as a multiple of equity
    trend_lookback: int = TRADING_DAYS
    vol_window: int = 60
    min_trade_usd: float = 200.0
    buffer: float = 0.10              # leave a position alone within 10% of its target
    max_drawdown: float = 0.15        # the book closes everything and stops past this
    trend_on_excess: bool = True      # trend counts the year's interest gap, as the paper does
    rate_max_age_days: int = 120      # an older published rate gives way to fallback_rates_pct
    fallback_rates_pct: Dict[str, float] = field(default_factory=dict)

    @property
    def bars_needed(self) -> int:
        return self.trend_lookback + self.vol_window + 10


def config_from_dict(cfg: Optional[Dict[str, Any]]) -> FxConfig:
    """Read `forex_carry_trend`. Pairs without the dollar on one side are
    left out: the book's value is kept in dollars."""
    cfg = cfg or {}
    pairs = []
    for p in cfg.get('pairs', DEFAULT_PAIRS):
        p = str(p).upper()
        base, _, quote = p.partition('_')
        if 'USD' not in (base, quote) or base == quote or not quote:
            logger.warning(f"Forex carry/trend: {p} is not a dollar pair; left out")
            continue
        pairs.append(p)
    return FxConfig(
        enabled=bool(cfg.get('enabled', False)) and bool(pairs),
        pairs=tuple(pairs),
        vol_target=max(float(cfg.get('vol_target', 0.10)), 0.0),
        max_leverage=max(float(cfg.get('max_leverage', 3.0)), 0.0),
        trend_lookback=int(cfg.get('trend_lookback', TRADING_DAYS)),
        vol_window=int(cfg.get('vol_window', 60)),
        min_trade_usd=float(cfg.get('min_trade_usd', 200.0)),
        buffer=float(cfg.get('buffer', 0.10)),
        max_drawdown=float(cfg.get('max_drawdown', 0.15)),
        trend_on_excess=bool(cfg.get('trend_on_excess', True)),
        rate_max_age_days=int(cfg.get('rate_max_age_days', 120)),
        fallback_rates_pct={str(k).upper(): float(v)
                            for k, v in (cfg.get('fallback_rates_pct') or {}).items()
                            if not str(k).startswith('_')},
    )


def split(pair: str) -> Tuple[str, str]:
    base, _, quote = pair.upper().partition('_')
    return base, quote


def foreign(pair: str) -> str:
    """The non-dollar currency of a dollar pair."""
    base, quote = split(pair)
    return quote if base == 'USD' else base


def oriented(pair: str, closes: Sequence[float]) -> List[float]:
    """Closes turned so that up means the foreign currency rose against the
    dollar (USD_JPY up means the yen fell, so it is inverted)."""
    clean = [float(c) for c in closes if c and c > 0]
    return [1.0 / c for c in clean] if split(pair)[0] == 'USD' else clean


def sign(x: float) -> float:
    return 1.0 if x > 0 else -1.0 if x < 0 else 0.0


def trend_signal(closes: Sequence[float], lookback: int, interest: float = 0.0) -> Optional[float]:
    """+1 if the (oriented) excess return over the last `lookback` days was
    positive, -1 if negative; None without enough history. `interest` is
    the interest-rate gap earned over the same days, as a fraction (0.02 for
    a 2-point gap held a year); 0 measures the price move alone."""
    if len(closes) <= lookback:
        return None
    return sign(closes[-1] / closes[-1 - lookback] - 1.0 + interest)


def carry_signals(rate_gaps: Dict[str, float]) -> Dict[str, float]:
    """+1 for currencies whose interest-rate gap over the dollar is above
    the median of the pairs, -1 below, 0 at it (the cross-sectional carry
    rule the study tested)."""
    if not rate_gaps:
        return {}
    ordered = sorted(rate_gaps.values())
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {p: sign(g - median) for p, g in rate_gaps.items()}


def realised_vol(closes: Sequence[float], window: int) -> Optional[float]:
    """Yearly volatility of the last `window` daily returns."""
    if len(closes) <= window:
        return None
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(len(closes) - window, len(closes))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
    vol = math.sqrt(var) * math.sqrt(TRADING_DAYS)
    return vol if vol > 0 else None


def target_weights(closes_by_pair: Dict[str, Sequence[float]], rates_pct: Dict[str, float],
                   cfg: FxConfig, avg_rates_pct: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """This week's weights, each as a fraction of equity held in the foreign
    currency against the dollar (negative: short the foreign currency), and
    the signals behind them for the record.

    A pair without enough history, a volatility reading or an interest
    rate for both its currencies gets no position. `avg_rates_pct` is each
    currency's average rate over the trend's lookback, for the interest
    part of the excess return; a currency missing from it uses its rate in
    `rates_pct`."""
    usd = rates_pct.get('USD')
    series = {p: oriented(p, closes_by_pair.get(p) or []) for p in cfg.pairs}
    gaps = {p: rates_pct[foreign(p)] - usd for p in cfg.pairs
            if usd is not None and foreign(p) in rates_pct}
    carry = carry_signals(gaps)
    detail, raw = {}, {}
    n = len(cfg.pairs)
    avg = {**rates_pct, **(avg_rates_pct or {})}
    for p in cfg.pairs:
        interest = 0.0
        if cfg.trend_on_excess and 'USD' in avg and foreign(p) in avg:
            interest = (avg[foreign(p)] - avg['USD']) / 100.0 * cfg.trend_lookback / TRADING_DAYS
        trend = trend_signal(series[p], cfg.trend_lookback, interest)
        vol = realised_vol(series[p], cfg.vol_window)
        row = {'trend': trend, 'carry': carry.get(p), 'rate_gap_pct': gaps.get(p),
               'trend_interest_pct': round(interest * 100, 3),
               'vol': round(vol, 4) if vol else None}
        if trend is None or vol is None or p not in carry:
            row['signal'] = None
            row['why_none'] = ('history' if trend is None else 'volatility' if vol is None
                               else 'interest rate')
            detail[p] = row
            continue
        sig = (carry[p] + trend) / 2.0
        row['signal'] = sig
        raw[p] = sig * cfg.vol_target / vol / n
        detail[p] = row
    gross = sum(abs(w) for w in raw.values())
    scale = cfg.max_leverage / gross if cfg.max_leverage and gross > cfg.max_leverage else 1.0
    weights = {p: w * scale for p, w in raw.items()}
    for p, w in weights.items():
        detail[p]['weight'] = round(w, 4)
    return {'weights': weights, 'signals': detail, 'gross_leverage': round(gross * scale, 3),
            'scaled_by': round(scale, 3)}


def target_units(weights: Dict[str, float], equity: float, prices: Dict[str, float]) -> Dict[str, float]:
    """Weights turned into position sizes in the pair's base currency
    (positive long the pair, negative short). XXX_USD: units of XXX worth
    weight x equity dollars. USD_XXX: dollars, with the sign flipped,
    because long the foreign currency is short the pair."""
    out = {}
    for p, w in weights.items():
        px = prices.get(p)
        if not px or px <= 0:
            continue
        base, _ = split(p)
        out[p] = -w * equity if base == 'USD' else w * equity / px
    return out


def notional_usd(pair: str, units: float, price: float) -> float:
    """Size of a position in dollars (always positive)."""
    return abs(units) if split(pair)[0] == 'USD' else abs(units) * price


def plan_changes(current: Dict[str, float], targets: Dict[str, float], prices: Dict[str, float],
                 cfg: FxConfig) -> Dict[str, float]:
    """Unit changes to place this week: every pair whose position should go
    to zero or change side, and any other whose change is larger than both
    min_trade_usd and `buffer` of the target (a small drift is left alone)."""
    changes = {}
    for p in set(current) | set(targets):
        px = prices.get(p)
        if not px:
            continue
        have, want = current.get(p, 0.0), targets.get(p, 0.0)
        delta = want - have
        if abs(delta) < 1e-9:
            continue
        flips = want == 0.0 or (have != 0.0 and sign(want) != sign(have))
        size = notional_usd(p, delta, px)
        if not flips and (size < cfg.min_trade_usd or
                          size < cfg.buffer * notional_usd(p, want, px)):
            continue
        changes[p] = delta
    return changes


def due(last_rebalance: Optional[str], today: date) -> Optional[str]:
    """'build' before the first rebalance, 'weekly' in a new ISO week."""
    if not last_rebalance:
        return 'build'
    last = date.fromisoformat(last_rebalance[:10])
    return 'weekly' if last.isocalendar()[:2] != today.isocalendar()[:2] else None


# --------------------------------------------------------------- interest rates

def parse_fred_csv(text: str) -> List[Tuple[date, float]]:
    rows = []
    for line in text.strip().splitlines()[1:]:
        parts = line.split(',')
        if len(parts) < 2:
            continue
        try:
            rows.append((date.fromisoformat(parts[0]), float(parts[1])))
        except ValueError:
            continue                          # FRED writes '.' for a missing month
    return rows


def fetch_fred(code: str, timeout: float = 20.0) -> List[Tuple[date, float]]:
    with urllib.request.urlopen(FRED_URL.format(code=code), timeout=timeout) as r:
        return parse_fred_csv(r.read().decode('utf-8', 'replace'))


def current_rates(currencies: Sequence[str], cfg: FxConfig, cache_path: str,
                  today: Optional[date] = None,
                  fetch: Optional[Callable[[str], List[Tuple[date, float]]]] = None,
                  refresh_days: int = 7) -> Dict[str, Any]:
    """Each currency's 3-month interest rate in percent, and where it came
    from: the latest FRED month when it is under rate_max_age_days old,
    else fallback_rates_pct. Downloaded at most every `refresh_days` and
    kept in `cache_path`, so a FRED outage leaves the last good figures."""
    today = today or datetime.now(timezone.utc).date()
    fetch = fetch or (lambda c: fetch_fred(c))
    cache: Dict[str, Any] = {}
    try:
        with open(cache_path, encoding='utf-8') as f:
            cache = json.load(f) or {}
    except (OSError, ValueError):
        cache = {}
    fetched = cache.get('fetched_on')
    stale = not fetched or (today - date.fromisoformat(fetched)).days >= refresh_days
    series = cache.get('series') or {}
    if stale:
        for ccy in currencies:
            code = FRED_SERIES.get(ccy)
            if not code:
                continue
            try:
                rows = fetch(code)
                if rows:
                    month, value = rows[-1]
                    series[ccy] = {'month': month.isoformat(), 'pct': value,
                                   'history': [[m.isoformat(), v] for m, v in rows[-15:]]}
            except Exception as e:
                logger.warning(f"Interest rate for {ccy} not fetched from FRED: {e}")
        cache = {'fetched_on': today.isoformat(), 'series': series}
        try:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with open(cache_path + '.tmp', 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2)
            os.replace(cache_path + '.tmp', cache_path)
        except OSError as e:
            logger.warning(f"Interest-rate cache not saved: {e}")
    rates, sources, averages = {}, {}, {}
    for ccy in currencies:
        row = series.get(ccy)
        if row and (today - date.fromisoformat(row['month'])).days <= cfg.rate_max_age_days:
            rates[ccy], sources[ccy] = float(row['pct']), f"FRED {row['month'][:7]}"
        elif ccy in cfg.fallback_rates_pct:
            rates[ccy], sources[ccy] = cfg.fallback_rates_pct[ccy], 'config fallback_rates_pct'
        else:
            continue
        averages[ccy] = average_rate(row, rates[ccy], today)
    return {'rates_pct': rates, 'sources': sources, 'avg_12m_pct': averages}


def average_rate(row: Optional[Dict[str, Any]], current: float, today: date) -> float:
    """A currency's average rate over the last twelve months, for the
    interest part of the trend's excess return: the published months in that
    year, with each month missing from FRED (a series that stopped, or not
    yet published) counted at `current`."""
    published = {}
    for month, pct in (row or {}).get('history') or []:
        try:
            published[month[:7]] = float(pct)
        except (TypeError, ValueError):
            continue
    values = []
    y, m = today.year, today.month
    for _ in range(12):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        values.append(published.get(f"{y:04d}-{m:02d}", current))
    return sum(values) / len(values)
