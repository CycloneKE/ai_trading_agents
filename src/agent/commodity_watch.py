"""Watch-only markets: commodities, and the UK, Japanese, French and South
African stock markets. Learn their patterns from live prices without risking
a cent.

The agent trades stocks, crypto, currencies and Kenyan shares. It has never
looked at metals, oil or farm produce, and it cannot trade them directly
(Alpaca offers no commodities). This watches them anyway, and puts three
well-known patterns to the test against what each commodity then did:

* trend_200d    price crosses its 200-day average (up is a buy, down a sell).
* momentum_12m  on the first trading day of each month, the commodity is
                bought if it rose over the past 12 months and sold if it fell.
* seasonal      on the first trading day of each month, bought if the same
                calendar month has risen in at least 70% of at least five
                earlier years, sold if it has fallen in at least 70%.

Every signal is scored against the price 5 and 20 trading days later and
against how often the commodity simply went up anyway (the same measure the
agent uses for its own strategies, signal_ledger.py), so a rising market does
not make every buy look clever.

Two kinds of evidence are kept apart and never mixed:

* "backfill": signals on days before the watch began, replayed over years of
  history. The seasonal pattern only ever uses years before the signal day, so
  it is never allowed to peek ahead. This is quick, but a pattern found by
  searching many ideas can look good by luck, so a backfill result is a
  candidate, not proof.
* "live": signals on days after the watch began, scored as the weeks go by.
  Only this confirms a pattern.

This module has no connection to any broker and nothing here places an order.
The commodities it watches are deliberately not in data_manager.symbols, so the
trading loop never sees them. Whether a pattern is ever traded is a separate
decision for the operator, after the live record is in.
"""
import json
import logging
import os
import threading
import time
from bisect import bisect_right
from datetime import date, datetime, timezone
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

HORIZON = 20                      # the pattern signals are monthly; judged a month on
HORIZONS = (20, 60, 120)          # also judged a quarter and half a year on (about 1, 3 and 6 months)
SOURCES = ('trend_200d', 'momentum_12m', 'seasonal')
BACKFILL, LIVE = 'backfill', 'live'
TREND_DAYS = 200
MOMENTUM_DAYS = 252
SEASONAL_MIN_YEARS = 5
SEASONAL_SHARE = 0.7

EQUITY_DEFAULTS: Dict[str, Any] = {
    'enabled': True,
    # Assumed cost of one round trip, per market, as a percentage: the market's
    # tax on purchases plus an allowance for spread and commission. Checked
    # against public sources on 18 October 2026:
    #   UK            stamp duty reserve tax 0.5% on purchases (HMRC 2026/27 table), plus a
    #                 PTM levy of GBP 1.50 on trades over GBP 10,000; the tax is to be
    #                 replaced in 2027 by a securities transfer tax whose rate was not found.
    #   France        financial transaction tax 0.4% on purchases of shares of French
    #                 companies worth over EUR 1 billion (raised from 0.3% in 2025). Funds
    #                 are not subject to it.
    #   South Africa  securities transfer tax 0.25% (SARS), plus settlement and levy fees of
    #                 about 0.006% (undated JSE page).
    #   Japan         no securities transaction tax is believed to exist, but that was NOT
    #                 confirmed from a source.
    # The allowance for broker commission and spread (about 0.1% a side, more in
    # Johannesburg) is an assumption, not any broker's schedule. Replace these with
    # your own broker's costs; config.json can change them.
    'costs_pct': {'uk_equities': 0.7, 'japan_equities': 0.3, 'france_equities': 0.6,
                  'south_africa_equities': 0.7},
}

DEFAULTS: Dict[str, Any] = {
    'enabled': True,
    'refresh_hours': 12,
    'history_period': 'max',      # a yfinance period: how much history to replay
    'min_signals': 30,
    'assumed_cost_pct': 0.1,      # taken off each signal's average move; a fund's spread and fees
}

# What is watched, by group. Futures give the cleanest long history of the
# commodity itself but are data only (Yahoo's continuous contract jumps when it
# rolls to the next month); the funds are what the US book could actually trade,
# and the oil and gas funds lose value just from rolling contracts, so judge
# them by their own price, never by the commodity's.
INSTRUMENTS: List[Dict[str, str]] = [
    {'symbol': 'GC=F', 'name': 'Gold futures', 'group': 'metals', 'kind': 'futures'},
    {'symbol': 'SI=F', 'name': 'Silver futures', 'group': 'metals', 'kind': 'futures'},
    {'symbol': 'PL=F', 'name': 'Platinum futures', 'group': 'metals', 'kind': 'futures'},
    {'symbol': 'PA=F', 'name': 'Palladium futures', 'group': 'metals', 'kind': 'futures'},
    {'symbol': 'HG=F', 'name': 'Copper futures', 'group': 'metals', 'kind': 'futures'},
    {'symbol': 'CL=F', 'name': 'US oil (WTI) futures', 'group': 'energy', 'kind': 'futures'},
    {'symbol': 'BZ=F', 'name': 'Brent oil futures', 'group': 'energy', 'kind': 'futures'},
    {'symbol': 'NG=F', 'name': 'Natural gas futures', 'group': 'energy', 'kind': 'futures'},
    {'symbol': 'ZW=F', 'name': 'Wheat futures', 'group': 'agriculture', 'kind': 'futures'},
    {'symbol': 'ZC=F', 'name': 'Corn futures', 'group': 'agriculture', 'kind': 'futures'},
    {'symbol': 'ZS=F', 'name': 'Soybean futures', 'group': 'agriculture', 'kind': 'futures'},
    {'symbol': 'KC=F', 'name': 'Coffee futures', 'group': 'agriculture', 'kind': 'futures'},
    {'symbol': 'CC=F', 'name': 'Cocoa futures', 'group': 'agriculture', 'kind': 'futures'},
    {'symbol': 'SB=F', 'name': 'Sugar futures', 'group': 'agriculture', 'kind': 'futures'},
    {'symbol': 'GLD', 'name': 'Gold fund', 'group': 'metals', 'kind': 'fund'},
    {'symbol': 'SLV', 'name': 'Silver fund', 'group': 'metals', 'kind': 'fund'},
    {'symbol': 'CPER', 'name': 'Copper fund', 'group': 'metals', 'kind': 'fund'},
    {'symbol': 'USO', 'name': 'US oil fund', 'group': 'energy', 'kind': 'fund'},
    {'symbol': 'UNG', 'name': 'Natural gas fund', 'group': 'energy', 'kind': 'fund'},
    {'symbol': 'DBA', 'name': 'Farm produce basket fund', 'group': 'agriculture', 'kind': 'fund'},
    {'symbol': 'WEAT', 'name': 'Wheat fund', 'group': 'agriculture', 'kind': 'fund'},
    {'symbol': 'CORN', 'name': 'Corn fund', 'group': 'agriculture', 'kind': 'fund'},
    {'symbol': 'SOYB', 'name': 'Soybean fund', 'group': 'agriculture', 'kind': 'fund'},
]

# Stock markets watched, by market: the index (data only), an index fund, and
# several of the market's largest companies. Yahoo Finance tickers; one that
# returns no prices is skipped, and the dashboard shows which. Two cautions that
# apply to all of them. First, these are TODAY's large companies, so a replay
# over their history is flattered by survivorship (the ones that fell out of the
# index are missing); the index and index-fund series carry no such flattery and
# are the fairer test. Second, prices are in each market's own currency and unit
# (London quotes in pence, Johannesburg in cents), so a result is the move in
# local currency, before the exchange rate a dollar-based investor would also bear.
EQUITY_INSTRUMENTS: List[Dict[str, str]] = [
    {'symbol': '^FTSE', 'name': 'FTSE 100 index', 'group': 'uk_equities', 'kind': 'index', 'quote': 'points'},
    {'symbol': 'ISF.L', 'name': 'FTSE 100 index fund', 'group': 'uk_equities', 'kind': 'fund', 'quote': 'pence'},
    {'symbol': 'HSBA.L', 'name': 'HSBC', 'group': 'uk_equities', 'kind': 'stock', 'quote': 'pence'},
    {'symbol': 'SHEL.L', 'name': 'Shell', 'group': 'uk_equities', 'kind': 'stock', 'quote': 'pence'},
    {'symbol': 'AZN.L', 'name': 'AstraZeneca', 'group': 'uk_equities', 'kind': 'stock', 'quote': 'pence'},
    {'symbol': 'ULVR.L', 'name': 'Unilever', 'group': 'uk_equities', 'kind': 'stock', 'quote': 'pence'},
    {'symbol': 'BP.L', 'name': 'BP', 'group': 'uk_equities', 'kind': 'stock', 'quote': 'pence'},
    {'symbol': 'GSK.L', 'name': 'GSK', 'group': 'uk_equities', 'kind': 'stock', 'quote': 'pence'},
    {'symbol': 'RIO.L', 'name': 'Rio Tinto', 'group': 'uk_equities', 'kind': 'stock', 'quote': 'pence'},
    {'symbol': 'LLOY.L', 'name': 'Lloyds Banking Group', 'group': 'uk_equities', 'kind': 'stock', 'quote': 'pence'},
    {'symbol': 'BARC.L', 'name': 'Barclays', 'group': 'uk_equities', 'kind': 'stock', 'quote': 'pence'},
    {'symbol': '^N225', 'name': 'Nikkei 225 index', 'group': 'japan_equities', 'kind': 'index', 'quote': 'points'},
    {'symbol': '1321.T', 'name': 'Nikkei 225 index fund', 'group': 'japan_equities', 'kind': 'fund', 'quote': 'yen'},
    {'symbol': '1306.T', 'name': 'TOPIX index fund', 'group': 'japan_equities', 'kind': 'fund', 'quote': 'yen'},
    {'symbol': '7203.T', 'name': 'Toyota', 'group': 'japan_equities', 'kind': 'stock', 'quote': 'yen'},
    {'symbol': '6758.T', 'name': 'Sony', 'group': 'japan_equities', 'kind': 'stock', 'quote': 'yen'},
    {'symbol': '9984.T', 'name': 'SoftBank Group', 'group': 'japan_equities', 'kind': 'stock', 'quote': 'yen'},
    {'symbol': '8306.T', 'name': 'Mitsubishi UFJ Financial', 'group': 'japan_equities', 'kind': 'stock', 'quote': 'yen'},
    {'symbol': '6861.T', 'name': 'Keyence', 'group': 'japan_equities', 'kind': 'stock', 'quote': 'yen'},
    {'symbol': '9432.T', 'name': 'NTT', 'group': 'japan_equities', 'kind': 'stock', 'quote': 'yen'},
    {'symbol': '8035.T', 'name': 'Tokyo Electron', 'group': 'japan_equities', 'kind': 'stock', 'quote': 'yen'},
    {'symbol': '^FCHI', 'name': 'CAC 40 index', 'group': 'france_equities', 'kind': 'index', 'quote': 'points'},
    {'symbol': 'CAC.PA', 'name': 'CAC 40 index fund', 'group': 'france_equities', 'kind': 'fund', 'quote': 'euro'},
    {'symbol': 'MC.PA', 'name': 'LVMH', 'group': 'france_equities', 'kind': 'stock', 'quote': 'euro'},
    {'symbol': 'TTE.PA', 'name': 'TotalEnergies', 'group': 'france_equities', 'kind': 'stock', 'quote': 'euro'},
    {'symbol': 'SAN.PA', 'name': 'Sanofi', 'group': 'france_equities', 'kind': 'stock', 'quote': 'euro'},
    {'symbol': 'AIR.PA', 'name': 'Airbus', 'group': 'france_equities', 'kind': 'stock', 'quote': 'euro'},
    {'symbol': 'OR.PA', 'name': "L'Oreal", 'group': 'france_equities', 'kind': 'stock', 'quote': 'euro'},
    {'symbol': 'BNP.PA', 'name': 'BNP Paribas', 'group': 'france_equities', 'kind': 'stock', 'quote': 'euro'},
    {'symbol': 'SU.PA', 'name': 'Schneider Electric', 'group': 'france_equities', 'kind': 'stock', 'quote': 'euro'},
    {'symbol': 'CS.PA', 'name': 'AXA', 'group': 'france_equities', 'kind': 'stock', 'quote': 'euro'},
    {'symbol': '^J203.JO', 'name': 'FTSE/JSE All Share index', 'group': 'south_africa_equities', 'kind': 'index', 'quote': 'points'},
    {'symbol': 'STX40.JO', 'name': 'Satrix 40 index fund', 'group': 'south_africa_equities', 'kind': 'fund', 'quote': 'cents'},
    {'symbol': 'NPN.JO', 'name': 'Naspers', 'group': 'south_africa_equities', 'kind': 'stock', 'quote': 'cents'},
    {'symbol': 'SBK.JO', 'name': 'Standard Bank', 'group': 'south_africa_equities', 'kind': 'stock', 'quote': 'cents'},
    {'symbol': 'FSR.JO', 'name': 'FirstRand', 'group': 'south_africa_equities', 'kind': 'stock', 'quote': 'cents'},
    {'symbol': 'AGL.JO', 'name': 'Anglo American', 'group': 'south_africa_equities', 'kind': 'stock', 'quote': 'cents'},
    {'symbol': 'MTN.JO', 'name': 'MTN Group', 'group': 'south_africa_equities', 'kind': 'stock', 'quote': 'cents'},
    {'symbol': 'SOL.JO', 'name': 'Sasol', 'group': 'south_africa_equities', 'kind': 'stock', 'quote': 'cents'},
    {'symbol': 'CFR.JO', 'name': 'Richemont', 'group': 'south_africa_equities', 'kind': 'stock', 'quote': 'cents'},
    {'symbol': 'GFI.JO', 'name': 'Gold Fields', 'group': 'south_africa_equities', 'kind': 'stock', 'quote': 'cents'},
]


def settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The watch's settings. `enabled` is true when either the commodities or the
    equity markets are on; `commodities_enabled` and `equities_enabled` say which."""
    cfg = ((((config or {}).get('watch_only') or {}).get('commodities')) or {})
    out = {**DEFAULTS, **{k: v for k, v in cfg.items() if not str(k).startswith('_') and k != 'instruments'}}
    eq = equity_settings(config)
    out['commodities_enabled'] = bool(out['enabled'])
    out['equities_enabled'] = bool(eq['enabled'])
    out['enabled'] = out['commodities_enabled'] or out['equities_enabled']
    return out


def equity_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = ((((config or {}).get('watch_only') or {}).get('equity_markets')) or {})
    out = {**EQUITY_DEFAULTS, **{k: v for k, v in cfg.items() if not str(k).startswith('_')
                                 and k not in ('instruments', 'costs_pct')}}
    out['costs_pct'] = {**EQUITY_DEFAULTS['costs_pct'], **{k: float(v) for k, v in (cfg.get('costs_pct') or {}).items()}}
    return out


def costs(config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """The assumed round-trip cost, in percent, by group; groups without one use
    the commodities' `assumed_cost_pct`."""
    return dict(equity_settings(config)['costs_pct'])


def family(group: str) -> str:
    """'equities' for a stock market group, else 'commodities'."""
    return 'equities' if str(group).endswith('_equities') else 'commodities'


def _items(listed: Any, builtin: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for i in (listed if isinstance(listed, list) and listed else builtin):
        if isinstance(i, dict) and i.get('symbol'):
            out.append({'symbol': str(i['symbol']), 'name': str(i.get('name') or i['symbol']),
                        'group': str(i.get('group') or 'other'), 'kind': str(i.get('kind') or 'futures'),
                        'quote': str(i.get('quote') or '')})
    return out


def instruments(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """The watch list: commodities (when on) then stock markets (when on), each
    from config.json when it has a list, else the built-in one."""
    wo = (config or {}).get('watch_only') or {}
    com, eq = wo.get('commodities') or {}, wo.get('equity_markets') or {}
    out: List[Dict[str, str]] = []
    if com.get('enabled', DEFAULTS['enabled']):
        out += _items(com.get('instruments'), INSTRUMENTS)
    if eq.get('enabled', EQUITY_DEFAULTS['enabled']):
        out += _items(eq.get('instruments'), EQUITY_INSTRUMENTS)
    return out


# ------------------------------------------------------------ data quality

def fix_unit_steps(v: List[float]) -> Tuple[List[float], int]:
    """Undo the 100-fold jumps Yahoo sometimes gives a price that switches between
    pounds and pence (or rand and cents). A real one-day move of 100 times never
    happens, so a step of about 100 or 1/100 between neighbouring closes is a unit
    change: everything before it is rescaled to match what follows. Returns the
    corrected prices and how many steps were found. Only the unit change (exactly
    100) is undone, so the genuine move on the day of the step is kept."""
    fixed, steps = list(v), 0
    factor = 1.0
    for i in range(len(v) - 1, 0, -1):
        r = v[i] / v[i - 1] if v[i - 1] else 1.0
        if 90 <= r <= 110 or 1 / 110 <= r <= 1 / 90:
            factor *= 100.0 if r > 1 else 0.01          # exactly the unit change; the day's real move stays
            steps += 1
        fixed[i - 1] = v[i - 1] * factor
    return fixed, steps


def clean_closes(closes: Dict[date, float]) -> Tuple[Dict[date, float], int]:
    """The closes with unit steps undone, and how many were found."""
    days = sorted(d for d, c in closes.items() if c and c > 0)
    fixed, steps = fix_unit_steps([float(closes[d]) for d in days])
    return dict(zip(days, fixed)), steps


# ----------------------------------------------------------------- patterns

def _series(closes: Dict[date, float]) -> Tuple[List[date], List[float]]:
    days = sorted(d for d, c in closes.items() if c and c > 0)
    return days, [float(closes[d]) for d in days]


def _month_opens(days: List[date]) -> List[int]:
    """Index of the first trading day of each month in the series."""
    return [i for i in range(1, len(days)) if days[i].month != days[i - 1].month]


def trend_signals(days: List[date], v: List[float], n: int = TREND_DAYS) -> List[Dict[str, Any]]:
    """A buy on the day the close moves above its n-day average, a sell on the
    day it moves below."""
    out, total, prev = [], 0.0, None
    for i, x in enumerate(v):
        total += x
        if i >= n:
            total -= v[i - n]
        if i < n - 1:
            continue
        above = x > total / n
        if prev is not None and above != prev:
            out.append({'day': days[i], 'source': 'trend_200d', 'action': 'buy' if above else 'sell'})
        prev = above
    return out


def momentum_signals(days: List[date], v: List[float], n: int = MOMENTUM_DAYS) -> List[Dict[str, Any]]:
    """On the first trading day of each month: buy after a rise over the past
    n trading days, sell after a fall."""
    out = []
    for i in _month_opens(days):
        if i < n:
            continue
        r = v[i] / v[i - n] - 1
        if r != 0:
            out.append({'day': days[i], 'source': 'momentum_12m', 'action': 'buy' if r > 0 else 'sell'})
    return out


def _seasonal_table(days: List[date], v: List[float]) -> List[Tuple[int, int, int, float]]:
    """(year, month, index of the month's first day, the month's return) for
    each complete month; the return is known only at the next month's first day."""
    opens = _month_opens(days)
    return [(days[a].year, days[a].month, b, v[b] / v[a] - 1) for a, b in zip(opens, opens[1:])]


def seasonal_signals(days: List[date], v: List[float], min_years: int = SEASONAL_MIN_YEARS,
                     share: float = SEASONAL_SHARE) -> List[Dict[str, Any]]:
    """On the first trading day of each month, the signal that month's own
    history gives, using only earlier years (nothing is peeked at)."""
    table = _seasonal_table(days, v)
    by_month: Dict[int, List[Tuple[int, int, float]]] = {}
    for year, month, known_at, r in table:
        by_month.setdefault(month, []).append((year, known_at, r))
    out = []
    for i in _month_opens(days):
        month, year = days[i].month, days[i].year
        past = [r for (y, known_at, r) in by_month.get(month, []) if y < year and known_at <= i]
        if len(past) < min_years:
            continue
        up = sum(1 for r in past if r > 0) / len(past)
        if up >= share:
            out.append({'day': days[i], 'source': 'seasonal', 'action': 'buy'})
        elif up <= 1 - share:
            out.append({'day': days[i], 'source': 'seasonal', 'action': 'sell'})
    return out


def signals_for(symbol: str, group: str, closes: Dict[date, float], version: Optional[str] = None
                ) -> List[Dict[str, Any]]:
    """Every pattern signal in `closes`, in the shape the signal ledger scores."""
    days, v = _series(closes)
    if len(days) < TREND_DAYS + 2:
        return []
    out = []
    for s in trend_signals(days, v) + momentum_signals(days, v) + seasonal_signals(days, v):
        out.append({**s, 'symbol': symbol, 'market': group, 'version': version})
    return out


def snapshot(symbol: str, closes: Dict[date, float]) -> Dict[str, Any]:
    """What each pattern says about a commodity right now."""
    days, v = _series(closes)
    if len(days) < TREND_DAYS + 2:
        return {'symbol': symbol, 'enough_history': False, 'bars': len(days)}
    sma = sum(v[-TREND_DAYS:]) / TREND_DAYS
    snap: Dict[str, Any] = {
        'symbol': symbol, 'enough_history': True, 'bars': len(days), 'last_day': days[-1].isoformat(),
        'last_close': round(v[-1], 4), 'above_200d': v[-1] > sma,
        'pct_from_200d': round((v[-1] / sma - 1) * 100, 2),
        'momentum_12m_pct': round((v[-1] / v[-1 - MOMENTUM_DAYS] - 1) * 100, 2) if len(v) > MOMENTUM_DAYS else None,
        'seasonal': None,
    }
    month, year = days[-1].month, days[-1].year
    past = [r for (y, m, _k, r) in _seasonal_table(days, v) if m == month and y < year]
    if past:
        up = sum(1 for r in past if r > 0) / len(past)
        reading = 'none'
        if len(past) >= SEASONAL_MIN_YEARS:
            reading = 'buy' if up >= SEASONAL_SHARE else 'sell' if up <= 1 - SEASONAL_SHARE else 'none'
        snap['seasonal'] = {'month': month, 'years': len(past), 'share_up': round(up, 2),
                            'avg_pct': round(mean(past) * 100, 2), 'reading': reading}
    return snap


# --------------------------------------------------------------------- study

def _verdict(n: int, z: float, early_lift: Optional[float], late_lift: Optional[float], min_signals: int) -> str:
    if n < min_signals:
        return 'too_few'
    if z >= 2 and (early_lift or 0) > 0 and (late_lift or 0) > 0:
        return 'candidate'
    if z <= -2 and (early_lift or 0) < 0 and (late_lift or 0) < 0:
        return 'wrong_way'
    return 'no_clear_edge'


def _as_day(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def thin(rows: Iterable[Dict[str, Any]], horizon: int) -> List[Dict[str, Any]]:
    """Keep, for each instrument and pattern, only signals whose outcome window
    does not overlap an earlier kept one.

    The patterns signal about once a month. Judged 60 or 120 trading days on,
    each outcome window covers the next two to five signals, so counting every
    signal would count the same stretch of price several times and make a
    little luck look like skill (z inflated by roughly the square root of the
    overlap). A signal is set aside if the previous kept one, for the same
    instrument and pattern, is closer than the horizon (taken as 7 calendar days
    per 5 trading days). A trend signal that whipsaws, or a monthly one after a
    longer horizon's first, is counted once."""
    gap = -(-int(horizon) * 7 // 5)
    last: Dict[Tuple[str, str], date] = {}
    out = []
    for r in sorted(rows, key=lambda r: str(r['day'])):
        key, day = (r['symbol'], r['source']), _as_day(r['day'])
        if key in last and (day - last[key]).days < gap:
            continue
        last[key] = day
        out.append(r)
    return out


def study(rows: Iterable[Dict[str, Any]], horizon: int = HORIZON, min_signals: int = 30,
          cost_pct: float = 0.1, group_costs: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """The skill of each pattern in each group, and across all commodities and
    all stock markets, with a verdict. A pattern is a 'candidate' only when it
    beats chance by two standard errors AND beats it in both the earlier and the
    later half of the history, which weeds out a pattern that worked in one
    decade only. Signals whose outcome window overlaps an earlier one are counted
    once (see `thin`). `group_costs` gives each group's assumed round-trip cost in
    percent; a group without one uses `cost_pct`."""
    from src.agent.signal_ledger import skill
    group_costs = group_costs or {}
    rows = thin([r for r in rows if int(r['horizon']) == int(horizon)], horizon)
    pooled = [{**r, 'market': f"all_{family(r.get('market') or '')}"} for r in rows]
    table = skill(rows, horizon, min_signals) + skill(pooled, horizon, min_signals)
    days = sorted(str(r['day']) for r in rows)
    mid = days[len(days) // 2] if days else ''
    out = []
    for t in table:
        is_pool = str(t['market']).startswith('all_')
        grp = [r for r in (pooled if is_pool else rows)
               if r['source'] == t['source'] and r.get('market') == t['market']]
        halves = []
        for part in ([r for r in grp if str(r['day']) < mid], [r for r in grp if str(r['day']) >= mid]):
            halves.append(round(sum(1 for r in part if r['ret'] > 0) / len(part) - mean(r['chance'] for r in part), 3)
                          if len(part) >= 5 else None)
        if is_pool:
            cost = mean(group_costs.get(g, cost_pct) for g in
                        (r['market'] for r in rows if r['source'] == t['source']
                         and f"all_{family(r['market'])}" == t['market'])) if grp else cost_pct
        else:
            cost = group_costs.get(t['market'], cost_pct)
        out.append({**t, 'tilt': 1.0, 'early_lift': halves[0], 'late_lift': halves[1],
                    'assumed_cost_pct': round(cost, 2), 'net_move_pct': round(t['avg_move_pct'] - cost, 2),
                    'verdict': _verdict(t['n'], t['z'], halves[0], halves[1], min_signals)})
    return out


def run_study(closes_by_symbol: Dict[str, Dict[date, float]], meta: Dict[str, Dict[str, str]],
              horizon: int = HORIZON, min_signals: int = 30, cost_pct: float = 0.1,
              group_costs: Optional[Dict[str, float]] = None,
              horizons: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    """Replay the patterns over history and score them, from daily closes, at
    `horizon` (the headline, in `table`) and at every one of `horizons`
    (`tables`, keyed by horizon)."""
    from src.agent.signal_ledger import outcomes_for
    wanted = list(dict.fromkeys([int(horizon)] + [int(h) for h in (horizons or ())]))
    cleaned, fixes = {}, {}
    for sym, closes in closes_by_symbol.items():
        cleaned[sym], n = clean_closes(closes)
        if n:
            fixes[sym] = n
    sigs = []
    for sym, closes in cleaned.items():
        sigs += signals_for(sym, (meta.get(sym) or {}).get('group', 'other'), closes, BACKFILL)
    rows = outcomes_for(sigs, lambda s: cleaned.get(s, {}), wanted)
    tables = {h: study(rows, h, min_signals, cost_pct, group_costs) for h in wanted}
    return {'signals': len(sigs), 'scored': len([r for r in rows if int(r['horizon']) == int(horizon)]),
            'instruments': len(closes_by_symbol), 'unit_fixes': fixes, 'table': tables[int(horizon)],
            'tables': tables}


# ----------------------------------------------------------------- the job

def default_closes_provider(period: str = 'max') -> Callable[[str], Dict[date, float]]:
    def closes_for(symbol: str) -> Dict[date, float]:
        from src.agent import chart_data
        from src.agent.scorecard import bars_to_closes
        try:
            return bars_to_closes(chart_data.yfinance_bars(symbol, period=period))
        except Exception as e:
            logger.debug(f"No daily closes for {symbol}: {e}")
            return {}
    return closes_for


class CommodityWatch:
    """Keeps the watch-only record up to date in the background. Its files are
    its own (watch_outcomes.db, watch_state.json), never the trading journals."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, data_dir: Optional[str] = None,
                 closes_for: Optional[Callable[[str], Dict[date, float]]] = None,
                 clock: Callable[[], float] = time.time):
        from src.agent.signal_ledger import SignalLedger
        self.config = config or {}
        self.rule = settings(self.config)
        self.items = instruments(self.config)
        self.dir = str(data_dir or DATA_DIR)
        os.makedirs(self.dir, exist_ok=True)
        self.ledger = SignalLedger(os.path.join(self.dir, 'watch_outcomes.db'))
        self.state_path = os.path.join(self.dir, 'watch_state.json')
        self.snapshot_path = os.path.join(self.dir, 'watch_snapshot.json')
        self.closes_for = closes_for or default_closes_provider(str(self.rule['history_period']))
        self._clock = clock
        self._last = 0.0
        self._thread: Optional[threading.Thread] = None
        self.group_started: Dict[str, date] = {}
        self.started = self._started_day()

    # -- when the watch began: the line between replayed history and live evidence.
    # The commodities share one start date; each stock market has its own, set the
    # first time it returns prices, so a market added later is not credited with
    # "live" days before it was actually being watched.
    def _today(self) -> date:
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc).date()

    def _save_state(self) -> None:
        try:
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump({'started': self.started.isoformat(),
                           'groups': {g: d.isoformat() for g, d in self.group_started.items()}}, f)
        except OSError as e:
            logger.warning(f"Could not save the watch start dates: {e}")

    def _started_day(self) -> date:
        try:
            with open(self.state_path, encoding='utf-8') as f:
                state = json.load(f)
            for g, d in (state.get('groups') or {}).items():
                try:
                    self.group_started[g] = date.fromisoformat(d)
                except ValueError:
                    continue
            return date.fromisoformat(state['started'])
        except (OSError, ValueError, KeyError):
            today = self._today()
            self.started = today
            self._save_state()
            return today

    def start_for(self, group: str) -> date:
        """The day the watch began for a group (set now if this is its first run)."""
        if family(group) != 'equities':
            return self.started
        if group not in self.group_started:
            self.group_started[group] = self._today()
            self._save_state()
        return self.group_started[group]

    def due(self) -> bool:
        return bool(self.rule['enabled']) and self._clock() - self._last >= float(self.rule['refresh_hours']) * 3600

    def maybe_update(self) -> bool:
        """Start an update in the background if one is due. Never blocks the loop."""
        if (self._thread is not None and self._thread.is_alive()) or not self.due():
            return False
        self._last = self._clock()
        self._thread = threading.Thread(target=self.run, daemon=True, name='commodity-watch')
        self._thread.start()
        return True

    def run(self) -> Dict[str, Any]:
        from src.agent.signal_ledger import outcomes_for
        result: Dict[str, Any] = {'instruments': 0, 'new_outcomes': 0}
        try:
            closes_by, snaps = {}, []
            meta = {i['symbol']: i for i in self.items}
            for item in self.items:
                raw = self.closes_for(item['symbol']) or {}
                if not raw:
                    continue
                closes, fixes = clean_closes(raw)
                closes_by[item['symbol']] = closes
                snaps.append({**snapshot(item['symbol'], closes), 'name': item['name'], 'group': item['group'],
                              'kind': item['kind'], 'quote': item.get('quote', ''), 'unit_fixes': fixes})
            for group in sorted({meta[sym]['group'] for sym in closes_by}):
                self.start_for(group)                      # a market's first prices start its live record
            sigs = []
            for sym, closes in closes_by.items():
                start = self.start_for(meta[sym]['group'])
                for s in signals_for(sym, meta[sym]['group'], closes):
                    s['version'] = BACKFILL if s['day'] < start else LIVE
                    sigs.append(s)
            rows = outcomes_for(sigs, lambda s: closes_by.get(s, {}), (5,) + tuple(HORIZONS))
            result = {'instruments': len(closes_by), 'new_outcomes': self.ledger.record(rows)}
            with open(self.snapshot_path, 'w', encoding='utf-8') as f:
                json.dump({'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                           'started': self.started.isoformat(),
                           'started_by_group': {g: d.isoformat() for g, d in self.group_started.items()},
                           'instruments': snaps}, f)
            if result['new_outcomes']:
                logger.info(f"Market watch: {result['new_outcomes']} new outcomes from "
                            f"{result['instruments']} instruments (watch only, no orders)")
        except Exception as e:
            logger.error(f"Market watch failed: {e}")
        return result

    def close(self) -> None:
        self.ledger.close()


# ------------------------------------------------------------ for the API

def report(data_dir: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What the watch has found, read from its own files only (read-only, safe
    against a live agent)."""
    from src.agent.signal_ledger import read_ledger
    rule = settings(config)
    base = str(data_dir or DATA_DIR)
    rows = read_ledger(os.path.join(base, 'watch_outcomes.db'))
    try:
        with open(os.path.join(base, 'watch_snapshot.json'), encoding='utf-8') as f:
            snap = json.load(f)
    except (OSError, ValueError):
        snap = {}
    out = {'enabled': bool(rule['enabled']), 'started': snap.get('started'), 'updated_at': snap.get('updated_at'),
           'started_by_group': snap.get('started_by_group') or {}, 'horizon': HORIZON, 'horizons': list(HORIZONS),
           'instruments': snap.get('instruments') or [], 'assumed_cost_pct': rule['assumed_cost_pct'],
           'costs_pct': costs(config)}
    for label in (BACKFILL, LIVE):
        part = [r for r in rows if r.get('version') == label]
        out[label] = {'signals': len({(r['symbol'], r['day'], r['source'], r['action']) for r in part}),
                      'table': study(part, HORIZON, int(rule['min_signals']), float(rule['assumed_cost_pct']),
                                     costs(config)),
                      'tables': {str(h): study(part, h, int(rule['min_signals']), float(rule['assumed_cost_pct']),
                                               costs(config)) for h in HORIZONS}}
    return out
