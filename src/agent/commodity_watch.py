"""Watch-only commodities: learn their patterns from live prices without
risking a cent.

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
SOURCES = ('trend_200d', 'momentum_12m', 'seasonal')
BACKFILL, LIVE = 'backfill', 'live'
TREND_DAYS = 200
MOMENTUM_DAYS = 252
SEASONAL_MIN_YEARS = 5
SEASONAL_SHARE = 0.7

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


def settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = ((((config or {}).get('watch_only') or {}).get('commodities')) or {})
    return {**DEFAULTS, **{k: v for k, v in cfg.items() if not str(k).startswith('_') and k != 'instruments'}}


def instruments(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """The watch list: the one in config.json when it has one, else the built-in."""
    listed = ((((config or {}).get('watch_only') or {}).get('commodities')) or {}).get('instruments')
    out = []
    for i in (listed if isinstance(listed, list) and listed else INSTRUMENTS):
        if isinstance(i, dict) and i.get('symbol'):
            out.append({'symbol': str(i['symbol']), 'name': str(i.get('name') or i['symbol']),
                        'group': str(i.get('group') or 'other'), 'kind': str(i.get('kind') or 'futures')})
    return out


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


def study(rows: Iterable[Dict[str, Any]], horizon: int = HORIZON, min_signals: int = 30,
          cost_pct: float = 0.1) -> List[Dict[str, Any]]:
    """The skill of each pattern in each group, and across all commodities, with
    a verdict. A pattern is a 'candidate' only when it beats chance by two
    standard errors AND beats it in both the earlier and the later half of the
    history, which weeds out a pattern that worked in one decade only."""
    from src.agent.signal_ledger import skill
    rows = [r for r in rows if int(r['horizon']) == int(horizon)]
    pooled = [{**r, 'market': 'all_commodities'} for r in rows]
    table = skill(rows, horizon, min_signals) + skill(pooled, horizon, min_signals)
    days = sorted(str(r['day']) for r in rows)
    mid = days[len(days) // 2] if days else ''
    out = []
    for t in table:
        grp = [r for r in (pooled if t['market'] == 'all_commodities' else rows)
               if r['source'] == t['source'] and (t['market'] == 'all_commodities' or r.get('market') == t['market'])]
        halves = []
        for part in ([r for r in grp if str(r['day']) < mid], [r for r in grp if str(r['day']) >= mid]):
            halves.append(round(sum(1 for r in part if r['ret'] > 0) / len(part) - mean(r['chance'] for r in part), 3)
                          if len(part) >= 5 else None)
        out.append({**t, 'tilt': 1.0, 'early_lift': halves[0], 'late_lift': halves[1],
                    'net_move_pct': round(t['avg_move_pct'] - cost_pct, 2),
                    'verdict': _verdict(t['n'], t['z'], halves[0], halves[1], min_signals)})
    return out


def run_study(closes_by_symbol: Dict[str, Dict[date, float]], meta: Dict[str, Dict[str, str]],
              horizon: int = HORIZON, min_signals: int = 30, cost_pct: float = 0.1,
              ) -> Dict[str, Any]:
    """Replay the patterns over history and score them, from daily closes."""
    from src.agent.signal_ledger import outcomes_for
    sigs = []
    for sym, closes in closes_by_symbol.items():
        sigs += signals_for(sym, (meta.get(sym) or {}).get('group', 'other'), closes, BACKFILL)
    rows = outcomes_for(sigs, lambda s: closes_by_symbol.get(s, {}), (horizon,))
    return {'signals': len(sigs), 'scored': len(rows), 'instruments': len(closes_by_symbol),
            'table': study(rows, horizon, min_signals, cost_pct)}


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
        self.started = self._started_day()

    # -- when the watch began: the line between replayed history and live evidence
    def _started_day(self) -> date:
        try:
            with open(self.state_path, encoding='utf-8') as f:
                return date.fromisoformat(json.load(f)['started'])
        except (OSError, ValueError, KeyError):
            today = datetime.fromtimestamp(self._clock(), tz=timezone.utc).date()
            try:
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump({'started': today.isoformat()}, f)
            except OSError as e:
                logger.warning(f"Could not save the commodity watch start date: {e}")
            return today

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
                closes = self.closes_for(item['symbol']) or {}
                if not closes:
                    continue
                closes_by[item['symbol']] = closes
                snaps.append({**snapshot(item['symbol'], closes), 'name': item['name'],
                              'group': item['group'], 'kind': item['kind']})
            sigs = []
            for sym, closes in closes_by.items():
                for s in signals_for(sym, meta[sym]['group'], closes):
                    s['version'] = BACKFILL if s['day'] < self.started else LIVE
                    sigs.append(s)
            rows = outcomes_for(sigs, lambda s: closes_by.get(s, {}), (5, HORIZON))
            result = {'instruments': len(closes_by), 'new_outcomes': self.ledger.record(rows)}
            with open(self.snapshot_path, 'w', encoding='utf-8') as f:
                json.dump({'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                           'started': self.started.isoformat(), 'instruments': snaps}, f)
            if result['new_outcomes']:
                logger.info(f"Commodity watch: {result['new_outcomes']} new outcomes from "
                            f"{result['instruments']} commodities (watch only, no orders)")
        except Exception as e:
            logger.error(f"Commodity watch failed: {e}")
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
           'horizon': HORIZON, 'instruments': snap.get('instruments') or [], 'assumed_cost_pct': rule['assumed_cost_pct']}
    for label in (BACKFILL, LIVE):
        part = [r for r in rows if r.get('version') == label]
        out[label] = {'signals': len({(r['symbol'], r['day'], r['source'], r['action']) for r in part}),
                      'table': study(part, HORIZON, int(rule['min_signals']), float(rule['assumed_cost_pct']))}
    return out
