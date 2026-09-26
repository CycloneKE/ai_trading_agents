"""Slower strategies from the gap analysis, one per market.

The three original strategies (momentum, mean reversion, RSI) are fast:
they react to days of price action, which costs little on a US broker and a
great deal on the NSE, where a round trip costs about 3.9%. These trade
rarely and only in the market they were designed for:

- crypto_trend: hold a coin only while it is above its 200-day average.
  Long-only, no leverage; the trading loop sizes it by volatility.
- nse_rotation: rank the NSE stocks by their return over about six months,
  skipping the latest month, and hold the leaders. Slow momentum, at a pace
  NSE costs allow.
- us_earnings_drift: after a company beats (or misses) its earnings
  estimate by a clear margin, its shares tend to keep drifting that way for
  weeks. Buys after a beat, exits after a miss.

Each votes only on its own market (`markets` in its settings, and the
`strategy_markets` map), and abstains, rather than voting "hold", when it
has no view: not enough history, no recent earnings report, a stock outside
the leaders. An abstaining strategy is left out of the ensemble entirely,
so it never waters down the others' votes.

Paper trading only, like everything else until the pilot passes.
"""
import json
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base_strategy import BaseStrategy
from .daily_bars import APPEND, REPLACE, classify

logger = logging.getLogger(__name__)

ABSTAIN = {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0, 'abstain': True}


def _abstain(why: str) -> Dict[str, Any]:
    return {**ABSTAIN, 'reason': why}


def _ramp(beyond: float, span: float) -> float:
    """0.5 at a threshold, rising to 1.0 at twice its distance (as TechnicalStrategy)."""
    return min(0.5 + 0.5 * max(beyond, 0.0) / span, 1.0) if span else 1.0


class DailyBarStrategy(BaseStrategy):
    """Keeps one close per trading day per symbol (see daily_bars.py)."""

    # Abstains until its history is deep enough, so a symbol short of that
    # history is not "unseeded": the agent does not keep refetching for it.
    abstains = True

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.markets = list(config.get('markets') or [])
        self.historical_data: Dict[str, List[float]] = {}
        self._bar_dates: Dict[str, str] = {}
        self.lookback_period = 1

    def _push(self, data: Dict[str, Any]) -> Optional[List[float]]:
        symbol = data.get('symbol', 'UNKNOWN')
        price = data.get('price') or data.get('close')
        if not price or price <= 0:
            return None
        hist = self.historical_data.setdefault(symbol, [])
        step = classify(self._bar_dates.get(symbol), data.get('bar_date'),
                        hist[-1] if hist else None, price)
        if step == APPEND:
            hist.append(float(price))
            if data.get('bar_date') is not None:
                self._bar_dates[symbol] = data['bar_date']
        elif step == REPLACE:
            hist[-1] = float(price)
        del hist[:-self.lookback_period]
        return hist

    def seed_history(self, symbol: str, closes: List[float]) -> int:
        cleaned = [float(c) for c in closes if c and c > 0]
        if not cleaned:
            return 0
        self.historical_data[symbol] = cleaned[-self.lookback_period:]
        self._bar_dates.pop(symbol, None)
        return len(self.historical_data[symbol])

    def update_model(self, data: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None):
        return True


class CryptoTrendStrategy(DailyBarStrategy):
    """Above the long average: hold. Below it: be in cash.

    A band around the average (2% by default) keeps a price hovering at the
    line from flipping the position every day.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.sma_period = int(config.get('sma_period', 200))
        self.band = float(config.get('band', 0.02))
        self.lookback_period = self.sma_period

    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        hist = self._push(data)
        if not hist or len(hist) < self.sma_period:
            return _abstain(f'needs {self.sma_period} daily bars')
        sma = sum(hist[-self.sma_period:]) / self.sma_period
        price = hist[-1]
        dist = price / sma - 1
        size = float(self.config.get('max_position_size', 0.05))
        if dist > self.band:
            conf = _ramp(dist - self.band, self.band)
            action = 'buy'
        elif dist < -self.band:
            conf = _ramp(-dist - self.band, self.band)
            action = 'sell'
        else:
            return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0,
                    'indicators': {'sma': sma, 'distance': dist}}
        return {'action': action, 'confidence': conf, 'position_size': conf * size,
                'indicators': {'sma': sma, 'distance': dist}}


class NseRotationStrategy(DailyBarStrategy):
    """Hold the NSE stocks that have risen most over about six months.

    Ranks every stock the strategy has enough history for by its return over
    `lookback` trading days, ending `skip` days ago (the latest month tends
    to reverse). The top `top_n` with a positive return are buys; a holding
    that falls out of the top `exit_rank`, or whose return turns negative,
    is a sell; stocks in between have no vote. The NSE paper account's
    30-day minimum holding and weekly limit on new holdings keep turnover
    down on top of this.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.lookback = int(config.get('lookback', 120))
        self.skip = int(config.get('skip', 20))
        self.top_n = int(config.get('top_n', 3))
        self.exit_rank = int(config.get('exit_rank', 2 * self.top_n))
        self.min_universe = int(config.get('min_universe', self.top_n + 2))
        self.lookback_period = self.lookback + self.skip + 1

    def score(self, hist: List[float]) -> Optional[float]:
        if len(hist) < self.lookback_period:
            return None
        end = hist[-1 - self.skip]
        start = hist[-1 - self.skip - self.lookback]
        return end / start - 1 if start > 0 else None

    def ranking(self) -> List[tuple]:
        scores = [(s, self.score(h)) for s, h in self.historical_data.items()]
        return sorted(((s, v) for s, v in scores if v is not None), key=lambda x: -x[1])

    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        symbol = data.get('symbol', 'UNKNOWN')
        if not self._push(data):
            return _abstain('no price')
        ranked = self.ranking()
        if len(ranked) < self.min_universe:
            return _abstain(f'{len(ranked)} stocks have {self.lookback_period} daily bars; '
                            f'{self.min_universe} needed to rank')
        rank = next((i + 1 for i, (s, _) in enumerate(ranked) if s == symbol), None)
        if rank is None:
            return _abstain(f'needs {self.lookback_period} daily bars')
        momentum = dict(ranked)[symbol]
        size = float(self.config.get('max_position_size', 0.05))
        info = {'rank': rank, 'of': len(ranked), 'momentum': momentum}
        if rank <= self.top_n and momentum > 0:
            conf = 1.0 - 0.5 * (rank - 1) / max(self.top_n, 1)   # 1.0 for the leader
            return {'action': 'buy', 'confidence': conf, 'position_size': conf * size,
                    'indicators': info}
        if rank > self.exit_rank or momentum < 0:
            return {'action': 'sell', 'confidence': 0.75, 'position_size': 0.0,
                    'indicators': info}
        return {**_abstain('between the leaders and the exit rank'), 'indicators': info}


class EarningsCalendar:
    """Recent earnings results from Finnhub, fetched once a day per symbol
    and kept in a small JSON cache so a restart does not refetch."""

    URL = 'https://finnhub.io/api/v1/calendar/earnings'

    def __init__(self, api_key: Optional[str], cache_path: Path,
                 fetch: Optional[Callable[[str, str, str], List[Dict[str, Any]]]] = None,
                 ttl_seconds: int = 86400):
        self.api_key = api_key
        self.cache_path = Path(cache_path)
        self.ttl = ttl_seconds
        self._fetch = fetch or self._finnhub
        self._custom_fetch = fetch is not None
        self._lock = threading.Lock()
        try:
            self._cache = json.loads(self.cache_path.read_text())
        except (OSError, ValueError):
            self._cache = {}

    @property
    def available(self) -> bool:
        return bool(self.api_key) or self._custom_fetch

    def _finnhub(self, symbol: str, start: str, end: str) -> List[Dict[str, Any]]:
        import requests
        r = requests.get(self.URL, params={'symbol': symbol, 'from': start, 'to': end,
                                           'token': self.api_key}, timeout=10)
        r.raise_for_status()
        return (r.json() or {}).get('earningsCalendar') or []

    def reports(self, symbol: str, today: date, days: int = 90) -> List[Dict[str, Any]]:
        """Reported results (actual and estimate known) in the last `days`, newest first."""
        with self._lock:
            entry = self._cache.get(symbol)
            fresh = entry and time.time() - entry.get('at', 0) < self.ttl
        if not fresh:
            try:
                rows = self._fetch(symbol, (today - timedelta(days=days)).isoformat(),
                                   today.isoformat())
            except Exception as e:
                logger.debug(f"Earnings calendar unavailable for {symbol}: {e}")
                rows = (entry or {}).get('rows', [])
            with self._lock:
                self._cache[symbol] = {'at': time.time(), 'rows': rows}
                try:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    self.cache_path.write_text(json.dumps(self._cache))
                except OSError:
                    pass
            entry = self._cache[symbol]
        out = [r for r in entry.get('rows', [])
               if r.get('date') and r.get('epsActual') is not None and r.get('epsEstimate')]
        return sorted(out, key=lambda r: r['date'], reverse=True)


class EarningsDriftStrategy(BaseStrategy):
    """Buy after a clear earnings beat; exit after a clear miss.

    The surprise is (actual - estimate) / |estimate|. A beat of at least
    `min_surprise` makes a buy for `hold_days` calendar days after the
    report, fading as the drift window closes; a miss of the same size is a
    sell. Otherwise, or with no Finnhub key, it abstains.
    """

    abstains = True

    def __init__(self, name: str, config: Dict[str, Any], calendar: Optional[EarningsCalendar] = None):
        super().__init__(name, config)
        self.markets = list(config.get('markets') or [])
        self.min_surprise = float(config.get('min_surprise', 0.05))
        self.hold_days = int(config.get('hold_days', 40))
        self.lookback_period = 1
        if calendar is None:
            from src.utils.paths import DATA_DIR
            key = os.getenv('TRADING_FINNHUB_API_KEY') or os.getenv('FINNHUB_API_KEY')
            calendar = EarningsCalendar(key, DATA_DIR / 'earnings_cache.json')
        self.calendar = calendar

    def generate_signals(self, data: Dict[str, Any], today: Optional[date] = None) -> Dict[str, Any]:
        if not self.calendar.available:
            return _abstain('no Finnhub API key')
        symbol = data.get('symbol', 'UNKNOWN')
        today = today or datetime.now(timezone.utc).date()
        reports = self.calendar.reports(symbol, today, days=self.hold_days + 7)
        if not reports:
            return _abstain('no recent earnings report')
        latest = reports[0]
        try:
            reported = date.fromisoformat(latest['date'][:10])
            actual, estimate = float(latest['epsActual']), float(latest['epsEstimate'])
        except (ValueError, TypeError):
            return _abstain('unreadable earnings report')
        age = (today - reported).days
        if age < 0 or age > self.hold_days or not estimate:
            return _abstain('outside the drift window')
        surprise = (actual - estimate) / abs(estimate)
        info = {'reported': latest['date'], 'surprise': surprise, 'days_since': age}
        if abs(surprise) < self.min_surprise:
            return {**_abstain('surprise too small'), 'indicators': info}
        # Strongest right after the report, fading to half by the end of
        # the window; a modest surprise drops below the ensemble's
        # confidence gate as the drift window closes.
        conf = _ramp(abs(surprise) - self.min_surprise, self.min_surprise) * \
            (1 - 0.5 * age / self.hold_days)
        size = float(self.config.get('max_position_size', 0.05))
        action = 'buy' if surprise > 0 else 'sell'
        return {'action': action, 'confidence': round(conf, 4),
                'position_size': round(conf * size, 4) if action == 'buy' else 0.0,
                'indicators': info}

    def update_model(self, data: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None):
        return True


SLOW_STRATEGIES = {
    'trend_following': CryptoTrendStrategy,
    'rotation': NseRotationStrategy,
    'earnings_drift': EarningsDriftStrategy,
}
