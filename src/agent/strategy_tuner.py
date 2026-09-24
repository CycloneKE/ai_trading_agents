"""Guarded re-tuning of the technical strategies' settings.

The ensemble already re-weights strategies by their realised results. That
changes how much each strategy is listened to, never what it does: the
momentum threshold, the mean-reversion band and the RSI levels were fixed
at whatever config.json said. An AutoML optimiser was created at startup and
never called.

This re-tunes those settings, carefully, because tuning to recent noise is
how an adaptive system loses money:

- When. Every `interval_days` (7) for every strategy, and sooner, at most
  once a day, for a strategy whose realised results have turned poor (win
  rate under `poor_win_rate` or a negative mean return, over at least
  `min_closed_trades` closed trades). That is where the learning comes in:
  results decide when a strategy's settings are questioned.
- How. The current settings and every setting one step either side (one
  parameter at a time, within fixed bounds) are run over the recent real
  daily history of every traded symbol with at least `min_bars` bars, long
  only, net of that market's round-trip costs (cost_model). Each symbol's
  trades are split by exit date into an earlier in-sample stretch and the
  latest `holdout_pct` held out.
- Adoption. A candidate replaces the current settings only if it beats them
  held out by at least `min_improvement` (return per symbol), is no worse
  in-sample, and made at least `min_trades` trades in each stretch. Of
  those, the best held-out result wins. Otherwise nothing changes.
- Record. Adopted settings are saved to data/strategy_params.json, applied
  at startup and at once, and every run is logged with its evidence.

What the simulation leaves out: the regime filter and the ensemble vote
(each strategy is tested alone), and adds and trims (entries and full exits
only). It judges a setting on its own signals, which is what is being set.
"""
import json
import logging
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.position_rules import rule

logger = logging.getLogger(__name__)

# family -> {config key: (strategy attribute, lower bound, upper bound, step)}
TUNABLE = {
    'momentum': {'threshold': ('momentum_threshold', 0.01, 0.06, 0.005)},
    'reversion': {'band': ('mean_reversion_band', 0.01, 0.06, 0.005)},
    'rsi': {'rsi_oversold': ('rsi_oversold', 20, 40, 5),
            'rsi_overbought': ('rsi_overbought', 60, 80, 5)},
}

DEFAULTS = {'enabled': True, 'interval_days': 7, 'holdout_pct': 0.3,
            'min_improvement': 0.005, 'min_trades': 5, 'min_bars': 120,
            'min_closed_trades': 10, 'poor_win_rate': 0.4}

Series = Dict[str, Tuple[List[float], float]]  # symbol -> (daily closes, round-trip cost)
Evaluation = Dict[str, Tuple[float, int]]       # 'is' / 'oos' -> (return per symbol, trades)


def family(name: str) -> Optional[str]:
    """The same dispatch TechnicalStrategy uses on the strategy's name."""
    if 'momentum' in name:
        return 'momentum'
    if 'reversion' in name:
        return 'reversion'
    if 'rsi' in name:
        return 'rsi'
    return None


def simulate(name: str, config: Dict[str, Any], params: Dict[str, Any],
             closes: List[float], round_trip: float) -> List[Tuple[int, float]]:
    """(exit bar, net return) for each trade one strategy makes on a series.

    Long only, as the live books are: buy when flat on a buy signal, sell
    when holding on a sell signal. A position still open at the end is
    closed at the last close, costs included, so a setting cannot look good
    by never selling.
    """
    from src.agent.technical_strategy import TechnicalStrategy
    strat = TechnicalStrategy(name, {**config, **params})
    trades, entry = [], None
    for i, close in enumerate(closes):
        action = strat.generate_signals({'symbol': 'TUNE', 'price': close}).get('action')
        if entry is None and action == 'buy':
            entry = close
        elif entry is not None and action == 'sell':
            trades.append((i, close / entry - 1 - round_trip))
            entry = None
    if entry is not None:
        trades.append((len(closes) - 1, closes[-1] / entry - 1 - round_trip))
    return trades


def evaluate(name: str, config: Dict[str, Any], params: Dict[str, Any],
             series: Series, holdout_pct: float) -> Evaluation:
    """In-sample and held-out return per symbol, and trade counts."""
    totals = {'is': [0.0, 0], 'oos': [0.0, 0]}
    for closes, round_trip in series.values():
        split = int(len(closes) * (1 - holdout_pct))
        for exit_bar, r in simulate(name, config, params, closes, round_trip):
            window = totals['is' if exit_bar < split else 'oos']
            window[0] += r
            window[1] += 1
    n = max(len(series), 1)
    return {k: (v[0] / n, v[1]) for k, v in totals.items()}


def neighbours(name: str, current: Dict[str, float]) -> List[Dict[str, float]]:
    """Settings one step from the current ones, one parameter at a time."""
    out = []
    for key, (_attr, lo, hi, step) in TUNABLE.get(family(name), {}).items():
        for delta in (-step, step):
            value = round(current[key] + delta, 6)
            if lo - 1e-9 <= value <= hi + 1e-9:
                out.append({**current, key: value})
    return out


def choose(current: Evaluation, candidates: List[Tuple[Dict[str, float], Evaluation]],
           r: Dict[str, Any]) -> Optional[Tuple[Dict[str, float], Evaluation]]:
    """The candidate that earns adoption under the rules above, or None."""
    qualified = [(params, ev) for params, ev in candidates
                 if ev['oos'][0] >= current['oos'][0] + r['min_improvement']
                 and ev['is'][0] >= current['is'][0]
                 and ev['oos'][1] >= r['min_trades'] and ev['is'][1] >= r['min_trades']]
    return max(qualified, key=lambda c: c[1]['oos'][0]) if qualified else None


class StrategyTuner:
    def __init__(self, strategy_manager, config: Dict[str, Any],
                 history_source: Callable[[], Series], params_path: Path,
                 evaluator: Callable[..., Evaluation] = evaluate,
                 clock: Callable[[], float] = time.time):
        self.sm = strategy_manager
        self.config = config
        self.rule = rule(DEFAULTS, config.get('strategy_tuning'))
        self.history_source = history_source
        self.path = Path(params_path)
        self.evaluator = evaluator
        self.clock = clock
        self._thread: Optional[threading.Thread] = None
        self.state = self._load()
        self.last_report: List[Dict[str, Any]] = []
        self.apply_saved()

    # ----------------------------------------------------------- persistence

    def _load(self) -> Dict[str, Any]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {'params': {}, 'last_run': 0.0, 'last_triggered': {}, 'log': []}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.state, indent=2))
        except OSError as e:
            logger.error(f"Could not save tuned strategy settings: {e}")

    def tunable(self) -> Dict[str, Any]:
        return {n: s for n, s in (getattr(self.sm, 'strategies', {}) or {}).items()
                if family(n) and all(hasattr(s, a) for a, *_ in TUNABLE[family(n)].values())}

    def current(self, name: str) -> Dict[str, float]:
        strat = self.tunable()[name]
        return {key: getattr(strat, attr) for key, (attr, *_r) in TUNABLE[family(name)].items()}

    def _apply(self, name: str, params: Dict[str, float]) -> None:
        strat = self.tunable().get(name)
        if strat is None:
            return
        for key, value in params.items():
            attr, lo, hi, _step = TUNABLE[family(name)][key]
            value = min(max(value, lo), hi)  # never outside the bounds
            setattr(strat, attr, value)
            if isinstance(getattr(strat, 'config', None), dict):
                strat.config[key] = value

    def apply_saved(self) -> None:
        for name, params in (self.state.get('params') or {}).items():
            if name in self.tunable():
                self._apply(name, params)
                logger.info(f"Strategy tuning: {name} runs with tuned settings {params}")

    # --------------------------------------------------------------- when

    def due(self) -> List[str]:
        """Strategies to re-tune now: all on schedule, or poor performers early."""
        if not self.rule['enabled']:
            return []
        now, names = self.clock(), list(self.tunable())
        if now - float(self.state.get('last_run') or 0) >= self.rule['interval_days'] * 86400:
            return names
        poor = []
        perf = getattr(self.sm, 'strategy_performance', {}) or {}
        for name in names:
            p = perf.get(name) or {}
            returns = p.get('returns') or []
            if (p.get('closed_trades') or 0) < self.rule['min_closed_trades']:
                continue
            losing = (p.get('win_rate') or 0) < self.rule['poor_win_rate'] or \
                (returns and sum(returns) / len(returns) < 0)
            recent = now - float((self.state.get('last_triggered') or {}).get(name, 0)) < 86400
            if losing and not recent:
                poor.append(name)
        return poor

    def maybe_tune(self) -> bool:
        """Start a tuning run in the background if one is due. Never blocks."""
        if self._thread is not None and self._thread.is_alive():
            return False
        names = self.due()
        if not names:
            return False
        self._thread = threading.Thread(target=self.run, args=(names,), daemon=True,
                                        name='strategy-tuner')
        self._thread.start()
        return True

    # ---------------------------------------------------------------- run

    def run(self, names: List[str]) -> List[Dict[str, Any]]:
        now = self.clock()
        scheduled = set(names) == set(self.tunable())
        if scheduled:
            self.state['last_run'] = now
        for name in names:
            self.state.setdefault('last_triggered', {})[name] = now
        try:
            series = {s: v for s, v in (self.history_source() or {}).items()
                      if len(v[0]) >= self.rule['min_bars']}
        except Exception as e:
            logger.error(f"Strategy tuning: history unavailable: {e}")
            self._save()
            return []
        report = [self.tune_one(name, series) for name in names]
        self.last_report = report
        self._save()
        return report

    def tune_one(self, name: str, series: Series) -> Dict[str, Any]:
        entry = {'strategy': name, 'at': datetime.utcnow().isoformat(timespec='seconds'),
                 'symbols': len(series)}
        if not series:
            entry['outcome'] = f"no symbol has {self.rule['min_bars']} daily bars yet"
            logger.info(f"Strategy tuning: {name} not tuned: {entry['outcome']}")
            return self._log(entry)
        strat_cfg = (self.config.get('strategies') or {}).get(name, {})
        now = self.current(name)
        base = self.evaluator(name, strat_cfg, now, series, self.rule['holdout_pct'])
        cands = [(p, self.evaluator(name, strat_cfg, p, series, self.rule['holdout_pct']))
                 for p in neighbours(name, now)]
        pick = choose(base, cands, self.rule)
        entry.update({'before': now, 'before_eval': base})
        if pick is None:
            entry['outcome'] = 'kept: no neighbouring setting beat it held out and in-sample'
            logger.info(f"Strategy tuning: {name} kept {now} (held out {base['oos'][0]:+.2%}, "
                        f"{base['oos'][1]} trades)")
            return self._log(entry)
        params, ev = pick
        self._apply(name, params)
        self.state.setdefault('params', {})[name] = params
        entry.update({'after': params, 'after_eval': ev, 'outcome': 'adopted'})
        logger.warning(
            f"Strategy tuning: {name} {now} -> {params}: held out {ev['oos'][0]:+.2%} vs "
            f"{base['oos'][0]:+.2%} ({ev['oos'][1]} trades), in-sample {ev['is'][0]:+.2%} vs "
            f"{base['is'][0]:+.2%}, over {len(series)} symbols")
        return self._log(entry)

    def _log(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        log = self.state.setdefault('log', [])
        log.append(entry)
        del log[:-50]  # the last fifty runs are plenty to audit
        return entry


def market_history(config: Dict[str, Any]) -> Series:
    """Daily closes and round-trip costs for every traded symbol.

    US and crypto from yfinance (two years); NSE from the scraper's real
    rows only. A market without enough real history is left out by the
    tuner's min_bars rather than tuned on too little.
    """
    from src.agent.cost_model import round_trip_pct
    from src.agent.history_warmstart import _yfinance_daily, fetch_daily_history, read_nse_history
    from src.connectors.nse_connector import NSE_CSV_DIR
    dm = config.get('data_manager', {})
    histories = fetch_daily_history(dm.get('symbols', []) or [], bars=520,
                                    fetch=lambda s: _yfinance_daily(s, period='2y'))
    histories.update(read_nse_history(dm.get('nse_symbols', []) or [], NSE_CSV_DIR, bars=520))
    return {s: (h['close'], round_trip_pct(s, config)) for s, h in histories.items()
            if h.get('close') and not any(math.isnan(c) for c in h['close'])}
