"""The agent's scorecard: where it is now, and where it needs to be.

The weekly paper-run report says what happened. This says whether that is
good enough: every measure has a target, and every measure is marked

    pass       at or beyond the target
    watch      short of it, or not yet convincing
    fail       clearly off it
    too_early  not enough evidence to judge either way (and how much more is needed)
    info       reported, no target

The targets are the bar the ninety-day paper run has to clear before real
money is worth discussing. They are proposals, kept in config.json
(`scorecard.targets`), so they can be argued with and changed in one place.

Six questions, in the order that matters:

1. Health. Is it running, and is it able to act? (coverage, degraded data,
   halts, workers, email alerts)
2. Evidence. Has it traded enough for any result to mean something?
3. Edge. Once it has, do its closed trades make money after costs, and is the
   win rate better than luck?
4. Returns and risk. Is the account ahead of simply holding the S&P 500, with
   drawdown inside its limit?
5. Forecasts. When it says buy or sell, does the price then move that way by
   more than it would by chance? Measured on every signal, traded or not, so
   it does not have to wait for trades to close.
6. Learning. Is it adjusting itself from results, and is it able to yet?

Nothing here is a prediction of the future. Every number is measured from what
the agent actually did and what the market then did.

`build()` is pure (data in, scorecard out). `read_inputs()` reads the data
from the journals, read-only, so it is safe against a live agent.
"""
import json
import math
import os
import sqlite3
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.agent.self_healing import DEGRADED_REASONS

PASS, WATCH, FAIL, EARLY, INFO = 'pass', 'watch', 'fail', 'too_early', 'info'

DEFAULT_TARGETS: Dict[str, Any] = {
    'planned_days': 90,
    'coverage_pct': 95.0,            # hours of the last week with a decision recorded
    'coverage_fail_pct': 80.0,
    'degraded_share_pct': 5.0,       # decisions blocked by missing data or a halt
    'degraded_fail_pct': 15.0,
    'min_closed_trades': 30,         # below this a win rate says nothing (the weekly report's rule)
    'profit_factor': 1.3,            # gains over losses, by return
    'max_concentration_pct': 70.0,   # share of the gains from one symbol
    'max_drawdown_pct': 10.0,
    'min_days_for_return': 14,
    'min_forecast_signals': 100,     # signals with a known outcome before a forecast verdict
    'min_forecast_watch': 30,
    'horizons': [5, 20],             # trading days ahead
    'tuner_max_age_days': 14,
    'evidence_gate_trades': 5,       # closed trades before results re-weight a strategy
}

DEGRADED = set(DEGRADED_REASONS) | {'halted'}


# ------------------------------------------------------------------ helpers

def wilson(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """A 95% interval for a true rate, given `hits` of `n`. Honest at small n,
    where the plain hits/n looks far more certain than it is."""
    if n <= 0:
        return 0.0, 1.0
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _m(mid: str, label: str, value: Any, display: str, status: str, target: str = '',
       note: str = '', detail: Any = None) -> Dict[str, Any]:
    out = {'id': mid, 'label': label, 'value': value, 'display': display, 'status': status,
           'target': target, 'note': note}
    if detail is not None:
        out['detail'] = detail
    return out


def _ts(value: Any) -> Optional[datetime]:
    """A journal timestamp (naive ISO, UTC) as an aware datetime."""
    try:
        d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _pct(x: Optional[float], digits: int = 1) -> str:
    return '—' if x is None else f"{x:.{digits}f}%"


def targets(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = ((config or {}).get('scorecard') or {}).get('targets') or {}
    out = {**DEFAULT_TARGETS, **{k: v for k, v in cfg.items() if not str(k).startswith('_')}}
    if 'max_drawdown_pct' not in cfg:                       # the account's own limit is the target
        limit = ((config or {}).get('risk_management') or {}).get('max_drawdown')
        if isinstance(limit, (int, float)) and limit > 0:
            out['max_drawdown_pct'] = float(limit) * 100
    return out


# ------------------------------------------------------------------- health

def _health(dec: List[Dict[str, Any]], start: datetime, now: datetime, healing: Optional[Dict[str, Any]],
            alerts: List[Dict[str, Any]], alerter: Optional[Dict[str, Any]], t: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    window_start = max(start, now - timedelta(days=7))
    hours = int((now - window_start).total_seconds() // 3600)
    rows = [d for d in dec if (ts := _ts(d.get('ts'))) and ts >= window_start]
    covered = len({str(d['ts'])[:13] for d in rows})
    if hours < 2:
        out.append(_m('coverage', 'Hours with a decision recorded', None, 'too soon', EARLY,
                      f"at least {t['coverage_pct']:.0f}%", 'The run is under two hours old.'))
    else:
        pct = min(100.0, 100.0 * covered / hours)
        status = PASS if pct >= t['coverage_pct'] else WATCH if pct >= t['coverage_fail_pct'] else FAIL
        gaps = max(hours - covered, 0)
        out.append(_m('coverage', 'Hours with a decision recorded', round(pct, 1), _pct(pct), status,
                      f"at least {t['coverage_pct']:.0f}%",
                      'It ran through every hour.' if not gaps else
                      f"{gaps} of the last {hours} hours have no record: the agent was stopped, frozen or redeploying."))
    if not rows:
        out.append(_m('degraded', 'Decisions blocked by missing data or a halt', None, 'no data', EARLY,
                      f"at most {t['degraded_share_pct']:.0f}%", 'No decisions in the last week.'))
    else:
        bad = [d for d in rows if d.get('skip_reason') in DEGRADED]
        share = 100.0 * len(bad) / len(rows)
        status = PASS if share <= t['degraded_share_pct'] else WATCH if share <= t['degraded_fail_pct'] else FAIL
        by = {}
        for d in bad:
            by[d['skip_reason']] = by.get(d['skip_reason'], 0) + 1
        worst = sorted(by.items(), key=lambda kv: -kv[1])
        syms = {}
        for d in bad:
            syms[d['symbol']] = syms.get(d['symbol'], 0) + 1
        top = ', '.join(f"{s} ({n})" for s, n in sorted(syms.items(), key=lambda kv: -kv[1])[:5])
        out.append(_m('degraded', 'Decisions blocked by missing data or a halt', round(share, 1), _pct(share), status,
                      f"at most {t['degraded_share_pct']:.0f}%",
                      ('Nothing was blocked.' if not bad else
                       f"{len(bad)} of {len(rows)}: " + ', '.join(f"{r} {n}" for r, n in worst) +
                       f". Most affected: {top}."),
                      detail={'by_reason': dict(worst), 'by_symbol': syms}))
    if healing is None:
        out.append(_m('halted', 'Trading halted right now', None, 'not known', INFO, 'not halted',
                      'Only the running agent knows this; the dashboard shows it.'))
    else:
        if healing.get('halted'):
            note = f"Halted: {healing.get('halt_reason') or 'reason unknown'}. " + (
                'This kind of halt lifts itself once the agent has been healthy for a while.'
                if healing.get('halt_clears_itself') else 'It will stay halted until you press Resume.')
            out.append(_m('halted', 'Trading halted right now', True, 'HALTED', FAIL, 'not halted', note))
        else:
            out.append(_m('halted', 'Trading halted right now', False, 'no', PASS, 'not halted'))
        dead = [w['worker'] for w in healing.get('workers', []) if w.get('alive') is False]
        gave_up = [w['worker'] for w in healing.get('workers', []) if w.get('gave_up')]
        stuck = healing.get('symbols_without_price') or {}
        if dead or gave_up:
            out.append(_m('workers', 'Background workers running', False, f"{len(dead)} down", FAIL, 'all running',
                          f"Not running: {', '.join(dead or gave_up)}."))
        else:
            out.append(_m('workers', 'Background workers running', True, 'all', PASS, 'all running',
                          f"{len(healing.get('workers', []))} watched."))
        if stuck:
            out.append(_m('no_price', 'Symbols with no real price', len(stuck), f"{len(stuck)} now", WATCH,
                          'none for over 30 minutes', ', '.join(f"{s} ({m:.0f} min)" for s, m in list(stuck.items())[:6])))
        if healing.get('consecutive_loop_errors'):
            out.append(_m('loop_errors', 'Trading loop failing', healing['consecutive_loop_errors'],
                          f"{healing['consecutive_loop_errors']} in a row", FAIL, 'none',
                          str(healing.get('last_loop_error') or '')))
    if alerter is not None:
        if alerter.get('configured'):
            out.append(_m('email', 'Email alerts', True, 'on', PASS, 'on',
                          f"To {', '.join(alerter.get('recipients') or [])}; {alerter.get('sent_24h', 0)} sent in the last day."
                          + (f" Last send error: {alerter['last_error']}" if alerter.get('last_error') else '')))
        else:
            out.append(_m('email', 'Email alerts', False, 'off', WATCH, 'on',
                          'Nobody is told when the agent halts or breaks. Set ALERT_EMAIL_TO and the SMTP_* '
                          'variables (see .env.example) and redeploy.'))
    week = (now - timedelta(days=7)).isoformat()
    recent = [a for a in alerts if str(a.get('ts', '')) >= week]
    restarts = sum(1 for a in recent if str(a.get('key', '')).endswith(':restarted'))
    resumes = sum(1 for a in recent if a.get('key') == 'heal:auto_resumed')
    crashes = sum(1 for a in recent if a.get('key') == 'heal:unclean_restart')
    critical = sum(1 for a in recent if a.get('severity') == 'critical' and a.get('status') != 'suppressed')
    out.append(_m('healing', 'Self-repairs and alerts, last 7 days', restarts + resumes + crashes,
                  f"{restarts} restarts, {resumes} auto-resumes, {crashes} crashes", INFO, '',
                  f"{critical} critical alert(s) raised." if critical else 'No critical alerts.'))
    return out


# ----------------------------------------------------------------- evidence

def _pace(n: int, need: int, days: float, planned: int) -> str:
    """Whether a count that must reach `need` by the end of the run is on pace."""
    if n >= need:
        return PASS
    if days < 0.33 * planned:
        return EARLY
    expected = need * min(days / planned, 1.0)
    return WATCH if n >= 0.5 * expected else FAIL


def _evidence(trips: List[Dict[str, Any]], attribution: Dict[str, Dict[str, Any]], days: float,
              t: Dict[str, Any]) -> List[Dict[str, Any]]:
    n, need, planned = len(trips), int(t['min_closed_trades']), int(t['planned_days'])
    status = _pace(n, need, days, planned)
    left = max(need - n, 0)
    note = ('Enough closed trades for a win rate to mean something.' if not left else
            f"{left} more closed trades needed. The agent is long-only and selective, so trades close slowly; "
            f"at the pace so far that is {'about ' + str(round(left / max(n, 1) * days)) + ' more days' if n else 'not yet estimable'}.")
    out = [_m('closed_trades', 'Closed round trips', n, str(n), status, f"at least {need} by day {planned}", note)]
    gate = int(t['evidence_gate_trades'])
    per = {k: v.get('closed_trades', 0) for k, v in (attribution or {}).items()}
    ready = sorted(k for k, c in per.items() if c >= gate)
    out.append(_m('strategies_with_evidence', f'Strategies with at least {gate} closed trades', len(ready),
                  f"{len(ready)} of {len(per) or 0} that have traded", INFO if ready else EARLY, '',
                  ('Their weights are being re-tilted from results: ' + ', '.join(ready)) if ready else
                  'Until a strategy has closed this many trades, its weight is not adjusted from results.',
                  detail=per))
    return out


# --------------------------------------------------------------------- edge

def _edge(trips: List[Dict[str, Any]], t: Dict[str, Any]) -> List[Dict[str, Any]]:
    n, need = len(trips), int(t['min_closed_trades'])
    if n < need:
        why = f"Judged once {need} trades have closed ({n} so far)."
        return [_m(mid, label, None, 'too early', EARLY, target, why) for mid, label, target in (
            ('win_rate', 'Win rate (95% range)', 'range above 50%'),
            ('avg_return', 'Average return per closed trade, after costs', 'above 0'),
            ('profit_factor', 'Profit factor (gains over losses)', f"at least {t['profit_factor']}"),
            ('concentration', 'Share of gains from one symbol', f"under {t['max_concentration_pct']:.0f}%"))]
    rets = [float(x['ret']) for x in trips]
    wins = sum(1 for r in rets if r > 0)
    lo, hi = wilson(wins, n)
    rate = wins / n
    out = [_m('win_rate', 'Win rate (95% range)', round(rate, 3), f"{rate:.0%} ({lo:.0%} to {hi:.0%})",
              PASS if lo > 0.5 else FAIL if hi < 0.5 else WATCH, 'range above 50%',
              'Better than a coin flip beyond reasonable doubt.' if lo > 0.5 else
              'Worse than a coin flip beyond reasonable doubt.' if hi < 0.5 else
              'The range includes 50%: luck cannot be ruled out yet.')]
    avg = mean(rets) * 100
    out.append(_m('avg_return', 'Average return per closed trade, after costs', round(avg, 2), f"{avg:+.2f}%",
                  PASS if avg > 0 else FAIL, 'above 0', 'Costs are already taken off.'))
    gains = sum(r for r in rets if r > 0)
    losses = -sum(r for r in rets if r < 0)
    pf = (gains / losses) if losses > 0 else None
    out.append(_m('profit_factor', 'Profit factor (gains over losses)', None if pf is None else round(pf, 2),
                  'no losses' if pf is None else f"{pf:.2f}",
                  PASS if pf is None or pf >= t['profit_factor'] else WATCH if pf >= 1 else FAIL,
                  f"at least {t['profit_factor']}", 'By return, so trades in different currencies compare fairly.'))
    by_sym: Dict[str, float] = {}
    for x in trips:
        if x['ret'] > 0:
            by_sym[x['symbol']] = by_sym.get(x['symbol'], 0.0) + float(x['ret'])
    if gains > 0:
        top, share = max(by_sym.items(), key=lambda kv: kv[1])
        share = 100.0 * share / gains
        out.append(_m('concentration', 'Share of gains from one symbol', round(share, 1), f"{share:.0f}% ({top})",
                      PASS if share <= t['max_concentration_pct'] else WATCH,
                      f"under {t['max_concentration_pct']:.0f}%",
                      'One symbol over one period is an anecdote, not an edge.' if share > t['max_concentration_pct'] else ''))
    return out


# -------------------------------------------------------- returns and risk

def _drawdown(values: List[float]) -> float:
    peak, worst = 0.0, 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak - v) / peak)
    return worst * 100


def _close_on_or_before(closes: Dict[date, float], day: date, ordered: List[date]) -> Optional[float]:
    i = bisect_right(ordered, day)
    return closes[ordered[i - 1]] if i else None


def _returns(equity: List[Tuple[datetime, float]], spy: Optional[Dict[date, float]], t: Dict[str, Any]
             ) -> List[Dict[str, Any]]:
    pts = sorted((d, v) for d, v in equity if v and v > 0)
    if len(pts) < 2:
        return [_m('vs_spy', 'Account return against the S&P 500', None, 'too early', EARLY, 'at or above SPY',
                   'No equity history since the run began.'),
                _m('drawdown', 'Worst drop from a peak', None, 'too early', EARLY,
                   f"under {t['max_drawdown_pct']:.0f}%", 'No equity history since the run began.')]
    first, last = pts[0], pts[-1]
    days = (last[0] - first[0]).total_seconds() / 86400
    ret = (last[1] / first[1] - 1) * 100
    out: List[Dict[str, Any]] = []
    spy_ret = None
    if spy:
        ordered = sorted(spy)
        a = _close_on_or_before(spy, first[0].date(), ordered)
        b = _close_on_or_before(spy, last[0].date(), ordered)
        if a and b:
            spy_ret = (b / a - 1) * 100
    if days < t['min_days_for_return']:
        out.append(_m('vs_spy', 'Account return against the S&P 500', round(ret, 2),
                      f"{ret:+.2f}%" + (f" vs SPY {spy_ret:+.2f}%" if spy_ret is not None else ''), EARLY,
                      'at or above SPY',
                      f"{days:.0f} days of history; judged from {t['min_days_for_return']}. "
                      "The account includes its index-fund core."))
    elif spy_ret is None:
        out.append(_m('vs_spy', 'Account return against the S&P 500', round(ret, 2), f"{ret:+.2f}%", INFO,
                      'at or above SPY', 'The S&P 500 prices could not be fetched, so no comparison.'))
    else:
        gap = ret - spy_ret
        status = PASS if gap >= 0 else FAIL if (gap < -2 and days >= 30) else WATCH
        out.append(_m('vs_spy', 'Account return against the S&P 500', round(gap, 2),
                      f"{ret:+.2f}% vs SPY {spy_ret:+.2f}% ({gap:+.2f} points)", status, 'at or above SPY',
                      f"Over {days:.0f} days. Holding an index fund is the effort-free alternative."))
    dd = _drawdown([v for _, v in pts])
    cap = float(t['max_drawdown_pct'])
    out.append(_m('drawdown', 'Worst drop from a peak', round(dd, 2), f"{dd:.2f}%",
                  PASS if dd <= 0.8 * cap else WATCH if dd <= cap else FAIL, f"under {cap:.0f}%",
                  'Measured on the account\'s own equity readings.'))
    return out


# ---------------------------------------------------------------- forecasts

def _market_tz(symbol: str, config: Optional[Dict[str, Any]]) -> Optional[str]:
    from src.agent.cost_model import classify
    m = classify(symbol, config)
    return {'nse': 'Africa/Nairobi', 'crypto': 'UTC'}.get(m, 'America/New_York')


def signals_from(decisions: Iterable[Dict[str, Any]], config: Optional[Dict[str, Any]] = None
                 ) -> List[Dict[str, Any]]:
    """Every buy or sell signal, taken or not, once per symbol, day and
    direction (a signal that persists is journaled repeatedly)."""
    from src.agent.ai_scorecard import _day
    seen, out = set(), []
    for d in decisions:
        action = str(d.get('action') or '').lower()
        if action not in ('buy', 'sell') or not d.get('price'):
            continue
        day = _day(d.get('ts'), _market_tz(d.get('symbol'), config))
        key = (d.get('symbol'), day, action)
        if day is None or key in seen:
            continue
        seen.add(key)
        out.append({'symbol': d['symbol'], 'day': day, 'action': action})
    return out


def _base_rate(closes: Dict[date, float], ordered: List[date], horizon: int) -> Optional[float]:
    """How often this symbol simply went up over `horizon` trading days."""
    if len(ordered) <= horizon:
        return None
    ups = sum(1 for i in range(len(ordered) - horizon) if closes[ordered[i + horizon]] > closes[ordered[i]])
    return ups / (len(ordered) - horizon)


def _forecast(sigs: List[Dict[str, Any]], closes_for: Optional[Callable[[str], Dict[date, float]]],
              config: Optional[Dict[str, Any]], t: Dict[str, Any]) -> List[Dict[str, Any]]:
    from src.agent.ai_scorecard import forward_return
    from src.agent.cost_model import classify
    horizons = [int(h) for h in t['horizons']]
    if closes_for is None or not sigs:
        why = 'Price history for the check was not available.' if sigs and closes_for is None else \
              'No buy or sell signals recorded yet.'
        return [_m(f'forecast_{h}d', f'Signals that moved the right way within {h} trading days', None,
                   'too early', EARLY, 'above chance', why) for h in horizons]
    cache: Dict[str, Tuple[Dict[date, float], List[date]]] = {}
    out = []
    for h in horizons:
        hits = 0
        n = 0
        rets: List[float] = []
        chance: List[float] = []
        by_market: Dict[str, List[int]] = {}
        for s in sigs:
            sym = s['symbol']
            if sym not in cache:
                try:
                    c = closes_for(sym) or {}
                except Exception:
                    c = {}
                cache[sym] = (c, sorted(c))
            closes, ordered = cache[sym]
            r = forward_return(closes, s['day'], h, ordered)
            if r is None:
                continue
            directional = r if s['action'] == 'buy' else -r
            base = _base_rate(closes, ordered, h)
            if base is None:
                continue
            n += 1
            ok = 1 if directional > 0 else 0
            hits += ok
            rets.append(directional)
            chance.append(base if s['action'] == 'buy' else 1 - base)
            by_market.setdefault(classify(sym, config), []).append(ok)
        label = f'Signals that moved the right way within {h} trading days'
        if n < t['min_forecast_watch']:
            out.append(_m(f'forecast_{h}d', label, None, f"too early ({n} with a known outcome)", EARLY,
                          'above the chance rate',
                          f"Judged from {int(t['min_forecast_watch'])} signals whose {h}-day outcome is known; "
                          f"{n} so far. A signal's outcome is known {h} trading days after it."))
            continue
        rate = hits / n
        base = mean(chance)
        lo, hi = wilson(hits, n)
        if n < t['min_forecast_signals']:
            status, note = WATCH, (f"{n} signals; a verdict needs {int(t['min_forecast_signals'])}. "
                                   f"Chance rate for the same symbols and days: {base:.0%}.")
        elif lo > base:
            status, note = PASS, f"Beyond doubt better than the {base:.0%} that chance (the market's own drift) gives."
        elif hi < base:
            status, note = FAIL, f"Worse than the {base:.0%} that chance gives."
        else:
            status, note = WATCH, f"The range includes the {base:.0%} that chance gives: no proven skill yet."
        out.append(_m(f'forecast_{h}d', label, round(rate, 3), f"{rate:.0%} of {n} ({lo:.0%} to {hi:.0%})", status,
                      f"above {base:.0%} (chance)", note + f" Average move in the signal's direction: {mean(rets) * 100:+.2f}%.",
                      detail={m: {'n': len(v), 'hit_rate': round(sum(v) / len(v), 3)} for m, v in by_market.items()}))
    return out


# ----------------------------------------------------------------- learning

def _learning(tuner: Optional[Dict[str, Any]], trips: int, attribution: Dict[str, Dict[str, Any]], days: float,
              now: datetime, config: Optional[Dict[str, Any]], t: Dict[str, Any]) -> List[Dict[str, Any]]:
    enabled = ((config or {}).get('strategy_tuning') or {}).get('enabled', True)
    out = []
    gate = int(t['evidence_gate_trades'])
    ready = sum(1 for v in (attribution or {}).values() if v.get('closed_trades', 0) >= gate)
    out.append(_m('learning_gate', 'Learning from its own results', ready, f"{ready} strategies ready",
                  PASS if ready else EARLY if days < 30 else WATCH, f"a strategy needs {gate} closed trades",
                  ('Results are re-weighting the strategies that have enough trades.' if ready else
                   f"No strategy has {gate} closed trades, so nothing can be learned from results yet "
                   f"({trips} trades closed in all). Learning from every signal's later price move, not only "
                   f"closed trades, would remove this wait."),))
    if not enabled:
        out.append(_m('tuner', 'Weekly settings review', None, 'switched off', INFO, 'weekly'))
        return out
    last = (tuner or {}).get('last_run')
    log = (tuner or {}).get('log') or []
    adopted = sum(1 for e in log if e.get('outcome') == 'adopted')
    if not last:
        out.append(_m('tuner', 'Weekly settings review', None, 'not run yet', EARLY if days < 7 else WATCH, 'weekly',
                      'It runs when the agent has been up a week, or sooner for a losing strategy.'))
    else:
        age = (now.timestamp() - float(last)) / 86400
        out.append(_m('tuner', 'Weekly settings review', round(age, 1), f"last ran {age:.0f} days ago",
                      PASS if age <= t['tuner_max_age_days'] else WATCH, f"within {t['tuner_max_age_days']} days",
                      f"{len(log)} runs recorded, {adopted} changes adopted (a change is adopted only if it beats the "
                      f"current settings on data it was not tuned on)."))
    return out


# ------------------------------------------------------------------ verdict

def _verdict(sections: List[Dict[str, Any]], days: float, planned: int) -> Dict[str, str]:
    metrics = {m['id']: m for s in sections for m in s['metrics']}
    health = [m for m in next(s for s in sections if s['id'] == 'health')['metrics']]
    if any(m['status'] == FAIL for m in health):
        bad = [m['label'] for m in health if m['status'] == FAIL]
        return {'level': 'needs_attention', 'title': 'Needs attention',
                'text': 'The agent is not fully healthy: ' + '; '.join(bad) + '. Fix this first; nothing below means much until it runs.'}
    fails = [m for m in metrics.values() if m['status'] == FAIL]
    if fails:
        return {'level': 'off_track', 'title': 'Off track',
                'text': 'Running, but short of the bar on: ' + '; '.join(m['label'] for m in fails) + '.'}
    early = [m for m in metrics.values() if m['status'] == EARLY]
    passing = [m for m in metrics.values() if m['status'] == PASS]
    if early:
        return {'level': 'too_early', 'title': 'Too early to judge',
                'text': f"Day {days:.0f} of {planned}. Nothing is failing, but {len(early)} measures "
                        f"have too little evidence to call: " + '; '.join(m['label'] for m in early[:4]) +
                        ('…' if len(early) > 4 else '') + '.'}
    if all(m['status'] in (PASS, INFO) for m in metrics.values()):
        return {'level': 'meets_bar', 'title': 'Meets the paper-run bar',
                'text': 'Every measure is at or beyond its target. That is a reason to discuss a small live pilot, '
                        'not proof the edge will last.'}
    return {'level': 'on_track', 'title': 'On track',
            'text': f"{len(passing)} measures pass and none fail; the rest are short of target or not yet convincing."}


# -------------------------------------------------------------------- build

SECTIONS = (
    ('health', 'Health', 'Is it running, and able to act?'),
    ('evidence', 'Evidence', 'Has it traded enough for a result to mean anything?'),
    ('edge', 'Edge', 'Do its closed trades make money after costs, better than luck?'),
    ('returns', 'Returns and risk', 'Is the account ahead of holding the S&P 500, inside its drawdown limit?'),
    ('forecast', 'Forecasts', 'When it says buy or sell, does the price then move that way, beyond chance?'),
    ('learning', 'Learning', 'Is it adjusting itself from results, and is it able to yet?'),
)


def build(inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None,
          now: Optional[datetime] = None) -> Dict[str, Any]:
    """The scorecard from `inputs` (see read_inputs for the keys)."""
    from src.agent.round_trips import closed_round_trips
    from src.agent.strategy_attribution import compute_attribution
    now = now or datetime.now(timezone.utc)
    t = targets(config)
    dec = inputs.get('decisions') or []
    fills = inputs.get('fills') or []
    start = _ts(inputs.get('run_started_at'))
    if start is None:
        firsts = [ts for d in dec if (ts := _ts(d.get('ts')))]
        start = min(firsts) if firsts else now
    days = max((now - start).total_seconds() / 86400, 0.0)
    planned = int(inputs.get('planned_days') or t['planned_days'])
    trips = closed_round_trips(fills, config)
    attribution = compute_attribution(fills) if fills else {}
    sigs = signals_from(dec, config)

    body = {
        'health': _health(dec, start, now, inputs.get('healing'), inputs.get('alerts') or [],
                          inputs.get('alerter'), t),
        'evidence': _evidence(trips, attribution, days, t),
        'edge': _edge(trips, t),
        'returns': _returns(inputs.get('equity') or [], inputs.get('spy'), t),
        'forecast': _forecast(sigs, inputs.get('closes_for'), config, t),
        'learning': _learning(inputs.get('tuner'), len(trips), attribution, days, now, config, t),
    }
    sections = [{'id': i, 'title': title, 'question': q, 'metrics': body[i]} for i, title, q in SECTIONS]
    counts = {s: sum(1 for sec in sections for m in sec['metrics'] if m['status'] == s)
              for s in (PASS, WATCH, FAIL, EARLY, INFO)}
    return {
        'generated_at': now.isoformat(timespec='seconds'),
        'run': {'started_at': start.isoformat(timespec='seconds'), 'day': round(days, 1), 'planned_days': planned,
                'percent_through': round(min(100.0, 100 * days / planned), 1) if planned else None},
        'verdict': _verdict(sections, days, planned),
        'counts': counts,
        'sections': sections,
        'targets': {k: v for k, v in t.items()},
    }


# ---------------------------------------------------------------- reading

def _ro(path: str) -> Optional[sqlite3.Connection]:
    if not os.path.exists(path):
        return None
    try:
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=10)
    except sqlite3.Error:
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: Optional[sqlite3.Connection], sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    if conn is None:
        return []
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except sqlite3.Error:
        return []


def latest_manifest(data_dir: str) -> Dict[str, Any]:
    folder = os.path.join(data_dir, 'paper_runs')
    try:
        files = sorted(f for f in os.listdir(folder) if f.endswith('.json'))
        with open(os.path.join(folder, files[-1]), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError, IndexError):
        return {}


def bars_to_closes(bars: Iterable[Dict[str, Any]]) -> Dict[date, float]:
    out = {}
    for b in bars or []:
        try:
            out[date.fromisoformat(str(b['time'])[:10])] = float(b['close'])
        except (KeyError, ValueError, TypeError):
            continue
    return out


def read_inputs(data_dir: str, config: Optional[Dict[str, Any]] = None,
                closes_for: Optional[Callable[[str], Dict[date, float]]] = None,
                spy: Optional[Dict[date, float]] = None,
                healing: Optional[Dict[str, Any]] = None,
                alerter: Optional[Dict[str, Any]] = None,
                now: Optional[datetime] = None) -> Dict[str, Any]:
    """Gather what `build` needs from the data folder, read-only: safe to run
    against a live agent. `healing` and `alerter` are the running agent's own
    status, when there is one."""
    from src.agent.alerts import AlertLog
    manifest = latest_manifest(data_dir)
    started = manifest.get('started_at')
    since_iso = str(started)[:19] if started else ''
    d_conn = _ro(os.path.join(data_dir, 'decision_journal.db'))
    # Holds are most of the journal. The forecast check needs only buys and
    # sells, over the whole run; health needs everything, but only the last week.
    week_ago = ((now or datetime.now(timezone.utc)) - timedelta(days=7)).replace(tzinfo=None).isoformat()
    decisions = _rows(
        d_conn, "SELECT ts, symbol, action, skip_reason, executed, price FROM decisions WHERE ts >= ?"
                " AND (action IN ('buy', 'sell') OR ts >= ?) ORDER BY ts", (since_iso or '', week_ago))
    o_conn = _ro(os.path.join(data_dir, 'order_journal.db'))
    fills = _rows(o_conn, "SELECT * FROM orders WHERE status = 'filled' AND filled_quantity > 0 ORDER BY created_at")
    if since_iso:
        fills = [f for f in fills if str(f.get('created_at') or '')[:19] >= since_iso]
    p_conn = _ro(os.path.join(data_dir, 'escalations.db'))
    equity = []
    for r in _rows(p_conn, "SELECT timestamp, value FROM portfolio_history ORDER BY timestamp"):
        ts = _ts(r['timestamp'])
        if ts and (not since_iso or (ts >= (_ts(started) or ts))):
            equity.append((ts, float(r['value'])))
    tuner = None
    try:
        with open(os.path.join(data_dir, 'strategy_params.json'), encoding='utf-8') as f:
            tuner = json.load(f)
    except (OSError, ValueError):
        pass
    alerts = AlertLog(os.path.join(data_dir, 'alerts.jsonl')).recent(limit=500)
    for c in (d_conn, o_conn, p_conn):
        if c is not None:
            c.close()
    return {'run_started_at': started, 'planned_days': manifest.get('planned_days'), 'decisions': decisions,
            'fills': fills, 'equity': equity, 'spy': spy, 'closes_for': closes_for, 'healing': healing,
            'alerter': alerter, 'alerts': alerts, 'tuner': tuner}
