"""Anomaly surfacing: 'what's unusual today', for a solo operator.

Pure functions over the decision + order journals. Each detector returns
zero or more anomalies (severity, symbol, one-line message) so the
dashboard can lead with what changed rather than a wall of charts.

Thresholds are deliberately conservative defaults — they want tuning once
a few days of real market-hours decisions have accumulated. Override via
the `config` dict (config.json 'anomalies' section).
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

DEFAULTS = {
    'blocked_intent_min': 5,     # N directional decisions blocked -> flag
    'high_slippage_bps': 20.0,   # avg adverse slippage on a symbol -> flag
    'persistent_skip_min': 4,    # same systemic skip reason N times -> flag
    'disagreement_conf': 0.6,    # both a strong buy AND strong sell signal
    'drawdown_warn_frac': 0.8,   # drawdown within 80% of the cap -> flag
}

# Skip reasons that indicate a *systemic* problem worth surfacing (vs. the
# normal 'hold' / 'below_confidence' quiet).
SYSTEMIC_REASONS = {
    'fallback_price': 'price feed is serving synthetic fallback data',
    'no_price': 'no valid price available',
    'no_account_info': 'broker account info unavailable',
    'min_notional': 'orders sized below the minimum notional',
    'pdt_guard': 'pattern-day-trader limit blocking exits',
    'duplicate': 'duplicate orders being blocked',
}


def _anom(severity, symbol, kind, message, **detail):
    return {'severity': severity, 'symbol': symbol, 'type': kind,
            'message': message, 'detail': detail}


def detect_blocked_intent(decisions: List[Dict[str, Any]], threshold: int) -> List[Dict[str, Any]]:
    """A symbol where the agent repeatedly WANTED to trade (action != hold)
    but nothing executed — it's trying and being blocked."""
    by_symbol = defaultdict(list)
    for d in decisions:
        by_symbol[d['symbol']].append(d)
    out = []
    for symbol, rows in by_symbol.items():
        blocked = [r for r in rows if r.get('action') not in (None, 'hold') and not r.get('executed')]
        if len(blocked) >= threshold:
            reasons = defaultdict(int)
            for r in blocked:
                reasons[r.get('skip_reason') or 'unknown'] += 1
            top = max(reasons, key=reasons.get)
            out.append(_anom('high', symbol, 'blocked_intent',
                             f"{symbol}: agent tried to trade {len(blocked)}× but was blocked (mostly '{top}')",
                             count=len(blocked), reasons=dict(reasons)))
    return out


def detect_high_slippage(orders: List[Dict[str, Any]], threshold_bps: float) -> List[Dict[str, Any]]:
    """Symbols whose recent fills show adverse average slippage."""
    by_symbol = defaultdict(list)
    for o in orders:
        if o.get('status') == 'filled' and o.get('filled_avg_price') and o.get('limit_price'):
            ref = o['limit_price']
            if ref <= 0:
                continue
            raw = (o['filled_avg_price'] - ref) / ref
            adverse = raw if o.get('side') == 'buy' else -raw
            by_symbol[o['symbol']].append(adverse * 10_000)
    out = []
    for symbol, slips in by_symbol.items():
        avg = sum(slips) / len(slips)
        if avg >= threshold_bps:
            out.append(_anom('medium', symbol, 'high_slippage',
                             f"{symbol}: avg slippage {avg:.0f} bps across {len(slips)} fills (execution cost leak)",
                             avg_bps=round(avg, 1), fills=len(slips)))
    return out


def detect_persistent_skip(decisions: List[Dict[str, Any]], threshold: int) -> List[Dict[str, Any]]:
    """A symbol repeatedly hitting the SAME systemic skip reason — a data or
    account problem, not normal market quiet."""
    counts = defaultdict(lambda: defaultdict(int))
    for d in decisions:
        r = d.get('skip_reason')
        if r in SYSTEMIC_REASONS:
            counts[d['symbol']][r] += 1
    out = []
    for symbol, reasons in counts.items():
        for reason, n in reasons.items():
            if n >= threshold:
                sev = 'high' if reason in ('fallback_price', 'no_price', 'no_account_info') else 'medium'
                out.append(_anom(sev, symbol, 'persistent_skip',
                                 f"{symbol}: {SYSTEMIC_REASONS[reason]} ({n}× recently)",
                                 reason=reason, count=n))
    return out


def detect_strategy_disagreement(decisions: List[Dict[str, Any]], conf: float) -> List[Dict[str, Any]]:
    """The most recent decision per symbol where strategies strongly disagree
    (a confident buy AND a confident sell at once) — signal instability."""
    seen = set()
    out = []
    for d in decisions:  # newest first
        symbol = d['symbol']
        if symbol in seen:
            continue
        seen.add(symbol)
        ps = d.get('per_strategy') or {}
        buys = [s.get('confidence', 0) for s in ps.values() if s.get('action') == 'buy']
        sells = [s.get('confidence', 0) for s in ps.values() if s.get('action') == 'sell']
        if buys and sells and max(buys) >= conf and max(sells) >= conf:
            out.append(_anom('low', symbol, 'strategy_disagreement',
                             f"{symbol}: strategies split — a strong buy and a strong sell at once",
                             max_buy=max(buys), max_sell=max(sells)))
    return out


def detect_drawdown(risk_report: Optional[Dict[str, Any]], warn_frac: float,
                    max_drawdown_cap: float) -> List[Dict[str, Any]]:
    """Portfolio drawdown approaching or past the configured cap."""
    if not risk_report or not max_drawdown_cap:
        return []
    dd = abs(risk_report.get('drawdown') or
             risk_report.get('current_metrics', {}).get('max_drawdown') or 0)
    if dd >= max_drawdown_cap:
        return [_anom('high', None, 'drawdown',
                      f"Portfolio drawdown {dd:.1%} has reached the {max_drawdown_cap:.0%} cap",
                      drawdown=dd, cap=max_drawdown_cap)]
    if dd >= warn_frac * max_drawdown_cap:
        return [_anom('medium', None, 'drawdown',
                      f"Portfolio drawdown {dd:.1%} approaching the {max_drawdown_cap:.0%} cap",
                      drawdown=dd, cap=max_drawdown_cap)]
    return []


SEVERITY_RANK = {'high': 0, 'medium': 1, 'low': 2}


def scan(decisions: List[Dict[str, Any]], orders: List[Dict[str, Any]],
         risk_report: Optional[Dict[str, Any]] = None,
         max_drawdown_cap: float = 0.10,
         config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Run all detectors and return anomalies, most severe first."""
    cfg = {**DEFAULTS, **(config or {})}
    anomalies = []
    anomalies += detect_blocked_intent(decisions, cfg['blocked_intent_min'])
    anomalies += detect_high_slippage(orders, cfg['high_slippage_bps'])
    anomalies += detect_persistent_skip(decisions, cfg['persistent_skip_min'])
    anomalies += detect_strategy_disagreement(decisions, cfg['disagreement_conf'])
    anomalies += detect_drawdown(risk_report, cfg['drawdown_warn_frac'], max_drawdown_cap)
    anomalies.sort(key=lambda a: SEVERITY_RANK.get(a['severity'], 9))
    return anomalies
