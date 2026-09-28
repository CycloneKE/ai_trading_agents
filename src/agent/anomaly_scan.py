"""Anomaly surfacing: 'what's unusual today', for a solo operator.

Pure functions over the decision + order journals. Each detector returns
zero or more anomalies (severity, symbol, one-line message) so the
dashboard can lead with what changed rather than a wall of charts.

Thresholds are deliberately conservative defaults — they want tuning once
a few days of real market-hours decisions have accumulated. Override via
the `config` dict (config.json 'anomalies' section).
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

DEFAULTS = {
    'blocked_intent_min': 5,     # N directional decisions blocked -> flag
    'high_slippage_bps': 20.0,   # avg adverse slippage on a symbol -> flag
    'persistent_skip_min': 4,    # same systemic skip reason N times -> flag
    'disagreement_conf': 0.6,    # both a strong buy AND strong sell signal
    'drawdown_warn_frac': 0.8,   # drawdown within 80% of the cap -> flag
    'window_hours': 12,          # only checks this recent count; older ones have been acted on
}

# Skip reasons that indicate a *systemic* problem worth surfacing (vs. the
# normal 'hold' / 'below_confidence' quiet).
SYSTEMIC_REASONS = {
    'fallback_price': 'price feed is serving synthetic fallback data',
    'no_price': 'no current price to size an order with',
    'no_account_info': 'broker account info unavailable',
    'min_notional': 'orders sized below the minimum notional',
    'pdt_guard': 'pattern-day-trader limit blocking exits',
    'duplicate': 'duplicate orders being blocked',
    'stale_price': 'the latest real price is too old to act on',
}

# Why a buy or sell signal was not acted on, in plain words, and whether that
# needs the operator. Most are the agent's own rules and checks doing their
# job: not adding to a holding until it has earned it, never short-selling,
# the AI review or the bias check saying no. Those are shown for information.
# The rest mean something stopped a trade the agent should have made.
HELD_BACK = {
    'add_not_profitable': "it is already held, and the agent adds to a holding only once it is 5% "
                          "above the last purchase and in profit after costs",
    'already_held': "it is already held, and this holding is not added to",
    'max_adds': "it has already been added to the most times allowed",
    'position_cap': "the holding is already at its size limit",
    'no_position': "the agent holds none to sell, and it never bets on a fall (short selling)",
    'trimmed_today': "the position was already trimmed once today",
    'min_holding': "it has been held for less than the minimum holding period",
    'turnover_budget': "this week's allowance of new NSE holdings is used up",
    'order_pending': "an earlier order for it is still working",
    'below_confidence': "the combined signal was not confident enough to trade",
    'bias_downgrade': "the bias check found the signal one-sided and lowered its confidence "
                      "below the level needed to trade",
    'llm_veto': "the AI review advised against it",
    'dissent': "no strategy agreed with the direction",
    'liquidity_cap': "the stock trades too thinly to buy even one share within the volume limit",
    'market_closed': "its market is closed (US stocks or forex); the signal is looked at again when it opens",
}
BLOCKED = {
    'insufficient_cash': ("there was not enough paper cash",
                          "Normal while the account is fully invested; if it persists, positions may be too large."),
    'min_notional': ("the order came out smaller than the minimum order size",
                     "Position sizing may be too small for this price; check the risk settings."),
    'pdt_guard': ("the pattern-day-trader rule blocked it", "Expected on a small US account; it clears after five trading days."),
    'halted': ("trading is halted", "Press Resume when you are ready to trade again."),
    'position_unknown': ("the agent could not read its holdings, so it skipped to be safe",
                         "Check the broker connection on the Risk & System page."),
    'no_account_info': ("the broker account could not be read", "Check the broker connection on the Risk & System page."),
    'fallback_price': ("the price was not a real market price", "Check the price feeds on the Risk & System page."),
    'stale_price': ("the latest price is too old (the stock may be suspended)", "No action unless the stock trades again."),
    'no_price': ("the agent could not get a current price to size the order, so it placed nothing",
                 "Harmless while the market is closed. If it continues during trading hours, the price "
                 "source is not answering; check the data feeds on the Risk & System page."),
    'duplicate': ("the same order was already sent", "Usually harmless; if it keeps happening, check the order journal."),
}


def _anom(severity, symbol, kind, message, hint=None, **detail):
    """One notice. `severity` high or medium needs the operator; low and
    info are for information."""
    return {'severity': severity, 'symbol': symbol, 'type': kind,
            'message': message, 'hint': hint,
            'attention': severity in ('high', 'medium'), 'detail': detail}


def detect_blocked_intent(decisions: List[Dict[str, Any]], threshold: int) -> List[Dict[str, Any]]:
    """A symbol with a buy or sell signal, again and again, that was not
    acted on. Counted in the decision log's rows, which are the agent's
    checks (a row is written when the outcome changes, and every half hour
    or so while it does not): none of them sent an order.

    When something stopped a trade the agent should have made (no cash, no
    real price, the broker unreadable), it needs a look. When the agent's
    own rules held it back, it is for information only.
    """
    by_symbol = defaultdict(list)
    for d in decisions:
        by_symbol[d['symbol']].append(d)
    out = []
    for symbol, rows in by_symbol.items():
        signalled = [r for r in rows if r.get('action') not in (None, 'hold') and not r.get('executed')]
        if len(signalled) < threshold:
            continue
        reasons = defaultdict(int)
        sides = defaultdict(int)
        for r in signalled:
            reasons[r.get('skip_reason') or 'unknown'] += 1
            sides[r.get('action')] += 1
        side = max(sides, key=sides.get)
        blocked = {k: n for k, n in reasons.items() if k not in HELD_BACK}
        n_blocked = sum(blocked.values())
        last_at = max((str(r.get('ts')) for r in signalled if r.get('ts')), default=None)
        if n_blocked >= threshold:
            top = max(blocked, key=blocked.get)
            why, hint = BLOCKED.get(top, (f"of '{top}'", None))
            out.append(_anom('high', symbol, 'blocked_intent',
                             f"{symbol}: a {side} signal could not be carried out on {n_blocked} checks, "
                             f"because {why}.", hint=hint,
                             count=n_blocked, reasons=dict(reasons), last_at=last_at))
            continue
        top = max(reasons, key=reasons.get)
        out.append(_anom('info', symbol, 'held_back',
                         f"{symbol}: {side} signal on {len(signalled)} checks, held back on purpose "
                         f"because {HELD_BACK.get(top, top)}.",
                         hint='No action needed: this is a rule working as designed.',
                         count=len(signalled), reasons=dict(reasons), last_at=last_at))
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
                             f"{symbol}: fills averaged {avg / 100:.2f}% worse than the order price over {len(slips)} trades.",
                             hint='Trading costs are higher than planned; consider trading it less often.',
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
                                 f"{symbol}: {SYSTEMIC_REASONS[reason]} ({n} recent checks).",
                                 hint=BLOCKED.get(reason, (None, None))[1],
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
                             f"{symbol}: the strategies disagree, with a strong buy and a strong sell signal at the same time.",
                             hint='No action needed: the agent does not trade on a split vote.',
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
                      f"The account is {dd:.1%} below its peak, which reaches the {max_drawdown_cap:.0%} limit.",
                      hint='The risk manager stops new buying at this limit; review before resuming.',
                      drawdown=dd, cap=max_drawdown_cap)]
    if dd >= warn_frac * max_drawdown_cap:
        return [_anom('medium', None, 'drawdown',
                      f"The account is {dd:.1%} below its peak, close to the {max_drawdown_cap:.0%} limit.",
                      hint='No action needed yet; new buying stops if it reaches the limit.',
                      drawdown=dd, cap=max_drawdown_cap)]
    return []


SEVERITY_RANK = {'high': 0, 'medium': 1, 'low': 2, 'info': 3}


def recent_only(decisions: List[Dict[str, Any]], hours: float,
                now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """The decisions logged in the last `hours` (times are UTC). A row
    without a time is kept. Older checks describe problems already fixed or
    gone, and kept a notice alive for days."""
    if not hours:
        return decisions
    cutoff = ((now or datetime.utcnow()) - timedelta(hours=float(hours))).isoformat()
    return [d for d in decisions if not d.get('ts') or str(d['ts'])[:26] >= cutoff[:26]]


def scan(decisions: List[Dict[str, Any]], orders: List[Dict[str, Any]],
         risk_report: Optional[Dict[str, Any]] = None,
         max_drawdown_cap: float = 0.10,
         config: Optional[Dict[str, Any]] = None,
         now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Run all detectors and return anomalies, most severe first."""
    cfg = {**DEFAULTS, **(config or {})}
    decisions = recent_only(decisions, cfg['window_hours'], now)
    anomalies = []
    anomalies += detect_blocked_intent(decisions, cfg['blocked_intent_min'])
    anomalies += detect_high_slippage(orders, cfg['high_slippage_bps'])
    anomalies += detect_persistent_skip(decisions, cfg['persistent_skip_min'])
    anomalies += detect_strategy_disagreement(decisions, cfg['disagreement_conf'])
    anomalies += detect_drawdown(risk_report, cfg['drawdown_warn_frac'], max_drawdown_cap)
    # A stock blocked for a reason already says so once; the same reason
    # counted again as a "persistent skip" would only repeat it.
    said = {(a['symbol'], r) for a in anomalies if a['type'] == 'blocked_intent'
            for r in a['detail'].get('reasons', {})}
    anomalies = [a for a in anomalies if not (a['type'] == 'persistent_skip'
                                              and (a['symbol'], a['detail'].get('reason')) in said)]
    anomalies.sort(key=lambda a: SEVERITY_RANK.get(a['severity'], 9))
    return anomalies
