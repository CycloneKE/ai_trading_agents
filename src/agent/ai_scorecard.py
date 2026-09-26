"""Does the AI's review of trades help?

Every buy or sell signal is shown to the language model, which can approve
it or veto it (turn it into a hold). That costs money and time, and a veto
also costs the trade. This compares what happened next to the signals it
approved with what happened next to the ones it vetoed: if vetoed signals
went on to do as well, the review is adding nothing.

Returns are measured on daily closes, from the close on the decision day to
the close a fixed number of trading days later, in the signal's direction (a
sell that was followed by a fall counts as a gain), before costs. Both ends
come from the same price series, so a split or dividend adjustment in that
series cannot show up as a return.

Only verdicts the model actually gave count. When the model is off or every
provider is failing, the agent passes the strategy's signal through
unchanged; that is not an approval, and counting it as one would dilute
exactly the group being measured. A real verdict carries the model's
reasoning; a passed-through signal does not.
"""
from bisect import bisect_right
from datetime import date, datetime, timezone
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional

HORIZONS = (5, 20)
# Fewer vetoes than this and the comparison is noise; the dashboard says so.
MIN_FOR_VERDICT = 30


def _day(ts: Any, tz: Optional[str] = None) -> Optional[date]:
    """The date of a UTC journal timestamp, in the exchange's own time zone.

    A US decision logged at 21:00 New York time is already the next day in
    UTC; dating it by UTC would skip a trading day in its forward return.
    """
    try:
        text = str(ts)
        if not tz or tz == 'UTC' or len(text) <= 10:
            return date.fromisoformat(text[:10])
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo(tz)).date()
    except (ValueError, KeyError):
        return None


def reviewed_signals(decisions: Iterable[Dict[str, Any]],
                     tz_for: Optional[Callable[[str], Optional[str]]] = None) -> List[Dict[str, Any]]:
    """Buy and sell signals the AI reviewed, one per symbol, day and direction.

    A vetoed signal carries skip_reason 'llm_veto'; an approved one has a
    verdict in the same direction. Either way the verdict must carry the
    model's reasoning (see the module note). The journal can hold a
    persisting signal more than once a day, which would count it several
    times. `tz_for` gives each symbol's exchange time zone for dating.
    """
    seen = set()
    out = []
    for d in decisions:
        verdict = d.get('llm_verdict') or {}
        proposed = (d.get('action') or '').lower()
        day = _day(d.get('ts'), tz_for(d.get('symbol')) if tz_for else None)
        if not verdict or not verdict.get('reasoning') or proposed not in ('buy', 'sell') \
                or not d.get('price') or not day:
            continue
        if d.get('skip_reason') == 'llm_veto' or (verdict.get('action') or '').lower() == 'hold':
            outcome = 'vetoed'
        elif (verdict.get('action') or '').lower() == proposed:
            outcome = 'approved'
        else:
            continue
        key = (d.get('symbol'), day, proposed)
        if key in seen:
            continue
        seen.add(key)
        out.append({'symbol': d.get('symbol'), 'day': day, 'action': proposed,
                    'price': float(d['price']), 'outcome': outcome})
    return out


def forward_return(closes: Dict[date, float], day: date, horizon: int,
                   ordered: Optional[List[date]] = None) -> Optional[float]:
    """Return from the close on the decision day (the last close on or
    before it) to the close `horizon` trading days after it, or None if
    that day has not come yet or there is no close to start from."""
    ordered = ordered if ordered is not None else sorted(closes)
    i = bisect_right(ordered, day)
    if i == 0 or i - 1 + horizon >= len(ordered):
        return None
    base = closes[ordered[i - 1]]
    return closes[ordered[i - 1 + horizon]] / base - 1 if base > 0 else None


def _group(returns: List[float]) -> Dict[str, Any]:
    if not returns:
        return {'n': 0, 'avg_return_pct': None, 'win_rate': None}
    return {'n': len(returns), 'avg_return_pct': round(mean(returns) * 100, 2),
            'win_rate': round(sum(1 for r in returns if r > 0) / len(returns), 3)}


def scorecard(decisions: Iterable[Dict[str, Any]],
              closes_for: Callable[[str], Dict[date, float]],
              horizons=HORIZONS,
              tz_for: Optional[Callable[[str], Optional[str]]] = None) -> Dict[str, Any]:
    """Approved versus vetoed signals at each horizon, with a plain verdict."""
    signals = reviewed_signals(decisions, tz_for)
    cache: Dict[str, Any] = {}
    by_h: Dict[int, Dict[str, List[float]]] = {h: {'approved': [], 'vetoed': []} for h in horizons}
    for s in signals:
        sym = s['symbol']
        if sym not in cache:
            try:
                closes = closes_for(sym) or {}
            except Exception:
                closes = {}
            cache[sym] = (closes, sorted(closes))
        closes, ordered = cache[sym]
        for h in horizons:
            r = forward_return(closes, s['day'], h, ordered)
            if r is not None:
                by_h[h][s['outcome']].append(r if s['action'] == 'buy' else -r)

    result: Dict[str, Any] = {'reviewed': len(signals),
                              'approved': sum(1 for s in signals if s['outcome'] == 'approved'),
                              'vetoed': sum(1 for s in signals if s['outcome'] == 'vetoed'),
                              'horizons': {}}
    for h in horizons:
        a, v = _group(by_h[h]['approved']), _group(by_h[h]['vetoed'])
        edge = (round(a['avg_return_pct'] - v['avg_return_pct'], 2)
                if a['n'] and v['n'] else None)
        result['horizons'][str(h)] = {'approved': a, 'vetoed': v, 'veto_edge_pct': edge}
    result['verdict'] = _verdict(result, max(horizons))
    return result


def _verdict(result: Dict[str, Any], h: int) -> str:
    row = result['horizons'][str(h)]
    vetoed = row['vetoed']['n']
    if vetoed < MIN_FOR_VERDICT:
        return (f"Too early to judge: {vetoed} vetoed signal(s) have a {h}-day result; "
                f"at least {MIN_FOR_VERDICT} are needed.")
    edge = row['veto_edge_pct']
    if edge is None:
        return 'No approved signals to compare with yet.'
    if edge > 0.5:
        return (f"The review helps: over {h} trading days, approved signals beat vetoed ones "
                f"by {edge:.2f} points on average.")
    if edge < -0.5:
        return (f"The review is costing money: vetoed signals did {abs(edge):.2f} points better "
                f"over {h} trading days than approved ones.")
    return f"No clear effect: approved and vetoed signals ended within 0.5 points over {h} days."
