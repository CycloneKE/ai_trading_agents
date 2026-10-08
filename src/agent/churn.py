"""Positions bought and sold again within the hour: the pattern that burns money.

A strategy that trades daily bars should hold a position for days. When the same
symbol is opened and closed again and again within minutes, every pass pays the
spread and the fees and earns nothing; a week of it turned a flat paper account
into a 2.7% loss. Something is wrong (here, a trailing stop inheriting the
peak of a position that was already closed), and whatever the cause, it is
visible in the order journal long before it is visible in the profit.

This counts, from the filled orders, the round trips that closed within
`quick_hours` of opening, per symbol, so the scorecard can show them, the agent
can raise an alert, and scripts/churn_report.py can lay them out.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

QUICK_HOURS = 1.0


def _when(value: Any) -> Optional[datetime]:
    try:
        d = datetime.fromisoformat(str(value)[:26])
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def quick_trips(trips: Iterable[Dict[str, Any]], since: Optional[datetime] = None,
                quick_hours: float = QUICK_HOURS) -> List[Dict[str, Any]]:
    """The round trips that closed within `quick_hours` of opening, no earlier than `since`."""
    out = []
    for t in trips:
        days = t.get('days')
        closed = _when(t.get('closed_at'))
        if days is None or days * 24 >= quick_hours:
            continue
        if since is not None and (closed is None or closed < since):
            continue
        out.append(t)
    return out


def by_symbol(quick: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per symbol: how many, the mean return, and how each was closed."""
    out: Dict[str, Dict[str, Any]] = {}
    for t in quick:
        rec = out.setdefault(t['symbol'], {'n': 0, 'rets': [], 'exits': {}, 'minutes': []})
        rec['n'] += 1
        rec['rets'].append(float(t['ret']))
        rec['exits'][t['exit']] = rec['exits'].get(t['exit'], 0) + 1
        rec['minutes'].append(float(t['days']) * 1440)
    for rec in out.values():
        rets, mins = rec.pop('rets'), sorted(rec.pop('minutes'))
        rec['mean_return_pct'] = round(sum(rets) / len(rets) * 100, 3)
        rec['median_minutes'] = round(mins[len(mins) // 2], 1)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]['n']))


def recent(fills: List[Dict[str, Any]], config: Optional[Dict[str, Any]], hours: float = 24.0,
           now: Optional[datetime] = None, quick_hours: float = QUICK_HOURS,
           since: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
    """Quick round trips by symbol over the last `hours`, from filled orders,
    and no earlier than `since` when given."""
    from src.agent.round_trips import closed_round_trips
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    if since is not None and since > start:
        start = since
    return by_symbol(quick_trips(closed_round_trips(fills, config), start, quick_hours))
