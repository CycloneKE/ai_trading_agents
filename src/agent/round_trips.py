"""Closed round trips, matched from the order journal's filled orders.

One definition of "a closed trade" for everything that counts them: the
AI self-review (self_assessment.closed_trade_evidence) and the dashboard's
Agent Track Record (/api/performance). The dashboard used to read a list
(performance_analytics.trades_history) that nothing in the agent ever
filled, so its closed-trade count and win rate stayed at 0 however many
trades had closed.
"""
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

# How the order journal labels an exit that a rule forced, not a signal.
STOP_TAGS = ('stop_loss', 'trailing_stop', 'kill_switch')


def _when(order: Dict[str, Any]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(order.get('created_at'))[:26])
    except ValueError:
        return None


def closed_round_trips(fills: List[Dict[str, Any]],
                       config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The closed round trips in `fills` (filled orders, oldest first).

    FIFO-matched and long only: a sell closes the oldest held lots of that
    symbol first, and a sell with nothing held closes nothing. Each trip
    is {'symbol', 'exit', 'ret', 'cost', 'days', 'closed_at'}:

    - `exit` is what closed it: 'stop_loss', 'trailing_stop' or
      'kill_switch' when the journal says a rule forced it, else 'signal';
    - `ret` is the return on the entry price after the modelled costs a
      fill price leaves out (cost_model.fill_gap_pct), as a fraction;
    - `cost` is what the matched shares cost to buy, so `ret * cost` is the
      profit or loss in the account's own currency;
    - `days` is how long the lot was held, or None when a time is missing;
    - `closed_at` is when it was sold (ISO, UTC), or None.

    A sell that closes several lots is several trips: the lots were bought
    separately, so each is its own result.
    """
    from src.agent.cost_model import fill_gap_pct

    lots: Dict[str, Any] = {}
    trips: List[Dict[str, Any]] = []
    for o in fills:
        sym = str(o.get('symbol') or '').upper()
        qty, price = float(o.get('filled_quantity') or 0), o.get('filled_avg_price')
        if not sym or qty <= 0 or price is None or float(price) <= 0:
            continue
        price, side = float(price), str(o.get('side') or '').lower()
        if side == 'buy':
            lots.setdefault(sym, deque()).append([qty, price, _when(o)])
            continue
        if side != 'sell':
            continue
        tag = o.get('strategy') if o.get('strategy') in STOP_TAGS else 'signal'
        remaining, held = qty, lots.setdefault(sym, deque())
        sold_at = _when(o)
        gap = fill_gap_pct(sym, config)
        while remaining > 1e-9 and held:
            lot = held[0]
            matched = min(remaining, lot[0])
            ratio = price / lot[1]
            days = ((sold_at - lot[2]).total_seconds() / 86400.0) if lot[2] and sold_at else None
            trips.append({'symbol': sym, 'exit': tag,
                          'ret': ratio - 1.0 - gap * (1.0 + ratio),
                          'cost': matched * lot[1], 'days': days,
                          'closed_at': sold_at.isoformat() if sold_at else None})
            lot[0] -= matched
            remaining -= matched
            if lot[0] <= 1e-9:
                held.popleft()
    return trips


def trade_counts(trips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The headline counts for the dashboard: closed, winning and losing
    trips (a flat one counts as losing: costs make it a loss) and the win
    rate as a 0-1 fraction."""
    n = len(trips)
    wins = sum(1 for t in trips if t['ret'] > 0)
    return {'total_trades': n, 'winning_trades': wins, 'losing_trades': n - wins,
            'win_rate': (wins / n) if n else 0.0}
