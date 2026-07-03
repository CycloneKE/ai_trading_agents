"""Per-strategy P&L attribution computed from the order journal.

Every filled order carries the strategy that decided it, so realized P&L,
win rate and open inventory can be attributed per strategy with an
average-cost book per (strategy, symbol) — no broker state needed beyond
optional live prices for unrealized P&L.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_attribution(fills: List[Dict[str, Any]],
                        price_lookup: Optional[Callable[[str], Optional[float]]] = None
                        ) -> Dict[str, Dict[str, Any]]:
    """Attribute realized/unrealized P&L to strategies.

    Args:
        fills: journal rows (status=filled), each with strategy, symbol,
            side, filled_quantity, filled_avg_price, created_at.
        price_lookup: optional callable symbol -> current price, used for
            unrealized P&L on open inventory.

    Returns:
        {strategy: {realized_pnl, closed_trades, win_rate, unrealized_pnl,
                    open_positions: [{symbol, quantity, avg_entry_price, ...}]}}
    """
    # Average-cost book per (strategy, symbol).
    books: Dict[tuple, Dict[str, float]] = {}

    for f in sorted(fills, key=lambda r: r.get('created_at') or ''):
        strategy = f.get('strategy') or 'unknown'
        symbol = f.get('symbol')
        qty = float(f.get('filled_quantity') or 0)
        price = float(f.get('filled_avg_price') or 0)
        side = f.get('side')
        if not symbol or qty <= 0 or price <= 0 or side not in ('buy', 'sell'):
            continue

        book = books.setdefault((strategy, symbol), {
            'qty': 0.0, 'avg': 0.0, 'realized': 0.0, 'closed': 0, 'wins': 0,
        })
        signed = qty if side == 'buy' else -qty
        pos = book['qty']

        if pos == 0 or (pos > 0) == (signed > 0):
            # Opening or extending in the same direction: blend average cost.
            total = abs(pos) + qty
            book['avg'] = (abs(pos) * book['avg'] + qty * price) / total
            book['qty'] = pos + signed
        else:
            # Reducing / closing / crossing through zero.
            close_qty = min(qty, abs(pos))
            direction = 1 if pos > 0 else -1
            pnl = (price - book['avg']) * close_qty * direction
            book['realized'] += pnl
            book['closed'] += 1
            if pnl > 0:
                book['wins'] += 1
            book['qty'] = pos + signed
            if book['qty'] == 0:
                book['avg'] = 0.0
            elif (book['qty'] > 0) != (pos > 0):
                # Crossed zero: the remainder is a fresh position at this fill.
                book['avg'] = price

    # Aggregate books per strategy.
    result: Dict[str, Dict[str, Any]] = {}
    for (strategy, symbol), book in books.items():
        agg = result.setdefault(strategy, {
            'realized_pnl': 0.0, 'closed_trades': 0, 'wins': 0,
            'unrealized_pnl': 0.0, 'open_positions': [],
        })
        agg['realized_pnl'] += book['realized']
        agg['closed_trades'] += book['closed']
        agg['wins'] += book['wins']

        if book['qty'] != 0:
            open_pos = {
                'symbol': symbol,
                'quantity': round(book['qty'], 6),
                'avg_entry_price': round(book['avg'], 4),
            }
            if price_lookup:
                try:
                    current = price_lookup(symbol)
                except Exception:
                    current = None
                if current:
                    upl = (current - book['avg']) * book['qty']
                    open_pos['current_price'] = current
                    open_pos['unrealized_pl'] = round(upl, 2)
                    agg['unrealized_pnl'] += upl
            agg['open_positions'].append(open_pos)

    for strategy, agg in result.items():
        closed = agg['closed_trades']
        agg['win_rate'] = round(agg['wins'] / closed, 4) if closed else 0.0
        agg['realized_pnl'] = round(agg['realized_pnl'], 2)
        agg['unrealized_pnl'] = round(agg['unrealized_pnl'], 2)
        del agg['wins']

    return result
