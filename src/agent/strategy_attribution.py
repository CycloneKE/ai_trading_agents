"""Per-strategy P&L attribution computed from the order journal.

Every filled order records who decided it, so realized P&L, win rate and
open inventory can be attributed per strategy with an average-cost book per
(strategy, symbol) — no broker state needed beyond optional live prices for
unrealized P&L.

Two kinds of journal row:

- Rows with `strategy_weights` (JSON {strategy: share}) were written by the
  long-only book. A buy is credited to the strategies that voted for it, in
  proportion to their confidence (credit_weights). A sell closes whatever
  the strategies hold in that symbol, in proportion to what each holds,
  because in a long-only account a sell is always an exit: the strategies
  that chose to buy own the outcome, whoever or whatever (a stop, a weaker
  sell signal) triggered the exit. Before this, every blended order was
  tagged 'ensemble', so no individual strategy was ever credited with
  anything and the performance-weighted ensemble had nothing to learn from.
- Older rows without weights keep the original behaviour: each order's own
  `strategy` book, independent of the others, which can go short.

Each closing trade's return (P&L over its cost) is kept alongside the P&L.
Returns pool across markets; P&L in KES and USD does not, unless `fx`
converts it.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def credit_weights(per_strategy: Optional[Dict[str, Dict[str, Any]]],
                   action: str) -> Optional[Dict[str, float]]:
    """Each strategy's share of the credit for `action`: the strategies that
    voted for it, in proportion to their confidence. None if none did."""
    votes = {name: float(v.get('confidence') or 0.0)
             for name, v in (per_strategy or {}).items()
             if isinstance(v, dict) and v.get('action') == action
             and float(v.get('confidence') or 0.0) > 0}
    total = sum(votes.values())
    return {n: round(c / total, 6) for n, c in votes.items()} if total > 0 else None


def _weights(fill: Dict[str, Any]) -> Optional[Dict[str, float]]:
    raw = fill.get('strategy_weights')
    if not raw:
        return None
    try:
        w = json.loads(raw) if isinstance(raw, str) else dict(raw)
        w = {str(k): float(v) for k, v in w.items() if float(v) > 0}
    except (ValueError, TypeError, AttributeError):
        return None
    total = sum(w.values())
    return {k: v / total for k, v in w.items()} if total > 0 else None


def _trade(books: Dict[tuple, Dict[str, Any]], strategy: str, symbol: str,
           signed: float, price: float, fx: float) -> None:
    """Apply a signed quantity to one (strategy, symbol) average-cost book."""
    book = books.setdefault((strategy, symbol), {
        'qty': 0.0, 'avg': 0.0, 'realized': 0.0, 'closed': 0, 'wins': 0,
        'pnls': [], 'returns': [], 'fx': fx,
    })
    book['fx'] = fx
    qty, pos = abs(signed), book['qty']
    if pos == 0 or (pos > 0) == (signed > 0):
        # Opening or extending in the same direction: blend average cost.
        total = abs(pos) + qty
        book['avg'] = (abs(pos) * book['avg'] + qty * price) / total
        book['qty'] = pos + signed
        return
    # Reducing / closing / crossing through zero.
    close_qty = min(qty, abs(pos))
    direction = 1 if pos > 0 else -1
    pnl = (price - book['avg']) * close_qty * direction * fx
    book['realized'] += pnl
    book['closed'] += 1
    book['pnls'].append(round(pnl, 4))
    book['returns'].append(round((price / book['avg'] - 1) * direction, 6) if book['avg'] else 0.0)
    if pnl > 0:
        book['wins'] += 1
    book['qty'] = pos + signed
    if abs(book['qty']) < 1e-9:
        book['qty'], book['avg'] = 0.0, 0.0
    elif (book['qty'] > 0) != (pos > 0):
        # Crossed zero: the remainder is a fresh position at this fill.
        book['avg'] = price


def compute_attribution(fills: List[Dict[str, Any]],
                        price_lookup: Optional[Callable[[str], Optional[float]]] = None,
                        fx: Optional[Callable[[str], float]] = None
                        ) -> Dict[str, Dict[str, Any]]:
    """Attribute realized/unrealized P&L to strategies.

    Args:
        fills: journal rows (status=filled), each with strategy, symbol,
            side, filled_quantity, filled_avg_price, created_at, and
            optionally strategy_weights.
        price_lookup: optional callable symbol -> current price, used for
            unrealized P&L on open inventory.
        fx: optional callable symbol -> multiplier into one currency (e.g.
            KES to USD for NSE symbols), so P&L sums across markets.

    Returns:
        {strategy: {realized_pnl, closed_trades, win_rate, unrealized_pnl,
                    trade_pnls, trade_returns,
                    open_positions: [{symbol, quantity, avg_entry_price, ...}]}}
    """
    books: Dict[tuple, Dict[str, Any]] = {}

    for f in sorted(fills, key=lambda r: r.get('created_at') or ''):
        strategy = f.get('strategy') or 'unknown'
        symbol = f.get('symbol')
        qty = float(f.get('filled_quantity') or 0)
        price = float(f.get('filled_avg_price') or 0)
        side = f.get('side')
        if not symbol or qty <= 0 or price <= 0 or side not in ('buy', 'sell'):
            continue
        rate = float(fx(symbol)) if fx else 1.0
        weights = _weights(f)

        if weights is None:  # an older row: its own book, as before
            _trade(books, strategy, symbol, qty if side == 'buy' else -qty, price, rate)
        elif side == 'buy':
            for name, share in weights.items():
                _trade(books, name, symbol, qty * share, price, rate)
        else:
            holders = {k[0]: b['qty'] for k, b in books.items() if k[1] == symbol and b['qty'] > 1e-9}
            held = sum(holders.values())
            closing = min(qty, held)
            for name, holding in holders.items():
                _trade(books, name, symbol, -closing * holding / held, price, rate)
            rest = qty - closing
            if rest > 1e-9:  # more sold than any strategy held: the old behaviour
                for name, share in weights.items():
                    _trade(books, name, symbol, -rest * share, price, rate)

    # Aggregate books per strategy.
    result: Dict[str, Dict[str, Any]] = {}
    for (strategy, symbol), book in books.items():
        agg = result.setdefault(strategy, {
            'realized_pnl': 0.0, 'closed_trades': 0, 'wins': 0,
            'unrealized_pnl': 0.0, 'open_positions': [], 'trade_pnls': [],
            'trade_returns': [],
        })
        agg['realized_pnl'] += book['realized']
        agg['closed_trades'] += book['closed']
        agg['wins'] += book['wins']
        agg['trade_pnls'].extend(book['pnls'])
        agg['trade_returns'].extend(book['returns'])

        if abs(book['qty']) > 1e-9:
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
                    upl = (current - book['avg']) * book['qty'] * book['fx']
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


def compute_symbol_attribution(fills: List[Dict[str, Any]],
                               price_lookup: Optional[Callable[[str], Optional[float]]] = None
                               ) -> Dict[str, Dict[str, Any]]:
    """Attribute realized P&L to symbols rather than strategies.

    The question this answers is "is this name earning its place in the
    universe", which strategy-level attribution cannot: a symbol can be
    quietly losing money across every strategy that touches it and still be
    invisible in a per-strategy view.

    Uses the same average-cost book as `compute_attribution`, keyed by
    symbol alone.
    """
    books: Dict[str, Dict[str, float]] = {}
    out: Dict[str, Dict[str, Any]] = {}

    for f in sorted(fills, key=lambda r: r.get('created_at') or ''):
        symbol = f.get('symbol')
        qty = float(f.get('filled_quantity') or 0)
        price = float(f.get('filled_avg_price') or 0)
        side = f.get('side')
        if not symbol or qty <= 0 or price <= 0 or side not in ('buy', 'sell'):
            continue

        book = books.setdefault(symbol, {'qty': 0.0, 'avg': 0.0})
        rec = out.setdefault(symbol, {
            'realized_pnl': 0.0, 'closed_trades': 0, 'wins': 0,
            'buys': 0, 'sells': 0, 'first_seen': f.get('created_at'),
            'last_seen': f.get('created_at'),
        })
        rec['last_seen'] = f.get('created_at') or rec['last_seen']

        if side == 'buy':
            rec['buys'] += 1
            total = book['avg'] * book['qty'] + price * qty
            book['qty'] += qty
            book['avg'] = (total / book['qty']) if book['qty'] else 0.0
        else:
            rec['sells'] += 1
            closing = min(qty, book['qty'])
            if closing > 0:
                pnl = (price - book['avg']) * closing
                rec['realized_pnl'] += pnl
                rec['closed_trades'] += 1
                if pnl > 0:
                    rec['wins'] += 1
                book['qty'] -= closing
                if book['qty'] <= 1e-9:
                    book['qty'], book['avg'] = 0.0, 0.0

    for symbol, rec in out.items():
        book = books.get(symbol, {'qty': 0.0, 'avg': 0.0})
        rec['open_quantity'] = book['qty']
        rec['avg_entry_price'] = book['avg']
        rec['win_rate'] = (rec['wins'] / rec['closed_trades']
                           if rec['closed_trades'] else 0.0)
        rec['unrealized_pnl'] = 0.0
        if price_lookup and book['qty'] > 0:
            try:
                current = price_lookup(symbol)
                if current:
                    rec['unrealized_pnl'] = (float(current) - book['avg']) * book['qty']
            except Exception as e:
                logger.debug(f"Price lookup failed for {symbol}: {e}")
        rec['total_pnl'] = rec['realized_pnl'] + rec['unrealized_pnl']
    return out
