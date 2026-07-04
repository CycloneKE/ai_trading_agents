"""Assemble the per-symbol drill-down payload: the agent's story on one stock.

Pure functions over journal rows + a price lookup, so the API endpoint stays
thin and this stays unit-testable. Answers: what did the agent decide and
why, how well did it execute, and is it beating buy-and-hold on this name.
"""

from typing import Any, Callable, Dict, List, Optional

from src.agent.strategy_attribution import compute_attribution


def execution_quality(orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Slippage of fills vs. the decision (limit) price, in basis points.

    Positive bps = adverse (paid up on a buy / sold cheap). This is the
    cost leak that matters most at small size.
    """
    fills = [o for o in orders
             if o.get('status') == 'filled' and o.get('filled_avg_price')]
    slips = []
    per_order = []
    for o in fills:
        ref = o.get('limit_price') or o.get('filled_avg_price')
        fill = o['filled_avg_price']
        if not ref or ref <= 0:
            continue
        # sign so buys and sells both read "positive = worse"
        raw = (fill - ref) / ref
        adverse = raw if o.get('side') == 'buy' else -raw
        bps = round(adverse * 10_000, 1)
        slips.append(bps)
        per_order.append({
            'client_order_id': o.get('client_order_id'),
            'side': o.get('side'), 'ref_price': ref, 'fill_price': fill,
            'slippage_bps': bps, 'strategy': o.get('strategy'),
        })
    avg = round(sum(slips) / len(slips), 1) if slips else 0.0
    return {'fills': len(fills), 'avg_slippage_bps': avg, 'orders': per_order}


def alpha_vs_hold(symbol: str, attribution: Dict[str, Any],
                  first_entry_price: Optional[float],
                  current_price: Optional[float]) -> Dict[str, Any]:
    """Agent's total P&L on the symbol vs. simply holding it since first entry.

    The single BI number: if the agent can't beat buy-and-hold on a name,
    the operator should see it at a glance.
    """
    agent_pnl = round(sum(
        (a.get('realized_pnl', 0) or 0) + (a.get('unrealized_pnl', 0) or 0)
        for a in attribution.values()), 2)
    hold_return_pct = None
    if first_entry_price and current_price and first_entry_price > 0:
        hold_return_pct = round((current_price - first_entry_price) / first_entry_price * 100, 2)
    return {
        'agent_pnl': agent_pnl,
        'buy_hold_return_pct': hold_return_pct,
        'first_entry_price': first_entry_price,
        'current_price': current_price,
    }


def build_payload(symbol: str,
                  decisions: List[Dict[str, Any]],
                  orders: List[Dict[str, Any]],
                  price_lookup: Optional[Callable[[str], Optional[float]]] = None
                  ) -> Dict[str, Any]:
    """Full drill-down payload for one symbol."""
    symbol = symbol.upper()
    current_price = None
    if price_lookup:
        try:
            current_price = price_lookup(symbol)
        except Exception:
            current_price = None

    # Per-strategy books on THIS symbol (filter fills to the symbol first).
    symbol_fills = [o for o in orders
                    if o.get('status') == 'filled' and o.get('symbol') == symbol]
    attribution = compute_attribution(
        symbol_fills, price_lookup=(lambda s: current_price) if current_price else None)

    # First entry price = oldest fill's price (chronological).
    first_entry = None
    chrono = sorted(symbol_fills, key=lambda o: o.get('created_at') or '')
    if chrono:
        first_entry = chrono[0].get('filled_avg_price')

    # Decision tape: decisions + orders, already time-ordered by the journal.
    return {
        'symbol': symbol,
        'current_price': current_price,
        'decisions': decisions,               # the "why did/didn't it trade" log
        'orders': orders,                     # fills/attempts
        'strategy_books': attribution,        # momentum vs mean_reversion on this name
        'execution_quality': execution_quality(orders),
        'alpha_vs_hold': alpha_vs_hold(symbol, attribution, first_entry, current_price),
        'decision_count': len(decisions),
        'order_count': len(orders),
    }
