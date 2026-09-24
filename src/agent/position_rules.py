"""Position management rules shared by the NSE paper account and the US book.

Three decisions, one place, so both books behave the same way:

Adding to a winner. A buy signal on a held position becomes an add only when
the position has proved itself: selling it all now would be profitable after
the sale's own costs, the price is at least `min_gain_pct` above the last
entry (so each add needs the move to continue and adds cannot bunch at one
price), fewer than `max_adds` adds have been made, and the position stays
within `max_position_pct` of the account. An add is `add_size_pct` of a new
position's size.

Selling part of a position. A sell signal at or above `full_exit_confidence`
closes the position. A weaker one trims `trim_fraction` of it to reduce
exposure, at most once per day per position, so a weak sell signal that
persists all day cannot sell the position down cycle by cycle. A trim that
would leave less than `min_remaining_pct` of the account in the position
closes it instead: a sliver costs as much to watch as a whole position.

Stops (see stop_triggered) always close the whole position.
"""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

ADD_DEFAULTS = {'enabled': True, 'min_gain_pct': 0.05, 'max_adds': 2,
                'add_size_pct': 0.5, 'max_position_pct': 0.4}
EXIT_DEFAULTS = {'partial_exits': True, 'full_exit_confidence': 0.75,
                 'trim_fraction': 0.5, 'min_remaining_pct': 0.02}

# The US and crypto book sizes entries at up to risk_limits.max_position_size
# (5%) of equity, so its adds are capped just above that rather than at the
# NSE account's 40%. Adds apply to the listed markets only.
LIVE_ADD_DEFAULTS = {**ADD_DEFAULTS, 'max_position_pct': 0.08, 'markets': ['us_equity']}
LIVE_EXIT_DEFAULTS = {**EXIT_DEFAULTS, 'min_remaining_pct': 0.01}

# Refusals decided by the position alone, whatever the order's size. The
# callers check these before asking the LLM, so a trade the book cannot take
# never costs a model call.
HOLDING_RULE_REASONS = frozenset({'already_held', 'no_position', 'add_not_profitable',
                                  'max_adds', 'position_cap', 'trimmed_today'})


def rule(defaults: Dict[str, Any], configured: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Defaults overlaid with a config block, ignoring `_comment` keys."""
    return {**defaults, **{k: v for k, v in (configured or {}).items() if not k.startswith('_')}}


@dataclass
class PositionState:
    quantity: float = 0.0
    cost: float = 0.0             # cost basis, fees included
    entries: int = 0              # buys since the position opened
    last_entry_price: float = 0.0
    opened_at: Optional[str] = None
    last_trim_day: Optional[str] = None  # YYYY-MM-DD of the last partial sell
    realised: float = 0.0         # all closed P&L on this symbol, net of fees

    @property
    def held(self) -> bool:
        return self.quantity > 1e-9


def replay(fills: Iterable[Dict[str, Any]]) -> Dict[str, PositionState]:
    """Position state per symbol from fills, oldest first.

    Each fill: symbol, side, quantity, price, and optionally fees and time
    (ISO timestamp). A sell larger than the holding is treated as closing it:
    long-only books cannot be short, and older records that were are not
    this book's business.
    """
    book: Dict[str, PositionState] = {}
    for f in fills:
        s = book.setdefault(str(f['symbol']).upper(), PositionState())
        qty, price = float(f['quantity']), float(f['price'])
        fees, when = float(f.get('fees') or 0.0), f.get('time')
        if qty <= 0 or price <= 0:
            continue
        if f['side'] == 'buy':
            if not s.held:
                s.quantity, s.cost, s.entries, s.opened_at, s.last_trim_day = 0.0, 0.0, 0, when, None
            s.quantity += qty
            s.cost += qty * price + fees
            s.entries += 1
            s.last_entry_price = price
        elif s.held:
            sold = min(qty, s.quantity)
            basis = s.cost * sold / s.quantity
            s.realised += sold * price - fees - basis
            s.cost -= basis
            s.quantity -= sold
            if s.held:
                s.last_trim_day = (when or '')[:10] or None
            else:
                s.quantity, s.cost = 0.0, 0.0
    return book


def plan_add(state: PositionState, price: float, exit_cost_pct: float,
             account_value: float, new_position_value: float,
             add_rule: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """Value to add to a held position, or 0 and why not.

    `exit_cost_pct` is the fraction a sale loses to commission and slippage;
    `account_value` is what the position cap is a share of.
    """
    if not add_rule.get('enabled'):
        return 0.0, 'already_held'
    if state.entries - 1 >= int(add_rule['max_adds']):
        return 0.0, 'max_adds'
    exit_value = state.quantity * price * (1 - exit_cost_pct)
    if exit_value <= state.cost or price < state.last_entry_price * (1 + add_rule['min_gain_pct']):
        return 0.0, 'add_not_profitable'
    room = add_rule['max_position_pct'] * account_value - state.quantity * price
    wanted = new_position_value * add_rule['add_size_pct']
    value = min(wanted, room)
    if value <= 0:
        return 0.0, 'position_cap'
    return value, None


def plan_exit(state: PositionState, confidence: float, price: float,
              account_value: float, today: str,
              exit_rule: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """Quantity to sell on a sell signal, or 0 and why not."""
    if not state.held:
        return 0.0, 'no_position'
    if not exit_rule.get('partial_exits') or confidence >= exit_rule['full_exit_confidence']:
        return state.quantity, None
    if state.last_trim_day == today:
        return 0.0, 'trimmed_today'
    remaining = state.quantity * (1 - exit_rule['trim_fraction'])
    if remaining * price < exit_rule['min_remaining_pct'] * account_value:
        return state.quantity, None
    return state.quantity * exit_rule['trim_fraction'], None


def stop_triggered(price: float, avg_cost: float, high_since_entry: float,
                   stop_pct: float, trail_pct: float) -> Optional[str]:
    """'stop_loss' below the cost basis, 'trailing_stop' below the peak, else None."""
    if avg_cost > 0 and stop_pct > 0 and price <= avg_cost * (1 - stop_pct):
        return 'stop_loss'
    if high_since_entry > 0 and trail_pct > 0 and price <= high_since_entry * (1 - trail_pct):
        return 'trailing_stop'
    return None


def state_from_journal(journal, symbol: str, held: float, avg_entry: float) -> PositionState:
    """A broker-held position's state, its history taken from the order journal.

    The broker is the authority on quantity; the journal adds what it does
    not know (entries since the position opened, the last entry price, the
    last trim). If the journal's replay disagrees with the broker's quantity
    (orders placed before the journal existed, say), its history cannot be
    trusted, and the position is treated as one entry at the broker's
    average price.
    """
    fills = []
    if journal is not None:
        try:
            fills = [{'symbol': r['symbol'], 'side': r['side'],
                      'quantity': float(r.get('filled_quantity') or 0),
                      'price': float(r.get('filled_avg_price') or 0),
                      'time': r.get('updated_at') or r.get('created_at')}
                     for r in journal.filled_orders() if r.get('symbol') == symbol]
        except Exception:
            fills = []
    state = replay(fills).get(symbol.upper())
    if state is None or abs(state.quantity - held) > max(1e-6, 0.01 * held):
        # The last journaled sell, with no buy after it and shares still
        # held, was a trim whatever the journal knows of the history.
        trim_day = None
        if held > 0 and fills and fills[-1]['side'] == 'sell':
            trim_day = (fills[-1].get('time') or '')[:10] or None
        state = PositionState(quantity=held, cost=held * avg_entry, entries=1,
                              last_entry_price=avg_entry, last_trim_day=trim_day)
    return state
