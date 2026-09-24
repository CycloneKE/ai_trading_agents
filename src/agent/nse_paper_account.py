"""The agent's NSE paper account.

NSE decisions become order tickets, and with `nse_auto_paper_trade` on, the
agent fills its own tickets as paper trades. Those fills used to be booked
at the decision price with no cost, no cash and no rules, which overstated
every result:

- a buy signal that persisted all day bought another full position every
  evaluation cycle (the ticket dedupe only looked at pending tickets, and an
  auto-filled ticket is not pending);
- a sell signal with nothing held opened a short, which a Kenyan retail
  account cannot do;
- about 3.4% of round-trip cost (see cost_model.py) was never charged;
- there was no cash, so it could buy without limit.

This account holds KES `nse_paper_trading.starting_capital_kes`. It never
shorts and never spends more cash than it has; it adds to a holding only
once the holding has proved profitable, and a weaker sell signal trims a
position rather than closing it (both in position_rules.py, shared with the
US book). A stop-loss below the cost basis and a trailing stop below the
high since entry close a position whatever the signals say (`stop_loss`).
Each fill pays slippage (in the price) and commission (in `fees_kes`).

Its state is replayed from the filled tickets in the 'trading' book since
the account opened, so there is one record of what happened and nothing to
keep in step with it.
"""
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from src.agent.cost_model import market_costs
from src.agent.position_rules import (
    ADD_DEFAULTS, EXIT_DEFAULTS, HOLDING_RULE_REASONS, PositionState,
    plan_add, plan_exit, replay, rule, stop_triggered)

logger = logging.getLogger(__name__)

BOOK = 'trading'
RESOLVED_BY = 'auto_paper_trader'

__all__ = ['NsePaperAccount', 'order_size', 'execute_stop', 'HOLDING_RULE_REASONS', 'BOOK']

# NSE stops sit wider than the US ones. A round trip costs about 3.4%, and
# a daily-bar position stopped out by an ordinary day's noise pays that for
# nothing. Where the stock's ATR is known the stop scales with it, but never
# tighter than these.
STOP_DEFAULTS = {'enabled': True, 'stop_loss_pct': 0.08, 'trailing_stop_pct': 0.10}


class NsePaperAccount:
    def __init__(self, queue, config: Dict[str, Any]):
        self.queue = queue
        self.config = config
        cfg = config.get('nse_paper_trading') or {}
        self.starting_capital = float(cfg.get('starting_capital_kes') or 0.0)
        costs = market_costs('nse', config)
        self.commission_pct = float(costs.get('commission_pct') or 0.0)
        self.min_commission = float(costs.get('min_commission') or 0.0)
        self.slippage_pct = float(costs.get('slippage_pct') or 0.0)
        self.costs_verified = bool(costs.get('verified'))
        self.add_rule = rule(ADD_DEFAULTS, cfg.get('add_to_winners'))
        self.exit_rule = rule(EXIT_DEFAULTS, cfg.get('exits'))
        self.stop_rule = rule(STOP_DEFAULTS, cfg.get('stop_loss'))
        self.started_at = queue.paper_account_started_at() if self.enabled else None

    @property
    def enabled(self) -> bool:
        return self.starting_capital > 0

    # ------------------------------------------------------------ pricing

    def fill_price(self, side: str, price: float) -> float:
        """The decision price moved against us by slippage, to the cent."""
        sign = 1 if side == 'buy' else -1
        return round(price * (1 + sign * self.slippage_pct), 2)

    def fees(self, notional: float) -> float:
        return round(max(notional * self.commission_pct, self.min_commission), 2)

    # ------------------------------------------------------------- state

    def _fills(self):
        return self.queue.fills(book=BOOK, since=self.started_at)

    def states(self) -> Dict[str, PositionState]:
        return replay({'symbol': f['symbol'], 'side': f['side'], 'quantity': f['quantity'],
                       'price': f['price'], 'fees': f['fees_kes'], 'time': f['fill_at']}
                      for f in self._fills())

    def _ledger(self) -> Dict[str, Any]:
        """Cash, fees paid, realised P&L and positions, from the fills."""
        cash, fees = self.starting_capital, 0.0
        for f in self._fills():
            notional = f['quantity'] * f['price']
            fees += f['fees_kes']
            cash += -(notional + f['fees_kes']) if f['side'] == 'buy' else notional - f['fees_kes']
        states = self.states()
        positions = {
            s: {'quantity': int(round(p.quantity)), 'cost_kes': round(p.cost, 2),
                'avg_cost_kes': round(p.cost / p.quantity, 4) if p.held else 0.0,
                'entries': p.entries, 'last_entry_kes': p.last_entry_price}
            for s, p in states.items()}
        return {'cash': round(cash, 2), 'fees': round(fees, 2), 'states': states,
                'realised': round(sum(p.realised for p in states.values()), 2),
                'positions': positions}

    def positions(self) -> Dict[str, Dict[str, Any]]:
        """Held symbols: quantity and cost basis including fees."""
        return {s: p for s, p in self._ledger()['positions'].items() if p['quantity'] > 0}

    def cash(self) -> float:
        return self._ledger()['cash']

    @staticmethod
    def _account_at_cost(ledger: Dict[str, Any]) -> float:
        return ledger['cash'] + sum(p['cost_kes'] for p in ledger['positions'].values()
                                    if p['quantity'] > 0)

    # ------------------------------------------------------------- rules

    def plan(self, symbol: str, side: str, price: float, target_notional: float,
             confidence: float = 1.0) -> Tuple[int, Optional[str]]:
        """Quantity to trade under the account's rules, or 0 and why not.

        Reasons match the US book's (see TradingAgent._position_gate) so the
        dashboard reads them the same way.
        """
        ledger = self._ledger()
        state = ledger['states'].get(symbol.upper()) or PositionState()
        account = self._account_at_cost(ledger)
        if side == 'sell':
            qty, reason = plan_exit(state, confidence, price, account,
                                    datetime.utcnow().date().isoformat(), self.exit_rule)
            if reason:
                return 0, reason
            held = int(round(state.quantity))
            shares = int(qty) if int(qty) >= 1 else held  # a trim under one share closes it
            return min(shares, held), None
        if not state.held:
            return self._buyable(price, float(target_notional), ledger['cash'], target_notional)
        value, reason = plan_add(state, price, self.slippage_pct + self.commission_pct,
                                 account, float(target_notional), self.add_rule)
        if reason:
            return 0, reason
        qty, reason = self._buyable(price, value, ledger['cash'], value)
        wanted = float(target_notional) * self.add_rule['add_size_pct']
        if reason == 'min_notional' and value < wanted:
            reason = 'position_cap'  # the cap, not the order, left too little room
        return qty, reason

    def _buyable(self, price: float, budget: float, cash: float,
                 wanted: float) -> Tuple[int, Optional[str]]:
        """Whole shares that `budget` buys, fees included, within the cash."""
        px = self.fill_price('buy', price)
        budget = min(budget, cash)
        qty = int(budget // (px * (1 + self.commission_pct))) if px > 0 else 0
        while qty > 0 and qty * px + self.fees(qty * px) > cash:
            qty -= 1  # a minimum commission can push the last share over
        if qty < 1:
            return 0, 'insufficient_cash' if cash < wanted else 'min_notional'
        return qty, None

    def stop_distances(self, atr_pct: Optional[float]) -> Tuple[float, float]:
        """(stop_loss_pct, trailing_stop_pct): ATR-scaled, never under the floors."""
        stop, trail = float(self.stop_rule['stop_loss_pct']), float(self.stop_rule['trailing_stop_pct'])
        if atr_pct:
            from src.agent.volatility import stop_distances
            d = stop_distances(self.config, atr_pct)
            stop, trail = max(stop, d['stop_loss_pct']), max(trail, d['trailing_stop_pct'])
        return stop, trail

    def stop_check(self, symbol: str, price: float,
                   atr_pct: Optional[float] = None) -> Optional[Tuple[str, int, str]]:
        """(reason, quantity, explanation) when a stop has been hit, else None.

        The high since entry is kept in the database, so a restart does not
        forget how far a position had run.
        """
        if not self.stop_rule.get('enabled'):
            return None
        state = self.states().get(symbol.upper())
        if not state or not state.held:
            return None
        high = self.queue.paper_high(symbol.upper(), state.opened_at or '',
                                     max(price, state.last_entry_price))
        stop, trail = self.stop_distances(atr_pct)
        avg = state.cost / state.quantity
        reason = stop_triggered(price, avg, high, stop, trail)
        if not reason:
            return None
        why = (f"stop-loss: {price:.2f} is {1 - price / avg:.1%} below the cost {avg:.2f}"
               if reason == 'stop_loss' else
               f"trailing stop: {price:.2f} is {1 - price / high:.1%} below the high {high:.2f}")
        return reason, int(round(state.quantity)), why

    def fill(self, ticket_id: int, side: str, price: float, quantity: int, order_journal=None):
        """Book an auto paper fill: slippage in the price, commission as fees."""
        px = self.fill_price(side, price)
        return self.queue.mark_filled(
            ticket_id, fill_price=px, fill_quantity=quantity, resolved_by=RESOLVED_BY,
            notes=f'Paper fill at {px:.2f} (decision price {price:.2f})',
            order_journal=order_journal, fees_kes=self.fees(quantity * px))

    # ----------------------------------------------------------- summary

    def summary(self, prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Account figures for the API: cash, holdings at market, P&L, costs.

        A holding with no current price is valued at cost and flagged, rather
        than dropped, so equity never silently loses a position.
        """
        ledger = self._ledger()
        prices = {k.upper(): v for k, v in (prices or {}).items()}
        holdings, value = [], 0.0
        for sym, p in sorted(ledger['positions'].items()):
            if p['quantity'] <= 0:
                continue
            last = prices.get(sym)
            mv = p['quantity'] * last if last else p['cost_kes']
            value += mv
            holdings.append({'symbol': sym, 'quantity': p['quantity'],
                             'avg_cost_kes': p['avg_cost_kes'], 'last_price_kes': last,
                             'market_value_kes': round(mv, 2),
                             'unrealised_pnl_kes': round(mv - p['cost_kes'], 2),
                             'entries': p['entries'], 'priced': bool(last)})
        equity = ledger['cash'] + value
        return {
            'enabled': self.enabled, 'started_at': self.started_at,
            'starting_capital_kes': self.starting_capital,
            'cash_kes': ledger['cash'], 'holdings_value_kes': round(value, 2),
            'equity_kes': round(equity, 2),
            'return_pct': round((equity / self.starting_capital - 1) * 100, 2) if self.enabled else None,
            'realised_pnl_kes': ledger['realised'], 'fees_paid_kes': ledger['fees'],
            'holdings': holdings, 'costs_verified': self.costs_verified,
        }


def order_size(symbol: str, action: str, price: float, notional: float,
               queue, paper: Optional[NsePaperAccount],
               confidence: float = 1.0) -> Tuple[int, Optional[str]]:
    """Quantity for an NSE ticket, or 0 and a skip reason.

    With the paper account on, its rules and cash decide. Without it (the
    manual AIB-AXYS workflow) the recorded trading-book fills decide: no
    shorting, a sell closes the position, and no adding, since there is no
    cost record there to prove a holding profitable; cash is the operator's.
    """
    if paper is not None and paper.enabled:
        return paper.plan(symbol, action, price, notional, confidence)
    held = (queue.positions(book=BOOK).get(symbol.upper()) or {}).get('quantity', 0)
    if action == 'sell':
        return (held, None) if held > 0 else (0, 'no_position')
    if held > 0:
        return 0, 'already_held'
    qty = int(notional / price) if price > 0 else 0
    return (qty, None) if qty >= 1 else (0, 'min_notional')


def execute_stop(paper: NsePaperAccount, queue, symbol: str, price: float, reason: str,
                 quantity: int, why: str, order_journal=None) -> Optional[Dict[str, Any]]:
    """Sell a stopped-out paper position in full. Returns the fill, or None.

    Journaled as `reason` ('stop_loss' / 'trailing_stop') with that as its
    credit, so attribution closes the position against the strategies that
    bought it rather than opening a book for the stop.
    """
    ticket = queue.create_ticket(symbol, 'sell', quantity, suggested_limit_price=round(price, 2),
                                 rationale=why, strategy=reason, strategy_weights={reason: 1.0})
    if not ticket:
        return None
    ok, fill = paper.fill(ticket, 'sell', price, quantity, order_journal=order_journal)
    return fill if ok else None
