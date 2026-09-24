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

This account holds KES `nse_paper_trading.starting_capital_kes` and follows
the same rules the backtest and the US paper book do: buy only when nothing
is held, sell only what is held and all of it, never short, never spend
more cash than there is. Each fill pays slippage (in the price) and
commission (in `fees_kes`). Its state is derived from the filled tickets in
the 'trading' book since the account opened, so there is one record of what
happened and nothing to keep in step with it.
"""
import logging
import math
from typing import Any, Dict, Optional, Tuple

from src.agent.cost_model import market_costs

logger = logging.getLogger(__name__)

BOOK = 'trading'
RESOLVED_BY = 'auto_paper_trader'


class NsePaperAccount:
    def __init__(self, queue, config: Dict[str, Any]):
        self.queue = queue
        cfg = config.get('nse_paper_trading') or {}
        self.starting_capital = float(cfg.get('starting_capital_kes') or 0.0)
        costs = market_costs('nse', config)
        self.commission_pct = float(costs.get('commission_pct') or 0.0)
        self.min_commission = float(costs.get('min_commission') or 0.0)
        self.slippage_pct = float(costs.get('slippage_pct') or 0.0)
        self.costs_verified = bool(costs.get('verified'))
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

    def positions(self) -> Dict[str, Dict[str, Any]]:
        """Held symbols: quantity and average cost per share including fees."""
        return {s: p for s, p in self._ledger()['positions'].items() if p['quantity'] > 0}

    def cash(self) -> float:
        return self._ledger()['cash']

    def _ledger(self) -> Dict[str, Any]:
        """Replay the account's fills: cash, holdings at cost, realised P&L."""
        cash, realised, fees = self.starting_capital, 0.0, 0.0
        book: Dict[str, Dict[str, float]] = {}
        for f in self._fills():
            p = book.setdefault(f['symbol'], {'quantity': 0, 'cost': 0.0})
            notional = f['quantity'] * f['price']
            fees += f['fees_kes']
            if f['side'] == 'buy':
                cash -= notional + f['fees_kes']
                p['quantity'] += f['quantity']
                p['cost'] += notional + f['fees_kes']
            else:
                cash += notional - f['fees_kes']
                qty = min(f['quantity'], p['quantity'])
                basis = p['cost'] * qty / p['quantity'] if p['quantity'] > 0 else 0.0
                realised += notional - f['fees_kes'] - basis
                p['quantity'] -= f['quantity']
                p['cost'] -= basis
        positions = {
            s: {'quantity': p['quantity'], 'cost_kes': round(p['cost'], 2),
                'avg_cost_kes': round(p['cost'] / p['quantity'], 4) if p['quantity'] > 0 else 0.0}
            for s, p in book.items()}
        return {'cash': round(cash, 2), 'realised': round(realised, 2),
                'fees': round(fees, 2), 'positions': positions}

    # ------------------------------------------------------------- rules

    def plan(self, symbol: str, side: str, price: float,
             target_notional: float) -> Tuple[int, Optional[str]]:
        """Quantity to trade under the account's rules, or 0 and why not.

        Reasons match the US book's (see TradingAgent._position_gate) so the
        dashboard reads them the same way.
        """
        held = self.positions().get(symbol.upper(), {}).get('quantity', 0)
        if side == 'sell':
            return (held, None) if held > 0 else (0, 'no_position')
        if held > 0:
            return 0, 'already_held'
        px = self.fill_price('buy', price)
        budget = min(float(target_notional), self.cash())
        qty = int(budget // (px * (1 + self.commission_pct))) if px > 0 else 0
        while qty > 0 and qty * px + self.fees(qty * px) > self.cash():
            qty -= 1  # a minimum commission can push the last share over
        if qty < 1:
            return 0, 'insufficient_cash' if self.cash() < target_notional else 'min_notional'
        return qty, None

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
                             'priced': bool(last)})
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
               queue, paper: Optional[NsePaperAccount]) -> Tuple[int, Optional[str]]:
    """Quantity for an NSE ticket, or 0 and a skip reason.

    With the paper account on, its rules and cash decide. Without it (the
    manual AIB-AXYS workflow) the same holding rules apply to the recorded
    trading-book fills, since the NSE does not let a retail account short
    and the backtest never added to a position; cash is then the operator's.
    """
    if paper is not None and paper.enabled:
        return paper.plan(symbol, action, price, notional)
    held = (queue.positions(book=BOOK).get(symbol.upper()) or {}).get('quantity', 0)
    if action == 'sell':
        return (held, None) if held > 0 else (0, 'no_position')
    if held > 0:
        return 0, 'already_held'
    qty = int(notional / price) if price > 0 else 0
    return (qty, None) if qty >= 1 else (0, 'min_notional')
