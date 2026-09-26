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
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agent.cost_model import market_costs
from src.agent.position_rules import (
    ADD_DEFAULTS, EXIT_DEFAULTS, HOLDING_RULE_REASONS, PositionState,
    plan_add, plan_exit, replay, rule, stop_triggered)

logger = logging.getLogger(__name__)

BOOK = 'trading'
RESOLVED_BY = 'auto_paper_trader'

__all__ = ['NsePaperAccount', 'order_size', 'execute_stop', 'HOLDING_RULE_REASONS', 'BOOK']

# NSE stops sit wider than the US ones. A round trip costs about 3.9%, and
# a daily-bar position stopped out by an ordinary day's noise pays that for
# nothing. Where the stock's ATR is known the stop scales with it, but never
# tighter than these.
STOP_DEFAULTS = {'enabled': True, 'stop_loss_pct': 0.08, 'trailing_stop_pct': 0.10}

# How often and how big the account may trade. At about 3.9% a round trip,
# frequent NSE trading loses to costs; a paper fill for more shares than the
# market trades in a day could never happen; and sale proceeds are not cash
# until CDSC settles them three trading days later. Stops ignore the first
# three: they sell whenever they trigger.
# Kenyan withholding tax on dividends from NSE-listed companies for a
# resident individual: final, deducted at source (KRA).
DIVIDEND_WHT_DEFAULT = 0.05
DIVIDEND_EVENTS_DEFAULT = 'config/nse_dividend_events.json'

LIMIT_DEFAULTS = {'max_adv_fraction': 0.10, 'adv_days': 20, 'min_holding_days': 30,
                  'max_new_positions_per_week': 2, 'settlement_days': 3}


def _parse(ts: Optional[str]) -> datetime:
    """An ISO timestamp as a naive UTC datetime; the epoch when missing."""
    if not ts:
        return datetime(1970, 1, 1)
    dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
    return dt.replace(tzinfo=None) if dt.tzinfo is None else \
        dt.astimezone(timezone.utc).replace(tzinfo=None)


def subtract_trading_days(day: datetime, n: int) -> datetime:
    """`day` moved back `n` weekdays."""
    while n > 0:
        day -= timedelta(days=1)
        if day.weekday() < 5:
            n -= 1
    return day


def trading_days_since(ts: Optional[str], now: datetime) -> int:
    """Weekdays after the day of `ts`, up to and including `now`'s day.

    NSE settles T+3 trading days. Public holidays are not counted out, so a
    holiday week settles a day early here: close enough for a paper account.
    """
    day, end, n = _parse(ts).date(), now.date(), 0
    while day < end:
        day += timedelta(days=1)
        if day.weekday() < 5:
            n += 1
    return n


def held_days(opened_at: Optional[str], now: datetime) -> float:
    """Calendar days since the position opened; unknown counts as long held."""
    if not opened_at:
        return float('inf')
    return (now - _parse(opened_at)).total_seconds() / 86400


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
        self.costs = costs
        # The broker's yearly account fee (KES 200 at AIB-AXYS), charged at
        # the start of each account year.
        self.annual_fee_kes = float(costs.get('annual_fee_kes') or 0.0)
        self.add_rule = rule(ADD_DEFAULTS, cfg.get('add_to_winners'))
        self.exit_rule = rule(EXIT_DEFAULTS, cfg.get('exits'))
        self.stop_rule = rule(STOP_DEFAULTS, cfg.get('stop_loss'))
        self.limits = rule(LIMIT_DEFAULTS, cfg.get('trading_limits'))
        self.dividend_wht = float(cfg.get('dividend_withholding_pct', DIVIDEND_WHT_DEFAULT))
        self.dividend_events_path = cfg.get('dividend_events_path', DIVIDEND_EVENTS_DEFAULT)
        self.started_at = queue.paper_account_started_at() if self.enabled else None
        if self.enabled and not queue.paper_equity_history():
            # The equity curve starts where the account does.
            queue.record_paper_equity(datetime.utcnow().date().isoformat(), self.starting_capital, 0.0)

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

    def account_fees(self, now: Optional[datetime] = None) -> float:
        """Yearly account fees charged so far: one per account year begun."""
        if not self.annual_fee_kes or not self.started_at:
            return 0.0
        start = datetime.fromisoformat(self.started_at[:19])
        years = max(((now or datetime.utcnow()) - start).days, 0) // 365 + 1
        return round(self.annual_fee_kes * years, 2)

    def dividend_events(self) -> List[Dict[str, Any]]:
        """Announced dividends from the operator's events file; [] if none."""
        path = Path(self.dividend_events_path)
        if not path.is_absolute():
            from src.utils.paths import PROJECT_ROOT
            path = PROJECT_ROOT / path
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return []
        events = data.get('events', []) if isinstance(data, dict) else data
        return [e for e in events if isinstance(e, dict) and e.get('symbol')
                and e.get('dividend_kes') and e.get('payment_date')
                and (e.get('ex_date') or e.get('book_closure'))]

    def dividends(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Dividends the account has been paid: shares held the day before
        the ex-date, credited on the payment date, net of withholding tax.

        Without an ex-date the book-closure date stands in, less three
        trading days: NSE settles T+3, so a buyer must trade that early to be
        on the register when the books close.
        """
        now = now or datetime.utcnow()
        opened = _parse(self.started_at) if self.started_at else None
        fills = list(self._fills())
        out = []
        for e in self.dividend_events():
            try:
                paid = _parse(e['payment_date'])
                ex = _parse(e['ex_date']) if e.get('ex_date') else \
                    subtract_trading_days(_parse(e['book_closure']), 3)
            except ValueError:
                continue
            if paid > now or (opened and ex < opened):
                continue
            sym = str(e['symbol']).upper()
            held = 0.0
            for f in fills:
                if f['symbol'] == sym and _parse(f['fill_at']) < ex:
                    held += f['quantity'] if f['side'] == 'buy' else -f['quantity']
            shares = int(round(max(held, 0.0)))
            if shares <= 0:
                continue
            gross = round(shares * float(e['dividend_kes']), 2)
            tax = round(gross * self.dividend_wht, 2)
            out.append({'symbol': sym, 'shares': shares, 'dividend_kes': float(e['dividend_kes']),
                        'ex_date': ex.date().isoformat(), 'payment_date': paid.date().isoformat(),
                        'gross_kes': gross, 'tax_kes': tax, 'net_kes': round(gross - tax, 2)})
        return out

    def _ledger(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Cash, fees paid, realised P&L and positions, from the fills."""
        now = now or datetime.utcnow()
        account_fees = self.account_fees(now)
        paid = self.dividends(now)
        dividends_net = round(sum(d['net_kes'] for d in paid), 2)
        cash, fees, unsettled = self.starting_capital - account_fees + dividends_net, account_fees, 0.0
        settle = int(self.limits.get('settlement_days') or 0)
        for f in self._fills():
            notional = f['quantity'] * f['price']
            fees += f['fees_kes']
            cash += -(notional + f['fees_kes']) if f['side'] == 'buy' else notional - f['fees_kes']
            if f['side'] == 'sell' and settle and trading_days_since(f['fill_at'], now) < settle:
                unsettled += notional - f['fees_kes']
        states = self.states()
        positions = {
            s: {'quantity': int(round(p.quantity)), 'cost_kes': round(p.cost, 2),
                'avg_cost_kes': round(p.cost / p.quantity, 4) if p.held else 0.0,
                'entries': p.entries, 'last_entry_kes': p.last_entry_price}
            for s, p in states.items()}
        return {'cash': round(cash, 2), 'fees': round(fees, 2), 'account_fees': account_fees,
                'dividends': paid, 'dividends_net': dividends_net,
                'dividend_tax': round(sum(d['tax_kes'] for d in paid), 2),
                'unsettled': round(unsettled, 2),
                'available': round(cash - unsettled, 2),
                'states': states,
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
             confidence: float = 1.0, adv: Optional[float] = None,
             now: Optional[datetime] = None) -> Tuple[int, Optional[str]]:
        """Quantity to trade under the account's rules, or 0 and why not.

        Reasons match the US book's (see TradingAgent._position_gate) so the
        dashboard reads them the same way. `adv` is the stock's average daily
        volume in shares, which caps the order (trading_limits).
        """
        now = now or datetime.utcnow()
        ledger = self._ledger(now)
        state = ledger['states'].get(symbol.upper()) or PositionState()
        account = self._account_at_cost(ledger)
        if side == 'sell':
            if state.held and held_days(state.opened_at, now) < self.limits['min_holding_days']:
                return 0, 'min_holding'
            qty, reason = plan_exit(state, confidence, price, account,
                                    now.date().isoformat(), self.exit_rule)
            if reason:
                return 0, reason
            held = int(round(state.quantity))
            shares = int(qty) if int(qty) >= 1 else held  # a trim under one share closes it
            return self._liquidity_cap(min(shares, held), adv)
        if not state.held:
            if self.new_positions_since(now - timedelta(days=7)) >= \
                    self.limits['max_new_positions_per_week']:
                return 0, 'turnover_budget'
            qty, reason = self._buyable(price, float(target_notional), ledger['available'],
                                        target_notional)
            return (qty, reason) if reason else self._liquidity_cap(qty, adv)
        value, reason = plan_add(state, price, self.slippage_pct + self.commission_pct,
                                 account, float(target_notional), self.add_rule)
        if reason:
            return 0, reason
        qty, reason = self._buyable(price, value, ledger['available'], value)
        wanted = float(target_notional) * self.add_rule['add_size_pct']
        if reason == 'min_notional' and value < wanted:
            reason = 'position_cap'  # the cap, not the order, left too little room
        return (qty, reason) if reason else self._liquidity_cap(qty, adv)

    def _liquidity_cap(self, qty: int, adv: Optional[float]) -> Tuple[int, Optional[str]]:
        """At most max_adv_fraction of a day's average volume; no cap without volume data."""
        frac = float(self.limits.get('max_adv_fraction') or 0.0)
        if qty <= 0 or not adv or frac <= 0:
            return qty, None
        cap = int(adv * frac)
        return (min(qty, cap), None) if cap >= 1 else (0, 'liquidity_cap')

    def new_positions_since(self, since: datetime) -> int:
        """Buys that opened a position (not adds) at or after `since`."""
        held: Dict[str, float] = {}
        count = 0
        for f in self._fills():
            before = held.get(f['symbol'], 0.0)
            if f['side'] == 'buy':
                if before <= 0 and _parse(f['fill_at']) >= since:
                    count += 1
                held[f['symbol']] = before + f['quantity']
            else:
                held[f['symbol']] = max(before - f['quantity'], 0.0)
        return count

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

    def stop_levels(self, symbol: str, atr_pct: Optional[float] = None) -> Optional[Dict[str, float]]:
        """Where a holding's stops sit now, for display. Updates nothing."""
        state = self.states().get(symbol.upper())
        if not state or not state.held or not self.stop_rule.get('enabled'):
            return None
        stop, trail = self.stop_distances(atr_pct)
        avg = state.cost / state.quantity
        high = self.queue.paper_high_peek(symbol.upper(), state.opened_at or '') or state.last_entry_price
        return {'stop_loss_kes': round(avg * (1 - stop), 2),
                'trailing_stop_kes': round(high * (1 - trail), 2),
                'high_since_entry_kes': round(high, 2), 'stop_loss_pct': stop,
                'trailing_stop_pct': trail}

    def record_equity(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """Today's reading of the account's value, for the equity curve."""
        s = self.summary(prices)
        self.queue.record_paper_equity(datetime.utcnow().date().isoformat(),
                                       s['cash_kes'], s['holdings_value_kes'])
        return s

    def fill(self, ticket_id: int, side: str, price: float, quantity: int, order_journal=None):
        """Book an auto paper fill: slippage in the price, commission as fees."""
        px = self.fill_price(side, price)
        return self.queue.mark_filled(
            ticket_id, fill_price=px, fill_quantity=quantity, resolved_by=RESOLVED_BY,
            notes=f'Paper fill at {px:.2f} (decision price {price:.2f})',
            order_journal=order_journal, fees_kes=self.fees(quantity * px))

    # ----------------------------------------------------------- summary

    def summary(self, prices: Optional[Dict[str, float]] = None,
                atr_lookup=None) -> Dict[str, Any]:
        """Account figures for the API: cash, holdings at market, P&L, costs.

        A holding with no current price is valued at cost and flagged, rather
        than dropped, so equity never silently loses a position. With an
        `atr_lookup` (symbol -> ATR fraction) each holding carries its stops.
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
            holding = {'symbol': sym, 'quantity': p['quantity'],
                       'avg_cost_kes': p['avg_cost_kes'], 'last_price_kes': last,
                       'cost_kes': p['cost_kes'], 'market_value_kes': round(mv, 2),
                       'unrealised_pnl_kes': round(mv - p['cost_kes'], 2),
                       'unrealised_pnl_pct': round((mv / p['cost_kes'] - 1) * 100, 2) if p['cost_kes'] else 0.0,
                       'entries': p['entries'], 'priced': bool(last)}
            if atr_lookup is not None:
                try:
                    atr = atr_lookup(sym)
                except Exception:
                    atr = None
                holding['stops'] = self.stop_levels(sym, atr)
            holdings.append(holding)
        equity = ledger['cash'] + value
        return {
            'enabled': self.enabled, 'started_at': self.started_at,
            'starting_capital_kes': self.starting_capital,
            'cash_kes': ledger['cash'], 'holdings_value_kes': round(value, 2),
            'unsettled_kes': ledger['unsettled'], 'available_cash_kes': ledger['available'],
            'equity_kes': round(equity, 2),
            'return_pct': round((equity / self.starting_capital - 1) * 100, 2) if self.enabled else None,
            'realised_pnl_kes': ledger['realised'], 'fees_paid_kes': ledger['fees'],
            'account_fees_kes': ledger['account_fees'],
            'dividends_net_kes': ledger['dividends_net'],
            'dividend_tax_kes': ledger['dividend_tax'],
            'dividends': ledger['dividends'],
            'holdings': holdings, 'costs_verified': self.costs_verified,
        }


def order_size(symbol: str, action: str, price: float, notional: float,
               queue, paper: Optional[NsePaperAccount],
               confidence: float = 1.0, adv: Optional[float] = None) -> Tuple[int, Optional[str]]:
    """Quantity for an NSE ticket, or 0 and a skip reason.

    With the paper account on, its rules and cash decide. Without it (the
    manual AIB-AXYS workflow) the recorded trading-book fills decide: no
    shorting, a sell closes the position, and no adding, since there is no
    cost record there to prove a holding profitable; cash is the operator's.
    """
    if paper is not None and paper.enabled:
        return paper.plan(symbol, action, price, notional, confidence, adv=adv)
    held = (queue.positions(book=BOOK).get(symbol.upper()) or {}).get('quantity', 0)
    if action == 'sell':
        return (held, None) if held > 0 else (0, 'no_position')
    if held > 0:
        return 0, 'already_held'
    qty = int(notional / price) if price > 0 else 0
    if qty < 1:
        return 0, 'min_notional'
    if adv:  # a ticket for more than a tenth of a day's volume will not fill
        cap = int(adv * LIMIT_DEFAULTS['max_adv_fraction'])
        return (min(qty, cap), None) if cap >= 1 else (0, 'liquidity_cap')
    return qty, None


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
