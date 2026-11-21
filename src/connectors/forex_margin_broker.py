"""The forex margin book: a simulated currency account that can sell short.

The old forex book was a copy of the cash paper broker: it could only buy a
pair with the dollars it held, so it could only ever bet on the dollar
falling. A currency account at a real broker is a margin account: selling
EUR_USD is as ordinary as buying it, positions are sized against equity
rather than cash, and every position open at 17:00 New York time pays or
earns the interest-rate gap between its two currencies (rollover).

This book does the same, at live prices only:

- positions are held in the pair's base currency, positive long, negative
  short; profit is kept in dollars (USD_JPY profit, made in yen, is turned
  into dollars at the current rate);
- every fill pays `spread_per_side` of what is traded;
- an order is refused if gross exposure would pass `max_leverage` times
  equity (a closing or reducing order is never refused);
- `accrue_rollover` charges or credits a day's interest per calendar day a
  position is held, at the two currencies' rates, less `rollover_markup_pct`
  a year (brokers keep a markup on both sides);
- past `max_drawdown` from its highest equity it closes every position and
  refuses new ones until an operator clears `halted` in its state file.

It keeps its own trade log and closed trades (long and short), because the
order journal's round trips are long only. It manages its own risk: the
agent's daily stop-loss loop leaves it alone (`self_managed`), as it leaves
the core and the momentum sleeve.

State is kept in data/<state_file>, written after every change.
"""
import json
import logging
import os
import threading
import uuid
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.connectors.base_broker import AccountInfo, BaseBroker, OrderRequest, OrderResponse, Position
from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)


def _split(pair: str):
    base, _, quote = pair.upper().partition('_')
    return base, quote


def _usd(pair: str, amount_in_quote: float, price: float) -> float:
    """An amount in the pair's quote currency, in dollars."""
    return amount_in_quote if _split(pair)[1] == 'USD' else amount_in_quote / price


def _notional(pair: str, units: float, price: float) -> float:
    return abs(units) if _split(pair)[0] == 'USD' else abs(units) * price


class ForexMarginBroker(BaseBroker):
    """See the module docstring."""

    self_managed = True

    def __init__(self, config: Dict[str, Any], price_for: Optional[Callable[[str], Optional[float]]] = None,
                 state_dir: Optional[str] = None):
        super().__init__(config)
        self.broker_name = 'forex_margin_paper'
        self.is_paper_trading = True
        self.initial_cash = float(config.get('initial_cash', 10000.0))
        self.spread_per_side = float(config.get('spread_per_side', 0.0001))
        self.max_leverage = float(config.get('max_leverage', 3.0))
        self.rollover_markup = float(config.get('rollover_markup_pct', 0.5)) / 100.0
        self.max_drawdown = float(config.get('max_drawdown', 0.15))
        self.state_path = os.path.join(state_dir or str(DATA_DIR),
                                       config.get('state_file', 'forex_margin_state.json'))
        if price_for is None:
            from src.utils.real_price_feed import price_feed
            price_for = price_feed.get_price
        self._price_for = price_for
        self._lock = threading.RLock()
        self.balance = self.initial_cash        # dollars: realised profit, costs and interest included
        self.positions: Dict[str, Dict[str, Any]] = {}   # pair -> {units, entry, opened_at}
        self.fills: List[Dict[str, Any]] = []
        self.closed: List[Dict[str, Any]] = []
        self.last_prices: Dict[str, float] = {}
        self.last_accrual: Optional[str] = None
        self.peak_equity = self.initial_cash
        self.halted: Optional[str] = None
        self.interest_total = 0.0
        self.costs_total = 0.0
        self.fill_count = 0
        self._orders: Dict[str, OrderResponse] = {}
        self._load()

    # ------------------------------------------------------------ plumbing
    def connect(self) -> bool:
        self.is_connected = True
        self.last_connection_time = datetime.now(timezone.utc)
        return True

    def disconnect(self) -> bool:
        self._save()
        self.is_connected = False
        return True

    def price(self, pair: str, live_only: bool = False) -> Optional[float]:
        try:
            px = self._price_for(pair)
        except Exception as e:
            logger.debug(f"Forex margin book: no live price for {pair}: {e}")
            px = None
        if px and px > 0:
            self.last_prices[pair] = float(px)
            return float(px)
        return None if live_only else self.last_prices.get(pair)

    # ------------------------------------------------------------ valuation
    def unrealised(self, pair: str, price: Optional[float] = None) -> float:
        pos = self.positions.get(pair)
        px = price or self.price(pair)
        if not pos or not px:
            return 0.0
        return _usd(pair, pos['units'] * (px - pos['entry']), px)

    def equity(self) -> float:
        with self._lock:
            return self.balance + sum(self.unrealised(p) for p in list(self.positions))

    def gross_exposure(self, overrides: Optional[Dict[str, float]] = None) -> float:
        units = {p: v['units'] for p, v in self.positions.items()}
        units.update(overrides or {})
        total = 0.0
        for p, u in units.items():
            px = self.price(p)
            if u and px:
                total += _notional(p, u, px)
        return total

    def get_account_info(self) -> Optional[AccountInfo]:
        with self._lock:
            eq = self.equity()
            gross = self.gross_exposure()
            margin = gross / self.max_leverage if self.max_leverage else gross
            info = AccountInfo(
                account_id='forex_margin_paper', cash=eq - margin, equity=eq,
                buying_power=max(eq * self.max_leverage - gross, 0.0), initial_margin=margin,
                maintenance_margin=margin / 2, day_trade_count=0,
                last_updated=datetime.now(timezone.utc), broker_name=self.broker_name)
            self.account_info = info
            return info

    def get_positions(self) -> List[Position]:
        out = []
        with self._lock:
            for pair, pos in self.positions.items():
                px = self.price(pair) or pos['entry']
                units = pos['units']
                cost = _notional(pair, units, pos['entry'])
                pnl = self.unrealised(pair, px)
                direction = 1.0 if units > 0 else -1.0
                out.append(Position(
                    symbol=pair, quantity=units, avg_entry_price=pos['entry'], current_price=px,
                    market_value=direction * _notional(pair, units, px), unrealized_pl=pnl,
                    unrealized_pl_percent=(pnl / cost) if cost else 0.0, cost_basis=cost,
                    broker_name=self.broker_name))
        return out

    def get_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        return [o for o in self._orders.values()
                if (symbol is None or o.symbol == symbol) and o.status not in ('filled', 'rejected', 'canceled')]

    def get_order_by_client_order_id(self, client_order_id: str):
        return next((o for o in self._orders.values() if o.client_order_id == client_order_id), None)

    def cancel_order(self, order_id: str) -> bool:
        return False                          # every order fills or is refused at once

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        px = self.price(symbol, live_only=True)
        if not px:
            return None
        return {'symbol': symbol, 'price': px, 'bid': px * (1 - self.spread_per_side),
                'ask': px * (1 + self.spread_per_side), 'timestamp': datetime.now(timezone.utc).isoformat()}

    # ------------------------------------------------------------ trading
    def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        """Fill a market order at once at the live price, or refuse it."""
        side = str(order.side).lower()
        qty = abs(float(order.quantity or 0))
        delta = qty if side == 'buy' else -qty
        resp = self.trade(order.symbol.upper(), delta, client_order_id=order.client_order_id)
        return resp

    def trade(self, pair: str, delta_units: float, client_order_id: Optional[str] = None,
              reason: str = 'rebalance', now: Optional[datetime] = None) -> Optional[OrderResponse]:
        """Change a position by `delta_units` (base currency, signed).
        Returns the filled order, a rejected one, or None without a price."""
        now = now or datetime.now(timezone.utc)
        if not self.is_connected or not delta_units:
            return None
        px = self.price(pair, live_only=True)
        if not px:
            logger.warning(f"Forex margin book: no live price for {pair}; order not placed")
            return None
        with self._lock:
            have = self.positions.get(pair, {}).get('units', 0.0)
            new = have + delta_units
            reduces = abs(new) <= abs(have) and (new == 0 or (new > 0) == (have > 0))
            side = 'buy' if delta_units > 0 else 'sell'
            resp = OrderResponse(
                order_id=str(uuid.uuid4()), client_order_id=client_order_id or f'fx_{uuid.uuid4().hex[:12]}',
                symbol=pair, quantity=abs(delta_units), filled_quantity=0.0, side=side,
                order_type='market', status='rejected', created_at=now, updated_at=now,
                broker_name=self.broker_name)
            self._orders[resp.order_id] = resp
            if not reduces:
                if self.halted:
                    logger.warning(f"Forex margin book is halted ({self.halted}); {side} {pair} refused")
                    return resp
                eq = self.equity()
                if eq <= 0 or self.gross_exposure({pair: new}) > self.max_leverage * eq * 1.0001:
                    logger.warning(f"Forex margin book: {side} {pair} refused, gross exposure would pass "
                                   f"{self.max_leverage:g}x equity")
                    return resp
            fill = px * (1 + self.spread_per_side) if delta_units > 0 else px * (1 - self.spread_per_side)
            self._apply(pair, delta_units, fill, now, reason)
            cost = _notional(pair, delta_units, px) * self.spread_per_side
            self.costs_total += cost
            self.fill_count += 1
            resp.status, resp.filled_quantity, resp.filled_avg_price = 'filled', abs(delta_units), fill
            self.fills.append({'id': resp.client_order_id, 'symbol': pair, 'side': side,
                               'quantity': abs(delta_units), 'price': fill, 'strategy': reason,
                               'spread_cost_usd': round(cost, 4), 'at': now.isoformat()})
            self.fills = self.fills[-500:]
            self._save()
            logger.info(f"Forex margin book: {side} {abs(delta_units):,.0f} {pair} @ {fill:.5f} ({reason})")
            return resp

    def _apply(self, pair: str, delta: float, fill: float, now: datetime, reason: str) -> None:
        """Book a fill: close what it closes (realising profit in dollars),
        open what is left at the fill price."""
        pos = self.positions.get(pair)
        have = pos['units'] if pos else 0.0
        if have and (have > 0) != (delta > 0):
            closing = min(abs(delta), abs(have)) * (1 if have > 0 else -1)
            pnl = _usd(pair, closing * (fill - pos['entry']), fill)
            self.balance += pnl
            cost = _notional(pair, closing, pos['entry'])
            opened = pos.get('opened_at')
            days = None
            try:
                days = (now - datetime.fromisoformat(opened)).days if opened else None
            except ValueError:
                pass
            self.closed.append({'symbol': pair, 'direction': 'long' if have > 0 else 'short',
                                'exit': reason, 'ret': (pnl / cost) if cost else 0.0, 'cost': cost,
                                'pnl_usd': pnl, 'days': days, 'closed_at': now.isoformat()})
            self.closed = self.closed[-500:]
            remaining = have - closing
            delta -= -closing                       # the part of the order not used to close
            delta = delta if abs(delta) > 1e-9 else 0.0
            if abs(remaining) > 1e-9:
                pos['units'] = remaining
                return
            self.positions.pop(pair, None)
            have, pos = 0.0, None
        if delta:
            if pos:                                  # adding to a position on the same side
                units = have + delta
                pos['entry'] = (have * pos['entry'] + delta * fill) / units
                pos['units'] = units
            else:
                self.positions[pair] = {'units': delta, 'entry': fill, 'opened_at': now.isoformat()}

    # ------------------------------------------------------------ interest and risk
    def accrue_rollover(self, rates_pct: Dict[str, float], today: Optional[date] = None) -> float:
        """Interest for every calendar day since the last accrual, at
        today's prices and rates. Returns the dollars credited (negative:
        charged). The first call only starts the clock."""
        today = today or datetime.now(timezone.utc).date()
        with self._lock:
            last = date.fromisoformat(self.last_accrual) if self.last_accrual else None
            self.last_accrual = today.isoformat()
            if last is None or today <= last:
                self._save()
                return 0.0
            days = (today - last).days
            total = 0.0
            for pair, pos in self.positions.items():
                base, quote = _split(pair)
                px = self.price(pair)
                if not px or base not in rates_pct or quote not in rates_pct:
                    continue
                size = _notional(pair, pos['units'], px)
                direction = 1.0 if pos['units'] > 0 else -1.0
                gap = (rates_pct[base] - rates_pct[quote]) / 100.0
                total += size * (direction * gap - self.rollover_markup) * days / 365.0
            self.balance += total
            self.interest_total += total
            self._save()
            return total

    def check_drawdown(self, now: Optional[datetime] = None) -> Optional[str]:
        """Close everything and halt past max_drawdown from the peak.
        Uses live prices only; returns the halt reason when it halts."""
        with self._lock:
            if any(self.price(p, live_only=True) is None for p in self.positions):
                return None
            eq = self.equity()
            self.peak_equity = max(self.peak_equity, eq)
            if self.halted or not self.max_drawdown or eq >= self.peak_equity * (1 - self.max_drawdown):
                self._save()
                return None
            reason = (f"drawdown {1 - eq / self.peak_equity:.1%} passed the {self.max_drawdown:.0%} limit "
                      f"(equity {eq:,.2f}, peak {self.peak_equity:,.2f})")
            for pair in list(self.positions):
                self.trade(pair, -self.positions[pair]['units'], reason='drawdown_halt', now=now)
            self.halted = reason
            self._save()
            logger.critical(f"Forex margin book halted: {reason}")
            return reason

    def current_units(self) -> Dict[str, float]:
        return {p: v['units'] for p, v in self.positions.items()}

    # ------------------------------------------------------------ for the dashboard
    def own_fills(self) -> List[Dict[str, Any]]:
        return list(reversed(self.fills))[:100]

    def own_round_trips(self) -> List[Dict[str, Any]]:
        return list(self.closed)

    def summary(self) -> Dict[str, Any]:
        return {'interest_usd': round(self.interest_total, 2), 'spread_costs_usd': round(self.costs_total, 2),
                'gross_exposure_usd': round(self.gross_exposure(), 2), 'peak_equity': round(self.peak_equity, 2),
                'halted': self.halted, 'max_leverage': self.max_leverage, 'max_drawdown': self.max_drawdown}

    # ------------------------------------------------------------ state
    def _save(self) -> None:
        state = {'balance': self.balance, 'positions': self.positions, 'fills': self.fills,
                 'closed': self.closed, 'last_prices': self.last_prices, 'last_accrual': self.last_accrual,
                 'peak_equity': self.peak_equity, 'halted': self.halted,
                 'interest_total': self.interest_total, 'costs_total': self.costs_total,
                 'fill_count': self.fill_count, 'initial_cash': self.initial_cash,
                 'saved_at': datetime.now(timezone.utc).isoformat()}
        try:
            os.makedirs(os.path.dirname(self.state_path) or '.', exist_ok=True)
            tmp = self.state_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, self.state_path)
        except OSError as e:
            logger.error(f"Forex margin book state not saved: {e}")

    def _load(self) -> None:
        try:
            with open(self.state_path, encoding='utf-8') as f:
                s = json.load(f) or {}
        except (OSError, ValueError):
            return
        self.balance = float(s.get('balance', self.initial_cash))
        self.positions = {p: {'units': float(v['units']), 'entry': float(v['entry']),
                              'opened_at': v.get('opened_at')}
                          for p, v in (s.get('positions') or {}).items() if float(v.get('units', 0))}
        self.fills = s.get('fills') or []
        self.closed = s.get('closed') or []
        self.last_prices = {k: float(v) for k, v in (s.get('last_prices') or {}).items()}
        self.last_accrual = s.get('last_accrual')
        self.peak_equity = float(s.get('peak_equity', self.initial_cash))
        self.halted = s.get('halted')
        self.interest_total = float(s.get('interest_total', 0.0))
        self.costs_total = float(s.get('costs_total', 0.0))
        self.fill_count = int(s.get('fill_count', len(self.fills)))
