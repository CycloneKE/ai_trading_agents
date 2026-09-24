"""Live orders must follow the rules of the backtest that justified them.

The backtest (scripts/run_backtest.py) buys only when nothing is held ("this
book does not pyramid") and sells only to exit a held long; the engine
refuses any sell larger than the position. Live trading had neither rule.
Its idempotency key includes the minute, so a buy signal that stayed on all
day would place a fresh full-size buy every cycle, and a sell with nothing
held opened a short. That was invisible only because no signal ever fired.

These drive the real TradingAgent._execute_trades against a broker that
fills instantly, so the position changes between cycles exactly as it would.
"""
import os
import tempfile
import types
from types import SimpleNamespace

import pytest

from src.agent.main import TradingAgent
from src.agent.order_journal import OrderJournal
from src.connectors.base_broker import OrderResponse, Position
from src.utils.config_validator import load_config


class FillingBroker:
    """Fills every market order immediately at the given price."""

    broker_name = 'fake'

    def __init__(self, price, held=0.0, fail_positions=False, pending=()):
        self.is_connected = True
        self.price = price
        self.cash = 100000.0
        self.held = {'SPY': held} if held else {}
        self.fail_positions = fail_positions
        self.pending = list(pending)
        self.orders = []

    def get_account_info(self):
        equity = self.cash + sum(q * self.price for q in self.held.values())
        return SimpleNamespace(equity=equity, cash=self.cash)

    def fetch_positions(self):
        if self.fail_positions:
            raise RuntimeError('503 from broker')
        return [Position(symbol=s, quantity=q, avg_entry_price=self.price,
                         current_price=self.price, market_value=q * self.price,
                         unrealized_pl=0.0, unrealized_pl_percent=0.0,
                         cost_basis=q * self.price) for s, q in self.held.items() if q]

    def fetch_open_orders(self, symbol=None):
        return [o for o in self.pending if symbol is None or o.symbol == symbol]

    def place_order(self, order):
        order.fill_price = self.price
        self.orders.append(order)
        qty = float(order.quantity)
        if order.side == 'buy':
            self.held[order.symbol] = self.held.get(order.symbol, 0.0) + qty
            self.cash -= qty * self.price
        else:
            self.held[order.symbol] = self.held.get(order.symbol, 0.0) - qty
            self.cash += qty * self.price
        return OrderResponse(order_id=f'o{len(self.orders)}', client_order_id=order.client_order_id,
                             symbol=order.symbol, quantity=qty, filled_quantity=qty,
                             side=order.side, order_type='market', status='filled',
                             created_at=None, updated_at=None, filled_avg_price=self.price)


class _NoOp:
    def __getattr__(self, _):
        return lambda *a, **k: None


@pytest.fixture
def agent_for(tmp_path, monkeypatch):
    clock = {'t': 1_790_000_000.0}
    monkeypatch.setattr('time.time', lambda: clock['t'])

    def build(broker):
        journal = OrderJournal(db_path=str(tmp_path / 'orders.db'))
        agent = SimpleNamespace(
            config=load_config('config/config.json'),
            components={'broker_manager': SimpleNamespace(get_broker=lambda *a: broker)},
            order_journal=journal, monitoring_service=None, risk_manager=_NoOp(),
            event_calendar=SimpleNamespace(risk_multiplier=lambda: 1.0),
            trading_halted=False, _cycle_decisions={},
            _pdt_blocks_order=lambda *a: False,
        )
        agent._position_gate = types.MethodType(TradingAgent._position_gate, agent)

        def cycle(action, symbol='SPY', confidence=0.9, per_strategy=None):
            clock['t'] += 60           # a new loop cycle: a new idempotency key
            agent._cycle_decisions = {symbol: {'symbol': symbol}}
            placed = len(broker.orders)
            TradingAgent._execute_trades(agent, {symbol: {
                'action': action, 'confidence': confidence, 'position_size': 0.05,
                'price': broker.price, 'strategy': 'ensemble',
                'per_strategy': per_strategy or {}}}, {})
            # The live loop syncs fills into the journal before deciding
            # (order_journal.sync_fills); do the same for this broker.
            for o in broker.orders[placed:]:
                journal.mark_final(o.client_order_id, 'filled', filled_quantity=float(o.quantity),
                                   filled_avg_price=o.fill_price)
            return agent._cycle_decisions[symbol]
        cycle.journal = journal
        return cycle
    return build


def test_a_buy_signal_held_for_five_cycles_buys_once(agent_for):
    broker = FillingBroker(price=500.0)
    cycle = agent_for(broker)
    reasons = [cycle('buy').get('skip_reason') for _ in range(5)]
    buys = [o for o in broker.orders if o.side == 'buy']
    assert len(buys) == 1, f'expected exactly one buy, got {len(buys)}'
    # A US holding may be added to, but only once it has proved profitable.
    assert reasons[1:] == ['add_not_profitable'] * 4


def test_a_sell_signal_exits_the_whole_position_once(agent_for):
    broker = FillingBroker(price=500.0, held=7.25)
    cycle = agent_for(broker)
    for _ in range(3):
        cycle('sell')
    sells = [o for o in broker.orders if o.side == 'sell']
    assert len(sells) == 1 and float(sells[0].quantity) == 7.25
    assert broker.held['SPY'] == 0


def test_a_sell_signal_with_nothing_held_never_opens_a_short(agent_for):
    broker = FillingBroker(price=500.0)
    cycle = agent_for(broker)
    assert cycle('sell').get('skip_reason') == 'no_position'
    assert broker.orders == []


def test_no_second_order_while_one_is_still_working(agent_for):
    working = SimpleNamespace(symbol='SPY', status='new')
    broker = FillingBroker(price=500.0, pending=[working])
    cycle = agent_for(broker)
    assert cycle('buy').get('skip_reason') == 'order_pending'
    assert broker.orders == []


def test_unknown_holdings_mean_no_order(agent_for):
    """get_positions() returns [] on an API error; "could not check" must
    never read as "holds nothing"."""
    broker = FillingBroker(price=500.0, fail_positions=True)
    cycle = agent_for(broker)
    assert cycle('buy').get('skip_reason') == 'position_unknown'
    assert broker.orders == []


def test_buy_then_exit_then_buy_again(agent_for):
    """The full round trip the backtest trades."""
    broker = FillingBroker(price=500.0)
    cycle = agent_for(broker)
    cycle('buy'); cycle('buy'); cycle('sell'); cycle('sell'); cycle('buy')
    assert [o.side for o in broker.orders] == ['buy', 'sell', 'buy']


# ------------------------------------------------ adds, trims and credit

def test_a_profitable_holding_is_added_to_once_it_has_run(agent_for):
    broker = FillingBroker(price=500.0)
    cycle = agent_for(broker)
    cycle('buy')
    first = float(broker.orders[0].quantity)
    broker.price = 520.0                     # +4%: not enough yet
    assert cycle('buy').get('skip_reason') == 'add_not_profitable'
    broker.price = 530.0                     # +6% above the entry
    assert cycle('buy').get('executed') is True
    add = float(broker.orders[1].quantity)
    assert broker.orders[1].side == 'buy' and add < first
    # The next add needs another 5% above the add's own price.
    broker.price = 540.0
    assert cycle('buy').get('skip_reason') == 'add_not_profitable'


def test_a_weak_sell_trims_once_a_day_and_a_strong_one_closes(agent_for):
    broker = FillingBroker(price=500.0, held=8.0)
    cycle = agent_for(broker)
    cycle('sell', confidence=0.6)
    assert [o.side for o in broker.orders] == ['sell']
    assert float(broker.orders[0].quantity) == 4.0 and broker.held['SPY'] == 4.0
    assert cycle('sell', confidence=0.6).get('skip_reason') == 'trimmed_today'
    cycle('sell', confidence=0.9)
    assert broker.held['SPY'] == 0 and float(broker.orders[1].quantity) == 4.0


def test_crypto_holdings_are_not_added_to(agent_for):
    broker = FillingBroker(price=60000.0)
    broker.held['BTC-USD'] = 0.5
    cycle = agent_for(broker)
    broker.price = 70000.0
    assert cycle('buy', symbol='BTC-USD').get('skip_reason') == 'already_held'
    assert broker.orders == []


def test_a_blended_order_credits_the_strategies_that_voted_for_it(agent_for):
    import json
    broker = FillingBroker(price=500.0)
    cycle = agent_for(broker)
    cycle('buy', per_strategy={'momentum': {'action': 'buy', 'confidence': 0.6},
                               'rsi_strategy': {'action': 'buy', 'confidence': 0.2},
                               'mean_reversion': {'action': 'sell', 'confidence': 0.7}})
    (row,) = cycle.journal.filled_orders()
    assert json.loads(row['strategy_weights']) == {'momentum': 0.75, 'rsi_strategy': 0.25}
