"""Integration tests driving the REAL TradingAgent._execute_trades.

The rest of the suite mirrors the execution-plan logic in a local helper
rather than calling the method, which is precisely how the pyramiding defect
survived 232 passing tests. These drive the actual code path with a fake
broker so the regression is caught where it happened.

Reproduces the 2026-07-24 journal: the same buy signal repeating every
60-second cycle, with nothing checking the position already held.
"""

import os
import sys
import tempfile
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.exposure import ExposureGuard
from src.agent.main import TradingAgent
from src.agent.order_journal import OrderJournal
from src.connectors.base_broker import AccountInfo, OrderResponse, Position

EQUITY = 100_000.0
PRICE = 310.0


class FakeBroker:
    """Accumulates fills into a position, the way a real broker does."""

    broker_name = 'fake'
    is_connected = True

    def __init__(self, equity=EQUITY):
        self.equity = equity
        self.qty = 0.0
        self.placed = []

    def get_account_info(self):
        return AccountInfo(
            account_id='a', cash=self.equity * 0.9, equity=self.equity,
            buying_power=self.equity, initial_margin=0, maintenance_margin=0,
            day_trade_count=0, last_updated=datetime(2026, 7, 24),
            broker_name=self.broker_name)

    def get_positions(self):
        if not self.qty:
            return []
        return [Position(symbol='TSLA', quantity=self.qty, avg_entry_price=PRICE,
                         current_price=PRICE, market_value=self.qty * PRICE,
                         unrealized_pl=0.0, unrealized_pl_percent=0.0,
                         cost_basis=abs(self.qty) * PRICE,
                         broker_name=self.broker_name)]

    def get_orders(self, symbol=None):
        return []

    def place_order(self, order):
        self.placed.append(order)
        self.qty += order.quantity if order.side == 'buy' else -order.quantity
        return OrderResponse(
            order_id=f'o{len(self.placed)}', client_order_id=order.client_order_id,
            symbol=order.symbol, quantity=order.quantity,
            filled_quantity=order.quantity, side=order.side,
            order_type=order.order_type, status='filled',
            created_at=datetime(2026, 7, 24), updated_at=datetime(2026, 7, 24),
            filled_avg_price=PRICE, broker_name=self.broker_name)


class FakeBrokerManager:
    def __init__(self, broker):
        self._broker = broker

    def get_broker(self, name=None):
        return self._broker


def make_agent(broker, cfg_overrides=None):
    """Build a TradingAgent shell with only what _execute_trades touches.

    Deliberately avoids __init__ (which starts threads, connectors and the
    API server) while still exercising the real method under test.
    """
    agent = object.__new__(TradingAgent)
    agent.config = {
        'risk_limits': {'max_position_size': 0.05, 'reentry_cooldown_seconds': 900},
        'trading': {'allow_fractional': True, 'min_notional': 5.0},
        'trading_loop_interval': 60,
        'cash_policy': {'enabled': False},
    }
    if cfg_overrides:
        agent.config.update(cfg_overrides)
    agent.components = {'broker_manager': FakeBrokerManager(broker)}
    agent.trading_halted = False
    agent.monitoring_service = None
    agent._cycle_decisions = {}

    fd, jpath = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    agent.order_journal = OrderJournal(db_path=jpath)

    fd, epath = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    agent.exposure_guard = ExposureGuard(agent.config, db_path=epath)

    agent._pdt_blocks_order = lambda *a, **k: False
    return agent, jpath, epath


@pytest.fixture
def rig():
    broker = FakeBroker()
    agent, jpath, epath = make_agent(broker)
    yield agent, broker
    agent.order_journal.close()
    agent.exposure_guard.close()
    os.unlink(jpath)
    os.unlink(epath)


def buy_signal(confidence=0.8, position_size=0.02):
    return {'TSLA': {'action': 'buy', 'confidence': confidence,
                     'position_size': position_size, 'price': PRICE,
                     'strategy': 'ensemble'}}


def run_cycles(agent, signals, n, cycle_start=1_000_000):
    """Drive n distinct decision cycles (distinct client_order_ids, so the
    journal's own dedup is not what stops the pyramiding)."""
    import src.agent.main as main_mod
    real_time = main_mod.time.time
    try:
        for i in range(n):
            main_mod.time.time = lambda i=i: float((cycle_start + i) * 60)
            agent._cycle_decisions = {
                s: {'symbol': s, 'executed': False, 'skip_reason': None}
                for s in signals
            }
            agent._execute_trades(dict(signals), {})
    finally:
        main_mod.time.time = real_time


def test_single_entry_still_executes_normally(rig):
    """Guard must not break the ordinary path."""
    agent, broker = rig
    run_cycles(agent, buy_signal(), 1)
    assert len(broker.placed) == 1
    assert broker.placed[0].side == 'buy'


def test_repeated_signal_stops_at_the_position_cap(rig):
    """THE regression: 20 identical buy cycles. Before the fix this produced
    20 fills and a position ~4x the configured 5% cap."""
    agent, broker = rig
    run_cycles(agent, buy_signal(), 20)

    final_notional = broker.qty * PRICE
    assert final_notional <= 0.05 * EQUITY + 0.01, (
        f"position grew to ${final_notional:,.0f} against a "
        f"${0.05 * EQUITY:,.0f} cap over {len(broker.placed)} orders")


def test_orders_stop_being_placed_once_capped(rig):
    agent, broker = rig
    run_cycles(agent, buy_signal(), 20)
    # Some orders fill the cap, the rest must be refused outright.
    assert len(broker.placed) < 20


def test_capped_cycle_records_a_readable_skip_reason(rig):
    agent, broker = rig
    run_cycles(agent, buy_signal(), 20)
    assert agent._cycle_decisions['TSLA']['skip_reason'] == 'exposure_cap'


def test_short_side_is_capped_the_same_way(rig):
    """XLF/XLV: repeated sells with no long to close stacked a short."""
    agent, broker = rig
    sell = {'TSLA': {'action': 'sell', 'confidence': 0.8, 'position_size': 0.02,
                     'price': PRICE, 'strategy': 'ensemble'}}
    run_cycles(agent, sell, 20)
    assert abs(broker.qty) * PRICE <= 0.05 * EQUITY + 0.01


def test_cooldown_blocks_rebuy_after_a_forced_exit(rig):
    """The 15:22:31 rebuy, ten seconds after the stop filled."""
    agent, broker = rig
    agent.exposure_guard.record_forced_exit('TSLA', 'stop_loss')
    run_cycles(agent, buy_signal(), 5)
    assert broker.placed == []
    assert agent._cycle_decisions['TSLA']['skip_reason'] == 'exposure_cap'


def test_trading_resumes_once_the_cooldown_expires(rig):
    """The gate must be a delay, not a permanent ban."""
    agent, broker = rig
    agent.exposure_guard.cooldown_seconds = 0.0     # window elapsed
    agent.exposure_guard.record_forced_exit('TSLA', 'stop_loss')
    run_cycles(agent, buy_signal(), 1)
    assert len(broker.placed) == 1


def test_net_exposure_is_passed_to_the_gate(rig):
    """The cap is worthless if the loop never tells the gate what the book
    currently looks like."""
    agent, broker = rig
    seen = {}
    real = agent.exposure_guard.permitted_notional

    def spy(*args, **kwargs):
        seen['net'] = kwargs.get('net_notional')
        return real(*args, **kwargs)

    agent.exposure_guard.permitted_notional = spy
    run_cycles(agent, buy_signal(), 1)
    assert seen.get('net') is not None


def test_net_exposure_reflects_existing_positions(rig):
    agent, broker = rig
    broker.qty = 100.0                      # 100 sh * $310 = $31,000 long
    seen = {}
    real = agent.exposure_guard.permitted_notional

    def spy(*args, **kwargs):
        seen['net'] = kwargs.get('net_notional')
        return real(*args, **kwargs)

    agent.exposure_guard.permitted_notional = spy
    run_cycles(agent, buy_signal(), 1)
    assert seen['net'] == pytest.approx(31_000.0)
