"""The exit path must obey the same broker constraints as the entry path.

On 2026-07-24 seventeen consecutive TSLA stop-loss orders were rejected with
`fractional orders must be DAY orders` over 91 minutes while the position sat
21% underwater. The entry path had this right all along — it routes through
position_sizing.size_order(), which returns time_in_force='day' for a
fractional quantity. _enforce_stop_losses bypassed that helper and hardcoded
'gtc'.

The connector now coerces the TIF as a backstop, but a backstop in one
broker adapter is not a fix: the exit path must use the same sizing helper
as the entry path so the next broker constraint is handled in one place.

These tests also cover the second half of the incident — the stop filling at
15:22:21 and the ensemble rebuying the same symbol at 15:22:31.
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
from src.connectors.base_broker import OrderResponse, Position

PRICE = 310.0


class RecordingBroker:
    broker_name = 'fake'
    is_connected = True

    def __init__(self, positions):
        self._positions = positions
        self.placed = []
        self.reject_non_day = True

    def get_positions(self):
        return list(self._positions)

    def get_orders(self, symbol=None):
        return []

    def place_order(self, order):
        # Mirror Alpaca: a fractional quantity with a non-DAY TIF is rejected.
        if self.reject_non_day and order.quantity % 1 != 0 \
                and order.time_in_force != 'day':
            self.placed.append(order)
            return None
        self.placed.append(order)
        return OrderResponse(
            order_id='o1', client_order_id=order.client_order_id,
            symbol=order.symbol, quantity=order.quantity,
            filled_quantity=order.quantity, side=order.side,
            order_type=order.order_type, status='filled',
            created_at=datetime(2026, 7, 24), updated_at=datetime(2026, 7, 24),
            filled_avg_price=PRICE, broker_name=self.broker_name)


class BrokerManager:
    def __init__(self, broker):
        self._b = broker

    def get_broker(self, name=None):
        return self._b


def losing_position(symbol='TSLA', qty=34.356, loss_pct=-0.21):
    cost = abs(qty) * 400.0
    return Position(symbol=symbol, quantity=qty, avg_entry_price=400.0,
                    current_price=PRICE, market_value=abs(qty) * PRICE,
                    unrealized_pl=cost * loss_pct, unrealized_pl_percent=loss_pct,
                    cost_basis=cost, broker_name='fake')


def make_agent(broker):
    agent = object.__new__(TradingAgent)
    agent.config = {
        'risk_limits': {'max_position_size': 0.05, 'stop_loss_pct': 0.05,
                        'trailing_stop_pct': 0.03,
                        'reentry_cooldown_seconds': 900},
        'trading': {'allow_fractional': True, 'min_notional': 5.0},
    }
    agent.components = {'broker_manager': BrokerManager(broker)}
    agent.monitoring_service = None
    agent.trading_halted = False

    fd, jpath = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    agent.order_journal = OrderJournal(db_path=jpath)

    fd, epath = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    agent.exposure_guard = ExposureGuard(agent.config, db_path=epath)
    return agent, jpath, epath


@pytest.fixture
def rig():
    broker = RecordingBroker([losing_position()])
    agent, jpath, epath = make_agent(broker)
    yield agent, broker
    agent.order_journal.close()
    agent.exposure_guard.close()
    os.unlink(jpath)
    os.unlink(epath)


def test_fractional_stop_loss_uses_a_day_order(rig):
    """THE regression: 34.356 shares with time_in_force='gtc' was rejected
    seventeen times in a row."""
    agent, broker = rig
    agent._enforce_stop_losses(0.05, 0.03)

    assert broker.placed, 'stop-loss never reached the broker'
    order = broker.placed[0]
    assert order.quantity % 1 != 0                  # genuinely fractional
    assert order.time_in_force == 'day', (
        f"fractional exit sent as {order.time_in_force!r}; Alpaca rejects this")


def test_fractional_stop_loss_actually_fills(rig):
    agent, broker = rig
    agent._enforce_stop_losses(0.05, 0.03)
    coid = broker.placed[0].client_order_id
    assert agent.order_journal.get(coid)['status'] == 'submitted'


def test_whole_share_stop_loss_keeps_gtc(rig):
    """Back-compat: a whole-share exit should still be GTC so it survives the
    session, exactly as before this change."""
    agent, broker = rig
    broker._positions = [losing_position(qty=30.0)]
    agent._enforce_stop_losses(0.05, 0.03)
    assert broker.placed[0].time_in_force == 'gtc'


def test_stop_loss_closes_the_entire_position(rig):
    """The gate must never shrink an exit — partial exits leave risk on."""
    agent, broker = rig
    agent._enforce_stop_losses(0.05, 0.03)
    assert broker.placed[0].quantity == pytest.approx(34.356)


def test_short_position_is_closed_with_a_buy(rig):
    agent, broker = rig
    broker._positions = [losing_position(qty=-34.356)]
    agent._enforce_stop_losses(0.05, 0.03)
    assert broker.placed[0].side == 'buy'
    assert broker.placed[0].quantity == pytest.approx(34.356)


def test_a_filled_stop_starts_the_reentry_cooldown(rig):
    """Without this the ensemble rebuys seconds later — it did, at 15:22:31."""
    agent, broker = rig
    agent._enforce_stop_losses(0.05, 0.03)
    assert agent.exposure_guard.in_cooldown('TSLA') is True


def test_a_rejected_stop_does_not_start_a_cooldown(rig):
    """Cooldown means 'we just exited'. If the exit never happened, blocking
    re-entry would only mask an unclosed losing position."""
    agent, broker = rig
    broker.place_order = lambda order: None
    agent._enforce_stop_losses(0.05, 0.03)
    assert agent.exposure_guard.in_cooldown('TSLA') is False


def test_healthy_position_is_left_alone(rig):
    agent, broker = rig
    broker._positions = [losing_position(loss_pct=-0.01)]
    agent._enforce_stop_losses(0.05, 0.03)
    assert broker.placed == []
    assert agent.exposure_guard.in_cooldown('TSLA') is False


# ---------------------------------------------------------------------------
# Trailing stop — same exit path, same broker rules
# ---------------------------------------------------------------------------

class FakeRiskManager:
    def __init__(self, symbols):
        self._symbols = symbols

    def check_trailing_stops(self, pct):
        return [{'symbol': s} for s in self._symbols]


def test_trailing_stop_on_a_fractional_position_uses_day(rig):
    agent, broker = rig
    broker._positions = [losing_position(loss_pct=-0.01)]   # not a hard stop
    agent.risk_manager = FakeRiskManager(['TSLA'])
    agent._enforce_stop_losses(0.05, 0.03)
    assert broker.placed[0].time_in_force == 'day'


def test_trailing_stop_closes_a_short_with_a_buy(rig):
    """Was hardcoded to 'sell', which would have doubled the short."""
    agent, broker = rig
    broker._positions = [losing_position(qty=-34.356, loss_pct=-0.01)]
    agent.risk_manager = FakeRiskManager(['TSLA'])
    agent._enforce_stop_losses(0.05, 0.03)
    assert broker.placed[0].side == 'buy'


def test_trailing_stop_starts_the_reentry_cooldown(rig):
    agent, broker = rig
    broker._positions = [losing_position(loss_pct=-0.01)]
    agent.risk_manager = FakeRiskManager(['TSLA'])
    agent._enforce_stop_losses(0.05, 0.03)
    assert agent.exposure_guard.in_cooldown('TSLA') is True


# ---------------------------------------------------------------------------
# Kill switch — the third exit path
# ---------------------------------------------------------------------------

class NoQuoteDataManager:
    """Forces the kill switch onto its market-order fallback branch."""
    connectors = {}


def kill_switch_rig(positions, with_price=True):
    broker = RecordingBroker(positions)
    agent, jpath, epath = make_agent(broker)
    agent.components['data_manager'] = NoQuoteDataManager()
    broker.cancel_order = lambda oid: True
    if not with_price:
        for p in positions:
            p.current_price = 0.0
    return agent, broker, jpath, epath


def test_kill_switch_market_fallback_uses_day_for_fractional():
    """The limit branch was already DAY; the no-price market fallback still
    hardcoded GTC — the same rejection waiting to happen."""
    agent, broker, jpath, epath = kill_switch_rig(
        [losing_position(qty=34.356)], with_price=False)
    try:
        agent.halt_trading(flatten=True, reason='test')
        assert broker.placed, 'kill switch placed nothing'
        assert broker.placed[0].order_type == 'market'
        assert broker.placed[0].time_in_force == 'day'
    finally:
        agent.order_journal.close()
        agent.exposure_guard.close()
        os.unlink(jpath)
        os.unlink(epath)


def test_kill_switch_flatten_starts_the_reentry_cooldown():
    """Otherwise resume_trading() immediately rebuys everything just flattened."""
    agent, broker, jpath, epath = kill_switch_rig([losing_position(qty=34.356)])
    try:
        agent.halt_trading(flatten=True, reason='test')
        assert agent.exposure_guard.in_cooldown('TSLA') is True
    finally:
        agent.order_journal.close()
        agent.exposure_guard.close()
        os.unlink(jpath)
        os.unlink(epath)


def test_kill_switch_limit_branch_still_uses_extended_hours_day():
    """Back-compat: the working branch must not regress."""
    agent, broker, jpath, epath = kill_switch_rig([losing_position(qty=34.356)])
    try:
        agent.halt_trading(flatten=True, reason='test')
        order = broker.placed[0]
        assert order.order_type == 'limit'
        assert order.extended_hours is True
        assert order.time_in_force == 'day'
        assert order.limit_price == pytest.approx(round(PRICE * 0.995, 2))
    finally:
        agent.order_journal.close()
        agent.exposure_guard.close()
        os.unlink(jpath)
        os.unlink(epath)
