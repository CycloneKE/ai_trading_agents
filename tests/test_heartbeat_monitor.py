"""Tests for the Dead-Man's Switch Heartbeat Monitor."""
import time
import pytest
from unittest.mock import MagicMock, patch
from src.agent.heartbeat_monitor import HeartbeatMonitor


class FakeOrder:
    def __init__(self, oid):
        self.id = oid


class FakeBroker:
    def __init__(self):
        self.cancelled = []
        self.is_connected = True

    def get_orders(self):
        return [FakeOrder('ord_1'), FakeOrder('ord_2')]

    def cancel_order(self, oid):
        self.cancelled.append(oid)


class FakeBrokerManager:
    def __init__(self, broker):
        self._broker = broker

    def get_broker(self):
        return self._broker


def test_ping_resets_elapsed():
    hb = HeartbeatMonitor(MagicMock(), MagicMock(), MagicMock(), timeout_seconds=60)
    time.sleep(0.1)
    assert hb.elapsed() >= 0.1
    hb.ping()
    assert hb.elapsed() < 0.05


def test_is_healthy_within_timeout():
    hb = HeartbeatMonitor(MagicMock(), MagicMock(), MagicMock(), timeout_seconds=60)
    assert hb.is_healthy() is True


def test_is_healthy_false_after_timeout():
    hb = HeartbeatMonitor(MagicMock(), MagicMock(), MagicMock(), timeout_seconds=0.05)
    time.sleep(0.1)
    assert hb.is_healthy() is False


def test_status_returns_dict():
    hb = HeartbeatMonitor(MagicMock(), MagicMock(), MagicMock(), timeout_seconds=60)
    s = hb.status()
    assert 'healthy' in s
    assert 'elapsed_seconds' in s
    assert 'timeout_seconds' in s
    assert s['timeout_seconds'] == 60
    assert s['triggered'] is False


def test_watchdog_triggers_kill_switch_and_cancels_orders():
    risk = MagicMock()
    broker = FakeBroker()
    broker_mgr = FakeBrokerManager(broker)
    audit = MagicMock()

    hb = HeartbeatMonitor(risk, broker_mgr, audit, timeout_seconds=0.05)
    # Don't start the daemon thread, call _check_timeout directly
    time.sleep(0.1)
    # Simulate one watchdog check
    hb._check_timeout()
    assert hb._triggered is True
    risk.set_persistent_kill_switch.assert_called_once()
    assert 'ord_1' in broker.cancelled
    assert 'ord_2' in broker.cancelled
    audit.log_event.assert_called_once()


def test_watchdog_does_not_retrigger():
    risk = MagicMock()
    audit = MagicMock()
    hb = HeartbeatMonitor(risk, MagicMock(), audit, timeout_seconds=0.05)
    time.sleep(0.1)
    hb._check_timeout()
    hb._check_timeout()
    # Should only trigger once
    assert risk.set_persistent_kill_switch.call_count == 1
