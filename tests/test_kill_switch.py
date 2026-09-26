"""The kill switch is enforced: it halts trading, survives a restart, and
only an operator's Resume clears it (src/agent/main.py, heartbeat_monitor.py,
realtime_risk_manager.py)."""
import sqlite3
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.utils.paths as paths
from src.agent.heartbeat_monitor import HeartbeatMonitor
from src.agent.main import TradingAgent
from src.agent.realtime_risk_manager import RealTimeRiskManager


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA_DIR', tmp_path)
    return tmp_path


def _agent(risk_manager, heartbeat=None):
    agent = SimpleNamespace(trading_halted=False, halt_reason=None, halted_at=None,
                            risk_manager=risk_manager, heartbeat_monitor=heartbeat,
                            components={}, order_journal=None)
    for name in ('halt_trading', 'resume_trading', '_restore_halt'):
        setattr(agent, name, types.MethodType(getattr(TradingAgent, name), agent))
    return agent


def test_a_heartbeat_timeout_halts_the_agent_itself(data_dir):
    """It used to write the switch to a database that nothing read."""
    risk = RealTimeRiskManager({}, None)
    agent = _agent(risk)
    hb = HeartbeatMonitor(risk, MagicMock(), MagicMock(), timeout_seconds=0.05,
                          on_trigger=lambda reason: agent.halt_trading(reason=reason))
    hb.ping()
    time.sleep(0.1)
    hb._check_timeout()
    assert agent.trading_halted is True
    assert agent.halt_reason.startswith('HEARTBEAT_TIMEOUT')


def test_a_halt_survives_a_restart_with_its_reason(data_dir):
    first = _agent(RealTimeRiskManager({}, None))
    first.halt_trading(reason='halted from the dashboard')
    reborn = _agent(RealTimeRiskManager({}, None))
    reborn._restore_halt()
    assert reborn.trading_halted is True
    assert reborn.halt_reason == 'halted from the dashboard'
    assert reborn.halted_at


def test_resume_clears_the_stored_switch_and_re_arms_the_watchdog(data_dir):
    risk = RealTimeRiskManager({}, None)
    hb = HeartbeatMonitor(risk, MagicMock(), MagicMock(), timeout_seconds=0.05)
    agent = _agent(risk, hb)
    hb.on_trigger = lambda reason: agent.halt_trading(reason=reason)
    hb.ping()
    time.sleep(0.1)
    hb._check_timeout()
    assert agent.trading_halted
    agent.resume_trading()
    assert (agent.trading_halted, agent.halt_reason, risk.emergency_stop) == (False, None, False)
    # Cleared in the database too: a restart comes back trading.
    reborn = _agent(RealTimeRiskManager({}, None))
    reborn._restore_halt()
    assert reborn.trading_halted is False
    # And a later freeze trips it again.
    hb.ping()
    time.sleep(0.1)
    hb._check_timeout()
    assert agent.trading_halted is True


def test_a_switch_set_before_reasons_were_recorded_still_halts(data_dir):
    conn = sqlite3.connect(str(data_dir / 'risk_state.db'))
    conn.execute("CREATE TABLE risk_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    conn.execute("INSERT INTO risk_state VALUES ('kill_switch_active', 'true', '2026-09-24T11:09:35')")
    conn.commit()
    conn.close()
    agent = _agent(RealTimeRiskManager({}, None))
    agent._restore_halt()
    assert agent.trading_halted is True
    assert 'reason not recorded' in agent.halt_reason
    assert agent.halted_at == '2026-09-24T11:09:35'


def test_without_a_callback_the_watchdog_still_records_the_switch():
    risk = MagicMock()
    hb = HeartbeatMonitor(risk, MagicMock(), MagicMock(), timeout_seconds=0.05)
    hb.ping()
    time.sleep(0.1)
    hb._check_timeout()
    risk.set_persistent_kill_switch.assert_called_once()
