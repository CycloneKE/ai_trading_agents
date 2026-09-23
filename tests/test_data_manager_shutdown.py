"""Shutdown must not log a connector with no teardown as an error.

RealDataConnector and NSEConnector hold no long-lived resource, so neither
defines disconnect(). DataManager.stop() used to call it unconditionally,
raising AttributeError on every shutdown, caught and logged at ERROR. That is
the wrong severity for "nothing to release", and it would bury a genuine
teardown failure in the noise.
"""
import logging

# The agent runs src.agent.data_manager (main.py imports it by that path),
# so this tests that module specifically, not the divergent scripts/ copy
# that the bare-name import happens to resolve to under conftest's sys.path.
import src.agent.data_manager as dm_module
from src.agent.data_manager import DataManager


def _running_dm(monkeypatch):
    monkeypatch.setattr(dm_module, 'FALLBACK_AVAILABLE', False)
    monkeypatch.setattr(dm_module, 'FallbackDataGenerator', None)
    dm = DataManager({'symbols': ['AAPL'], 'connectors': {}, 'use_fallback_only': True})
    dm.is_running = True  # stop() early-returns otherwise
    return dm


class _NoDisconnect:
    """A connector like RealDataConnector: no disconnect() at all."""


class _CleanDisconnect:
    def __init__(self):
        self.closed = False

    def disconnect(self):
        self.closed = True


class _RaisingDisconnect:
    def disconnect(self):
        raise RuntimeError('socket already gone')


def test_a_connector_without_disconnect_is_not_logged_as_an_error(monkeypatch, caplog):
    dm = _running_dm(monkeypatch)
    dm.connectors = {'real_data': _NoDisconnect(), 'nse': _NoDisconnect()}
    with caplog.at_level(logging.DEBUG):
        dm.stop()
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR
              and 'disconnect' in r.getMessage().lower()]
    assert not errors, f"shutdown logged an error for a no-op teardown: {errors}"


def test_a_real_disconnect_still_runs(monkeypatch):
    dm = _running_dm(monkeypatch)
    conn = _CleanDisconnect()
    dm.connectors = {'coinbase': conn}
    dm.stop()
    assert conn.closed is True


def test_a_genuinely_failing_disconnect_is_still_logged_as_an_error(monkeypatch, caplog):
    """The guard must not swallow real failures, only absent methods."""
    dm = _running_dm(monkeypatch)
    dm.connectors = {'coinbase': _RaisingDisconnect()}
    with caplog.at_level(logging.DEBUG):
        dm.stop()
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR
              and 'coinbase' in r.getMessage()]
    assert errors, 'a real teardown failure must still surface as an error'
