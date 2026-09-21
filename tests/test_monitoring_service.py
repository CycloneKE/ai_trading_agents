"""The liveness endpoint must survive a missing optional dependency.

psutil used to gate the entire monitoring service, so on any machine without
it the agent ran happily while /health did not exist. The log line said
"Monitoring service started on port 8080" either way. For an unattended
ninety-day run that is the difference between knowing the process died and
finding out months later.
"""
import base64
import json
import socket
import urllib.error
import urllib.request

import pytest

import src.utils.monitoring as monitoring
from src.utils.monitoring import MonitoringService


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def without_psutil(monkeypatch):
    monkeypatch.setattr(monitoring, 'PSUTIL_AVAILABLE', False)
    yield


def test_service_is_enabled_without_psutil(without_psutil):
    svc = MonitoringService({'monitoring': {'enabled': True}})
    assert svc.enabled is True
    assert svc.system_metrics_enabled is False


def test_system_metrics_are_the_only_casualty(without_psutil):
    svc = MonitoringService({'monitoring': {'enabled': True,
                                            'system_metrics_enabled': True}})
    assert svc.system_metrics_enabled is False
    assert svc.enabled is True


def test_config_still_wins_over_everything(without_psutil):
    svc = MonitoringService({'monitoring': {'enabled': False}})
    assert svc.enabled is False


def test_health_endpoint_answers_without_psutil(without_psutil, monkeypatch):
    """The regression this whole module exists to prevent."""
    port = _free_port()
    monkeypatch.setenv('MONITORING_PASSWORD', 'probe-password')
    svc = MonitoringService({'monitoring': {'enabled': True, 'port': port,
                                            'host': '127.0.0.1'}})
    svc.register_health_check('always_ok', lambda: {'status': 'ok'})
    svc.start(port=port)
    try:
        assert svc.is_running, 'service refused to start without psutil'
        token = base64.b64encode(b':probe-password').decode()
        req = urllib.request.Request(
            f'http://127.0.0.1:{port}/health',
            headers={'Authorization': f'Basic {token}'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode())
        assert body.get('status') in ('ok', 'running', 'degraded')
    finally:
        svc.stop()


def test_status_endpoint_returns_null_uptime_rather_than_failing(without_psutil,
                                                                 monkeypatch):
    port = _free_port()
    monkeypatch.setenv('MONITORING_PASSWORD', 'probe-password')
    svc = MonitoringService({'monitoring': {'enabled': True, 'port': port,
                                            'host': '127.0.0.1'}})
    svc.start(port=port)
    try:
        token = base64.b64encode(b':probe-password').decode()
        req = urllib.request.Request(
            f'http://127.0.0.1:{port}/status',
            headers={'Authorization': f'Basic {token}'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode())
        assert body['status'] == 'running'
        assert body['uptime'] is None
        assert body['process_uptime'] is None
    finally:
        svc.stop()
