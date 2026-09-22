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


# ------------------------------------------------- reachable from other hosts

def _routable_ipv4():
    """This machine's non-loopback address, or None.

    Opening a UDP socket toward a public address sends nothing; it just makes
    the kernel pick the interface it would route through.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 53))
            ip = s.getsockname()[0]
        return None if ip.startswith('127.') else ip
    except OSError:
        return None


def test_the_default_bind_address_is_not_loopback_only():
    """The production bug.

    monitoring.host was never set anywhere, so the 127.0.0.1 default always
    won. A container's loopback is invisible to every other container, so
    Coolify's proxy got connection refused and health.<domain> served 502,
    while the container's own healthcheck passed and the logs said started.
    """
    assert monitoring.DEFAULT_BIND_ADDRESS == '0.0.0.0'
    port = _free_port()
    svc = MonitoringService({'monitoring': {'enabled': True, 'port': port}})
    svc.start(port=port)
    try:
        assert svc.is_running
        assert svc.bind_address == '0.0.0.0'
        assert svc.server.server_address[0] == '0.0.0.0'
    finally:
        svc.stop()


def test_health_answers_on_a_non_loopback_address():
    """What the proxy actually does. Loopback-only binding fails this."""
    ip = _routable_ipv4()
    if ip is None:
        pytest.skip('no non-loopback IPv4 on this host')
    port = _free_port()
    svc = MonitoringService({'monitoring': {'enabled': True, 'port': port}})
    svc.register_health_check('always_ok', lambda: {'status': 'ok'})
    svc.start(port=port)
    try:
        with urllib.request.urlopen(f'http://{ip}:{port}/health', timeout=10) as r:
            assert r.status == 200
    finally:
        svc.stop()


def test_loopback_binding_is_still_reachable_only_on_loopback():
    """Proves the test above would have caught the bug, rather than passing
    because everything is reachable in this environment anyway."""
    ip = _routable_ipv4()
    if ip is None:
        pytest.skip('no non-loopback IPv4 on this host')
    port = _free_port()
    svc = MonitoringService({'monitoring': {'enabled': True, 'port': port,
                                            'host': '127.0.0.1'}})
    svc.start(port=port)
    try:
        assert svc.bind_address == '127.0.0.1'
        with pytest.raises((urllib.error.URLError, OSError)):
            urllib.request.urlopen(f'http://{ip}:{port}/health', timeout=5)
    finally:
        svc.stop()


def test_the_bind_address_can_be_overridden(monkeypatch):
    """Both knobs, for running outside a container. Env wins over config."""
    monkeypatch.setenv('MONITORING_HOST', '127.0.0.1')
    port = _free_port()
    svc = MonitoringService({'monitoring': {'enabled': True, 'port': port,
                                            'host': '0.0.0.0'}})
    svc.start(port=port)
    try:
        assert svc.bind_address == '127.0.0.1'
    finally:
        svc.stop()


def test_the_startup_log_names_the_address_not_just_the_port(caplog):
    """'started on port 8080' is true of a server nothing can reach. That
    line is why this went undiagnosed for an entire deployment."""
    port = _free_port()
    svc = MonitoringService({'monitoring': {'enabled': True, 'port': port}})
    with caplog.at_level('INFO'):
        svc.start(port=port)
    try:
        started = [r.getMessage() for r in caplog.records
                   if 'Monitoring service started' in r.getMessage()]
        assert started, 'no startup line logged'
        assert '0.0.0.0' in started[0]
    finally:
        svc.stop()
