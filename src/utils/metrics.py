"""Centralized metrics helper for Prometheus exposure.

Use `PROMETHEUS_ENABLED=true` and optionally `PROMETHEUS_PORT` to enable the HTTP /metrics endpoint.
This module is safe to import even if `prometheus_client` is not installed; functions become no-ops.
"""
import os
import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import start_http_server
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus metrics HTTP server if available and enabled.

    Controlled by the environment variable `PROMETHEUS_ENABLED` (true/1/yes).
    """
    enabled = os.getenv('PROMETHEUS_ENABLED', 'false').lower() in ('1', 'true', 'yes')
    if not enabled:
        logger.debug("Prometheus metrics disabled by PROMETHEUS_ENABLED")
        return

    if not _HAS_PROM:
        logger.warning("prometheus_client not available; metrics endpoint cannot be started")
        return

    try:
        start_http_server(int(os.getenv('PROMETHEUS_PORT', port)))
        logger.info(f"Prometheus metrics server started on port {os.getenv('PROMETHEUS_PORT', port)}")
    except Exception as e:
        logger.error(f"Failed to start Prometheus metrics server: {e}")
