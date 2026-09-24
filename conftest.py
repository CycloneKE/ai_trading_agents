"""Pytest configuration.

The codebase was refactored from flat top-level modules into ``src/`` (plus a
handful of helpers under ``scripts/`` and ``integrations/``), but many modules,
tests, and the live trading agent still import by the original bare names
(e.g. ``import live_risk_manager``, ``from backtest_engine import ...``).

To keep those imports resolvable during test collection — without rewriting
every call site — put each source root on ``sys.path``. This mirrors how the
modules are imported at runtime (their directories are expected on the path).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Repo root first (so ``src`` / ``integrations`` resolve as packages), then the
# individual source dirs that hold bare-name modules.
_SOURCE_ROOTS = [
    _ROOT,
    os.path.join(_ROOT, 'src'),
    os.path.join(_ROOT, 'src', 'agent'),
    os.path.join(_ROOT, 'src', 'api'),
    os.path.join(_ROOT, 'src', 'connectors'),
    os.path.join(_ROOT, 'src', 'utils'),
    os.path.join(_ROOT, 'scripts'),
    os.path.join(_ROOT, 'integrations'),
]

for _path in _SOURCE_ROOTS:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

# The API refuses to load its login module without a JWT secret and then
# answers every protected route with 503. Which test module imports the API
# first depends on collection order, so set a test secret before any does.
os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-the-test-suite-only')


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _offline_fx_rate(tmp_path_factory, monkeypatch):
    """Tests never fetch the live KES/USD rate or write data/fx_rate.json."""
    from src.connectors import fx_rate
    monkeypatch.setattr(fx_rate, '_shared', fx_rate.FxRate(
        tmp_path_factory.mktemp('fx') / 'fx_rate.json', sources=[], background=False))
