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


@pytest.fixture(autouse=True)
def _private_alert_log(tmp_path_factory, monkeypatch):
    """Tests never read or write the real data/alerts.jsonl.

    An alerter built without a log of its own keeps one under the data
    directory. A test that raised an alert that way (an NSE ticket) left it
    in the repository's data/alerts.jsonl, and the alerts API test, which
    expects an empty log, then failed whenever it ran after that test or on
    a machine where the file was left over from an earlier run.
    """
    from src.agent import alerts
    monkeypatch.setattr(alerts, 'DATA_DIR', tmp_path_factory.mktemp('alerts'))


@pytest.fixture(autouse=True)
def _private_loss_limits(tmp_path_factory, monkeypatch):
    """Tests never read or write the real data/loss_limits.json: a saved
    breach there would quietly refuse buys in any test that builds an agent."""
    from src.agent import loss_limits
    monkeypatch.setattr(loss_limits, 'DATA_DIR', tmp_path_factory.mktemp('loss_limits'))


@pytest.fixture(autouse=True)
def _private_data_files(request, tmp_path_factory, monkeypatch):
    """Tests never create or change the real databases under data/.

    The escalation, audit, improvement, sleeve and risk databases, and the
    learned NSE ISIN map, are opened under the data folder when a class is
    built without a path of its own. Tests that did so left those files in
    the repository's data/ folder, where a later run (or the live agent)
    read what an earlier test had written. Each test gets a private folder.

    Modules that import DATA_DIR inside a function read it from
    src.utils.paths at call time, so that is pointed at the private folder
    too, except for the tests marked real_data_dir, which check the real path.
    """
    import importlib
    private = tmp_path_factory.mktemp('data')
    if not request.node.get_closest_marker('real_data_dir'):
        monkeypatch.setattr('src.utils.paths.DATA_DIR', private)
    for name in ('src.agent.escalation_manager', 'src.agent.audit_journal',
                 'src.agent.self_assessment', 'src.agent.nse_order_queue',
                 'src.agent.sector_specialist', 'src.agent.performance_analytics',
                 'src.agent.sleeve.sleeve_manager',
                 'src.agent.sleeve.dividend_ledger'):
        monkeypatch.setattr(importlib.import_module(name), 'DATA_DIR', private)
    from src.connectors import nse_isin
    monkeypatch.setattr(nse_isin, '_default_path', lambda: private / 'nse_isin_map.json')
