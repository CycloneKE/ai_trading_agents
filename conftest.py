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
