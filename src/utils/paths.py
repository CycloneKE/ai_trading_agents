"""Absolute filesystem anchors. Import DATA_DIR instead of joining 'data'
relatively — relative paths silently fork state when a process starts from
a different working directory.

AGENT_DATA_DIR, when set, moves the data folder elsewhere (the smoke test
runs a whole agent against a temporary folder this way, so it never writes
the real one)."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ['AGENT_DATA_DIR']).resolve() if os.environ.get('AGENT_DATA_DIR') \
    else PROJECT_ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
