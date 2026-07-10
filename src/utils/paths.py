"""Absolute filesystem anchors. Import DATA_DIR instead of joining 'data'
relatively — relative paths silently fork state when a process starts from
a different working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
