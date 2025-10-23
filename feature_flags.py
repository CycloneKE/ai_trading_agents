"""
Simple in-process feature flags.
The flags are read from environment variables prefixed with FF_ or managed via the module API.
"""
import os
from typing import Dict

_flags: Dict[str, bool] = {}


def is_enabled(name: str, default: bool = False) -> bool:
    # First check runtime managed flags
    if name in _flags:
        return bool(_flags[name])
    # Then check environment variable FF_<NAME>
    env = os.getenv(f"FF_{name.upper()}")
    if env is not None:
        return env.lower() in ("1", "true", "yes", "on")
    return default


def set_flag(name: str, value: bool) -> None:
    _flags[name] = bool(value)


def clear_flags() -> None:
    _flags.clear()
