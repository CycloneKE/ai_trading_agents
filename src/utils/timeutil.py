"""Time and timezone utilities."""
from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()
