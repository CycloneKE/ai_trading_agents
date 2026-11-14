"""Time and timezone utilities."""
from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def quote_age_seconds(timestamp, now=None):
    """Seconds since an ISO-8601 timestamp, or None when it cannot be read.

    A timestamp without a timezone is taken as UTC (the data manager stamps
    batches with datetime.utcnow()). None, not a guess, for an unreadable or
    missing value, so callers decide what an unknown age means.
    """
    if not timestamp:
        return None
    try:
        ts = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(
            str(timestamp).replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - ts).total_seconds()
