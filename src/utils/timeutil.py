"""UTC clock helpers.

`datetime.utcnow()` is deprecated in Python 3.12+ and slated for removal.
The obvious replacement, `datetime.now(timezone.utc)`, is NOT a drop-in
here: it returns an *aware* datetime, so

    - `.isoformat()` gains a '+00:00' suffix, which changes the TEXT format
      stored in the order journal, decision journal, escalations and
      improvements databases — breaking lexicographic ORDER BY against the
      rows already on disk, and
    - subtracting it from a naive timestamp parsed out of one of those
      tables raises `TypeError: can't subtract offset-naive and
      offset-aware datetimes`.

These helpers use only non-deprecated APIs while producing output that is
byte-for-byte what `utcnow()` produced, so the migration is invisible to
every existing reader and to every row already written.

Use `utc_now()` where a datetime object is needed and `utc_now_iso()` where
a stored/serialised string is needed.
"""

from datetime import datetime, timezone

__all__ = ['utc_now', 'utc_now_iso']


def utc_now() -> datetime:
    """Current UTC time as a NAIVE datetime — identical to the value the
    deprecated `datetime.utcnow()` returned, so it stays comparable with
    timestamps already persisted by this system."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utc_now_iso() -> str:
    """ISO-8601 string in the exact legacy shape: no offset, no 'Z'.

    e.g. '2026-07-24T15:37:24.474158'
    """
    return utc_now().isoformat()
