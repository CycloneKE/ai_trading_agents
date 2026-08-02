"""UTC clock helpers that replace the deprecated `datetime.utcnow()`.

The constraint that makes this non-trivial: the order journal, decision
journal, escalations and improvements DBs all store `utcnow().isoformat()`
strings and sort/compare them as TEXT. Switching to timezone-aware
`datetime.now(timezone.utc)` would start emitting a '+00:00' suffix, which
- breaks lexicographic ORDER BY against every pre-existing row, and
- makes new rows unparseable by code expecting naive strings.

So the replacement must be byte-format-identical to the old output while
using only non-deprecated APIs.
"""

import re
from datetime import datetime, timedelta, timezone

from src.utils.timeutil import utc_now, utc_now_iso


def test_iso_format_is_identical_to_legacy_naive_utcnow():
    got = utc_now_iso()
    # Legacy shape: 2026-07-24T15:37:24.474158 — no offset, no 'Z'
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?', got), got
    assert '+' not in got
    assert not got.endswith('Z')


def test_iso_string_round_trips_through_naive_fromisoformat():
    """Existing readers call datetime.fromisoformat() and expect naive."""
    parsed = datetime.fromisoformat(utc_now_iso())
    assert parsed.tzinfo is None


def test_utc_now_returns_naive_datetime_in_utc():
    now = utc_now()
    assert now.tzinfo is None
    reference = datetime.now(timezone.utc).replace(tzinfo=None)
    assert abs(now - reference) < timedelta(seconds=5)


def test_utc_now_is_comparable_with_stored_legacy_values():
    """Arithmetic against previously-stored naive timestamps must still work
    — an aware/naive mix raises TypeError at runtime."""
    legacy = datetime.fromisoformat('2026-07-24T15:37:24.474158')
    delta = utc_now() - legacy
    assert delta.total_seconds() > 0


def test_iso_strings_sort_lexicographically_after_legacy_rows():
    legacy_row = '2026-07-24T15:37:24.474158'
    assert utc_now_iso() > legacy_row


def test_helpers_emit_no_deprecation_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        utc_now()
        utc_now_iso()
