"""Tests for the event-calendar risk windows."""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.event_calendar import EventCalendar

EVENTS = [
    {"name": "Kenya election", "date": "2027-08-09", "days_before": 45,
     "days_after": 21, "multiplier": 0.5, "markets": ["NSE"]},
    {"name": "FOMC", "date": "2026-07-29", "days_before": 2,
     "days_after": 1, "multiplier": 0.7, "markets": ["US"]},
]


def cal():
    return EventCalendar(events=EVENTS)


def test_quiet_day_multiplier_is_one():
    assert cal().risk_multiplier(date(2026, 7, 10), 'US') == 1.0
    assert cal().risk_multiplier(date(2026, 7, 10), 'NSE') == 1.0


def test_fomc_window_scales_us_only():
    c = cal()
    assert c.risk_multiplier(date(2026, 7, 28), 'US') == 0.7   # day before
    assert c.risk_multiplier(date(2026, 7, 30), 'US') == 0.7   # day after
    assert c.risk_multiplier(date(2026, 7, 31), 'US') == 1.0   # window over
    assert c.risk_multiplier(date(2026, 7, 28), 'NSE') == 1.0  # wrong market


def test_election_window_boundaries():
    c = cal()
    assert c.risk_multiplier(date(2027, 6, 25), 'NSE') == 0.5  # 45 days before
    assert c.risk_multiplier(date(2027, 8, 30), 'NSE') == 0.5  # 21 days after
    assert c.risk_multiplier(date(2027, 6, 24), 'NSE') == 1.0  # just outside
    assert c.risk_multiplier(date(2027, 8, 31), 'NSE') == 1.0


def test_lowest_multiplier_wins_and_bad_entries_skipped():
    c = EventCalendar(events=EVENTS + [
        {"name": "overlap", "date": "2026-07-29", "days_before": 2,
         "days_after": 1, "multiplier": 0.4, "markets": ["US"]},
        {"name": "garbage", "date": "not-a-date", "multiplier": 0.0},
    ])
    assert c.risk_multiplier(date(2026, 7, 29), 'US') == 0.4
