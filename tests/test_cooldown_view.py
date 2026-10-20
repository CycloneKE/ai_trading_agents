"""Blocked symbols must be visibly blocked.

A symbol inside a post-exit cooldown produces no orders — which on the
dashboard looks exactly like a symbol with no signal. That is the same class
of silent failure that let seventeen rejected stop-losses pass unnoticed for
91 minutes: the system was doing something surprising and the UI had no way
to say so.

This view turns "why isn't it trading TSLA?" into an answerable question.
"""

import os
import tempfile

import pytest

from src.agent.exposure import ExposureGuard
from src.api.cooldown_view import build_cooldown_view


@pytest.fixture
def guard():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    g = ExposureGuard(
        {'risk_limits': {'max_position_size': 0.05,
                         'reentry_cooldown_seconds': 900}}, db_path=path)
    yield g
    g.close()
    os.unlink(path)


def test_empty_when_nothing_is_cooling_down(guard):
    view = build_cooldown_view(guard)
    assert view['active'] == []
    assert view['count'] == 0


def test_a_stopped_symbol_appears(guard):
    guard.record_forced_exit('TSLA', 'stop_loss', now=1_000.0)
    view = build_cooldown_view(guard, now=1_060.0)
    assert view['count'] == 1
    entry = view['active'][0]
    assert entry['symbol'] == 'TSLA'
    assert entry['reason'] == 'stop_loss'


def test_remaining_seconds_are_reported(guard):
    guard.record_forced_exit('TSLA', 'stop_loss', now=1_000.0)
    view = build_cooldown_view(guard, now=1_100.0)
    assert view['active'][0]['seconds_remaining'] == pytest.approx(800.0)


def test_human_readable_duration_is_provided(guard):
    """The dashboard should not have to do arithmetic."""
    guard.record_forced_exit('TSLA', 'stop_loss', now=1_000.0)
    view = build_cooldown_view(guard, now=1_000.0)
    assert view['active'][0]['remaining_label'] == '15m 0s'


def test_label_drops_the_minutes_component_under_a_minute(guard):
    guard.record_forced_exit('TSLA', 'stop_loss', now=1_000.0)
    view = build_cooldown_view(guard, now=1_000.0 + 875)
    assert view['active'][0]['remaining_label'] == '25s'


def test_expired_cooldowns_disappear(guard):
    guard.record_forced_exit('TSLA', 'stop_loss', now=1_000.0)
    view = build_cooldown_view(guard, now=1_000.0 + 901)
    assert view['active'] == []
    assert view['count'] == 0


def test_multiple_symbols_sorted_by_time_remaining(guard):
    guard.record_forced_exit('AAPL', 'stop_loss', now=1_000.0)
    guard.record_forced_exit('TSLA', 'trailing_stop', now=1_200.0)
    guard.record_forced_exit('SPY', 'kill_switch', now=1_100.0)
    view = build_cooldown_view(guard, now=1_300.0)
    assert [e['symbol'] for e in view['active']] == ['AAPL', 'SPY', 'TSLA']


def test_reason_is_carried_through_for_each_symbol(guard):
    guard.record_forced_exit('AAPL', 'stop_loss', now=1_000.0)
    guard.record_forced_exit('TSLA', 'trailing_stop', now=1_000.0)
    view = build_cooldown_view(guard, now=1_010.0)
    by_symbol = {e['symbol']: e['reason'] for e in view['active']}
    assert by_symbol == {'AAPL': 'stop_loss', 'TSLA': 'trailing_stop'}


def test_cooldown_window_is_reported_so_the_ui_can_explain_itself(guard):
    view = build_cooldown_view(guard)
    assert view['cooldown_seconds'] == 900


def test_missing_guard_degrades_to_an_empty_view():
    """The API must not 500 when the agent has not built a guard yet."""
    view = build_cooldown_view(None)
    assert view == {'active': [], 'count': 0, 'cooldown_seconds': 0}


def test_disabled_cooldown_reports_zero_window():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    g = ExposureGuard({'risk_limits': {'reentry_cooldown_seconds': 0}}, db_path=path)
    try:
        g.record_forced_exit('TSLA', 'stop_loss')
        view = build_cooldown_view(g)
        assert view['cooldown_seconds'] == 0
        assert view['active'] == []
    finally:
        g.close()
        os.unlink(path)
