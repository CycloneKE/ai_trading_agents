"""Prices the agent can trust, or an alert.

A dead price feed and a selective strategy look identical from the outside:
both produce a run that does not trade. The difference is that one is a
result and the other is an outage, and this check is what separates them.
"""
import importlib.util
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    'check_data_freshness', os.path.join(_ROOT, 'scripts', 'check_data_freshness.py'))
fresh = importlib.util.module_from_spec(_SPEC)
sys.modules['check_data_freshness'] = fresh
_SPEC.loader.exec_module(fresh)

TODAY = date(2026, 9, 21)   # a Monday


def _csv(path, newest: date, source='nse', days=5):
    rows = ['date,symbol,open,high,low,close,volume,source']
    for i in range(days):
        d = newest - timedelta(days=days - 1 - i)
        rows.append(f'{d.isoformat()},SYM,10,11,9,10.5,1000,{source}')
    path.write_text('\n'.join(rows) + '\n')


# --------------------------------------------------------------- staleness

def test_recent_real_bars_pass(tmp_path):
    _csv(tmp_path / 'SCOM.csv', TODAY - timedelta(days=1), source='nse')
    r = fresh.check_history(str(tmp_path), 4, today=TODAY)
    assert not r['stale'] and not r['synthetic']
    assert not fresh.failures(r, {'available': False, 'total': 0, 'share': 0.0})


def test_a_weekend_does_not_trigger_an_alert(tmp_path):
    """Monday sees Friday's close as three days old. Crying wolf every
    weekend trains you to ignore the alert."""
    _csv(tmp_path / 'SCOM.csv', TODAY - timedelta(days=3), source='nse')
    assert not fresh.check_history(str(tmp_path), 4, today=TODAY)['stale']


def test_bars_older_than_the_threshold_are_stale(tmp_path):
    _csv(tmp_path / 'SCOM.csv', TODAY - timedelta(days=9), source='nse')
    r = fresh.check_history(str(tmp_path), 4, today=TODAY)
    assert [e['symbol'] for e in r['stale']] == ['SCOM']
    assert r['stale'][0]['age_days'] == 9


def test_age_is_measured_from_the_newest_row_not_the_last_one(tmp_path):
    """A re-scrape can append an older row; ordering is not guaranteed."""
    p = tmp_path / 'SCOM.csv'
    p.write_text(
        'date,symbol,close,source\n'
        f'{(TODAY - timedelta(days=1)).isoformat()},SCOM,10,nse\n'
        f'{(TODAY - timedelta(days=30)).isoformat()},SCOM,9,nse\n')
    assert not fresh.check_history(str(tmp_path), 4, today=TODAY)['stale']


# -------------------------------------------------------------- provenance

def test_synthetic_bars_are_flagged(tmp_path):
    """The failure this check exists for: every bar in the shipped store was
    generated, and nothing had ever said so."""
    _csv(tmp_path / 'SCOM.csv', TODAY, source='synthetic')
    r = fresh.check_history(str(tmp_path), 4, today=TODAY)
    assert [e['symbol'] for e in r['synthetic']] == ['SCOM']
    assert 'synthetic' in '; '.join(fresh.failures(r, {'available': False, 'total': 0, 'share': 0.0}))


def test_fresh_but_fabricated_still_fails(tmp_path):
    """Recency is not credibility. A synthetic bar dated today is worse than
    a real bar from last week, because it looks fine."""
    _csv(tmp_path / 'SCOM.csv', TODAY, source='synthetic')
    r = fresh.check_history(str(tmp_path), 4, today=TODAY)
    assert not r['stale']
    assert fresh.failures(r, {'available': False, 'total': 0, 'share': 0.0})


@pytest.mark.parametrize('marker', ['synthetic', 'fallback', 'generated', 'MOCK'])
def test_every_fabricated_marker_is_recognised(tmp_path, marker):
    _csv(tmp_path / 'SCOM.csv', TODAY, source=marker)
    assert fresh.check_history(str(tmp_path), 4, today=TODAY)['synthetic']


def test_allow_synthetic_is_available_for_a_fresh_install(tmp_path):
    _csv(tmp_path / 'SCOM.csv', TODAY, source='synthetic')
    r = fresh.check_history(str(tmp_path), 4, today=TODAY)
    suppressed = dict(r, synthetic=[])
    assert not fresh.failures(suppressed, {'available': False, 'total': 0, 'share': 0.0})


# ----------------------------------------------------------- bad input

def test_an_empty_directory_fails_rather_than_passing_silently(tmp_path):
    r = fresh.check_history(str(tmp_path), 4, today=TODAY)
    assert fresh.failures(r, {'available': False, 'total': 0, 'share': 0.0})


def test_a_missing_directory_is_reported(tmp_path):
    r = fresh.check_history(str(tmp_path / 'nope'), 4, today=TODAY)
    assert r.get('error')
    assert fresh.failures(r, {'available': False, 'total': 0, 'share': 0.0})


def test_an_unreadable_file_does_not_crash_the_check(tmp_path):
    (tmp_path / 'BROKEN.csv').write_text('not,a,price\nfile\n')
    _csv(tmp_path / 'GOOD.csv', TODAY, source='nse')
    r = fresh.check_history(str(tmp_path), 4, today=TODAY)
    assert 'BROKEN' in r['unreadable']
    assert [e['symbol'] for e in r['symbols']] == ['GOOD']


# ----------------------------------------------------------- the live feed

def _journal(path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute('CREATE TABLE decisions (ts TEXT, symbol TEXT, skip_reason TEXT)')
    conn.executemany('INSERT INTO decisions VALUES (?,?,?)', rows)
    conn.commit()
    conn.close()


def test_a_mostly_dead_feed_fails(tmp_path):
    now = datetime.utcnow()
    rows = [(now.isoformat(), 'SCOM', 'fallback_price') for _ in range(9)]
    rows += [(now.isoformat(), 'SCOM', 'hold')]
    _journal(tmp_path / 'd.db', rows)
    live = fresh.check_live_feed(str(tmp_path / 'd.db'))
    assert live['share'] == pytest.approx(0.9)
    assert fresh.failures({'symbols': [{'symbol': 'X', 'age_days': 0}],
                           'stale': [], 'synthetic': [], 'unreadable': []}, live)


def test_an_occasional_fallback_is_noted_but_not_a_failure(tmp_path):
    """One symbol briefly unavailable is normal and should not page anyone."""
    now = datetime.utcnow()
    rows = [(now.isoformat(), 'SCOM', 'fallback_price')]
    rows += [(now.isoformat(), 'SCOM', 'hold') for _ in range(9)]
    _journal(tmp_path / 'd.db', rows)
    live = fresh.check_live_feed(str(tmp_path / 'd.db'))
    assert live['share'] == pytest.approx(0.1)
    assert not fresh.failures({'symbols': [{'symbol': 'X', 'age_days': 0}],
                               'stale': [], 'synthetic': [], 'unreadable': []}, live)


def test_old_decisions_are_outside_the_window(tmp_path):
    old = (datetime.utcnow() - timedelta(days=5)).isoformat()
    _journal(tmp_path / 'd.db', [(old, 'SCOM', 'fallback_price')])
    assert fresh.check_live_feed(str(tmp_path / 'd.db'), hours=24)['total'] == 0


def test_a_missing_journal_is_not_an_error(tmp_path):
    """Before the first run there is nothing to read, which is not a fault."""
    live = fresh.check_live_feed(str(tmp_path / 'absent.db'))
    assert live['available'] is False
    assert not fresh.failures({'symbols': [{'symbol': 'X', 'age_days': 0}],
                               'stale': [], 'synthetic': [], 'unreadable': []}, live)


# ------------------------------------------------------------------ output

def test_render_names_the_problem_in_words(tmp_path):
    _csv(tmp_path / 'SCOM.csv', TODAY, source='synthetic')
    r = fresh.check_history(str(tmp_path), 4, today=TODAY)
    text = fresh.render(r, {'available': False, 'total': 0, 'share': 0.0}, 4)
    assert 'SYNTHETIC' in text
    assert 'invented data' in text
