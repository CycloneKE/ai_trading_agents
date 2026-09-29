"""The market probe (scripts/probe_markets.py): what it makes of what Yahoo
answered. The network part is not tested here; the judgement is."""
import importlib.util
import os
from datetime import date, timedelta

spec = importlib.util.spec_from_file_location(
    'probe_markets', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'probe_markets.py'))
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)

TODAY = date(2026, 9, 30)          # a Wednesday


def bars(n, last=date(2026, 9, 29)):
    return [last - timedelta(days=i) for i in range(n)]


def test_a_market_with_two_years_of_fresh_bars_is_ok():
    r = pm.assess('SAP.DE', 'EUR', 180.5, bars(510), today=TODAY)
    assert r['verdict'] == 'OK' and r['age_weekdays'] == 1 and r['notes'] == []


def test_a_short_or_stale_history_is_thin_and_says_why():
    r = pm.assess('NPN.JO', 'ZAc', 3200.0, bars(120, last=date(2026, 9, 10)), today=TODAY)
    assert r['verdict'] == 'THIN'
    assert any('only 120 daily bars' in n for n in r['notes']) and any('weekdays old' in n for n in r['notes'])


def test_no_answer_is_no_data():
    assert pm.assess('DANGCEM.LG', None, None, [], today=TODAY)['verdict'] == 'NO DATA'
    assert pm.assess('X', 'USD', None, bars(600), today=TODAY)['verdict'] == 'NO DATA'


def test_quotes_in_pence_and_cents_are_flagged():
    """London quotes in pence and Johannesburg in cents: taken at face value
    every trade would be priced 100 times too high."""
    for ccy, unit in (('GBp', 'pence'), ('ZAc', 'cents')):
        r = pm.assess('Y', ccy, 100.0, bars(600), today=TODAY)
        assert any(unit in n and 'divide by 100' in n for n in r['notes'])
    assert pm.assess('Y', 'USD', 100.0, bars(600), today=TODAY)['notes'] == []


def test_weekdays_skip_the_weekend():
    assert pm.business_days_between(date(2026, 9, 25), date(2026, 9, 28)) == 1     # Friday to Monday
    assert pm.business_days_between(date(2026, 9, 28), date(2026, 9, 28)) == 0


def test_every_group_has_symbols_and_the_pairs_use_the_agents_names():
    assert all(pm.CANDIDATES.values())
    assert all(len(s) == 7 and s[3] == '_' for s in pm.CANDIDATES['forex_crosses'])
