import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from src.agent.sleeve.fundamentals_store import Fundamentals, FundamentalsStore


def test_is_stale_true_past_threshold():
    old = (datetime.now(timezone.utc) - timedelta(days=500)).isoformat()
    f = Fundamentals('SCOM', 6.5, 2.3, 2.4, 0.48, 8, 'positive', 100000, old)
    assert f.is_stale(400) is True


def test_is_stale_false_within_threshold():
    recent = datetime.now(timezone.utc).isoformat()
    f = Fundamentals('SCOM', 6.5, 2.3, 2.4, 0.48, 8, 'positive', 100000, recent)
    assert f.is_stale(400) is False


@pytest.fixture
def dividends_file():
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    yield path
    os.remove(path)


def test_returns_none_for_symbol_with_no_operator_entry(dividends_file, monkeypatch):
    with open(dividends_file, 'w') as f:
        json.dump({}, f)
    store = FundamentalsStore(dividends_path=dividends_file)
    monkeypatch.setattr('src.agent.sleeve.fundamentals_store.scrape_afx_company_page',
                        lambda s: None)
    assert store.get('SCOM') is None


def test_merges_scraped_and_operator_data(dividends_file, monkeypatch):
    with open(dividends_file, 'w') as f:
        json.dump({'SCOM': {'years_consecutive_paid': 8, 'eps_trend': 'positive',
                            'last_updated': '2026-07-01'}}, f)
    store = FundamentalsStore(dividends_path=dividends_file)
    monkeypatch.setattr(
        'src.agent.sleeve.fundamentals_store.scrape_afx_company_page',
        lambda s: {'symbol': 'SCOM', 'eps': 2.3863, 'pe_ratio': 14.69,
                   'dividend_per_share': 2.30, 'dividend_yield_pct': 6.56})
    monkeypatch.setattr(
        'src.agent.sleeve.fundamentals_store.load_csv', lambda s: [])

    f = store.get('SCOM')
    assert f is not None
    assert f.symbol == 'SCOM'
    assert f.yield_ttm_pct == 6.56
    assert f.years_consecutive_paid == 8
    assert f.eps_trend == 'positive'
    assert round(f.payout_ratio, 4) == round(2.30 / 2.3863, 4)


def test_scrape_failure_falls_back_to_operator_fields(dividends_file, monkeypatch):
    with open(dividends_file, 'w') as f:
        json.dump({'SCOM': {'yield_ttm_pct': 6.0, 'dividend_per_share_kes': 2.0,
                            'eps_kes': 2.0, 'years_consecutive_paid': 5,
                            'eps_trend': 'flat', 'last_updated': '2026-07-01'}}, f)
    store = FundamentalsStore(dividends_path=dividends_file)
    monkeypatch.setattr('src.agent.sleeve.fundamentals_store.scrape_afx_company_page',
                        lambda s: None)
    monkeypatch.setattr('src.agent.sleeve.fundamentals_store.load_csv', lambda s: [])

    f = store.get('SCOM')
    assert f is not None
    assert f.yield_ttm_pct == 6.0
    assert f.payout_ratio == 1.0


def test_avg_daily_volume_from_local_history(dividends_file, monkeypatch):
    with open(dividends_file, 'w') as f:
        json.dump({'SCOM': {'yield_ttm_pct': 6.0, 'dividend_per_share_kes': 2.0,
                            'eps_kes': 2.0, 'years_consecutive_paid': 5,
                            'eps_trend': 'flat', 'last_updated': '2026-07-01'}}, f)
    store = FundamentalsStore(dividends_path=dividends_file, volume_window=3)
    monkeypatch.setattr('src.agent.sleeve.fundamentals_store.scrape_afx_company_page',
                        lambda s: None)
    monkeypatch.setattr(
        'src.agent.sleeve.fundamentals_store.load_csv',
        lambda s: [{'volume': '100'}, {'volume': '200'}, {'volume': '300'}, {'volume': '9999'}])

    f = store.get('SCOM')
    # volume_window=3 keeps the last 3 rows [200, 300, 9999]; mean = 3499.67,
    # int() truncates to 3499.
    assert f.avg_daily_volume == 3499


def test_get_all_skips_missing_symbols(dividends_file, monkeypatch):
    with open(dividends_file, 'w') as f:
        json.dump({'SCOM': {'yield_ttm_pct': 6.0, 'dividend_per_share_kes': 2.0,
                            'eps_kes': 2.0, 'years_consecutive_paid': 5,
                            'eps_trend': 'flat', 'last_updated': '2026-07-01'}}, f)
    store = FundamentalsStore(dividends_path=dividends_file)
    monkeypatch.setattr('src.agent.sleeve.fundamentals_store.scrape_afx_company_page',
                        lambda s: None)
    monkeypatch.setattr('src.agent.sleeve.fundamentals_store.load_csv', lambda s: [])

    result = store.get_all(['SCOM', 'UNKNOWN'])
    assert [f.symbol for f in result] == ['SCOM']
