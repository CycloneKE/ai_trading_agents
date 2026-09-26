"""Benchmark curves for the dashboard (src/agent/benchmarks.py)."""
import csv
from datetime import date

import pytest

from src.agent import benchmarks as bm


def test_tbills_compound_daily_after_withholding_tax():
    days = ['2026-01-01', '2026-07-02', '2027-01-01']
    values = bm.tbill_values(days, 100000, 0.10, withholding=0.15)
    assert values[0] == 100000
    assert values[-1] == pytest.approx(100000 * 1.085, abs=0.5)   # 10% less 15% tax, one year


def test_a_price_series_is_rebased_to_the_start_value_and_forward_filled():
    prices = {date(2026, 9, 21): 100.0, date(2026, 9, 23): 110.0}
    days = ['2026-09-20T15:00', '2026-09-21T15:00', '2026-09-22T15:00', '2026-09-23T15:00']
    assert bm.rebased_values(days, prices, 5000) == [None, 5000, 5000, 5500]


def test_the_basket_is_equal_weight_and_skips_stocks_without_prices():
    closes = {'A': {date(2026, 9, 1): 10.0, date(2026, 9, 2): 12.0},
              'B': {date(2026, 9, 1): 50.0, date(2026, 9, 2): 45.0},
              'C': {}}
    values = bm.basket_values(['2026-09-01', '2026-09-02'], closes, 200000)
    assert values == [200000, pytest.approx(100000 * 1.2 + 100000 * 0.9)]


def test_nse_closes_use_real_rows_only(tmp_path):
    with open(tmp_path / 'SCOM.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['date', 'close', 'source'])
        w.writeheader()
        w.writerow({'date': '2026-09-01', 'close': 25, 'source': 'synthetic'})
        w.writerow({'date': '2026-09-02', 'close': 26, 'source': 'nse_pricelist'})
    assert bm.nse_closes('SCOM', tmp_path) == {date(2026, 9, 2): 26.0}


def test_overlay_adds_values_only_where_known():
    points = [{'day': 'a'}, {'day': 'b'}]
    bm.overlay(points, 'tbill_kes', [None, 5.0])
    assert points == [{'day': 'a'}, {'day': 'b', 'tbill_kes': 5.0}]


def test_an_account_opened_on_a_weekend_starts_the_basket_at_its_own_value():
    # Closes Friday 100 and Monday 110; the account opened on Saturday. The
    # basket can only buy on Monday, so it starts level with the account.
    closes = {'A': {date(2026, 9, 18): 100.0, date(2026, 9, 21): 110.0}}
    values = bm.basket_values(['2026-09-19', '2026-09-20', '2026-09-21'], closes, 200000)
    assert values == [200000, 200000, 200000]
