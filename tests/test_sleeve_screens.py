"""Ex-dividend timing, payout sustainability and sector concentration."""
from dataclasses import dataclass
from datetime import date

import pytest

from src.agent.sleeve.screens import (APPROACHING, JUST_AFTER, NEUTRAL,
                                      UNKNOWN_DATE, apply_concentration_limit,
                                      assess_sustainability, dividend_cover,
                                      ex_date_timing)

TODAY = date(2026, 6, 15)


# --- ex-dividend timing -----------------------------------------------------

def test_just_before_ex_date_is_the_window_to_avoid():
    assert ex_date_timing(date(2026, 6, 20), TODAY) == APPROACHING
    assert ex_date_timing(date(2026, 6, 15), TODAY) == APPROACHING


def test_just_after_ex_date_is_the_window_to_prefer():
    assert ex_date_timing(date(2026, 6, 10), TODAY) == JUST_AFTER


def test_far_from_the_ex_date_is_neutral():
    assert ex_date_timing(date(2026, 9, 1), TODAY) == NEUTRAL
    assert ex_date_timing(date(2026, 1, 1), TODAY) == NEUTRAL


def test_unknown_or_unparseable_dates_are_not_a_factor():
    assert ex_date_timing(None, TODAY) == UNKNOWN_DATE
    assert ex_date_timing('not a date', TODAY) == UNKNOWN_DATE


def test_iso_strings_are_accepted():
    assert ex_date_timing('2026-06-20', TODAY) == APPROACHING


# --- dividend cover ---------------------------------------------------------

def test_cover_is_earnings_over_dividends():
    assert dividend_cover(10.0, 4.0) == pytest.approx(2.5)


def test_cover_is_none_without_a_dividend():
    assert dividend_cover(10.0, 0) is None
    assert dividend_cover(10.0, None) is None
    assert dividend_cover(None, 4.0) is None


def test_negative_earnings_give_negative_cover():
    assert dividend_cover(-5.0, 2.0) < 0


# --- sustainability ---------------------------------------------------------

def healthy(**over):
    kw = dict(eps=10.0, dividend_per_share=4.0, yield_pct=7.0,
              years_consecutive_paid=6, eps_trend='positive')
    kw.update(over)
    return assess_sustainability(**kw)


def test_a_well_covered_payout_passes():
    v = healthy()
    assert v.passed and v.cover == pytest.approx(2.5)
    assert v.summary == 'sustainable'


def test_thin_cover_fails():
    v = healthy(eps=4.2, dividend_per_share=4.0)     # cover 1.05
    assert not v.passed
    assert any('cover' in r for r in v.reasons)


def test_payout_exceeding_earnings_fails():
    v = healthy(eps=2.0, dividend_per_share=4.0)     # cover 0.5
    assert not v.passed


def test_negative_earnings_fail():
    v = healthy(eps=-1.0)
    assert not v.passed
    assert any('negative earnings' in r for r in v.reasons)


def test_yield_trap_is_rejected_despite_the_headline_number():
    """A 15% yield with thin cover is a cut being priced in, not a bargain."""
    v = healthy(yield_pct=15.0, eps=4.4, dividend_per_share=4.0)
    assert not v.passed
    assert any('cut priced in' in r for r in v.reasons)


def test_a_high_yield_with_strong_cover_is_not_a_trap():
    v = healthy(yield_pct=13.0, eps=12.0, dividend_per_share=4.0)  # cover 3.0
    assert v.passed


def test_short_payment_history_fails():
    v = healthy(years_consecutive_paid=1)
    assert not v.passed
    assert any('consecutive years' in r for r in v.reasons)


def test_falling_earnings_fail():
    v = healthy(eps_trend='negative')
    assert not v.passed


def test_all_failures_are_reported_not_just_the_first():
    v = assess_sustainability(eps=-1.0, dividend_per_share=4.0, yield_pct=20.0,
                              years_consecutive_paid=0, eps_trend='negative')
    assert len(v.reasons) >= 3


# --- concentration ----------------------------------------------------------

@dataclass
class Cand:
    symbol: str


BANKS = {'KCB': 'banking', 'EQTY': 'banking', 'COOP': 'banking',
         'ABSA': 'banking', 'SCOM': 'telecom', 'EABL': 'consumer',
         'BAT': 'consumer'}


def test_caps_holdings_per_sector():
    ranked = [Cand(s) for s in ['KCB', 'EQTY', 'COOP', 'SCOM', 'EABL']]
    kept = [c.symbol for c in apply_concentration_limit(ranked, BANKS, max_per_sector=2)]
    assert kept == ['KCB', 'EQTY', 'SCOM', 'EABL']


def test_ranked_order_is_preserved():
    ranked = [Cand(s) for s in ['SCOM', 'KCB', 'EABL', 'EQTY']]
    kept = [c.symbol for c in apply_concentration_limit(ranked, BANKS, max_per_sector=1)]
    assert kept == ['SCOM', 'KCB', 'EABL']


def test_the_whole_sleeve_can_be_one_sector_if_unlimited():
    ranked = [Cand(s) for s in ['KCB', 'EQTY', 'COOP', 'ABSA']]
    assert len(apply_concentration_limit(ranked, BANKS, max_per_sector=0)) == 4


def test_unknown_sectors_are_not_capped():
    """Silently limiting names we failed to classify would shrink the sleeve."""
    ranked = [Cand(s) for s in ['X1', 'X2', 'X3', 'X4']]
    assert len(apply_concentration_limit(ranked, BANKS, max_per_sector=1)) == 4


def test_accepts_a_callable_lookup():
    ranked = [Cand(s) for s in ['KCB', 'EQTY', 'SCOM']]
    kept = [c.symbol for c in apply_concentration_limit(
        ranked, lambda s: BANKS.get(s, 'unknown'), max_per_sector=1)]
    assert kept == ['KCB', 'SCOM']


def test_a_broken_lookup_does_not_drop_candidates():
    def boom(_symbol):
        raise RuntimeError("sector service down")
    ranked = [Cand(s) for s in ['KCB', 'EQTY']]
    assert len(apply_concentration_limit(ranked, boom, max_per_sector=1)) == 2


def test_no_lookup_means_no_capping():
    ranked = [Cand(s) for s in ['KCB', 'EQTY', 'COOP']]
    assert len(apply_concentration_limit(ranked, None, max_per_sector=1)) == 3
