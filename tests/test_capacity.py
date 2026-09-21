"""Universe capacity arithmetic."""
import pytest

from src.agent.capacity import (DEFAULT_PROVIDER_LIMITS, assess,
                                llm_requests_per_cycle, max_symbols_for_llm,
                                projection)


def test_llm_calls_scale_with_universe_and_signal_rate():
    assert llm_requests_per_cycle(100, 0.4) == pytest.approx(40)
    assert llm_requests_per_cycle(0, 0.4) == 0
    assert llm_requests_per_cycle(100, 0) == 0


def test_signal_rate_is_clamped_to_a_fraction():
    assert llm_requests_per_cycle(10, 5.0) == pytest.approx(10)
    assert llm_requests_per_cycle(10, -1.0) == 0


def test_quota_ceiling_matches_the_arithmetic():
    # 10 a minute over a 60s cycle is 10 calls; at a 50% signal rate that
    # supports 20 symbols.
    assert max_symbols_for_llm(10, 60, 0.5) == 20
    # Halving the cycle halves the allowance.
    assert max_symbols_for_llm(10, 30, 0.5) == 10


def test_quota_ceiling_is_unknown_without_usable_inputs():
    assert max_symbols_for_llm(10, 60, 0) is None
    assert max_symbols_for_llm(0, 60, 0.5) is None
    assert max_symbols_for_llm(10, 0, 0.5) is None


def test_a_small_universe_has_no_findings():
    r = assess(5, per_symbol_seconds=0.005, cycle_budget_seconds=60,
               signal_rate=0.4, llm_limit_per_minute=10)
    assert r.within_budget
    assert r.findings == []


def test_the_real_configuration_is_over_the_free_tier_quota():
    """32 symbols at the 43.9% signal rate measured on real GOOG bars needs
    about 14 validation calls a cycle against 10 allowed."""
    r = assess(32, per_symbol_seconds=0.0054, cycle_budget_seconds=60,
               signal_rate=0.439, llm_limit_per_minute=10)
    assert r.within_budget                      # time is fine
    assert r.binding_constraint == 'LLM quota'  # quota is not
    assert r.max_symbols_by_llm == 22
    assert any('validation calls' in f for f in r.findings)


def test_cycle_time_binds_when_per_symbol_cost_is_high():
    r = assess(50, per_symbol_seconds=2.0, cycle_budget_seconds=60,
               signal_rate=0.01, llm_limit_per_minute=1000)
    assert not r.within_budget
    assert r.binding_constraint == 'cycle time'
    assert any('Cycle takes' in f for f in r.findings)


def test_low_headroom_is_flagged_before_it_overruns():
    r = assess(10, per_symbol_seconds=5.0, cycle_budget_seconds=60,
               signal_rate=0.0, llm_enabled=False)
    assert r.within_budget                       # 50s of 60s
    assert any('headroom' in f for f in r.findings)


def test_disabling_the_llm_removes_the_quota_ceiling():
    r = assess(500, per_symbol_seconds=0.001, cycle_budget_seconds=60,
               signal_rate=0.9, llm_enabled=False)
    assert r.max_symbols_by_llm is None
    assert not any('validation calls' in f for f in r.findings)


def test_fixed_overhead_reduces_the_symbol_ceiling():
    without = assess(1, 0.1, fixed_overhead_seconds=0.0, cycle_budget_seconds=60)
    with_ = assess(1, 0.1, fixed_overhead_seconds=30.0, cycle_budget_seconds=60)
    assert with_.max_symbols_by_time < without.max_symbols_by_time


def test_projection_marks_where_each_limit_is_crossed():
    rows = projection(0.005, 0.0, 60, 0.439, 10, counts=[10, 20, 30, 50])
    assert all(r['fits'] for r in rows)          # time is never the issue here
    assert rows[0]['llm_ok'] and rows[1]['llm_ok']
    assert not rows[2]['llm_ok'] and not rows[3]['llm_ok']


def test_gemini_free_tier_default_is_documented():
    assert DEFAULT_PROVIDER_LIMITS['gemini'] == 10
