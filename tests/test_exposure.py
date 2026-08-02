"""Portfolio-level position gate.

Regression suite for the two defects the 2026-07-24 order journal recorded:

1. PYRAMIDING. `max_position_size: 0.05` was enforced per *order* and never
   against the position already held, so the agent bought TSLA 7x in a row
   (34.356 sh, ~14% of equity) and later 13x in 15 minutes (~$28.2k, ~29% of
   equity) against a 5% cap. Same for the short side: 15 XLF sells and 13 XLV
   sells stacked into concentrated shorts.

2. STOP-AND-REBUY. The stop-loss filled at 15:22:21 and the ensemble bought
   the same symbol back at 15:22:31 — ten seconds later — then kept buying
   every minute. Nothing prevented the entry logic from re-opening what the
   risk layer had just liquidated.

The gate must never block a REDUCING order: being unable to exit is worse
than any concentration.
"""

import os
import tempfile

import pytest

from src.agent.exposure import ExposureGuard

EQUITY = 100_000.0
PRICE = 310.0


def make_guard(**overrides):
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    cfg = {'risk_limits': {'max_position_size': 0.05}}
    cfg['risk_limits'].update(overrides.pop('risk_limits', {}))
    cfg.update(overrides)
    return ExposureGuard(cfg, db_path=path), path


@pytest.fixture
def guard():
    g, path = make_guard()
    yield g
    g.close()
    os.unlink(path)


# ---------------------------------------------------------------- cap: adds

def test_full_target_allowed_on_an_empty_book(guard):
    assert guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 2_000.0) == 2_000.0


def test_target_above_cap_is_trimmed_to_the_cap(guard):
    # 5% of 100k = 5,000
    assert guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 9_000.0) == 5_000.0


def test_position_already_at_the_cap_blocks_further_adds(guard):
    """The TSLA case: ~34.356 sh at ~$400 was already past 5%, and the agent
    kept buying. Must return 0."""
    at_cap_qty = 5_000.0 / PRICE
    assert guard.permitted_notional('TSLA', 'buy', at_cap_qty, PRICE, EQUITY, 2_000.0) == 0.0


def test_partial_position_leaves_only_the_remaining_headroom(guard):
    held_qty = 3_000.0 / PRICE          # $3,000 of a $5,000 cap
    got = guard.permitted_notional('TSLA', 'buy', held_qty, PRICE, EQUITY, 2_500.0)
    assert got == pytest.approx(2_000.0)


def test_short_side_accumulation_is_capped_too(guard):
    """XLF/XLV: repeated sells with no long to close stacked a short. A sell
    that grows a short is an ADD, not a reduce."""
    short_qty = -(5_000.0 / PRICE)
    assert guard.permitted_notional('XLF', 'sell', short_qty, PRICE, EQUITY, 2_000.0) == 0.0


def test_repeated_adds_converge_to_the_cap_not_past_it(guard):
    """Simulate the actual failure loop: same signal every cycle."""
    qty = 0.0
    for _ in range(20):
        allowed = guard.permitted_notional('TSLA', 'buy', qty, PRICE, EQUITY, 2_000.0)
        qty += allowed / PRICE
    assert qty * PRICE == pytest.approx(5_000.0)


# ------------------------------------------------------------ cap: reduces

def test_reducing_order_is_never_blocked_even_far_over_cap(guard):
    """Exits must always be permitted — this is the whole point of a stop."""
    huge_qty = 50_000.0 / PRICE
    got = guard.permitted_notional('TSLA', 'sell', huge_qty, PRICE, EQUITY, 50_000.0)
    assert got == 50_000.0


def test_buy_that_covers_a_short_is_a_reduce(guard):
    short_qty = -(40_000.0 / PRICE)
    assert guard.permitted_notional('XLF', 'buy', short_qty, PRICE, EQUITY, 40_000.0) == 40_000.0


def test_order_that_crosses_through_flat_caps_only_the_new_side(guard):
    """Long $2,000, sell $9,000 -> closes $2,000 and opens a $5,000-capped
    short, not an unbounded one."""
    held_qty = 2_000.0 / PRICE
    got = guard.permitted_notional('TSLA', 'sell', held_qty, PRICE, EQUITY, 9_000.0)
    assert got == pytest.approx(7_000.0)     # 2,000 to flat + 5,000 cap


def test_dust_headroom_is_refused_rather_than_returned(guard):
    """Leaving $0.40 of headroom makes the caller journal an intent (burning
    a client_order_id) only for size_order to reject it on min_notional.
    Unusable headroom is no headroom."""
    nearly_full = (5_000.0 - 0.40) / PRICE
    assert guard.permitted_notional('TSLA', 'buy', nearly_full, PRICE, EQUITY, 2_000.0) == 0.0


def test_headroom_at_the_min_notional_boundary_is_still_allowed(guard):
    held = (5_000.0 - 25.0) / PRICE
    got = guard.permitted_notional('TSLA', 'buy', held, PRICE, EQUITY, 2_000.0)
    assert got == pytest.approx(25.0)


def test_min_notional_comes_from_trading_config():
    g, path = make_guard(trading={'min_notional': 100.0})
    try:
        held = (5_000.0 - 50.0) / PRICE
        assert g.permitted_notional('TSLA', 'buy', held, PRICE, EQUITY, 2_000.0) == 0.0
    finally:
        g.close()
        os.unlink(path)


# ------------------------------------------------------------ bad inputs

@pytest.mark.parametrize('equity', [0.0, -5.0, None])
def test_unusable_equity_blocks_rather_than_sizing_off_garbage(guard, equity):
    assert guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, equity, 2_000.0) == 0.0


@pytest.mark.parametrize('price', [0.0, -1.0, None])
def test_unusable_price_blocks(guard, price):
    assert guard.permitted_notional('TSLA', 'buy', 0.0, price, EQUITY, 2_000.0) == 0.0


# ------------------------------------------------------------- cooldown

def test_forced_exit_blocks_re_entry(guard):
    guard.record_forced_exit('TSLA', 'stop_loss')
    assert guard.in_cooldown('TSLA') is True
    assert guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 2_000.0) == 0.0


def test_cooldown_is_symbol_scoped(guard):
    guard.record_forced_exit('TSLA', 'stop_loss')
    assert guard.in_cooldown('AAPL') is False
    assert guard.permitted_notional('AAPL', 'buy', 0.0, PRICE, EQUITY, 2_000.0) == 2_000.0


def test_cooldown_never_blocks_an_exit(guard):
    """A trailing stop firing after a hard stop must still be able to close."""
    guard.record_forced_exit('TSLA', 'stop_loss')
    held_qty = 10_000.0 / PRICE
    assert guard.permitted_notional('TSLA', 'sell', held_qty, PRICE, EQUITY, 10_000.0) == 10_000.0


def test_cooldown_expires_after_the_configured_window():
    g, path = make_guard(risk_limits={'reentry_cooldown_seconds': 100})
    try:
        g.record_forced_exit('TSLA', 'stop_loss', now=1_000.0)
        assert g.in_cooldown('TSLA', now=1_050.0) is True
        assert g.in_cooldown('TSLA', now=1_101.0) is False
    finally:
        g.close()
        os.unlink(path)


def test_cooldown_survives_a_restart():
    """The 15:22 rebuy happened seconds after a stop; an in-memory cooldown
    would also be defeated by the agent's frequent restarts."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    cfg = {'risk_limits': {'max_position_size': 0.05, 'reentry_cooldown_seconds': 900}}
    try:
        g1 = ExposureGuard(cfg, db_path=path)
        g1.record_forced_exit('TSLA', 'stop_loss')
        g1.close()

        g2 = ExposureGuard(cfg, db_path=path)
        assert g2.in_cooldown('TSLA') is True
        g2.close()
    finally:
        os.unlink(path)


def test_seconds_remaining_is_reported_for_logging():
    g, path = make_guard(risk_limits={'reentry_cooldown_seconds': 100})
    try:
        g.record_forced_exit('TSLA', 'stop_loss', now=1_000.0)
        assert g.cooldown_remaining('TSLA', now=1_040.0) == pytest.approx(60.0)
        assert g.cooldown_remaining('TSLA', now=1_200.0) == 0.0
    finally:
        g.close()
        os.unlink(path)


def test_a_later_exit_extends_the_window(guard):
    guard.record_forced_exit('TSLA', 'stop_loss', now=1_000.0)
    guard.record_forced_exit('TSLA', 'trailing_stop', now=1_500.0)
    assert guard.cooldown_remaining('TSLA', now=1_600.0) > 0


def test_cooldown_disabled_when_configured_to_zero():
    g, path = make_guard(risk_limits={'reentry_cooldown_seconds': 0})
    try:
        g.record_forced_exit('TSLA', 'stop_loss')
        assert g.in_cooldown('TSLA') is False
    finally:
        g.close()
        os.unlink(path)


# --------------------------------------------------------- back-compat

def test_defaults_match_the_shipped_config_when_keys_are_absent():
    """An operator config predating this feature must still get a sane cap
    and a cooldown, not an unbounded gate."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        g = ExposureGuard({}, db_path=path)
        assert 0 < g.max_position_pct <= 0.10
        assert g.cooldown_seconds > 0
        g.close()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Portfolio net exposure
#
# The objective is absolute return uncorrelated to the market. A per-symbol
# cap does not deliver that: eleven separate 5% longs is a 55% long book that
# tracks SPY. This bounds the directional bet across the whole portfolio.
# ---------------------------------------------------------------------------

def test_add_within_the_net_cap_is_allowed(guard):
    got = guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 2_000.0,
                                   net_notional=10_000.0)
    assert got == 2_000.0


def test_add_beyond_the_net_cap_is_trimmed(guard):
    """Net long $29k of a $30k cap leaves $1k of headroom."""
    got = guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 4_000.0,
                                   net_notional=29_000.0)
    assert got == pytest.approx(1_000.0)


def test_add_at_the_net_cap_is_refused(guard):
    got = guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 2_000.0,
                                   net_notional=30_000.0)
    assert got == 0.0


def test_short_side_net_cap_is_symmetric(guard):
    got = guard.permitted_notional('TSLA', 'sell', 0.0, PRICE, EQUITY, 2_000.0,
                                   net_notional=-30_000.0)
    assert got == 0.0


def test_an_order_that_reduces_net_exposure_always_passes(guard):
    """Net long $50k (over cap) and selling: this brings the book back toward
    neutral and must never be blocked."""
    got = guard.permitted_notional('TSLA', 'sell', 200.0, PRICE, EQUITY, 5_000.0,
                                   net_notional=50_000.0)
    assert got == 5_000.0


def test_exits_are_never_blocked_by_the_net_cap(guard):
    held = 40_000.0 / PRICE
    got = guard.permitted_notional('TSLA', 'sell', held, PRICE, EQUITY, 40_000.0,
                                   net_notional=60_000.0)
    assert got == 40_000.0


def test_omitting_net_notional_skips_the_check(guard):
    """Back-compat: every existing caller passes six arguments."""
    got = guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 2_000.0)
    assert got == 2_000.0


def test_the_tighter_of_the_two_caps_wins(guard):
    """Per-symbol headroom is $5,000; net headroom is $500."""
    got = guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 4_000.0,
                                   net_notional=29_500.0)
    assert got == pytest.approx(500.0)


def test_net_dust_headroom_is_refused(guard):
    got = guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 2_000.0,
                                   net_notional=29_999.0)
    assert got == 0.0


def test_net_cap_is_configurable():
    g, path = make_guard(risk_limits={'max_net_exposure': 0.10})
    try:
        assert g.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 2_000.0,
                                    net_notional=10_000.0) == 0.0
    finally:
        g.close()
        os.unlink(path)


def test_net_cap_default_is_thirty_percent(guard):
    assert guard.max_net_exposure == pytest.approx(0.30)
