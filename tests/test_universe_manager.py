"""Universe additions must be screened.

Today `universe_scout` proposes anything that moved 3%. The traded universe
holds SPY, QQQ and DIA — three near-identical instruments — which leaves
roughly seven independent names out of eleven. For an uncorrelated-return
objective that is the wrong shape, so a candidate that duplicates existing
exposure is rejected rather than added.
"""

import pytest

from src.agent.universe_manager import correlation, propose_additions

CFG = {'universe': {'min_avg_dollar_volume': 1_000_000, 'min_price': 5.0,
                    'max_correlation': 0.9}}

UP = [100.0 + i for i in range(60)]
ALSO_UP = [50.0 + i * 0.5 for i in range(60)]        # perfectly correlated
DOWN = [10000.0 / (100.0 + i) for i in range(60)]    # perfectly anti-correlated
NOISE = [100.0 + (7 * i % 11) for i in range(60)]


def candidate(symbol, **over):
    base = {'symbol': symbol, 'reason': 'moved 4% today', 'market': 'us',
            'avg_dollar_volume': 5_000_000, 'price': 50.0}
    base.update(over)
    return base


def test_a_clean_candidate_is_proposed():
    out = propose_additions({'AAPL'}, [candidate('NFLX')],
                            {'AAPL': UP, 'NFLX': NOISE}, CFG)
    assert [c['symbol'] for c in out] == ['NFLX']


def test_an_already_tracked_symbol_is_skipped():
    out = propose_additions({'NFLX'}, [candidate('NFLX')],
                            {'NFLX': NOISE}, CFG)
    assert out == []


def test_an_illiquid_candidate_is_rejected():
    """You cannot exit what does not trade."""
    out = propose_additions({'AAPL'}, [candidate('THIN', avg_dollar_volume=10_000)],
                            {'AAPL': UP, 'THIN': NOISE}, CFG)
    assert out == []


def test_a_sub_minimum_price_candidate_is_rejected():
    out = propose_additions({'AAPL'}, [candidate('PENNY', price=1.2)],
                            {'AAPL': UP, 'PENNY': NOISE}, CFG)
    assert out == []


def test_a_correlated_candidate_is_rejected():
    """The SPY/QQQ/DIA problem: adding a fourth index tracker buys no
    diversification but consumes a universe slot."""
    out = propose_additions({'SPY'}, [candidate('QQQ')],
                            {'SPY': UP, 'QQQ': ALSO_UP}, CFG)
    assert out == []


def test_an_anticorrelated_candidate_is_also_rejected():
    """|rho| > 0.9 either way is redundant exposure — a perfect inverse is
    the same bet with the sign flipped."""
    out = propose_additions({'SPY'}, [candidate('SH')],
                            {'SPY': UP, 'SH': DOWN}, CFG)
    assert out == []


def test_a_candidate_with_no_history_is_rejected():
    """Unscreenable is not the same as clean."""
    out = propose_additions({'AAPL'}, [candidate('NEW')], {'AAPL': UP}, CFG)
    assert out == []


def test_the_rejection_reason_is_not_silently_swallowed(caplog):
    import logging
    with caplog.at_level(logging.INFO):
        propose_additions({'SPY'}, [candidate('QQQ')], {'SPY': UP, 'QQQ': ALSO_UP}, CFG)
    assert 'QQQ' in caplog.text


def test_market_is_carried_through():
    out = propose_additions(set(), [candidate('SCOM', market='kenyan')],
                            {'SCOM': NOISE}, CFG)
    assert out[0]['market'] == 'kenyan'


def test_correlation_of_identical_series_is_one():
    assert correlation(UP, UP) == pytest.approx(1.0)


def test_correlation_of_inverse_series_is_minus_one():
    assert correlation(UP, DOWN) == pytest.approx(-1.0)


def test_correlation_of_a_flat_series_is_zero_not_nan():
    assert correlation(UP, [100.0] * 60) == 0.0


def test_correlation_of_short_series_is_zero():
    assert correlation([1.0, 2.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# Unscreenable input must REJECT, not admit
#
# Sixth occurrence of the NaN-defeats-comparison class in this codebase. Here
# the direction matters: a NaN correlation compares False against the
# threshold, so a maximally-correlated candidate is silently APPROVED. And
# returning 0.0 ("uncorrelated") is not a safe fallback either — 0.0 also
# admits. Unscreenable data must be refused, exactly as insufficient history
# already is.
# ---------------------------------------------------------------------------

import math

NAN = float('nan')
INF = float('inf')


def test_correlation_never_returns_nan_for_contaminated_input():
    """The contract is "never NaN", not "returns zero". Non-finite bars are
    dropped and the remaining clean bars are correlated normally — a NaN
    result is what would be dangerous, because it compares False against
    every threshold and waves the candidate through."""
    dirty = [NAN] + [100.0 + i for i in range(59)]
    got = correlation(UP, dirty)
    assert not math.isnan(got), 'NaN compares False against every threshold'
    assert -1.0 <= got <= 1.0


def test_correlation_never_returns_nan_for_infinite_input():
    dirty = [INF] + [100.0 + i for i in range(59)]
    got = correlation(UP, dirty)
    assert not math.isnan(got)
    assert -1.0 <= got <= 1.0


def test_a_zero_price_does_not_raise():
    """A zero close would divide by zero in the returns calculation."""
    got = correlation(UP, [0.0] + [100.0 + i for i in range(59)])
    assert not math.isnan(got)


def test_a_candidate_with_contaminated_history_is_rejected():
    """THE bug: a NaN in QQQ's series made the correlation NaN, so
    `abs(nan) > 0.9` was False and a duplicate of SPY was approved."""
    dirty = [NAN] + [100.0 + i for i in range(59)]
    out = propose_additions({'SPY'}, [candidate('QQQ')], {'SPY': UP, 'QQQ': dirty}, CFG)
    assert out == []


def test_a_tracked_symbol_with_contaminated_history_does_not_admit_a_duplicate():
    """The contamination can be on the HELD side too — same outcome required."""
    dirty = [NAN] + [100.0 + i for i in range(59)]
    out = propose_additions({'SPY'}, [candidate('QQQ')], {'SPY': dirty, 'QQQ': ALSO_UP}, CFG)
    assert out == []


# --------------------------------------------------- malformed input

def test_one_malformed_candidate_does_not_poison_the_batch():
    """universe_scout feeds this directly; one bad row must not block
    screening of every other candidate in the same call."""
    good = candidate('NFLX')
    bad = candidate('BAD', avg_dollar_volume='not-a-number')
    out = propose_additions({'AAPL'}, [bad, good],
                            {'AAPL': UP, 'NFLX': NOISE, 'BAD': NOISE}, CFG)
    assert [c['symbol'] for c in out] == ['NFLX']


def test_a_none_candidate_entry_is_skipped():
    out = propose_additions({'AAPL'}, [None, candidate('NFLX')],
                            {'AAPL': UP, 'NFLX': NOISE}, CFG)
    assert [c['symbol'] for c in out] == ['NFLX']


def test_a_candidate_missing_its_numeric_fields_is_rejected():
    out = propose_additions({'AAPL'}, [{'symbol': 'X', 'reason': 'r'}],
                            {'AAPL': UP, 'X': NOISE}, CFG)
    assert out == []


def test_a_non_finite_price_or_volume_is_rejected():
    for bad in ({'price': NAN}, {'avg_dollar_volume': NAN}, {'price': INF}):
        out = propose_additions({'AAPL'}, [candidate('X', **bad)],
                                {'AAPL': UP, 'X': NOISE}, CFG)
        assert out == [], f'admitted a candidate with {bad}'


# ---------------------------------------------------------------------------
# Removals
#
# Nothing proposes removals today, so a dead or persistently losing symbol
# stays in the universe forever, consuming API quota and LLM calls.
# ---------------------------------------------------------------------------

from src.agent.universe_manager import propose_removals

RCFG = {'universe': {'min_avg_dollar_volume': 1_000_000,
                     'max_consecutive_losses': 4, 'max_undefined_days': 30}}


def stat(**over):
    base = {'consecutive_losses': 0, 'undefined_regime_days': 0,
            'avg_dollar_volume': 5_000_000, 'attribution': 100.0}
    base.update(over)
    return base


def test_a_healthy_symbol_is_kept():
    assert propose_removals({'AAPL'}, {'AAPL': stat()}, RCFG) == []


def test_consecutive_losses_trigger_a_removal():
    out = propose_removals({'TSLA'}, {'TSLA': stat(consecutive_losses=5)}, RCFG)
    assert [r['symbol'] for r in out] == ['TSLA']
    assert 'consecutive' in out[0]['reason'].lower()


def test_a_symbol_stuck_in_undefined_regime_is_removed():
    """The router can never trade it, so it is pure overhead."""
    out = propose_removals({'XLE'}, {'XLE': stat(undefined_regime_days=45)}, RCFG)
    assert [r['symbol'] for r in out] == ['XLE']
    assert 'regime' in out[0]['reason'].lower()


def test_a_symbol_gone_illiquid_is_removed():
    out = propose_removals({'THIN'}, {'THIN': stat(avg_dollar_volume=5_000)}, RCFG)
    assert [r['symbol'] for r in out] == ['THIN']


def test_persistently_negative_attribution_triggers_a_removal():
    out = propose_removals({'XLV'}, {'XLV': stat(attribution=-500.0)}, RCFG)
    assert [r['symbol'] for r in out] == ['XLV']


def test_a_symbol_at_the_threshold_is_kept():
    """Boundary: 4 losses is the limit, not a breach."""
    assert propose_removals({'TSLA'}, {'TSLA': stat(consecutive_losses=4)}, RCFG) == []


def test_a_symbol_with_no_stats_is_kept():
    """Absence of evidence is not evidence of failure."""
    assert propose_removals({'NEW'}, {}, RCFG) == []


def test_only_one_proposal_per_symbol():
    out = propose_removals({'BAD'}, {'BAD': stat(consecutive_losses=9,
                                                 attribution=-900.0,
                                                 avg_dollar_volume=1.0)}, RCFG)
    assert len(out) == 1


def test_a_symbol_with_non_finite_stats_is_kept():
    """A NaN/Infinity stat is unreadable, not evidence of failure — treat it
    the same as a missing stat and keep the symbol rather than silently
    misjudging it."""
    out = propose_removals(
        {'CORRUPT'},
        {'CORRUPT': stat(consecutive_losses=float('nan'),
                          undefined_regime_days=float('inf'),
                          avg_dollar_volume=float('nan'),
                          attribution=float('nan'))},
        RCFG)
    assert out == []
