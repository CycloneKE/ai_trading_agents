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
