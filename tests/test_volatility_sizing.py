"""Volatility-scaled position sizing."""
import pytest

from src.agent.position_sizing import (DEFAULT_RISK_PER_TRADE,
                                       risk_contribution,
                                       volatility_scaled_value)

EQ = 100_000.0


def size(atr_pct, **kw):
    return volatility_scaled_value(EQ, atr_pct, **kw)


def test_higher_volatility_gets_a_smaller_position():
    quiet = size(0.045)
    wild = size(0.090)
    assert wild < quiet


def test_risk_is_equalised_above_the_cap_crossover():
    """Past the point where the cap stops binding, every position risks the
    same fraction of equity regardless of the instrument's volatility."""
    for atr in (0.045, 0.07, 0.12):
        r = risk_contribution(size(atr), EQ, atr)
        assert r == pytest.approx(DEFAULT_RISK_PER_TRADE, rel=1e-6)


def test_cap_binds_for_quiet_instruments():
    # A 1% ATR name would imply a 20% position at 0.5% risk; the cap holds it
    # to 5%, which is the concentration ceiling doing its job.
    assert size(0.010) == pytest.approx(EQ * 0.05)
    assert size(0.026) == pytest.approx(EQ * 0.05)


def test_falls_back_to_the_flat_cap_without_atr():
    assert size(None) == pytest.approx(EQ * 0.05)
    assert size(0) == pytest.approx(EQ * 0.05)


def test_confidence_scales_the_result():
    full = size(0.07, confidence=1.0)
    half = size(0.07, confidence=0.5)
    assert half == pytest.approx(full * 0.5)


def test_zero_or_negative_confidence_means_no_position():
    assert size(0.05, confidence=0.0) == 0.0
    assert size(0.05, confidence=-1.0) == 0.0


def test_confidence_above_one_is_clamped():
    assert size(0.07, confidence=5.0) == pytest.approx(size(0.07, confidence=1.0))


def test_never_exceeds_the_cap():
    for atr in (0.0001, 0.001, 0.01, 0.5):
        assert size(atr) <= EQ * 0.05 + 1e-9


def test_zero_equity_yields_nothing():
    assert volatility_scaled_value(0, 0.03) == 0.0
    assert volatility_scaled_value(-100, 0.03) == 0.0


def test_risk_contribution_is_zero_without_inputs():
    assert risk_contribution(1000, EQ, None) == 0.0
    assert risk_contribution(1000, 0, 0.03) == 0.0


def test_a_crypto_position_is_a_fraction_of_an_equity_one():
    """The point of the change: 5% in a 7%-ATR pair is not the same risk as
    5% in a 2.6%-ATR stock, and should not be the same size."""
    equity_like = size(0.026)
    crypto_like = size(0.070)
    assert crypto_like < equity_like / 1.5
