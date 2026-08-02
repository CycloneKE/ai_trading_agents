"""Unit tests for dollar-notional position sizing."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.position_sizing import size_order


def test_small_account_gets_fractional_quantity():
    # 2% of a $2,000 account into a $750 stock
    sized = size_order(40.0, 750.0)
    assert sized is not None
    assert sized.is_fractional
    assert sized.time_in_force == 'day'
    assert abs(sized.quantity - 0.0533) < 1e-4
    assert sized.notional == 39.98


def test_below_min_notional_skipped():
    assert size_order(3.0, 100.0) is None
    assert size_order(4.99, 100.0, min_notional=5.0) is None
    assert size_order(5.0, 100.0, min_notional=5.0) is not None


def test_whole_share_keeps_preferred_tif():
    sized = size_order(1000.0, 100.0, preferred_tif='gtc')
    assert sized.quantity == 10.0
    assert not sized.is_fractional
    assert sized.time_in_force == 'gtc'


def test_fractional_disabled_rounds_down_to_whole_shares():
    sized = size_order(250.0, 100.0, allow_fractional=False)
    assert sized.quantity == 2.0
    assert not sized.is_fractional
    # under one whole share -> skip entirely
    assert size_order(50.0, 100.0, allow_fractional=False) is None


def test_bad_inputs_return_none():
    assert size_order(0, 100.0) is None
    assert size_order(100.0, 0) is None
    assert size_order(-10, 100.0) is None
    assert size_order(100.0, -5) is None


# ---------------------------------------------------------------------------
# Broker TIF rule
#
# Alpaca rejects fractional orders that are not DAY. The entry path got this
# right via size_order(); the exit path hardcoded 'gtc' and lost seventeen
# consecutive TSLA stop-losses on 2026-07-24. The rule now lives in exactly
# one function so both paths share it.
# ---------------------------------------------------------------------------

from src.agent.position_sizing import time_in_force_for


def test_fractional_quantity_forces_day():
    assert time_in_force_for(34.356, preferred_tif='gtc') == 'day'


def test_whole_share_quantity_keeps_the_preferred_tif():
    assert time_in_force_for(30.0, preferred_tif='gtc') == 'gtc'
    assert time_in_force_for(30, preferred_tif='day') == 'day'


def test_negative_quantity_is_judged_on_magnitude():
    """Short closes arrive as signed quantities in some call sites."""
    assert time_in_force_for(-34.356, preferred_tif='gtc') == 'day'
    assert time_in_force_for(-30.0, preferred_tif='gtc') == 'gtc'


def test_string_quantity_is_still_classified_correctly():
    """Broker SDKs hand back qty as a string; isinstance checks silently
    skipped the coercion and let a fractional GTC order through."""
    assert time_in_force_for('34.356', preferred_tif='gtc') == 'day'
    assert time_in_force_for('30', preferred_tif='gtc') == 'gtc'


def test_unparseable_quantity_falls_back_to_day():
    """DAY is the safe default: it is accepted for both fractional and whole
    quantities, so an unknown type can never produce a rejected order."""
    assert time_in_force_for(None, preferred_tif='gtc') == 'day'
    assert time_in_force_for('abc', preferred_tif='gtc') == 'day'


def test_size_order_uses_the_same_rule():
    sized = size_order(1000.0, 310.0, preferred_tif='gtc')
    assert sized.is_fractional
    assert sized.time_in_force == 'day'
