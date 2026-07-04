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
