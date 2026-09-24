"""The position rules both books share (src/agent/position_rules.py)."""
from types import SimpleNamespace

import pytest

from src.agent.position_rules import (
    ADD_DEFAULTS, EXIT_DEFAULTS, PositionState, plan_add, plan_exit, replay,
    state_from_journal, stop_triggered)


def _fill(side, qty, price, time='2026-09-24T10:00:00', fees=0.0, symbol='X'):
    return {'symbol': symbol, 'side': side, 'quantity': qty, 'price': price,
            'fees': fees, 'time': time}


# ------------------------------------------------------------------ replay

def test_replay_tracks_entries_cost_and_the_last_entry():
    s = replay([_fill('buy', 10, 100, fees=2), _fill('buy', 5, 110)])['X']
    assert (s.quantity, s.cost, s.entries, s.last_entry_price) == (15, 1552, 2, 110)


def test_a_partial_sell_is_a_trim_and_realises_its_share():
    s = replay([_fill('buy', 10, 100), _fill('sell', 4, 120, '2026-09-25T11:00:00', fees=1)])['X']
    assert s.quantity == 6 and s.cost == pytest.approx(600)
    assert s.realised == pytest.approx(4 * 120 - 1 - 400)
    assert s.last_trim_day == '2026-09-25'


def test_a_closed_position_starts_afresh():
    s = replay([_fill('buy', 10, 100), _fill('buy', 5, 110), _fill('sell', 15, 120),
                _fill('buy', 3, 130, '2026-09-26T10:00:00')])['X']
    assert (s.quantity, s.entries, s.last_entry_price, s.opened_at) == (3, 1, 130, '2026-09-26T10:00:00')
    assert s.last_trim_day is None


def test_a_sell_of_nothing_is_ignored():
    assert not replay([_fill('sell', 5, 100)])['X'].held


# -------------------------------------------------------------------- adds

def _held(qty=100, price=100.0, entries=1, cost=None):
    return PositionState(quantity=qty, cost=cost if cost is not None else qty * price,
                         entries=entries, last_entry_price=price)


def test_no_add_until_a_sale_would_clear_its_costs():
    state = _held()
    # 5% up but a 6% round-trip exit cost leaves it underwater.
    assert plan_add(state, 105.0, 0.06, 1e6, 10000, ADD_DEFAULTS) == (0.0, 'add_not_profitable')


def test_an_add_is_half_a_new_position():
    value, why = plan_add(_held(), 106.0, 0.02, 1e6, 10000, ADD_DEFAULTS)
    assert why is None and value == 5000


def test_adds_stop_at_max_adds():
    assert plan_add(_held(entries=3), 200.0, 0.02, 1e6, 10000, ADD_DEFAULTS)[1] == 'max_adds'


def test_the_position_cap_limits_the_add():
    # 100 shares at 106 = 10,600 of a 30,000 account; 40% cap leaves 1,400.
    value, why = plan_add(_held(), 106.0, 0.02, 30000, 10000, ADD_DEFAULTS)
    assert why is None and value == pytest.approx(30000 * 0.4 - 10600)
    assert plan_add(_held(), 106.0, 0.02, 20000, 10000, ADD_DEFAULTS)[1] == 'position_cap'


def test_adding_can_be_disabled():
    rule = {**ADD_DEFAULTS, 'enabled': False}
    assert plan_add(_held(), 150.0, 0.02, 1e6, 10000, rule) == (0.0, 'already_held')


# ------------------------------------------------------------------- exits

TODAY = '2026-09-24'


def test_a_strong_sell_closes():
    assert plan_exit(_held(), 0.8, 100.0, 1e6, TODAY, EXIT_DEFAULTS) == (100, None)


def test_a_weak_sell_trims():
    assert plan_exit(_held(), 0.6, 100.0, 1e5, TODAY, EXIT_DEFAULTS) == (50, None)


def test_only_one_trim_a_day():
    state = _held()
    state.last_trim_day = TODAY
    assert plan_exit(state, 0.6, 100.0, 1e5, TODAY, EXIT_DEFAULTS) == (0.0, 'trimmed_today')
    assert plan_exit(state, 0.6, 100.0, 1e5, '2026-09-25', EXIT_DEFAULTS) == (50, None)


def test_a_trim_that_would_leave_a_sliver_closes_instead():
    # Half of 10,000 would leave 1% of a 500,000 account, under the 2% floor.
    assert plan_exit(_held(), 0.6, 100.0, 500000, TODAY, EXIT_DEFAULTS) == (100, None)


def test_partial_exits_can_be_disabled():
    rule = {**EXIT_DEFAULTS, 'partial_exits': False}
    assert plan_exit(_held(), 0.55, 100.0, 1e6, TODAY, rule) == (100, None)


def test_nothing_held_nothing_to_sell():
    assert plan_exit(PositionState(), 0.9, 100.0, 1e6, TODAY, EXIT_DEFAULTS) == (0.0, 'no_position')


# ------------------------------------------------------------------- stops

def test_stops():
    assert stop_triggered(93.0, 100.0, 100.0, 0.08, 0.10) is None
    assert stop_triggered(92.0, 100.0, 100.0, 0.08, 0.10) == 'stop_loss'
    assert stop_triggered(117.0, 100.0, 130.0, 0.08, 0.10) == 'trailing_stop'
    assert stop_triggered(118.0, 100.0, 130.0, 0.08, 0.10) is None


# ---------------------------------------------------- state from the journal

def _journal(rows):
    return SimpleNamespace(filled_orders=lambda: rows)


def _row(side, qty, price, when='2026-09-24T10:00:00', symbol='SPY'):
    return {'symbol': symbol, 'side': side, 'filled_quantity': qty,
            'filled_avg_price': price, 'created_at': when, 'updated_at': when}


def test_the_journal_supplies_a_positions_history():
    s = state_from_journal(_journal([_row('buy', 10, 100), _row('buy', 5, 110)]), 'SPY', 15, 103.33)
    assert (s.entries, s.last_entry_price) == (2, 110)


def test_a_journal_that_disagrees_with_the_broker_is_not_trusted():
    s = state_from_journal(_journal([_row('buy', 3, 100)]), 'SPY', 15, 104.0)
    assert (s.quantity, s.entries, s.last_entry_price, s.cost) == (15, 1, 104.0, 15 * 104.0)


def test_a_sell_that_left_shares_is_a_trim_even_without_history():
    s = state_from_journal(_journal([_row('sell', 4, 100, '2026-09-24T11:00:00')]), 'SPY', 4, 100.0)
    assert s.last_trim_day == '2026-09-24'
