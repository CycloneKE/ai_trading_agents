"""Cross-sectional momentum rotation."""
import pytest

from src.agent.momentum_rotation import (RotationConfig, config_from_dict,
                                         momentum_score, rank_universe,
                                         rebalance_orders, select,
                                         target_weights)

CFG = RotationConfig(lookback=12, skip_recent=1, top_n=3)


def series(*returns, start=100.0):
    """Build a price series from period returns."""
    out = [start]
    for r in returns:
        out.append(out[-1] * (1 + r))
    return out


def flat(n, price=100.0):
    return [price] * n


# --- momentum_score ---------------------------------------------------------

def test_returns_none_without_enough_history():
    assert momentum_score(flat(12), CFG) is None
    assert momentum_score([], CFG) is None
    assert momentum_score(None, CFG) is None


def test_skips_the_most_recent_period():
    """12-1: the window ends one period back, so a spike in the final period
    must not affect the score."""
    prices = [100.0 * (1.01 ** i) for i in range(14)]
    base = momentum_score(prices, CFG)
    spiked = momentum_score(prices[:-1] + [prices[-1] * 5], CFG)
    assert base == pytest.approx(spiked)


def test_measures_the_right_window():
    prices = flat(14)
    prices[1] = 50.0          # 13 back... inside the window start
    m = momentum_score(prices, CFG)
    assert m is not None


def test_rising_series_scores_positive_falling_negative():
    assert momentum_score([100 * (1.02 ** i) for i in range(20)], CFG) > 0
    assert momentum_score([100 * (0.98 ** i) for i in range(20)], CFG) < 0


def test_ignores_non_positive_prices():
    prices = [0, -5] + [100 * (1.02 ** i) for i in range(20)]
    assert momentum_score(prices, CFG) is not None


# --- ranking ----------------------------------------------------------------

def test_ranks_strongest_first():
    universe = {
        'WEAK': [100 * (1.001 ** i) for i in range(20)],
        'STRONG': [100 * (1.05 ** i) for i in range(20)],
        'MID': [100 * (1.02 ** i) for i in range(20)],
    }
    ranked = rank_universe(universe, CFG)
    assert [r.symbol for r in ranked] == ['STRONG', 'MID', 'WEAK']
    assert [r.rank for r in ranked] == [1, 2, 3]


def test_symbols_without_history_are_omitted_not_ranked_last():
    """Ranking a new listing last would systematically avoid new names."""
    universe = {'OLD': [100 * (1.02 ** i) for i in range(20)], 'NEW': flat(3)}
    ranked = rank_universe(universe, CFG)
    assert [r.symbol for r in ranked] == ['OLD']


def test_empty_universe_ranks_to_nothing():
    assert rank_universe({}, CFG) == []
    assert rank_universe(None, CFG) == []


# --- selection --------------------------------------------------------------

def test_holds_only_the_top_n():
    universe = {f'S{i}': [100 * ((1 + i / 100) ** j) for j in range(20)]
                for i in range(1, 6)}
    chosen = select(universe, RotationConfig(top_n=2))
    assert len(chosen) == 2
    assert [c.symbol for c in chosen] == ['S5', 'S4']


def test_weights_sum_to_one_when_fully_invested():
    universe = {f'S{i}': [100 * (1.02 ** j) for j in range(20)] for i in range(3)}
    chosen = select(universe, RotationConfig(top_n=3, max_weight=1.0))
    assert sum(c.weight for c in chosen) == pytest.approx(1.0)


def test_absolute_filter_drops_falling_leaders():
    """In a bear market the best name is still losing money. Relative
    strength alone would hold it; the absolute filter holds cash instead."""
    universe = {'A': [100 * (0.98 ** i) for i in range(20)],
                'B': [100 * (0.95 ** i) for i in range(20)]}
    assert select(universe, RotationConfig(absolute_filter=True)) == []
    assert len(select(universe, RotationConfig(absolute_filter=False))) == 2


def test_max_weight_caps_concentration():
    universe = {'ONLY': [100 * (1.05 ** i) for i in range(20)]}
    chosen = select(universe, RotationConfig(top_n=3, max_weight=0.4))
    assert chosen[0].weight == pytest.approx(0.4)


def test_target_weights_is_empty_when_nothing_qualifies():
    assert target_weights({'A': flat(20)}, CFG) == {}


# --- rebalancing ------------------------------------------------------------

def test_generates_buys_and_sells_toward_target():
    orders = rebalance_orders({'A': 0.5, 'B': 0.5}, {'A': 1.0},
                              100_000, {'A': 10, 'B': 10})
    assert orders['A'] == pytest.approx(50_000)
    assert orders['B'] == pytest.approx(-50_000)


def test_small_drifts_are_not_traded():
    """Churning a position by a fraction of a percent pays costs for nothing."""
    orders = rebalance_orders({'A': 0.500}, {'A': 0.502}, 100_000, {'A': 10})
    assert orders == {}


def test_unpriceable_symbols_are_left_alone():
    orders = rebalance_orders({'A': 0.5}, {'A': 1.0}, 100_000, {})
    assert orders == {}
    orders = rebalance_orders({'A': 0.5}, {'A': 1.0}, 100_000, {'A': 0})
    assert orders == {}


def test_zero_portfolio_value_trades_nothing():
    assert rebalance_orders({'A': 0.0}, {'A': 1.0}, 0, {'A': 10}) == {}


def test_full_exit_to_cash():
    orders = rebalance_orders({'A': 1.0}, {}, 100_000, {'A': 10})
    assert orders['A'] == pytest.approx(-100_000)


# --- config -----------------------------------------------------------------

def test_config_from_dict_clamps_nonsense():
    c = config_from_dict({'lookback': -5, 'skip_recent': -2, 'top_n': -1})
    assert c.lookback >= 1 and c.skip_recent >= 0 and c.top_n >= 0


def test_config_defaults_to_twelve_one():
    c = config_from_dict({})
    assert (c.lookback, c.skip_recent) == (12, 1)
    assert c.required_history == 13
