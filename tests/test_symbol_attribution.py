"""Per-symbol P&L attribution from the order journal."""
import pytest

from src.agent.strategy_attribution import compute_symbol_attribution


def fill(symbol, side, qty, price, at, strategy='momentum'):
    return {'symbol': symbol, 'side': side, 'filled_quantity': qty,
            'filled_avg_price': price, 'created_at': at, 'strategy': strategy}


def test_a_simple_round_trip_is_attributed_to_its_symbol():
    a = compute_symbol_attribution([
        fill('AAPL', 'buy', 10, 100.0, '2026-01-01'),
        fill('AAPL', 'sell', 10, 110.0, '2026-01-05'),
    ])
    assert a['AAPL']['realized_pnl'] == pytest.approx(100.0)
    assert a['AAPL']['closed_trades'] == 1
    assert a['AAPL']['win_rate'] == 1.0
    assert a['AAPL']['open_quantity'] == 0


def test_losses_are_attributed_too():
    a = compute_symbol_attribution([
        fill('MSFT', 'buy', 10, 100.0, '2026-01-01'),
        fill('MSFT', 'sell', 10, 90.0, '2026-01-05'),
    ])
    assert a['MSFT']['realized_pnl'] == pytest.approx(-100.0)
    assert a['MSFT']['win_rate'] == 0.0


def test_averages_cost_across_multiple_buys():
    a = compute_symbol_attribution([
        fill('X', 'buy', 10, 100.0, '2026-01-01'),
        fill('X', 'buy', 10, 200.0, '2026-01-02'),   # average 150
        fill('X', 'sell', 20, 160.0, '2026-01-03'),
    ])
    assert a['X']['realized_pnl'] == pytest.approx(200.0)


def test_a_partial_sell_leaves_the_rest_open():
    a = compute_symbol_attribution([
        fill('X', 'buy', 10, 100.0, '2026-01-01'),
        fill('X', 'sell', 4, 120.0, '2026-01-02'),
    ])
    assert a['X']['realized_pnl'] == pytest.approx(80.0)
    assert a['X']['open_quantity'] == pytest.approx(6)
    assert a['X']['avg_entry_price'] == pytest.approx(100.0)


def test_symbols_are_kept_separate_even_within_one_strategy():
    """The point of this view: a losing symbol can hide inside a winning
    strategy's aggregate."""
    a = compute_symbol_attribution([
        fill('WIN', 'buy', 10, 100.0, '2026-01-01', strategy='s'),
        fill('WIN', 'sell', 10, 150.0, '2026-01-02', strategy='s'),
        fill('LOSE', 'buy', 10, 100.0, '2026-01-01', strategy='s'),
        fill('LOSE', 'sell', 10, 80.0, '2026-01-02', strategy='s'),
    ])
    assert a['WIN']['realized_pnl'] > 0
    assert a['LOSE']['realized_pnl'] < 0


def test_unrealized_pnl_uses_the_price_lookup():
    a = compute_symbol_attribution(
        [fill('X', 'buy', 10, 100.0, '2026-01-01')],
        price_lookup=lambda s: 130.0)
    assert a['X']['unrealized_pnl'] == pytest.approx(300.0)
    assert a['X']['total_pnl'] == pytest.approx(300.0)


def test_a_broken_price_lookup_does_not_lose_the_realized_figures():
    def boom(_):
        raise RuntimeError("price feed down")
    a = compute_symbol_attribution([
        fill('X', 'buy', 10, 100.0, '2026-01-01'),
        fill('X', 'sell', 5, 120.0, '2026-01-02'),
    ], price_lookup=boom)
    assert a['X']['realized_pnl'] == pytest.approx(100.0)
    assert a['X']['unrealized_pnl'] == 0.0


def test_malformed_rows_are_skipped():
    a = compute_symbol_attribution([
        {'symbol': None, 'side': 'buy', 'filled_quantity': 1, 'filled_avg_price': 1},
        {'symbol': 'X', 'side': 'buy', 'filled_quantity': 0, 'filled_avg_price': 10},
        {'symbol': 'X', 'side': 'hold', 'filled_quantity': 1, 'filled_avg_price': 10},
        fill('X', 'buy', 1, 10.0, '2026-01-01'),
    ])
    assert a['X']['buys'] == 1


def test_fills_are_processed_in_time_order_not_list_order():
    out_of_order = [
        fill('X', 'sell', 10, 120.0, '2026-01-05'),
        fill('X', 'buy', 10, 100.0, '2026-01-01'),
    ]
    a = compute_symbol_attribution(out_of_order)
    assert a['X']['realized_pnl'] == pytest.approx(200.0)


def test_no_fills_means_no_attribution():
    assert compute_symbol_attribution([]) == {}
