"""Unit tests for per-strategy P&L attribution."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.strategy_attribution import compute_attribution


def fill(strategy, symbol, side, qty, price, ts):
    return {
        'strategy': strategy, 'symbol': symbol, 'side': side,
        'filled_quantity': qty, 'filled_avg_price': price,
        'created_at': f'2026-07-02T{ts}',
    }


def test_round_trip_realized_pnl():
    fills = [
        fill('momentum', 'AAPL', 'buy', 10, 100.0, '10:00'),
        fill('momentum', 'AAPL', 'sell', 10, 105.0, '11:00'),
    ]
    out = compute_attribution(fills)
    assert out['momentum']['realized_pnl'] == 50.0
    assert out['momentum']['closed_trades'] == 1
    assert out['momentum']['win_rate'] == 1.0
    assert out['momentum']['open_positions'] == []


def test_average_cost_and_partial_close():
    fills = [
        fill('s', 'SPY', 'buy', 10, 100.0, '10:00'),
        fill('s', 'SPY', 'buy', 10, 110.0, '10:30'),   # avg cost 105
        fill('s', 'SPY', 'sell', 5, 120.0, '11:00'),   # +75 realized
    ]
    out = compute_attribution(fills)
    assert out['s']['realized_pnl'] == 75.0
    pos = out['s']['open_positions'][0]
    assert pos['quantity'] == 15
    assert pos['avg_entry_price'] == 105.0


def test_strategies_are_independent_on_same_symbol():
    fills = [
        fill('momentum', 'AAPL', 'buy', 10, 100.0, '10:00'),
        fill('mean_rev', 'AAPL', 'sell', 5, 100.0, '10:01'),  # separate short book
        fill('momentum', 'AAPL', 'sell', 10, 110.0, '11:00'),
        fill('mean_rev', 'AAPL', 'buy', 5, 90.0, '11:01'),    # cover at profit
    ]
    out = compute_attribution(fills)
    assert out['momentum']['realized_pnl'] == 100.0
    assert out['mean_rev']['realized_pnl'] == 50.0  # short 100 -> cover 90
    assert out['momentum']['open_positions'] == []
    assert out['mean_rev']['open_positions'] == []


def test_losing_trade_and_win_rate():
    fills = [
        fill('s', 'AAPL', 'buy', 10, 100.0, '09:00'),
        fill('s', 'AAPL', 'sell', 10, 95.0, '10:00'),   # -50
        fill('s', 'MSFT', 'buy', 10, 200.0, '11:00'),
        fill('s', 'MSFT', 'sell', 10, 210.0, '12:00'),  # +100
    ]
    out = compute_attribution(fills)
    assert out['s']['realized_pnl'] == 50.0
    assert out['s']['closed_trades'] == 2
    assert out['s']['win_rate'] == 0.5


def test_unrealized_with_price_lookup():
    fills = [fill('s', 'AAPL', 'buy', 10, 100.0, '10:00')]
    out = compute_attribution(fills, price_lookup=lambda sym: 104.0)
    pos = out['s']['open_positions'][0]
    assert pos['unrealized_pl'] == 40.0
    assert out['s']['unrealized_pnl'] == 40.0


def test_cross_through_zero_opens_new_book():
    fills = [
        fill('s', 'AAPL', 'buy', 5, 100.0, '10:00'),
        fill('s', 'AAPL', 'sell', 8, 110.0, '11:00'),  # close 5 (+50), short 3 @110
    ]
    out = compute_attribution(fills)
    assert out['s']['realized_pnl'] == 50.0
    pos = out['s']['open_positions'][0]
    assert pos['quantity'] == -3
    assert pos['avg_entry_price'] == 110.0


def test_garbage_rows_ignored():
    fills = [
        fill('s', 'AAPL', 'buy', 0, 100.0, '10:00'),      # zero qty
        fill('s', 'AAPL', 'buy', 5, 0, '10:01'),          # zero price
        {'strategy': 's', 'symbol': None, 'side': 'buy',  # no symbol
         'filled_quantity': 1, 'filled_avg_price': 1, 'created_at': 'x'},
    ]
    assert compute_attribution(fills) == {}
