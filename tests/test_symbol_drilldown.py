"""Tests for the per-symbol drill-down payload assembly."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.symbol_drilldown import execution_quality, alpha_vs_hold, build_payload


def order(symbol='AAPL', side='buy', status='filled', limit=100.0, fill=100.0,
          qty=1.0, strategy='momentum', ts='a', coid=None):
    return {'symbol': symbol, 'side': side, 'status': status,
            'limit_price': limit, 'filled_avg_price': fill,
            'filled_quantity': qty, 'strategy': strategy, 'created_at': ts,
            'client_order_id': coid or f'{symbol}-{ts}'}


def test_slippage_sign_buy_and_sell():
    # buy filled above limit -> adverse (positive bps); sell below -> adverse too
    eq = execution_quality([
        order(side='buy', limit=100.0, fill=100.5),    # +50 bps adverse
        order(side='sell', limit=100.0, fill=99.5),    # +50 bps adverse
    ])
    assert eq['fills'] == 2
    assert eq['avg_slippage_bps'] == 50.0


def test_favorable_fill_is_negative_bps():
    eq = execution_quality([order(side='buy', limit=100.0, fill=99.9)])  # price improvement
    assert eq['avg_slippage_bps'] == -10.0


def test_non_filled_orders_ignored_in_execution_quality():
    eq = execution_quality([
        order(status='canceled', fill=None),
        order(status='filled', limit=100, fill=101),
    ])
    assert eq['fills'] == 1


def test_alpha_vs_hold_computes_hold_return():
    attribution = {'momentum': {'realized_pnl': 30.0, 'unrealized_pnl': 5.0}}
    out = alpha_vs_hold('AAPL', attribution, first_entry_price=100.0, current_price=110.0)
    assert out['agent_pnl'] == 35.0
    assert out['buy_hold_return_pct'] == 10.0


def test_alpha_vs_hold_handles_missing_prices():
    out = alpha_vs_hold('AAPL', {}, first_entry_price=None, current_price=None)
    assert out['agent_pnl'] == 0
    assert out['buy_hold_return_pct'] is None


def test_build_payload_filters_to_symbol():
    decisions = [{'symbol': 'AAPL', 'action': 'hold', 'skip_reason': 'llm_veto'}]
    orders = [
        order('AAPL', side='buy', fill=100.0, ts='1'),
        order('AAPL', side='sell', fill=110.0, ts='2'),
        order('MSFT', side='buy', fill=200.0, ts='1'),   # different symbol: excluded from books
    ]
    payload = build_payload('AAPL', decisions, orders, price_lookup=lambda s: 112.0)
    assert payload['symbol'] == 'AAPL'
    assert payload['current_price'] == 112.0
    # strategy book P&L only reflects AAPL fills (100 -> 110 = +10 realized)
    assert payload['strategy_books']['momentum']['realized_pnl'] == 10.0
    assert payload['alpha_vs_hold']['first_entry_price'] == 100.0
    assert payload['decision_count'] == 1


def test_build_payload_empty_is_safe():
    payload = build_payload('AAPL', [], [], price_lookup=lambda s: None)
    assert payload['strategy_books'] == {}
    assert payload['execution_quality']['fills'] == 0
    assert payload['alpha_vs_hold']['agent_pnl'] == 0
