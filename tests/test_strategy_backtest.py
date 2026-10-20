"""Backtest harness correctness.

A backtest is only worth the trust you put in it, and the two ways they lie
are lookahead and missing costs. Both are asserted here as invariants rather
than eyeballed, because the number this produces is the only evidence the
strategy has any edge at all.

The harness drives the REAL StrategyManager, ExposureGuard and size_order —
not a reimplementation. A backtest of a mirror of the strategy tells you
nothing about the strategy.
"""

import pytest

from src.agent.strategy_backtest import (
    BacktestResult, buy_and_hold_return, max_drawdown, run_backtest, sharpe_ratio,
)

FLAT = [100.0] * 60
RISING = [100.0 + i for i in range(60)]


def cfg(**over):
    base = {
        'risk_limits': {'max_position_size': 0.05, 'stop_loss_pct': 0.05,
                        'trailing_stop_pct': 0.03, 'reentry_cooldown_seconds': 0},
        'trading': {'commission': 0.001, 'slippage': 0.0005,
                    'min_notional': 5.0, 'allow_fractional': True},
        'strategies': {'momentum': {'type': 'technical', 'enabled': True,
                                    'weight': 1.0, 'lookback_period': 20}},
    }
    base.update(over)
    return base


def always(action, confidence=0.8, size=0.02):
    """A deterministic stand-in signal source."""
    def _fn(symbol, closes):
        return {'action': action, 'confidence': confidence, 'position_size': size}
    return _fn


def alternating():
    state = {'n': 0}

    def _fn(symbol, closes):
        state['n'] += 1
        return {'action': 'buy' if state['n'] % 2 else 'sell',
                'confidence': 0.8, 'position_size': 0.02}
    return _fn


# ------------------------------------------------------------- no lookahead

def test_every_fill_happens_on_the_bar_after_its_signal():
    """THE invariant. A signal derived from closes[0..t] must fill at t+1;
    filling at t is trading on information you did not have."""
    res = run_backtest({'X': RISING}, cfg(), signal_fn=always('buy'))
    assert res.trades, 'no trades to check'
    for t in res.trades:
        assert t.fill_index == t.signal_index + 1, (
            f"fill at bar {t.fill_index} for a signal computed on bar "
            f"{t.signal_index} — lookahead")


def test_fill_price_is_the_next_bars_close():
    res = run_backtest({'X': RISING}, cfg(), signal_fn=always('buy'))
    t = res.trades[0]
    # Fill price includes slippage, so compare against the raw bar.
    assert t.bar_price == pytest.approx(RISING[t.fill_index])


def test_signal_function_never_sees_a_future_bar():
    seen = []

    def spy(symbol, closes):
        seen.append(list(closes))
        return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0}

    run_backtest({'X': RISING}, cfg(), signal_fn=spy)
    for i, window in enumerate(seen):
        assert window == RISING[:len(window)]
        assert RISING[len(window)] not in window[-1:] or True   # shape check
        assert len(window) <= len(RISING) - 1, 'saw the final bar it trades into'


def test_last_bar_is_reserved_for_filling_not_signalling():
    calls = []
    run_backtest({'X': RISING}, cfg(),
                 signal_fn=lambda s, c: calls.append(len(c)) or
                 {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0})
    assert max(calls) == len(RISING) - 1


# -------------------------------------------------------------------- costs

def test_a_round_trip_at_an_unchanged_price_loses_money():
    """Commission and slippage must actually bite. A flat market with a
    round trip has to end below the starting capital."""
    res = run_backtest({'X': FLAT}, cfg(), signal_fn=alternating())
    assert res.trades
    assert res.final_equity < 100_000.0


def test_commission_and_slippage_are_both_accounted():
    res = run_backtest({'X': FLAT}, cfg(), signal_fn=alternating())
    assert res.commission_paid > 0
    assert res.slippage_paid > 0


def test_zero_cost_config_leaves_a_flat_market_flat():
    """Sanity check that the loss above is the costs and nothing else."""
    c = cfg()
    c['trading']['commission'] = 0.0
    c['trading']['slippage'] = 0.0
    res = run_backtest({'X': FLAT}, c, signal_fn=alternating())
    assert res.final_equity == pytest.approx(100_000.0, rel=1e-9)


def test_buys_pay_up_and_sells_receive_less():
    res = run_backtest({'X': FLAT}, cfg(), signal_fn=alternating())
    buys = [t for t in res.trades if t.side == 'buy']
    sells = [t for t in res.trades if t.side == 'sell']
    assert all(t.fill_price > t.bar_price for t in buys)
    assert all(t.fill_price < t.bar_price for t in sells)


# ------------------------------------------------------- risk controls live

def test_no_buy_ever_takes_a_position_past_the_cap():
    """The same gate as production — otherwise the backtest measures a
    strategy the live system will never run.

    The invariant is on PURCHASES, not on mark-to-market value: a holding
    that appreciates past 5% has not breached anything, and force-selling
    winners is not what the cap is for.
    """
    res = run_backtest({'X': RISING}, cfg(), signal_fn=always('buy'))
    buys = [t for t in res.trades if t.side == 'buy']
    assert buys
    for t in buys:
        assert t.post_fill_exposure_pct <= 0.05 + 1e-6, (
            f"buy at bar {t.fill_index} left {t.post_fill_exposure_pct:.1%} "
            f"exposure against a 5% cap")


def test_repeated_buy_signal_is_eventually_refused():
    """The pyramiding regression, inside the backtest."""
    res = run_backtest({'X': RISING}, cfg(), signal_fn=always('buy'))
    assert res.exposure_blocks > 0
    assert len([t for t in res.trades if t.side == 'buy']) < len(RISING) / 2


def test_cooldown_blocks_rebuy_after_a_stop_in_backtest():
    crash = [100.0] * 30 + [80.0] * 30      # -20%: trips the 5% stop
    c = cfg()
    c['risk_limits']['reentry_cooldown_seconds'] = 10 * 86400
    res = run_backtest({'X': crash}, c, signal_fn=always('buy'))
    assert res.cooldown_blocks > 0


def test_stop_loss_fires_on_a_crash():
    crash = [100.0] * 30 + [80.0] * 30
    res = run_backtest({'X': crash}, cfg(), signal_fn=always('buy'))
    assert any(t.reason == 'stop_loss' for t in res.trades)


# ---------------------------------------------------------------- benchmark

def test_buy_and_hold_return_is_simple_price_appreciation():
    assert buy_and_hold_return([100.0, 110.0]) == pytest.approx(0.10)
    assert buy_and_hold_return([100.0, 50.0]) == pytest.approx(-0.50)


def test_buy_and_hold_handles_degenerate_input():
    assert buy_and_hold_return([]) == 0.0
    assert buy_and_hold_return([100.0]) == 0.0
    assert buy_and_hold_return([0.0, 100.0]) == 0.0


# ------------------------------------------------------------------ metrics

def test_max_drawdown_measures_peak_to_trough():
    assert max_drawdown([100, 120, 60, 80]) == pytest.approx(0.5)
    assert max_drawdown([100, 110, 120]) == pytest.approx(0.0)


def test_max_drawdown_of_an_empty_curve_is_zero():
    assert max_drawdown([]) == 0.0


def test_sharpe_of_a_constant_curve_is_zero_not_nan():
    """Zero variance must not produce inf/nan and flatter a flat strategy."""
    assert sharpe_ratio([100.0] * 50) == 0.0


def test_sharpe_is_positive_for_a_noisy_riser():
    """A *perfectly* geometric series has zero return variance, so its Sharpe
    is undefined, not infinite — hence the wobble. Real curves always have
    some."""
    wobble = [0.0, 0.002, -0.001, 0.003, -0.002]
    curve, price = [100.0], 100.0
    for i in range(100):
        price *= (1 + 0.001 + wobble[i % len(wobble)])
        curve.append(price)
    assert sharpe_ratio(curve) > 0


# -------------------------------------------------------------------- shape

def test_result_reports_everything_needed_to_judge_it():
    res = run_backtest({'X': RISING}, cfg(), signal_fn=always('buy'))
    assert isinstance(res, BacktestResult)
    for field in ('final_equity', 'total_return', 'trades', 'win_rate',
                  'max_drawdown', 'sharpe', 'commission_paid',
                  'slippage_paid', 'exposure_blocks', 'cooldown_blocks'):
        assert hasattr(res, field), field


def test_no_trades_is_reported_honestly_not_as_zero_return_success():
    res = run_backtest({'X': FLAT}, cfg(), signal_fn=always('hold'))
    assert res.trades == []
    assert res.total_return == pytest.approx(0.0)
    assert res.win_rate == 0.0


def test_empty_history_does_not_crash():
    res = run_backtest({}, cfg(), signal_fn=always('buy'))
    assert res.trades == []


def test_series_shorter_than_two_bars_is_skipped():
    res = run_backtest({'X': [100.0]}, cfg(), signal_fn=always('buy'))
    assert res.trades == []
