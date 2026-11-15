"""Backtests over markets with different calendars (backtest_engine.py,
scripts/run_backtest.py).

Found in the operator's first review pack: with stocks and crypto together,
a stock holding was valued at zero on every weekend (crypto trades, stocks
do not), so the equity curve fell and recovered each week: single days of
-58% and +142%, a 342% "volatility", and a final value without the stocks
because the run ended on a Sunday. Interest and the length of the test were
counted at 252 bars a year, so six years read as 8.7.
"""
import importlib.util
import os
from datetime import date, timedelta

import pandas as pd
import pytest

from src.agent.backtest_engine import BacktestEngine, Order, OrderSide, OrderType

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_backtest_module():
    spec = importlib.util.spec_from_file_location('run_backtest', os.path.join(ROOT, 'scripts', 'run_backtest.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frame(days, price):
    return pd.DataFrame({'close': [price] * len(days)}, index=pd.to_datetime(days))


def _calendar(start, n, weekdays_only):
    out, d = [], start
    while len(out) < n:
        if not weekdays_only or d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_a_stock_keeps_its_value_on_a_day_only_crypto_trades():
    days = _calendar(date(2026, 1, 5), 14, weekdays_only=False)                # Monday to Sunday, twice
    data = {'STOCK': _frame([d for d in days if d.weekday() < 5], 100.0), 'COIN': _frame(days, 50.0)}
    engine = BacktestEngine({'initial_capital': 10000, 'commission_rate': 0, 'slippage_rate': 0,
                             'cash_yield': 0.0})
    engine.add_market_data(data)

    def buy_once(ts, prices, portfolio):
        if ts == pd.Timestamp(days[0]):
            return [Order(symbol='STOCK', side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50, timestamp=ts)]
        return []
    engine.run_backtest(buy_once)
    values = [round(p['portfolio_value']) for p in engine.portfolio_history]
    assert min(values[1:]) == max(values[1:])                                  # flat over the weekend too
    assert values[-1] == pytest.approx(10000, rel=0.01)                        # a Sunday close keeps the stock


def test_a_year_is_measured_on_the_calendar():
    rb = _run_backtest_module()
    stocks = _calendar(date(2025, 1, 6), 504, weekdays_only=True)              # two years of weekdays
    assert rb.bars_per_year({"S": _frame(stocks, 1.0)}) == pytest.approx(261, rel=0.02)     # weekdays, no holidays here
    every_day = _calendar(date(2025, 1, 6), 730, weekdays_only=False)
    mixed = {'S': _frame(stocks, 1.0), 'C': _frame(every_day, 1.0)}
    assert rb.bars_per_year(mixed) == pytest.approx(365, rel=0.02)
    eq = [100.0] * 730
    assert rb.analyse(eq, [], 100.0, rb.bars_per_year(mixed))['years'] == pytest.approx(2.0, abs=0.02)


def test_idle_cash_earns_a_years_interest_in_a_year_whatever_the_bars():
    every_day = _calendar(date(2025, 1, 1), 366, weekdays_only=False)          # one calendar year, daily
    engine = BacktestEngine({'initial_capital': 100000, 'cash_yield': 0.10, 'periods_per_year': 252})
    engine.add_market_data({'C': _frame(every_day, 1.0)})
    engine.run_backtest(lambda ts, prices, portfolio: [])
    assert engine.interest_earned == pytest.approx(100000 * ((1 + 0.10 / 365.25) ** 365 - 1), rel=0.01)


def test_the_momentum_sleeve_is_compared_from_its_first_trade(tmp_path):
    """It needs about a year of history before it can rank; comparing it with
    a buy-and-hold that was invested all along charged it a year in cash."""
    rb = _run_backtest_module()
    days = _calendar(date(2020, 1, 6), 700, weekdays_only=True)
    import numpy as np
    rng = np.random.default_rng(3)
    hist = tmp_path / 'h'
    hist.mkdir()
    for i, sym in enumerate(['XLK', 'XLI', 'XLU', 'TLT']):
        closes = 100 * np.cumprod(1 + rng.normal(0.0005 * (i + 1), 0.01, len(days)))
        pd.DataFrame({'date': [d.isoformat() for d in days], 'close': closes}).to_csv(hist / f'{sym}.csv', index=False)
    out = tmp_path / 'out'
    import sys
    argv = sys.argv
    sys.argv = ['run_backtest', '--csv-dir', str(hist), '--strategy', 'sleeve', '--leave-one-out', '--out', str(out)]
    try:
        rb.main()
    finally:
        sys.argv = argv
    report = (out / 'report.md').read_text()
    assert '- Measured from the first trade, 2021-' in report
    assert '## Does it rest on one fund?' in report and 'Leave-one-out robustness' in report
    import json
    meta = json.loads((out / 'results.json').read_text())['meta']
    assert meta['evaluated_from'] > meta['start']
