"""scripts/crypto_study.py: the band rule holds its state inside the band,
and no rule profits from seeing the day it trades on."""
import numpy as np
import pandas as pd

from scripts import crypto_study as cs


def test_the_band_rule_buys_above_sells_below_and_holds_in_between():
    idx = pd.date_range('2020-01-01', periods=260)
    price = [100.0] * 200 + [110.0] * 20 + [100.0] * 20 + [80.0] * 20
    px = pd.DataFrame({'BTC-USD': price}, index=idx)
    held = cs.band_rule(px, 200, 0.05)['BTC-USD']
    assert held.iloc[199] == 0.0                      # at the average: not bought
    assert held.iloc[205] == 1.0                      # 10% above: held
    assert held.iloc[225] == 1.0                      # back near the average, inside the band: still held
    assert held.iloc[-1] == 0.0                       # well below: sold


def test_a_rule_that_peeks_at_todays_return_gains_nothing_once_lagged():
    rng = np.random.default_rng(5)
    idx = pd.date_range('2020-01-01', periods=800)
    px = pd.DataFrame({'BTC-USD': 100 * np.cumprod(1 + rng.normal(0, 0.03, 800))}, index=idx)
    rets = px.pct_change()
    cheat = (rets > 0).astype(float)
    r = cs.evaluate(cheat, rets, pd.Series(0.0, index=idx), idx[250])
    assert abs(r['sharpe']) < 1.0
