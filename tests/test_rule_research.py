"""scripts/rule_research.py: the rules run on any price table and never see
the day they trade on."""
import numpy as np
import pandas as pd

from scripts import rule_research as rr


def _prices(days=700, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2010-01-01', periods=days)
    cols = ['SPY', 'TLT', 'GLD']
    rets = rng.normal(0.0002, 0.01, (days, len(cols)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=cols)


def test_every_rule_produces_weights_for_every_day():
    px = _prices()
    for name, w in rr.rules(px).items():
        assert w.shape == px.shape, name
        assert np.isfinite(w.values).all(), name


def test_a_rule_that_peeks_at_todays_return_gains_nothing_once_lagged():
    px = _prices()
    rets = px.pct_change()
    cheat = np.sign(rets).fillna(0)
    rf = pd.Series(0.0, index=px.index)
    r = rr.evaluate(cheat, rets, rf, px.index[100])
    assert abs(r['sharpe']) < 1.0
