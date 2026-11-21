"""scripts/forex_factor_study.py must not see the future."""
import numpy as np
import pandas as pd

from scripts.forex_factor_study import evaluate


def _random_walk(seed=7, days=2000, pairs=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2015-01-01', periods=days)
    rets = pd.DataFrame(rng.normal(0, 0.005, (days, pairs)), index=idx,
                        columns=[f'P{i}' for i in range(pairs)])
    vol = rets.rolling(60).std() * np.sqrt(252)
    return rets, vol


def test_a_same_day_signal_is_traded_the_next_day():
    # A signal that "knows" today's return would be a perfect strategy if it
    # were traded today. Traded tomorrow, as the study does, it is noise.
    rets, vol = _random_walk()
    cheat = np.sign(rets)
    r = evaluate(cheat, rets, vol, rets.index[100])
    assert abs(r['sharpe']) < 1.0


def test_a_known_drift_is_found():
    # A steady upward drift should be picked up by being long.
    rets, vol = _random_walk()
    rets = rets + 0.001
    long = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
    r = evaluate(long, rets, vol, rets.index[100])
    assert r['sharpe'] > 1.0 and r['cagr'] > 0
