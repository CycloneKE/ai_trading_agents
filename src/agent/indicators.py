"""Vendored technical indicators.

`pandas_ta` is an optional dependency: the only releases still on PyPI
(0.4.67b0, 0.4.71b0) require Python >= 3.12, so on 3.9-3.11 it cannot be
installed at all. Without it `technical_strategy` fails to import and
`StrategyManager` silently substitutes `MockStrategy`, which holds forever —
a trading agent that looks healthy and never trades.

These implementations remove that failure mode. `technical_strategy` prefers
pandas_ta when it is importable and falls back to these otherwise.
"""

import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

    Seeds the average gain/loss with a simple mean of the first `length`
    deltas, then applies Wilder recursive smoothing. Values before the seed
    are NaN, matching pandas_ta's `min_periods` behaviour. The first few
    post-seed values can differ marginally from pandas_ta's ewm variant;
    the series converges within ~3*length bars.
    """
    close = pd.Series(close, dtype='float64').reset_index(drop=True)
    n = close.size
    if length <= 0 or n <= length:
        return pd.Series([float('nan')] * n)

    delta = close.diff()
    gain = delta.clip(lower=0.0).to_numpy()
    loss = (-delta).clip(lower=0.0).to_numpy()

    out = [float('nan')] * n
    # Seed: simple mean of the first `length` deltas (indices 1..length).
    avg_gain = float(gain[1:length + 1].mean())
    avg_loss = float(loss[1:length + 1].mean())

    def _rsi_from(g, l):
        total = g + l
        return 50.0 if total == 0.0 else 100.0 * g / total

    out[length] = _rsi_from(avg_gain, avg_loss)
    for i in range(length + 1, n):
        avg_gain = (avg_gain * (length - 1) + gain[i]) / length
        avg_loss = (avg_loss * (length - 1) + loss[i]) / length
        out[i] = _rsi_from(avg_gain, avg_loss)

    res = pd.Series(out, name=f"RSI_{length}")
    return res
