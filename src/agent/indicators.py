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


def atr(high, low, close, length: int = 14) -> pd.Series:
    """Wilder's Average True Range.

    True range is the greatest of the current bar's range, the gap up from
    the previous close, and the gap down from it. Smoothed the same way as
    `rsi` above: a simple mean seed over the first `length` values, then
    Wilder recursive smoothing.

    Returned in price units. Divide by price for a percentage.
    """
    high = pd.Series(high, dtype='float64').reset_index(drop=True)
    low = pd.Series(low, dtype='float64').reset_index(drop=True)
    close = pd.Series(close, dtype='float64').reset_index(drop=True)
    n = close.size
    if length <= 0 or n <= length or high.size != n or low.size != n:
        return pd.Series([float('nan')] * n)

    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1).to_numpy()

    out = [float('nan')] * n
    # tr[0] has no previous close; seed from indices 1..length.
    avg = float(pd.Series(tr[1:length + 1]).mean())
    out[length] = avg
    for i in range(length + 1, n):
        avg = (avg * (length - 1) + tr[i]) / length
        out[i] = avg
    return pd.Series(out, name=f"ATR_{length}")


def atr_from_closes(close, length: int = 14) -> pd.Series:
    """ATR when only closes are available.

    The live loop often holds a close-only price buffer, so true range
    degrades to |close - prev_close|. This understates the real range,
    typically by a third or so for daily equity bars, which makes stops
    derived from it tighter rather than looser. Prefer `atr` whenever
    high/low are on hand.
    """
    close = pd.Series(close, dtype='float64').reset_index(drop=True)
    return atr(close, close, close, length)
