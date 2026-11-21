#!/usr/bin/env python3
"""Test the textbook forex approaches on real history, net of costs.

Answers one question before any forex code is changed: which of the
approaches professionals use on currencies (trend, carry, both) would have
made money on the major pairs, and does it matter that the agent's forex
book is long only?

Every setting is the standard published one, never tuned on this data, so
the result is not fitted to the sample:

- Trend: time-series momentum, the sign of the 12-month return
  (Moskowitz, Ooi and Pedersen, "Time Series Momentum", 2012), and the
  50/200-day moving-average rule.
- Carry: hold currencies whose 3-month interest rate is higher, against
  those whose rate is lower (Koijen, Moskowitz, Pedersen and Vrugt,
  "Carry", 2018). Interest is counted in the return, as a broker's
  overnight rollover would pay or charge it.
- Each pair is sized to 10% annual volatility (at most 3x leverage), and
  pays about one pip of spread per unit traded.

A signal from day t's close is traded on day t+1, and each month's interest
rate is used only from the following month (when it had been published), so
nothing is known before it happened.

Usage:
    python scripts/forex_factor_study.py                 # download, then test
    python scripts/forex_factor_study.py --skip-download  # reuse the files

Prices come from Yahoo Finance (scripts/fetch_history.py); interest rates
are the OECD 3-month interbank rates published on FRED (fred.stlouisfed.org).
"""
import argparse
import os
import sys
import urllib.request
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pair -> the FRED country code of its non-USD currency. A pair is turned
# round where needed so that "long" always means long the foreign currency
# against the dollar (USD_JPY up means the yen fell, so it is inverted).
PAIRS: Dict[str, str] = {'EUR_USD': 'EZ', 'GBP_USD': 'GB', 'AUD_USD': 'AU',
                         'NZD_USD': 'NZ', 'USD_JPY': 'JP'}
FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=IR3TIB01{code}M156N&cosd=2000-01-01'
COST = 0.0001        # one pip a side on a major pair, about 0.01% of what is traded
VOL_TARGET = 0.10    # each pair sized to 10% annual volatility
MAX_LEVERAGE = 3.0
TRADING_DAYS = 252


def download(price_dir: str, rate_dir: str, years: int) -> None:
    from scripts.fetch_history import fetch
    fetch(list(PAIRS), price_dir, years=years)
    os.makedirs(rate_dir, exist_ok=True)
    for code in ['US'] + sorted(set(PAIRS.values())):
        path = os.path.join(rate_dir, f'{code}.csv')
        try:
            with urllib.request.urlopen(FRED_URL.format(code=code), timeout=30) as r, \
                    open(path, 'wb') as f:
                f.write(r.read())
            print(f"  rate {code}: saved")
        except Exception as e:
            print(f"  rate {code}: failed ({e})")


def load_prices(price_dir: str) -> pd.DataFrame:
    out = {}
    for pair in PAIRS:
        df = pd.read_csv(os.path.join(price_dir, f'{pair}.csv'), parse_dates=['date'])
        s = df.set_index('date')['close'].astype(float)
        out[pair] = 1 / s if pair.startswith('USD_') else s
    return pd.DataFrame(out).dropna()


def load_rate(rate_dir: str, code: str, index: pd.DatetimeIndex) -> pd.Series:
    df = pd.read_csv(os.path.join(rate_dir, f'{code}.csv'))
    df.columns = ['date', 'rate']
    df['date'] = pd.to_datetime(df['date'])
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    monthly = df.set_index('date')['rate'].shift(1)          # usable a month later
    return monthly.reindex(index, method='ffill').ffill() / 100


def evaluate(signal: pd.DataFrame, returns: pd.DataFrame, vol: pd.DataFrame,
             start: pd.Timestamp, lag: int = 1) -> Dict[str, float]:
    held = signal.shift(lag).fillna(0)                        # traded `lag` days later
    weight = (held * (VOL_TARGET / vol.shift(lag))).clip(-MAX_LEVERAGE, MAX_LEVERAGE).fillna(0)
    net = (weight * returns).mean(axis=1) - weight.diff().abs().mean(axis=1).fillna(0) * COST
    net = net[net.index >= start]
    equity = (1 + net).cumprod()
    years = len(net) / TRADING_DAYS

    def sharpe(x):
        return float(x.mean() / x.std() * np.sqrt(TRADING_DAYS)) if x.std() > 0 else 0.0

    half = len(net) // 2
    return {'cagr': float(equity.iloc[-1] ** (1 / years) - 1),
            'sharpe': sharpe(net),
            'max_dd': float((equity / equity.cummax() - 1).min()),
            'first_half': sharpe(net[:half]),
            'second_half': sharpe(net[half:])}


def study(price_dir: str, rate_dir: str) -> Dict[str, Dict[str, float]]:
    px = load_prices(price_dir)
    spot = px.pct_change().fillna(0)
    vol = spot.rolling(60).std() * np.sqrt(TRADING_DAYS)
    us = load_rate(rate_dir, 'US', px.index)
    rate_gap = pd.DataFrame({p: load_rate(rate_dir, c, px.index) - us for p, c in PAIRS.items()})
    total = spot + rate_gap / TRADING_DAYS                    # price move plus interest
    start = px.index[min(len(px) - 1, TRADING_DAYS + 10)]     # after the 12-month warm-up

    trend = np.sign(px.pct_change(TRADING_DAYS))
    ma = np.sign(px.rolling(50).mean() - px.rolling(200).mean())
    rank = rate_gap.rank(axis=1)
    carry = np.sign(rank.sub(rank.median(axis=1), axis=0))
    blend = (carry + trend) / 2
    # The papers' trend: the sign of the 12-month EXCESS return, i.e. what a
    # currency forward earned (price move plus the interest-rate gap).
    trend_x = np.sign((1 + total).rolling(TRADING_DAYS).apply(np.prod, raw=True) - 1)
    tests = {
        'Trend 12m, long and short': trend,
        'Trend 12m on excess return, L/S': trend_x,
        'Trend 12m, long only': trend.clip(lower=0),
        'MA 50/200, long and short': ma,
        'MA 50/200, long only': ma.clip(lower=0),
        'Carry, long and short': carry,
        'Carry + trend, long and short': blend,
        'Carry + excess-return trend, L/S': (carry + trend_x) / 2,
        'Carry + trend, long only': blend.clip(lower=0),
        'Buy and hold the currencies': pd.DataFrame(1.0, index=px.index, columns=px.columns),
    }
    print(f"Period {start.date()} to {px.index[-1].date()}; pairs {', '.join(PAIRS)}")
    print(f"{'Approach':32s} {'CAGR':>7s} {'Sharpe':>7s} {'Max DD':>8s}   Sharpe by half   "
          f"Sharpe, 1-5 day delays")
    results = {}
    for name, sig in tests.items():
        r = results[name] = evaluate(sig, total, vol, start)
        lags = [evaluate(sig, total, vol, start, lag=k)['sharpe'] for k in range(1, 6)]
        r['sharpe_lag_mean'], r['sharpe_lag_min'] = sum(lags) / len(lags), min(lags)
        print(f"{name:32s} {r['cagr']:7.2%} {r['sharpe']:7.2f} {r['max_dd']:8.1%}"
              f"   {r['first_half']:5.2f} / {r['second_half']:5.2f}"
              f"     mean {r['sharpe_lag_mean']:5.2f}, worst {r['sharpe_lag_min']:5.2f}")
    print("\nSharpe is return per unit of risk (above 0.5 is respectable for one "
          "asset class). An edge should be positive in both halves, not in one lucky stretch, "
          "and should survive acting one to five days later (timing luck): a slow signal "
          "whose result swings with a day's delay is mostly noise.")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--price-dir', default='data/history/forex')
    ap.add_argument('--rate-dir', default='data/history/rates')
    ap.add_argument('--years', type=int, default=10)
    ap.add_argument('--skip-download', action='store_true')
    args = ap.parse_args(argv)
    if not args.skip_download:
        download(args.price_dir, args.rate_dir, args.years)
    study(args.price_dir, args.rate_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
