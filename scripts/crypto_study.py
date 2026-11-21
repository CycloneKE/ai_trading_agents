#!/usr/bin/env python3
"""Test the crypto rules on real history, net of costs.

The agent trades crypto (BTC, ETH, SOL on Alpaca) with one rule,
crypto_trend: hold a coin while it is more than `band` above its 200-day
average, sell it when it is more than `band` below. This compares that rule,
fully invested when in, with simply holding the coins and with published
alternatives, each at its stated settings, under the same checks as
scripts/forex_factor_study.py and scripts/rule_research.py:

- costs of 0.35% a side (Alpaca's 0.25% taker fee plus 0.10% slippage, the
  config's own figures);
- returns over cash (FRED 3-month US rate), 365 trading days a year;
- both halves of the period, and the same signal acted on one to five days
  late (timing luck).

Each coin gets an equal share of the money from the day it has enough
history; a coin out of the market holds cash.

Rules:
- 200-day average, 5% band (the agent's crypto_trend as configured);
- 200-day average, no band; and with a 2% band (the code's default);
- 12-month time-series momentum, long or cash (Moskowitz, Ooi and Pedersen
  2012, the sign of the excess return);
- 4-week time-series momentum, long or cash (Liu and Tsyvinski, "Risks and
  Returns of Cryptocurrency", Review of Financial Studies 2021, found
  crypto's own momentum strongest at one to four weeks);
- the 200-day rule with each coin sized to 50% yearly volatility (at most
  fully invested), the way trend funds size, so a coin moving twice as much
  gets half the money;
- buy and hold.

Usage:
    python scripts/crypto_study.py                 # download, then test
    python scripts/crypto_study.py --skip-download
"""
import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD']
COST = 0.0035          # a side: 0.25% Alpaca taker fee + 0.10% slippage
DAYS = 365             # crypto trades every day
VOL_TARGET = 0.50


def load(price_dir: str, rate_dir: str):
    px = {}
    for c in COINS:
        path = os.path.join(price_dir, f'{c}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['date'])
            px[c] = df.set_index('date')['close'].astype(float)
    px = pd.DataFrame(px).sort_index()
    rf = pd.Series(0.0, index=px.index)
    path = os.path.join(rate_dir, 'US.csv')
    if os.path.exists(path):
        r = pd.read_csv(path)
        r.columns = ['date', 'pct']
        r['date'] = pd.to_datetime(r['date'])
        r['pct'] = pd.to_numeric(r['pct'], errors='coerce')
        monthly = r.dropna().set_index('date')['pct'].shift(1)
        rf = monthly.reindex(px.index, method='ffill').fillna(0) / 100 / DAYS
    return px, rf


def band_rule(px: pd.DataFrame, days: int, band: float) -> pd.DataFrame:
    """1 while held, 0 in cash: buy above sma*(1+band), sell below sma*(1-band),
    hold the last state in between (the agent's crypto_trend)."""
    sma = px.rolling(days).mean()
    out = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    out[px > sma * (1 + band)] = 1.0
    out[px < sma * (1 - band)] = 0.0
    out[sma.isna()] = 0.0
    return out.ffill().fillna(0.0)


def weights(signal: pd.DataFrame, px: pd.DataFrame, sizing: str = 'equal') -> pd.DataFrame:
    live = px.notna() & px.rolling(200).mean().notna()
    n = live.sum(axis=1).replace(0, np.nan)
    sig = signal.where(live, 0.0).fillna(0.0)
    if sizing == 'vol':
        vol = px.pct_change().rolling(60).std() * np.sqrt(DAYS)
        sig = (sig * (VOL_TARGET / vol)).clip(upper=1.0).fillna(0.0)
    return sig.div(n, axis=0).fillna(0.0)


def trim_replay(px: pd.DataFrame, rf: pd.Series, held: pd.DataFrame, capped: bool, trim_buffer=None,
                lag: int = 1, start=None) -> Dict:
    """The live sizing, day by day: a coin is bought on the trend rule's buy
    with an equal share of the account (or, `capped`, that share scaled by
    VOL_TARGET over its 60-day volatility), left to grow, and sold on the
    rule's sell; with `trim_buffer`, each Monday a coin worth more than its
    cap by more than the buffer is sold back down to it (crypto_risk.py).
    Costs COST a side; idle cash earns the risk-free rate."""
    rets = px.pct_change().fillna(0.0)
    vol = (px.pct_change().rolling(60).std() * np.sqrt(DAYS)).shift(lag)
    live = (px.notna() & px.rolling(200).mean().notna()).shift(lag, fill_value=False)
    want = held.shift(lag).fillna(0.0)
    cash, val, equity = 1.0, {c: 0.0 for c in px.columns}, []
    for i, d in enumerate(px.index):
        for c in px.columns:
            val[c] *= 1 + rets[c].iloc[i]
        cash *= 1 + rf.iloc[i]
        total = cash + sum(val.values())
        n = max(int(live.iloc[i].sum()), 1)
        for c in px.columns:
            if not live[c].iloc[i]:
                continue
            v = vol[c].iloc[i]
            cap = total / n * (min(VOL_TARGET / v, 1.0) if capped and v == v and v > 0 else 1.0)
            if want[c].iloc[i] and val[c] == 0:
                amount = min(cap, cash)
                val[c] += amount * (1 - COST)
                cash -= amount
            elif not want[c].iloc[i] and val[c] > 0:
                cash += val[c] * (1 - COST)
                val[c] = 0.0
            elif trim_buffer is not None and want[c].iloc[i] and d.weekday() == 0 \
                    and val[c] > cap * (1 + trim_buffer):
                cut = val[c] - cap
                val[c] -= cut
                cash += cut * (1 - COST)
        equity.append(cash + sum(val.values()))
    eq = pd.Series(equity, index=px.index)
    eq = eq[eq.index >= (start if start is not None else px.index[0])]
    r = (eq.pct_change() - rf).dropna()
    years = len(r) / DAYS
    half = len(r) // 2
    sharpe = lambda x: float(x.mean() / x.std() * np.sqrt(DAYS)) if x.std() > 0 else 0.0
    return {'cagr': float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1), 'sharpe': sharpe(r),
            'max_dd': float((eq / eq.cummax() - 1).min()), 'halves': (sharpe(r[:half]), sharpe(r[half:]))}


def evaluate(w: pd.DataFrame, rets: pd.DataFrame, rf: pd.Series, start, lag: int = 1) -> Dict:
    held = w.shift(lag).fillna(0.0)
    excess = rets.sub(rf, axis=0).fillna(0.0)
    net = (held * excess).sum(axis=1) - held.diff().abs().sum(axis=1).fillna(0.0) * COST
    net = net[net.index >= start]
    eq = (1 + net + rf[rf.index >= start]).cumprod()
    years = len(net) / DAYS

    def sharpe(x):
        return float(x.mean() / x.std() * np.sqrt(DAYS)) if x.std() > 0 else 0.0

    half = len(net) // 2
    trades = float((held.diff().abs() > 1e-9).sum().sum() / years)
    return {'cagr': float(eq.iloc[-1] ** (1 / years) - 1), 'sharpe': sharpe(net),
            'max_dd': float((eq / eq.cummax() - 1).min()),
            'halves': (sharpe(net[:half]), sharpe(net[half:])),
            'exposure': float(held[held.index >= start].sum(axis=1).mean()), 'trades': trades}


def study(price_dir: str, rate_dir: str):
    px, rf = load(price_dir, rate_dir)
    rets = px.pct_change()
    start = px.index[min(len(px) - 1, 210)]
    tsmom12 = (np.sign((1 + rets.sub(rf, axis=0)).rolling(DAYS).apply(np.prod, raw=True) - 1)
               .clip(lower=0).fillna(0.0))
    tsmom4w = (np.sign((1 + rets.sub(rf, axis=0)).rolling(28).apply(np.prod, raw=True) - 1)
               .clip(lower=0).fillna(0.0))
    hold = pd.DataFrame(1.0, index=px.index, columns=px.columns)
    tests = {
        '200-day average, 5% band (as configured)': weights(band_rule(px, 200, 0.05), px),
        '200-day average, 2% band': weights(band_rule(px, 200, 0.02), px),
        '200-day average, no band': weights(band_rule(px, 200, 0.0), px),
        '12-month momentum, long or cash': weights(tsmom12, px),
        '4-week momentum, long or cash': weights(tsmom4w, px),
        '200-day 5% band, sized to 50% vol': weights(band_rule(px, 200, 0.05), px, 'vol'),
        'Buy and hold': weights(hold, px),
    }
    print(f"Period {start.date()} to {px.index[-1].date()}; {', '.join(px.columns)}; "
          f"costs {COST:.2%} a side; returns over cash")
    print(f"{'Rule':42s} {'CAGR':>7s} {'Sharpe':>7s} {'Max DD':>8s}  {'Halves':>13s}  "
          f"{'1-5 day delays':>20s}  {'Invested':>8s} {'Trades/yr':>9s}")
    out = {}
    for name, w in tests.items():
        r = evaluate(w, rets, rf, start)
        lags = [evaluate(w, rets, rf, start, lag=k)['sharpe'] for k in range(1, 6)]
        r['lag_mean'], r['lag_min'] = float(np.mean(lags)), float(min(lags))
        out[name] = r
        print(f"{name:42s} {r['cagr']:7.1%} {r['sharpe']:7.2f} {r['max_dd']:8.1%}  "
              f"{r['halves'][0]:5.2f} / {r['halves'][1]:5.2f}  mean {r['lag_mean']:5.2f}, "
              f"worst {r['lag_min']:5.2f}  {r['exposure']:8.0%} {r['trades']:9.1f}")
    held = band_rule(px, 200, 0.05)
    print("\nThe agent's sizing (200-day rule, 5% band), each coin an equal share, left to grow:")
    for name, kw in (('bought at an equal share, never trimmed', {'capped': False}),
                     ('bought at its volatility cap, never trimmed', {'capped': True}),
                     ('volatility cap, trimmed weekly above cap + 20%', {'capped': True, 'trim_buffer': 0.2})):
        r = trim_replay(px, rf, held, start=start, **kw)
        out['sizing: ' + name] = r
        print(f"  {name:46s} {r['cagr']:7.1%} {r['sharpe']:7.2f} {r['max_dd']:8.1%}  "
              f"{r['halves'][0]:5.2f} / {r['halves'][1]:5.2f}")
    print("\nSharpe is measured over cash. A rule earns its place by a better Sharpe or a much smaller "
          "worst fall than holding, in both halves, and at every delay.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--price-dir', default='data/history/crypto')
    ap.add_argument('--rate-dir', default='data/history/rates')
    ap.add_argument('--years', type=int, default=12)
    ap.add_argument('--skip-download', action='store_true')
    args = ap.parse_args(argv)
    if not args.skip_download:
        from scripts.fetch_history import fetch
        fetch(COINS, args.price_dir, years=args.years)
        from scripts.forex_factor_study import download
        download(os.path.join(args.price_dir, '_unused_fx'), args.rate_dir, 1)
    study(args.price_dir, args.rate_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
