#!/usr/bin/env python3
"""Test well-documented trading rules on funds the US account can trade.

The question this answers: beyond forex, which published rules would have
earned their risk after costs over the last twenty years? Every rule uses the
settings its authors published, never tuned here, and every result faces the
same three checks as scripts/forex_factor_study.py:

1. after costs (0.05% a side for these funds: spread and slippage; Alpaca
   charges no commission);
2. positive in both halves of the period, not one lucky stretch;
3. timing luck: the same rule acted on one to five days late should not
   collapse.

Returns are measured over cash (the 3-month US rate from FRED), so a rule
that sits in cash is not credited with the interest as if it were skill.

Rules (sources in docs/FOREX_AND_PROFESSIONAL_TRAINING_PLAN.md):
- Cross-asset trend, long and short: each fund's 12-month return sign,
  sized to 10% volatility (Moskowitz, Ooi and Pedersen 2012).
- Cross-asset trend, long or cash: the same, never short.
- Trend blend 1/3/12 months, long and short (Hurst, Ooi and Pedersen,
  "A Century of Evidence on Trend-Following Investing", 2017).
- Equal-risk buy and hold of the same funds (the benchmark a trend rule
  must beat).
- S&P 500 above its 10-month average, else cash (Faber 2007).
- S&P 500 turn of the month: hold from the last trading day of a month to
  the third of the next (Lakonishok and Smidt 1988; McConnell and Xu 2008).
- S&P 500 two-day RSI below 10 above the 200-day average, out above the
  5-day average (Connors and Alvarez, Short Term Trading Strategies That Work, 2008).
- S&P 500 buy and hold.

Usage:
    python scripts/rule_research.py                  # download, then test
    python scripts/rule_research.py --skip-download
"""
import argparse
import os
import sys
from typing import Callable, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UNIVERSE = ['SPY', 'EFA', 'EEM', 'IWM', 'VNQ',          # stock markets and property
            'TLT', 'IEF', 'LQD',                        # government and corporate bonds
            'GLD', 'SLV', 'USO', 'DBC', 'DBA',          # gold, silver, oil, broad and farm commodities
            'UUP', 'FXE', 'FXY']                        # the dollar, the euro, the yen
COST = 0.0005
VOL_TARGET = 0.10
MAX_WEIGHT = 2.0
TRADING_DAYS = 252


def load(price_dir: str, rate_dir: str):
    px = {}
    for s in UNIVERSE:
        path = os.path.join(price_dir, f'{s}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['date'])
            px[s] = df.set_index('date')['close'].astype(float)
    px = pd.DataFrame(px).sort_index()
    rf = pd.Series(0.0, index=px.index)
    path = os.path.join(rate_dir, 'US.csv')
    if os.path.exists(path):
        r = pd.read_csv(path)
        r.columns = ['date', 'pct']
        r['date'] = pd.to_datetime(r['date'])
        r['pct'] = pd.to_numeric(r['pct'], errors='coerce')
        monthly = r.dropna().set_index('date')['pct'].shift(1)
        rf = monthly.reindex(px.index, method='ffill').fillna(0) / 100 / TRADING_DAYS
    return px, rf


def evaluate(weights: pd.DataFrame, rets: pd.DataFrame, rf: pd.Series, start, lag: int = 1) -> Dict:
    """Weights decided at a day's close, held from `lag` days later."""
    w = weights.shift(lag).fillna(0)
    excess = rets.sub(rf, axis=0).fillna(0)
    net = (w * excess).sum(axis=1) - w.diff().abs().sum(axis=1).fillna(0) * COST
    net = net[net.index >= start]
    eq = (1 + net + rf[rf.index >= start]).cumprod()
    years = len(net) / TRADING_DAYS

    def sharpe(x):
        return float(x.mean() / x.std() * np.sqrt(TRADING_DAYS)) if x.std() > 0 else 0.0

    half = len(net) // 2
    return {'cagr': float(eq.iloc[-1] ** (1 / years) - 1), 'sharpe': sharpe(net),
            'max_dd': float((eq / eq.cummax() - 1).min()), 'halves': (sharpe(net[:half]), sharpe(net[half:])),
            'exposure': float(w[w.index >= start].abs().sum(axis=1).mean())}


def rules(px: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    rets = px.pct_change()
    vol = rets.rolling(60).std() * np.sqrt(TRADING_DAYS)
    live = px.notna() & vol.notna()
    n = live.sum(axis=1).replace(0, np.nan)

    def sized(sig):
        w = (sig * (VOL_TARGET / vol)).clip(-MAX_WEIGHT, MAX_WEIGHT)
        return w.where(live).div(n, axis=0).fillna(0)

    tsmom = np.sign(px.pct_change(252))
    blend = (np.sign(px.pct_change(21)) + np.sign(px.pct_change(63)) + np.sign(px.pct_change(252))) / 3
    spy = px['SPY']
    only_spy = lambda s: pd.DataFrame({c: (s if c == 'SPY' else 0.0) for c in px.columns}, index=px.index)

    faber = (spy > spy.rolling(210).mean()).astype(float)
    month = px.index.to_period('M')
    day_in_month = pd.Series(1, index=px.index).groupby(month).cumsum()
    last_day = pd.Series(px.index, index=px.index).groupby(month).transform('max') == pd.Series(px.index, index=px.index)
    tom = ((day_in_month <= 3) | last_day).astype(float)
    tom = tom.shift(-1).fillna(0)        # decided the evening before the day held
    delta = spy.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    rsi2 = 100 - 100 / (1 + up / down.replace(0, np.nan))
    pos, held = [], 0.0
    sma200, sma5 = spy.rolling(200).mean(), spy.rolling(5).mean()
    for t in range(len(spy)):
        if held and spy.iloc[t] > sma5.iloc[t]:
            held = 0.0
        elif not held and spy.iloc[t] > sma200.iloc[t] and rsi2.iloc[t] < 10:
            held = 1.0
        pos.append(held)
    connors = pd.Series(pos, index=px.index)

    return {
        'Cross-asset trend, long and short': sized(tsmom),
        'Cross-asset trend, long or cash': sized(tsmom.clip(lower=0)),
        'Trend blend 1/3/12m, long and short': sized(blend),
        'Equal-risk buy and hold (16 funds)': sized(pd.DataFrame(1.0, index=px.index, columns=px.columns)),
        'S&P 500 above 10-month average': only_spy(faber),
        'S&P 500 turn of the month': only_spy(tom),
        'S&P 500 RSI(2) pullbacks': only_spy(connors),
        'S&P 500 buy and hold': only_spy(pd.Series(1.0, index=px.index)),
    }


def study(price_dir: str, rate_dir: str) -> Dict[str, Dict]:
    px, rf = load(price_dir, rate_dir)
    rets = px.pct_change()
    start = px.index[min(len(px) - 1, 270)]
    print(f"Period {start.date()} to {px.index[-1].date()}; {len(px.columns)} funds; returns over cash")
    print(f"{'Rule':38s} {'CAGR':>7s} {'Sharpe':>7s} {'Max DD':>8s}  {'Halves':>13s}  "
          f"{'1-5 day delays':>20s}  Avg exposure")
    out = {}
    for name, w in rules(px).items():
        r = evaluate(w, rets, rf, start)
        lags = [evaluate(w, rets, rf, start, lag=k)['sharpe'] for k in range(1, 6)]
        r['lag_mean'], r['lag_min'] = float(np.mean(lags)), float(min(lags))
        out[name] = r
        print(f"{name:38s} {r['cagr']:7.2%} {r['sharpe']:7.2f} {r['max_dd']:8.1%}  "
              f"{r['halves'][0]:5.2f} / {r['halves'][1]:5.2f}  mean {r['lag_mean']:5.2f}, worst {r['lag_min']:5.2f}"
              f"  {r['exposure']:.2f}x")
    print("\nCAGR includes interest on cash; Sharpe is measured over cash. A rule earns a place only if it "
          "beats buy and hold on Sharpe, holds up in both halves and survives the delays.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--price-dir', default='data/history/funds')
    ap.add_argument('--rate-dir', default='data/history/rates')
    ap.add_argument('--years', type=int, default=20)
    ap.add_argument('--skip-download', action='store_true')
    args = ap.parse_args(argv)
    if not args.skip_download:
        from scripts.fetch_history import fetch
        fetch(UNIVERSE, args.price_dir, years=args.years)
        from scripts.forex_factor_study import download
        download(os.path.join(args.price_dir, '_unused_fx'), args.rate_dir, 1)
    study(args.price_dir, args.rate_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
