#!/usr/bin/env python3
"""Replay the live forex carry and trend book over history.

scripts/forex_factor_study.py tests the idea in a spreadsheet-like way. This
runs the code the agent actually trades with: forex_carry_trend.py decides
the weights each week and the ForexMarginBroker fills them, paying spread
and rollover interest with its markup, with its leverage cap and drawdown
halt. Each Monday's decision uses only closes up to the Friday before and
interest rates published by then. If the two scripts disagree, the live
code does not do what the research tested.

Usage (after forex_factor_study.py has downloaded the data):
    python scripts/forex_book_replay.py
    python scripts/forex_book_replay.py --price-dir data/history/forex --rate-dir data/history/rates
"""
import argparse
import math
import os
import sys
import tempfile
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import forex_carry_trend as fx  # noqa: E402
from src.connectors.forex_margin_broker import ForexMarginBroker  # noqa: E402
from src.utils.config_validator import load_config  # noqa: E402


def load(price_dir: str, rate_dir: str, pairs):
    closes = {}
    for p in pairs:
        df = pd.read_csv(os.path.join(price_dir, f'{p}.csv'), parse_dates=['date'])
        closes[p] = df.set_index('date')['close'].astype(float)
    px = pd.DataFrame(closes).dropna()
    rates = {}
    for ccy, code in fx.FRED_SERIES.items():
        path = os.path.join(rate_dir, f'{code}.csv')
        if os.path.exists(path):
            r = pd.read_csv(path)
            r.columns = ['date', 'pct']
            r['date'] = pd.to_datetime(r['date'])
            r['pct'] = pd.to_numeric(r['pct'], errors='coerce')
            rates[ccy] = r.dropna().set_index('date')['pct']
    return px, rates


def rates_on(rates, day: pd.Timestamp):
    """Each currency's rate as published by `day`: last month's figure."""
    out = {}
    cutoff = day - pd.DateOffset(months=1)
    for ccy, s in rates.items():
        known = s[s.index <= cutoff]
        if len(known):
            out[ccy] = float(known.iloc[-1])
    return out


def replay(price_dir: str, rate_dir: str, config_path: str = 'config/config.json'):
    full = load_config(config_path)
    cfg = fx.config_from_dict(full.get('forex_carry_trend'))
    book_cfg = dict((full.get('brokers') or {}).get('forex_paper') or {})
    px, rates = load(price_dir, rate_dir, cfg.pairs)
    today = {'prices': {}}
    with tempfile.TemporaryDirectory() as tmp:
        book = ForexMarginBroker({**book_cfg, 'state_file': 'replay.json'},
                                 price_for=lambda p: today['prices'].get(p), state_dir=tmp)
        book.connect()
        equity, last_week = [], None
        for i, day in enumerate(px.index):
            today['prices'] = px.iloc[i].to_dict()
            r = rates_on(rates, day)
            book.accrue_rollover(r, today=day.date())
            book.check_drawdown()
            week = day.isocalendar()[:2]
            if i > cfg.bars_needed and week != last_week and not book.halted:
                last_week = week
                hist = {p: list(px[p].iloc[:i]) for p in cfg.pairs}   # up to yesterday's close
                plan = fx.target_weights(hist, r, cfg)
                targets = fx.target_units(plan['weights'], book.equity(), today['prices'])
                changes = fx.plan_changes(book.current_units(), targets, today['prices'], cfg)
                cur = book.current_units()
                for p in sorted(changes, key=lambda p: abs(cur.get(p, 0) + changes[p]) - abs(cur.get(p, 0))):
                    book.trade(p, changes[p], reason='replay')
            if i > cfg.bars_needed:
                equity.append((day, book.equity()))
        curve = pd.Series([e for _, e in equity], index=[d for d, _ in equity])
        rets = curve.pct_change().dropna()
        years = len(rets) / 252
        sharpe = rets.mean() / rets.std() * math.sqrt(252) if rets.std() else 0.0
        half = len(rets) // 2
        sh = lambda x: x.mean() / x.std() * math.sqrt(252) if x.std() else 0.0
        trips = book.own_round_trips()
        print(f"Replay of the live forex book, {curve.index[0].date()} to {curve.index[-1].date()}")
        print(f"  start {curve.iloc[0]:,.2f}  end {curve.iloc[-1]:,.2f}  "
              f"yearly {(curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1:.2%}")
        print(f"  Sharpe {sharpe:.2f} (halves {sh(rets[:half]):.2f} / {sh(rets[half:]):.2f})  "
              f"worst fall {(curve / curve.cummax() - 1).min():.1%}")
        print(f"  fills {book.fill_count} ({book.fill_count / years:.0f} a year), closed trades {len(trips)}, "
              f"interest {book.interest_total:,.2f}, spread {book.costs_total:,.2f}, halted: {book.halted or 'no'}")
        return {'sharpe': sharpe, 'curve': curve, 'book': book}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--price-dir', default='data/history/forex')
    ap.add_argument('--rate-dir', default='data/history/rates')
    ap.add_argument('--config', default='config/config.json')
    args = ap.parse_args(argv)
    replay(args.price_dir, args.rate_dir, args.config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
