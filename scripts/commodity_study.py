#!/usr/bin/env python3
"""Do the commodity patterns work? Replay them over years of price history.

Takes the three watch-only patterns (trend_200d, momentum_12m, seasonal) and
replays each over the whole price history Yahoo Finance holds for metals, oil
and farm produce, then scores every signal against what the price did a month
later and against how often it simply went up anyway. Nothing is traded; this
only reads prices (it needs the internet).

A pattern is marked CANDIDATE only when it beats chance by two standard errors
in all the history AND in both the earlier and the later half of it. That is a
reason to keep watching it live, not a reason to trade it: the live record the
agent keeps (the dashboard's Scorecard page) is the real test.

Usage:
    python scripts/commodity_study.py
    python scripts/commodity_study.py --period 10y
    python scripts/commodity_study.py --only metals
    python scripts/commodity_study.py --json
    python scripts/commodity_study.py --out reports/commodity_study.md
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import commodity_watch as cw                  # noqa: E402
from src.utils.config_validator import load_config           # noqa: E402

VERDICT = {'candidate': 'CANDIDATE', 'no_clear_edge': 'no clear edge', 'wrong_way': 'WRONG WAY',
           'too_few': 'too few signals'}


def render(result, cost_pct):
    lines = ['# Commodity pattern study', '',
             f"{result['instruments']} commodities, {result['signals']} signals, {result['scored']} scored "
             f"at {cw.HORIZON} trading days. Nothing here was traded.", '',
             'Each row: how often the signal moved the right way, against how often a random day would have. '
             f'"Net" is the average move after an assumed cost of {cost_pct}% per signal.', '']
    for market in ('all_commodities', 'metals', 'energy', 'agriculture'):
        rows = [r for r in result['table'] if r['market'] == market]
        if not rows:
            continue
        lines += [f"## {market.replace('_', ' ')}", '']
        for r in sorted(rows, key=lambda r: r['source']):
            early = 'n/a' if r['early_lift'] is None else f"{r['early_lift']:+.1%}"
            late = 'n/a' if r['late_lift'] is None else f"{r['late_lift']:+.1%}"
            lines.append(f"{r['source']:<13} {r['hit_rate']:.0%} right of {r['n']} (chance {r['chance']:.0%}, "
                         f"z {r['z']:+.1f}); earlier half {early}, later half {late}; "
                         f"net move {r['net_move_pct']:+.2f}%  ->  {VERDICT[r['verdict']]}")
        lines.append('')
    lines += ['Read with care: the commodities move together, so the signals are not fully independent and the '
              'true count is lower than it looks. Yahoo\'s futures series jump when a contract rolls over. '
              'Testing many patterns will throw up some that look good by luck. A CANDIDATE earns a place on the '
              'live watch, nothing more.']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='config/config.json')
    p.add_argument('--period', default=None, help='how much history to replay (default: the config, "max")')
    p.add_argument('--only', choices=['metals', 'energy', 'agriculture'])
    p.add_argument('--kind', choices=['futures', 'fund'], help='only futures series, or only funds')
    p.add_argument('--json', action='store_true')
    p.add_argument('--out', help='also write the report to this file')
    args = p.parse_args()
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)

    config = load_config(args.config) if os.path.exists(args.config) else {}
    rule = cw.settings(config)
    items = [i for i in cw.instruments(config)
             if (not args.only or i['group'] == args.only) and (not args.kind or i['kind'] == args.kind)]
    closes_for = cw.default_closes_provider(args.period or str(rule['history_period']))
    closes, meta = {}, {i['symbol']: i for i in items}
    for item in items:
        c = closes_for(item['symbol'])
        if c:
            closes[item['symbol']] = c
        else:
            print(f"(no prices for {item['symbol']}, skipped)", file=sys.stderr)
    if not closes:
        print('No price history could be fetched. This needs the internet.', file=sys.stderr)
        return 1
    result = cw.run_study(closes, meta, cw.HORIZON, int(rule['min_signals']), float(rule['assumed_cost_pct']))
    text = json.dumps(result, indent=2, default=str) if args.json else render(result, rule['assumed_cost_pct'])
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
