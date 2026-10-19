#!/usr/bin/env python3
"""Do the patterns work? Replay them over years of price history.

Takes the three watch-only patterns (trend_200d, momentum_12m, seasonal) and
replays each over the whole price history Yahoo Finance holds for metals, oil,
farm produce and the UK, Japanese, French and South African stock markets, then
scores every signal against what the price did a month later and against how
often it simply went up anyway. Nothing is traded; this only reads prices (it
needs the internet).

A pattern is marked CANDIDATE only when it beats chance by two standard errors
in all the history AND in both the earlier and the later half of it. That is a
reason to keep watching it live, not a reason to trade it: the live record the
agent keeps (the dashboard's Scorecard page) is the real test.

Usage:
    python scripts/commodity_study.py
    python scripts/commodity_study.py --period 10y
    python scripts/commodity_study.py --horizons 20,60,120,250
    python scripts/commodity_study.py --only metals
    python scripts/commodity_study.py --only uk_equities --kind index
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
    tables = result.get('tables') or {cw.HORIZON: result['table']}
    horizons = sorted(int(h) for h in tables)
    lines = ['# Pattern study: commodities and stock markets', '',
             f"{result['instruments']} instruments, {result['signals']} signals. Judged "
             f"{', '.join(str(h) for h in horizons)} trading days after each signal (about "
             f"{', '.join(f'{round(h / 21)}' for h in horizons)} months). Nothing here was traded.", '',
             'Each row: how often the signal moved the right way, against how often a random day would have. '
             '"Net" is the average move after each market\'s assumed cost, shown beside it. The stock-market taxes were '
             'checked against public sources (UK stamp duty 0.5%, France 0.4% on single stocks, South Africa 0.25%); '
             'the allowance for commission and spread is an assumption, and Japan\'s lack of a transaction tax is not '
             'confirmed. The patterns signal about once a month, so over longer horizons a signal whose '
             'outcome window overlaps an earlier one is counted once: otherwise the same stretch of price would be '
             'counted several times and luck would look like skill. That is why the counts fall as the horizon grows.', '']
    order = ['all_commodities', 'metals', 'energy', 'agriculture', 'all_equities', 'uk_equities', 'japan_equities',
             'france_equities', 'south_africa_equities']
    for h in horizons:
        table = tables[h] if h in tables else tables[str(h)]
        lines += [f"# Judged {h} trading days on", '']
        for market in order + sorted({r['market'] for r in table} - set(order)):
            rows = [r for r in table if r['market'] == market]
            if not rows:
                continue
            lines += [f"## {market.replace('_', ' ')} ({h} days)", '']
            for r in sorted(rows, key=lambda r: r['source']):
                early = 'n/a' if r['early_lift'] is None else f"{r['early_lift']:+.1%}"
                late = 'n/a' if r['late_lift'] is None else f"{r['late_lift']:+.1%}"
                lines.append(f"{r['source']:<13} {r['hit_rate']:.0%} right of {r['n']} (chance {r['chance']:.0%}, "
                             f"z {r['z']:+.1f}); earlier half {early}, later half {late}; "
                             f"net move {r['net_move_pct']:+.2f}% (cost {r['assumed_cost_pct']}%)  ->  "
                             f"{VERDICT[r['verdict']]}")
            lines.append('')
    cands = [(h, r) for h in horizons for r in (tables[h] if h in tables else tables[str(h)])
             if r['verdict'] == 'candidate' and r['market'].startswith('all_')]
    lines += ['## Summary of the pooled rows', '']
    if cands:
        for h, r in cands:
            lines.append(f"CANDIDATE at {h} days: {r['source']} across {r['market'].replace('_', ' ')} "
                         f"(z {r['z']:+.1f}, {r['n']} signals)")
        lines.append(f"{len(cands)} of {3 * 2 * len(horizons)} pooled rows are candidates. Judging at "
                     f"{len(horizons)} horizons gives {len(horizons)} chances for luck, so a pattern that is a "
                     "candidate at one horizon only deserves doubt; one that holds at neighbouring horizons "
                     "deserves a closer look.")
    else:
        lines.append(f"None of the {3 * 2 * len(horizons)} pooled rows is a candidate.")
    lines.append('')
    if result.get('unit_fixes'):
        lines += ['Prices that switched between pounds and pence (or rand and cents) were corrected: ' +
                  ', '.join(f"{k} ({v})" for k, v in sorted(result['unit_fixes'].items())) + '.', '']
    lines += ['Read with care: markets move together, so the signals are not fully independent and the '
              'true count is lower than it looks. Yahoo\'s futures series jump when a contract rolls over. '
              'The listed companies are today\'s large ones, which flatters their history (survivorship); the index '
              'and index-fund rows are the fairer test. Results are in each market\'s own currency. '
              'The seasonal pattern is a one-month pattern, so longer horizons dilute it by design. '
              'Testing many patterns will throw up some that look good by luck. A CANDIDATE earns a place on the '
              'live watch, nothing more.']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='config/config.json')
    p.add_argument('--horizons', default=','.join(str(h) for h in cw.HORIZONS),
                   help='trading days after a signal to judge it at, comma separated (default: %(default)s)')
    p.add_argument('--period', default=None, help='how much history to replay (default: the config, "max")')
    p.add_argument('--only', choices=['metals', 'energy', 'agriculture', 'uk_equities', 'japan_equities',
                                       'france_equities', 'south_africa_equities'])
    p.add_argument('--kind', choices=['futures', 'fund', 'index', 'stock'],
                   help='only futures series, funds, indexes or single stocks')
    p.add_argument('--json', action='store_true')
    p.add_argument('--out', help='also write the report to this file')
    args = p.parse_args()
    try:
        horizons = sorted({int(x) for x in args.horizons.split(',') if x.strip()})
    except ValueError:
        p.error('--horizons must be whole numbers of trading days, like 20,60,120')
    if not horizons or horizons[0] < 1 or horizons[-1] > 500:
        p.error('--horizons must be between 1 and 500 trading days')
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
    result = cw.run_study(closes, meta, horizons[0], int(rule['min_signals']), float(rule['assumed_cost_pct']),
                          cw.costs(config), horizons=horizons)
    text = json.dumps(result, indent=2, default=str) if args.json else render(result, rule['assumed_cost_pct'])
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
