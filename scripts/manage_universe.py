#!/usr/bin/env python3
"""Add, remove, list and review the symbols the agent trades.

Adding a symbol is a one-line config change, which is exactly why it is
worth a tool: each new name costs time and API quota on every cycle, lands
in a market with its own cost structure, and may simply duplicate a bet
already held. This refuses to add a symbol the screen rejects unless told
to, and writes nothing without --apply.

Usage:
    python scripts/manage_universe.py list
    python scripts/manage_universe.py add SCOM --csv-dir data/history
    python scripts/manage_universe.py add SCOM --csv-dir data/history --apply
    python scripts/manage_universe.py remove MSFT --apply
    python scripts/manage_universe.py review
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import OrderedDict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.candidate_screen import REJECT, format_result, screen  # noqa: E402
from src.agent.cost_model import classify, costs_for, round_trip_pct, unverified_markets  # noqa: E402
from src.utils.price_files import load_price_dir  # noqa: E402

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# The config lists serve two different purposes, and conflating them is how
# an intentional overlap gets reported as a bug.
#
# TRADING lists say what the agent acts on:
#   data_manager.symbols      walked by the main loop every cycle
#   data_manager.nse_symbols  walked by the NSE pass on its own slower cadence
#
# CLASSIFICATION lists say how a symbol is costed and treated. They are
# expected to intersect the trading lists: a crypto pair belongs in `symbols`
# so the loop trades it AND in `crypto_symbols` so cost_model prices it as
# crypto rather than as a US equity.
TRADING_LISTS = ('symbols', 'nse_symbols')
CLASSIFICATION_LISTS = ('crypto_symbols',)
ALL_LISTS = TRADING_LISTS + CLASSIFICATION_LISTS

# Where `add` files a newly approved symbol, by market.
MARKET_LISTS = {'nse': 'nse_symbols', 'crypto': 'symbols',
                'us_equity': 'symbols'}


def load_config(path):
    with open(path) as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def save_config(cfg, path):
    """Write config, keeping a timestamped backup.

    This file drives what the agent trades with real money, so an edit that
    cannot be undone is not an acceptable convenience.
    """
    backup = f"{path}.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(path, backup)
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')
    return backup


def config_lists(cfg, keys=ALL_LISTS):
    """{config key: [symbols]} exactly as written."""
    dm = cfg.get('data_manager', {})
    return {key: [s.upper() for s in dm.get(key, []) or []] for key in keys}


def universe(cfg):
    """{market: [symbols]} grouped by what each symbol actually is.

    Grouped by `classify` rather than by which list it sits in, because
    `data_manager.symbols` is a general list that can hold any market. A
    crypto pair listed there is still a crypto pair for costing purposes.
    """
    by_market = {}
    for syms in config_lists(cfg).values():
        for sym in syms:
            by_market.setdefault(classify(sym, cfg), set()).add(sym)
    return {m: sorted(s) for m, s in by_market.items()}


def conflicts(cfg):
    """Symbols claimed by more than one TRADING list.

    This is a genuine conflict: two execution paths on different cadences
    would both act on the same name. An overlap between a trading list and a
    classification list is not a conflict and is not reported here.
    """
    seen = {}
    for key, syms in config_lists(cfg, TRADING_LISTS).items():
        for sym in syms:
            seen.setdefault(sym, []).append(key)
    return {sym: keys for sym, keys in seen.items() if len(keys) > 1}


def unclassified_crypto(cfg):
    """Dash-suffixed fiat pairs traded but not listed in `crypto_symbols`.

    These still classify as crypto through cost_model's suffix rule, but
    relying on that leaves the fee schedule to a naming convention.
    """
    traded = {s for syms in config_lists(cfg, TRADING_LISTS).values() for s in syms}
    declared = set(config_lists(cfg, CLASSIFICATION_LISTS)['crypto_symbols'])
    return sorted(s for s in traded - declared
                  if '-' in s and s.rsplit('-', 1)[-1] in
                  {'USD', 'USDT', 'USDC', 'EUR'})


def all_symbols(cfg):
    """Every distinct symbol the agent acts on.

    Classification lists are excluded: a name appears there to describe a
    symbol that is already in a trading list, so counting it again would
    inflate the universe.
    """
    return {s for syms in config_lists(cfg, TRADING_LISTS).values() for s in syms}


def cmd_list(args, cfg):
    by_market = universe(cfg)
    total = len(all_symbols(cfg))
    print(f"Universe: {total} distinct symbols\n")
    for market in sorted(by_market):
        syms = by_market[market]
        if not syms:
            continue
        c = costs_for(syms[0], cfg)
        verified = 'verified' if c.get('verified') else 'UNVERIFIED'
        print(f"{market}  ({len(syms)} symbols, round trip "
              f"{round_trip_pct(syms[0], cfg):.2%}, costs {verified})")
        for i in range(0, len(syms), 8):
            print("    " + "  ".join(syms[i:i + 8]))
        print()

    clashes = conflicts(cfg)
    if clashes:
        print("Claimed by two trading paths (a real conflict):")
        for sym, keys in sorted(clashes.items()):
            print(f"    {sym}: {', '.join(keys)}")
        print("    Both would act on it, on different cadences.\n")

    stray = unclassified_crypto(cfg)
    if stray:
        print(f"Traded crypto not declared in crypto_symbols: {', '.join(stray)}")
        print("    They fall back to cost_model's dash-suffix rule, which "
              "leaves the fee schedule to a naming convention.\n")

    unverified = unverified_markets(cfg)
    active = {m for m, s in by_market.items() if s}
    for market in sorted(set(unverified) & active):
        print(f"WARNING: '{market}' costs are a placeholder. Any backtest of "
              f"those symbols is provisional until confirmed with your broker.")

    print(f"\nCapacity: python scripts/monitor_capacity.py --symbols {total}")
    return 0


def cmd_add(args, cfg):
    targets = [s.strip().upper() for s in args.symbols if s.strip()]
    existing = all_symbols(cfg)
    already = [s for s in targets if s in existing]
    for s in already:
        print(f"{s}: already in the universe, skipping")
    targets = [s for s in targets if s not in existing]
    if not targets:
        return 0

    approved = targets
    if args.csv_dir:
        data = load_price_dir(args.csv_dir)
        tracked = {s: df for s, df in data.items() if s in existing}
        approved = []
        for sym in targets:
            if sym not in data:
                print(f"{sym}: no usable price data in {args.csv_dir}. "
                      f"Add data or pass --skip-screen to add it anyway.\n")
                continue
            result = screen(sym, data[sym], cfg, universe=tracked,
                            holding_periods=args.holding_periods,
                            periods_per_year=args.periods_per_year)
            print(format_result(result))
            print()
            if result.verdict == REJECT and not args.force:
                print(f"  -> not adding {sym}. Pass --force to override.\n")
                continue
            approved.append(sym)
    elif not args.skip_screen:
        raise SystemExit(
            "Pass --csv-dir to screen the candidates first, or --skip-screen "
            "to add without screening. Adding a symbol unscreened is how an "
            "untradeable one ends up in the live loop.")

    if not approved:
        print("Nothing to add.")
        return 1

    for sym in approved:
        market = classify(sym, cfg)
        key = MARKET_LISTS[market]
        cfg.setdefault('data_manager', OrderedDict()).setdefault(key, [])
        if sym not in cfg['data_manager'][key]:
            cfg['data_manager'][key].append(sym)
        print(f"  + {sym} -> data_manager.{key} ({market})")

    total = len(all_symbols(cfg))
    print(f"\nUniverse would be {total} symbols.")
    print(f"Check it still fits the cycle: "
          f"python scripts/monitor_capacity.py --symbols {total}")

    if args.apply:
        backup = save_config(cfg, args.config)
        print(f"\nWritten to {args.config} (backup at {backup})")
    else:
        print("\nDry run. Nothing written. Re-run with --apply to commit.")
    return 0


def cmd_remove(args, cfg):
    targets = [s.strip().upper() for s in args.symbols if s.strip()]
    removed = []
    for sym in targets:
        for key in ALL_LISTS:
            lst = cfg.get('data_manager', {}).get(key) or []
            matches = [s for s in lst if s.upper() == sym]
            for m in matches:
                lst.remove(m)
                removed.append((sym, key))
    for sym in targets:
        if not any(s == sym for s, _ in removed):
            print(f"{sym}: not in the universe")
    for sym, key in removed:
        print(f"  - {sym} from data_manager.{key}")

    if not removed:
        return 1
    if any(s in {x for x, _ in removed} for s in
           (cfg.get('sleeve', {}).get('universe') or [])):
        print("\nNote: one or more of these also appear in sleeve.universe, "
              "which this command does not touch. Remove them there too if "
              "the long-term sleeve should stop accumulating them.")

    if args.apply:
        backup = save_config(cfg, args.config)
        print(f"\nWritten to {args.config} (backup at {backup})")
    else:
        print("\nDry run. Nothing written. Re-run with --apply to commit.")
    return 0


def cmd_review(args, cfg):
    """Which symbols have earned their place, from realized fills."""
    from src.agent.strategy_attribution import compute_symbol_attribution
    try:
        from src.agent.order_journal import OrderJournal
        journal = OrderJournal()
        fills = journal.filled_orders()
    except Exception as e:
        print(f"Could not read the order journal: {e}")
        print("Nothing to review until the agent has traded.")
        return 1

    if not fills:
        print("No filled orders recorded yet, so there is nothing to review.")
        print("This becomes useful after the agent has traded for a while.")
        return 1

    attribution = compute_symbol_attribution(fills)
    tracked = all_symbols(cfg)

    rows = sorted(attribution.items(), key=lambda kv: kv[1]['total_pnl'])
    print(f"{'symbol':<10}{'realized':>12}{'unrealized':>12}{'total':>12}"
          f"{'closed':>8}{'win%':>7}  status")
    for sym, a in rows:
        status = '' if sym in tracked else '(no longer tracked)'
        print(f"{sym:<10}{a['realized_pnl']:>12,.2f}{a['unrealized_pnl']:>12,.2f}"
              f"{a['total_pnl']:>12,.2f}{a['closed_trades']:>8}"
              f"{a['win_rate'] * 100:>6.0f}%  {status}")

    losers = [s for s, a in rows
              if a['total_pnl'] < 0 and a['closed_trades'] >= args.min_trades
              and s in tracked]
    untraded = sorted(tracked - set(attribution))

    print()
    if losers:
        print(f"Losing money over at least {args.min_trades} closed trades: "
              f"{', '.join(losers)}")
        print(f"  Consider: python scripts/manage_universe.py remove "
              f"{' '.join(losers)}")
    else:
        print(f"No tracked symbol is down over {args.min_trades}+ closed trades.")
    if untraded:
        print(f"\nTracked but never traded: {', '.join(untraded)}")
        print("  These still cost data calls and cycle time every loop. If the "
              "strategies never act on them, they are overhead.")
    print("\nA symbol's P&L over a handful of trades is mostly luck. Treat this "
          "as a prompt to look, not a rule to act on.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config/config.json')
    sub = ap.add_subparsers(dest='command', required=True)

    sub.add_parser('list', help='show the universe by market, with costs')

    p_add = sub.add_parser('add', help='screen and add symbols')
    p_add.add_argument('symbols', nargs='+')
    p_add.add_argument('--csv-dir', help='price data used to screen the candidates')
    p_add.add_argument('--skip-screen', action='store_true',
                       help='add without screening (not recommended)')
    p_add.add_argument('--force', action='store_true',
                       help='add even when the screen rejects')
    p_add.add_argument('--holding-periods', type=int, default=20)
    p_add.add_argument('--periods-per-year', type=int, default=252)
    p_add.add_argument('--apply', action='store_true', help='write the config')

    p_rm = sub.add_parser('remove', help='remove symbols')
    p_rm.add_argument('symbols', nargs='+')
    p_rm.add_argument('--apply', action='store_true', help='write the config')

    p_rev = sub.add_parser('review', help='which symbols have earned their place')
    p_rev.add_argument('--min-trades', type=int, default=5,
                       help='closed trades before a symbol is judged (default 5)')

    args = ap.parse_args()
    cfg = load_config(args.config)
    return {'list': cmd_list, 'add': cmd_add,
            'remove': cmd_remove, 'review': cmd_review}[args.command](args, cfg)


if __name__ == '__main__':
    sys.exit(main())
