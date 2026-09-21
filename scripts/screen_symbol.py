#!/usr/bin/env python3
"""Screen a candidate symbol before adding it to the universe.

Answers whether the system can trade this symbol profitably, which is a
different question from whether it is a good company. The cost check is the
one that matters most and the one usually skipped: an NSE round trip runs
several percent, so a symbol whose typical move is smaller than that cannot
pay for itself however good the signal.

Usage:
    python scripts/screen_symbol.py --csv-dir data/history SCOM EQTY
    python scripts/screen_symbol.py --csv-dir data/history --all --json out.json
    python scripts/screen_symbol.py --csv-dir data/history NEWCO --holding-periods 60
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.candidate_screen import ADD, MARGINAL, REJECT, format_result, screen  # noqa: E402
from src.utils.price_files import load_price_dir  # noqa: E402

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('symbols', nargs='*', help='candidates to screen')
    ap.add_argument('--csv-dir', required=True, help='directory of <SYMBOL>.csv files')
    ap.add_argument('--config', default='config/config.json')
    ap.add_argument('--all', action='store_true',
                    help='screen every CSV in the directory that is not already tracked')
    ap.add_argument('--holding-periods', type=int, default=20,
                    help='bars a position is expected to be held; the cost check '
                         'compares the round trip against the typical move over '
                         'this span (default 20)')
    ap.add_argument('--periods-per-year', type=int, default=252)
    ap.add_argument('--min-move-to-cost', type=float, default=3.0,
                    help='how many times the round-trip cost the typical move '
                         'should be before the symbol is comfortable (default 3)')
    ap.add_argument('--max-correlation', type=float, default=0.85)
    ap.add_argument('--min-avg-volume', type=float, default=None)
    ap.add_argument('--json', help='also write the full results here')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    dm = cfg.get('data_manager', {})
    tracked = {s.upper() for s in
               (dm.get('symbols', []) + dm.get('nse_symbols', [])
                + dm.get('crypto_symbols', []))}

    everything = load_price_dir(args.csv_dir)
    if not everything:
        raise SystemExit(f"No usable price files in {args.csv_dir}")

    if args.all:
        candidates = [s for s in sorted(everything) if s not in tracked]
        if not candidates:
            print("Every symbol with price data is already tracked.")
            return 0
    else:
        candidates = [s.strip().upper() for s in args.symbols if s.strip()]
        if not candidates:
            raise SystemExit("Name at least one symbol, or pass --all")

    missing = [s for s in candidates if s not in everything]
    for s in missing:
        print(f"{s}: no usable price data in {args.csv_dir}, cannot screen")
    candidates = [s for s in candidates if s in everything]
    if not candidates:
        return 1

    # Compare each candidate against the symbols already tracked, not against
    # the other candidates: correlation with something you do not hold is not
    # a reason to reject.
    universe = {s: df for s, df in everything.items() if s in tracked}

    results = []
    for sym in candidates:
        r = screen(sym, everything[sym], cfg,
                   universe={k: v for k, v in universe.items() if k != sym},
                   periods_per_year=args.periods_per_year,
                   holding_periods=args.holding_periods,
                   min_move_to_cost=args.min_move_to_cost,
                   max_correlation=args.max_correlation,
                   min_avg_volume=args.min_avg_volume)
        results.append(r)
        print(format_result(r))
        print()

    by_verdict = {v: [r.symbol for r in results if r.verdict == v]
                  for v in (ADD, MARGINAL, REJECT)}
    print("-" * 60)
    for verdict in (ADD, MARGINAL, REJECT):
        if by_verdict[verdict]:
            print(f"{verdict:<9} {', '.join(by_verdict[verdict])}")
    if by_verdict[ADD]:
        print(f"\nTo add: python scripts/manage_universe.py add "
              f"{' '.join(by_verdict[ADD])}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump([{'symbol': r.symbol, 'market': r.market,
                        'verdict': r.verdict, 'reasons': r.reasons,
                        'warnings': r.warnings, 'metrics': r.metrics}
                       for r in results], f, indent=2, default=str)
        print(f"\nFull results written to {args.json}")

    # Non-zero when nothing is addable, so this can gate an automated flow.
    return 0 if by_verdict[ADD] or by_verdict[MARGINAL] else 1


if __name__ == '__main__':
    sys.exit(main())
