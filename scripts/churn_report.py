#!/usr/bin/env python3
"""Is the agent buying and selling the same position over and over?

Reads the order journal read-only (safe against a live agent), finds the round
trips that closed within an hour of opening, and says for each symbol how many
there were, how long they were held, what they made and what closed them. A
daily-bar strategy should hold for days; a symbol bought and sold again within
minutes, many times, is paying the spread and fees for nothing.

Usage:
    python scripts/churn_report.py
    python scripts/churn_report.py --hours 72 --symbol BTC-USD
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import churn                                       # noqa: E402
from src.agent.round_trips import closed_round_trips              # noqa: E402
from src.utils.config_validator import load_config                # noqa: E402
from src.utils.paths import DATA_DIR                              # noqa: E402


def read_fills(path):
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM orders WHERE status = 'filled' AND filled_quantity > 0 ORDER BY created_at")]
    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hours', type=float, default=168, help='look back this many hours (default a week)')
    p.add_argument('--symbol', help='show the order sequence for one symbol')
    p.add_argument('--config', default='config/config.json')
    p.add_argument('--data-dir', default=str(DATA_DIR))
    args = p.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    fills = read_fills(os.path.join(args.data_dir, 'order_journal.db'))
    trips = closed_round_trips(fills, config)
    since = datetime.now(timezone.utc).timestamp() - args.hours * 3600
    window = [t for t in trips if t.get('closed_at') and datetime.fromisoformat(t['closed_at']).replace(
        tzinfo=timezone.utc).timestamp() >= since]
    quick = churn.quick_trips(window)
    print(f"Last {args.hours:.0f} hours: {len(window)} round trips closed, {len(quick)} of them within an hour of opening"
          f" ({(len(quick) / len(window) if window else 0):.0%}).\n")
    rows = churn.by_symbol(quick)
    if not rows:
        print('No position was sold within an hour of being bought. Nothing looks like churn.')
    else:
        print(f"{'symbol':<10} {'times':>5} {'held (median)':>14} {'avg result':>11}  closed by")
        for sym, r in rows.items():
            print(f"{sym:<10} {r['n']:>5} {r['median_minutes']:>11.0f} min {r['mean_return_pct']:>+10.2f}%  {r['exits']}")
        total = sum(t['ret'] * t['cost'] for t in quick)
        print(f"\nRough result of those trades (mixed currencies, so indicative only): {total:+,.2f}")
    if args.symbol:
        sym = args.symbol.upper()
        print(f"\nLast orders for {sym} (oldest first):")
        for f in [f for f in fills if str(f.get('symbol')).upper() == sym][-16:]:
            print(f"  {str(f.get('created_at'))[:19]}  {f.get('side'):<4} {float(f.get('filled_quantity') or 0):>12.6f} "
                  f"@ {float(f.get('filled_avg_price') or 0):>12.4f}  {f.get('strategy')}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
