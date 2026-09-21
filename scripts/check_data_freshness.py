#!/usr/bin/env python3
"""Is the agent looking at real, current prices?

Three ways the answer is no, each of which the system survives quietly:

- **Stale bars.** The scraper logs "0/9 symbols updated" when its sources are
  unreachable and carries on. The agent then trades, or declines to trade, on
  last week's prices.
- **Synthetic bars.** The historical store seeds itself with generated prices
  so a fresh install has something to work with. Those rows are marked
  `source=synthetic`, and nothing has ever compared that marker against
  reality. Every bar in this repository's store is synthetic, which means any
  NSE analysis built on it is describing invented prices.
- **A dead live feed.** When the data layer falls back to synthetic quotes the
  agent correctly refuses to trade, recording `fallback_price`. A run that
  held four thousand times because it could not see prices has not tested the
  strategy at all, and from the outside it looks identical to a strategy that
  found nothing.

Exits non-zero when any of those is true, so a scheduled job goes visibly red
rather than succeeding quietly.

    python scripts/check_data_freshness.py
    python scripts/check_data_freshness.py --max-age-days 7 --json
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.paths import DATA_DIR  # noqa: E402

# Four days by default. A check running on Monday sees Friday's close as three
# days old through an ordinary weekend, and a public holiday adds another day.
# Tighter than this and the job cries wolf every long weekend, which trains
# you to ignore it.
DEFAULT_MAX_AGE_DAYS = 4

# Provenance markers that mean "this price was invented, not observed".
SYNTHETIC_SOURCES = {'synthetic', 'fallback', 'generated', 'mock'}


def read_bars(path):
    """(newest date, source of the newest row, row count), or None."""
    try:
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None
    if not rows:
        return None

    newest_date, newest_source = None, None
    for row in rows:
        raw = (row.get('date') or '').strip()
        if not raw:
            continue
        try:
            parsed = datetime.strptime(raw[:10], '%Y-%m-%d').date()
        except ValueError:
            continue
        if newest_date is None or parsed > newest_date:
            newest_date = parsed
            newest_source = (row.get('source') or '').strip().lower()
    if newest_date is None:
        return None
    return newest_date, newest_source, len(rows)


def check_history(csv_dir, max_age_days, today=None):
    """Age and provenance of every symbol's newest bar."""
    today = today or datetime.now(timezone.utc).date()
    out = {'symbols': [], 'stale': [], 'synthetic': [], 'unreadable': []}
    if not os.path.isdir(csv_dir):
        out['error'] = f'no such directory: {csv_dir}'
        return out

    for name in sorted(os.listdir(csv_dir)):
        if not name.lower().endswith('.csv'):
            continue
        symbol = os.path.splitext(name)[0].upper()
        parsed = read_bars(os.path.join(csv_dir, name))
        if parsed is None:
            out['unreadable'].append(symbol)
            continue
        newest, source, count = parsed
        age = (today - newest).days
        entry = {'symbol': symbol, 'newest': newest.isoformat(),
                 'age_days': age, 'source': source, 'bars': count}
        out['symbols'].append(entry)
        if age > max_age_days:
            out['stale'].append(entry)
        if source in SYNTHETIC_SOURCES:
            out['synthetic'].append(entry)
    return out


def check_live_feed(decisions_db, hours=24):
    """Share of recent decisions blocked because the price was not real."""
    out = {'total': 0, 'fallback': 0, 'share': 0.0, 'available': False}
    if not os.path.exists(decisions_db):
        return out
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    try:
        conn = sqlite3.connect(f'file:{decisions_db}?mode=ro', uri=True)
    except sqlite3.Error:
        return out
    try:
        rows = conn.execute(
            'SELECT skip_reason, COUNT(*) FROM decisions WHERE ts >= ?'
            ' GROUP BY skip_reason', (since,)).fetchall()
    except sqlite3.Error:
        return out
    finally:
        conn.close()

    out['available'] = True
    for reason, count in rows:
        out['total'] += count
        if reason == 'fallback_price':
            out['fallback'] += count
    if out['total']:
        out['share'] = out['fallback'] / out['total']
    return out


def render(history, live, max_age_days):
    lines = ['', 'Data freshness', '=' * 62]

    if history.get('error'):
        lines.append(f"[BLOCK]  historical store: {history['error']}")
    elif not history['symbols']:
        lines.append('[BLOCK]  historical store is empty')
    else:
        oldest = max(history['symbols'], key=lambda e: e['age_days'])
        lines.append(f"{len(history['symbols'])} symbol(s); oldest bar is "
                     f"{oldest['age_days']} day(s) old ({oldest['symbol']})")

        if history['stale']:
            lines.append('')
            lines.append(f"[BLOCK]  {len(history['stale'])} symbol(s) have no bar "
                         f"newer than {max_age_days} days:")
            for e in history['stale'][:8]:
                lines.append(f"         {e['symbol']:8} newest {e['newest']} "
                             f"({e['age_days']}d)")
        else:
            lines.append('[ok]     every symbol has a recent bar')

        if history['synthetic']:
            lines.append('')
            lines.append(f"[BLOCK]  {len(history['synthetic'])} symbol(s) have a "
                         f"SYNTHETIC newest bar. These prices were generated, "
                         f"not observed, so anything measured on them is "
                         f"describing invented data:")
            for e in history['synthetic'][:8]:
                lines.append(f"         {e['symbol']:8} source={e['source']}")
        else:
            lines.append('[ok]     newest bars come from real sources')

        if history['unreadable']:
            lines.append(f"[warn]   unreadable: {', '.join(history['unreadable'][:8])}")

    lines.append('')
    if not live['available']:
        lines.append('[warn]   no decision journal yet, so the live feed was '
                     'not checked')
    elif live['total'] == 0:
        lines.append('[warn]   no decisions in the last 24h; either the agent '
                     'is stopped or it has only just started')
    elif live['share'] > 0.5:
        lines.append(f"[BLOCK]  {live['fallback']}/{live['total']} decisions "
                     f"({live['share']:.0%}) in the last 24h saw synthetic "
                     f"prices. The vendor feed is down and the run is not "
                     f"testing the strategy.")
    elif live['fallback']:
        lines.append(f"[warn]   {live['fallback']}/{live['total']} decisions "
                     f"({live['share']:.0%}) saw synthetic prices")
    else:
        lines.append(f"[ok]     all {live['total']} recent decisions saw real prices")

    lines.append('=' * 62)
    return '\n'.join(lines)


def failures(history, live):
    """The conditions that make this exit non-zero."""
    problems = []
    if history.get('error'):
        problems.append(history['error'])
    elif not history['symbols']:
        problems.append('historical store is empty')
    else:
        if history['stale']:
            problems.append(f"{len(history['stale'])} symbol(s) with stale bars")
        if history['synthetic']:
            problems.append(f"{len(history['synthetic'])} symbol(s) with synthetic bars")
    if live['available'] and live['total'] and live['share'] > 0.5:
        problems.append(f"{live['share']:.0%} of recent decisions saw synthetic prices")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--csv-dir', default=str(DATA_DIR / 'nse_historical'),
                    help='directory of <SYMBOL>.csv price history')
    ap.add_argument('--decisions-db', default=str(DATA_DIR / 'decision_journal.db'))
    ap.add_argument('--max-age-days', type=int, default=DEFAULT_MAX_AGE_DAYS,
                    help=f'stale beyond this many days (default: '
                         f'{DEFAULT_MAX_AGE_DAYS}, which absorbs a weekend '
                         f'plus a public holiday)')
    ap.add_argument('--hours', type=int, default=24,
                    help='window for the live feed check (default: 24)')
    ap.add_argument('--allow-synthetic', action='store_true',
                    help='do not fail on generated bars. Only reasonable on a '
                         'fresh install that has not scraped yet.')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    history = check_history(args.csv_dir, args.max_age_days)
    live = check_live_feed(args.decisions_db, args.hours)

    if args.allow_synthetic:
        history = dict(history, synthetic=[])

    problems = failures(history, live)

    if args.json:
        print(json.dumps({'history': history, 'live_feed': live,
                          'problems': problems, 'ok': not problems},
                         indent=2, default=str))
    else:
        print(render(history, live, args.max_age_days))
        if problems:
            print(f"\nNOT FRESH: {'; '.join(problems)}.")
        else:
            print('\nFRESH. Prices are recent and real.')

    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
