"""Back up the trading agent's durable state: the order/decision journal
(SQLite), paper broker state, and user accounts.

    python scripts/backup_state.py                # backup now, prune old ones
    python scripts/backup_state.py --keep-days 30  # override retention

Safe to run while the agent is live: order_journal.db is written in WAL
mode, so a plain file copy can miss data still sitting in the -wal file.
This uses sqlite3's online backup API instead, which produces a
consistent snapshot regardless of concurrent writers.

Schedule it with Windows Task Scheduler or cron for unattended backups;
this script only takes one now and prunes — it doesn't loop or daemonize.
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = {
    'order_journal.db': os.path.join(ROOT, 'data', 'order_journal.db'),
    'paper_trading_state.json': os.path.join(ROOT, 'data', 'paper_trading_state.json'),
    'users.json': os.path.join(ROOT, 'users.json'),
}


def backup_sqlite(src_path: str, dest_path: str):
    """Consistent snapshot via SQLite's online backup API — safe even
    while another process holds the DB open in WAL mode."""
    src = sqlite3.connect(f'file:{src_path}?mode=ro', uri=True)
    try:
        dest = sqlite3.connect(dest_path)
        try:
            src.backup(dest)
        finally:
            dest.close()
    finally:
        src.close()


def run_backup(keep_days: int) -> str:
    stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(ROOT, 'backups', stamp)
    os.makedirs(out_dir, exist_ok=True)

    manifest = {'timestamp_utc': datetime.utcnow().isoformat(), 'files': []}
    for name, src_path in SOURCES.items():
        if not os.path.exists(src_path):
            print(f"  skip (not found): {src_path}")
            continue
        dest_path = os.path.join(out_dir, name)
        if name.endswith('.db'):
            backup_sqlite(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)
        manifest['files'].append(name)
        print(f"  backed up: {name}")

    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    prune_old(keep_days)
    return out_dir


def prune_old(keep_days: int):
    backups_root = os.path.join(ROOT, 'backups')
    cutoff = datetime.utcnow() - timedelta(days=keep_days)
    if not os.path.isdir(backups_root):
        return
    for entry in os.listdir(backups_root):
        entry_path = os.path.join(backups_root, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            stamp = datetime.strptime(entry, '%Y%m%d_%H%M%S')
        except ValueError:
            continue  # not one of ours; leave it alone
        if stamp < cutoff:
            shutil.rmtree(entry_path, ignore_errors=True)
            print(f"  pruned old backup: {entry}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--keep-days', type=int, default=14,
                         help='Delete backups older than this many days (default: 14)')
    args = parser.parse_args()

    print(f"Backing up trading agent state (retention: {args.keep_days}d)...")
    out_dir = run_backup(args.keep_days)
    print(f"Done: {out_dir}")


if __name__ == '__main__':
    sys.exit(main())
