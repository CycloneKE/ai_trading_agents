#!/usr/bin/env python3
"""Back up the trading agent's durable state.

    python scripts/backup_state.py                    # backup now, prune old ones
    python scripts/backup_state.py --keep-days 30     # override retention
    python scripts/backup_state.py --dest /mnt/backups
    python scripts/backup_state.py --verify-only      # check the newest backup

Safe to run while the agent is live: the journals are written in WAL mode, so
a plain file copy can miss data still sitting in the -wal file. This uses
sqlite3's online backup API instead, which produces a consistent snapshot
regardless of concurrent writers, and then runs an integrity check on the
copy. A backup that cannot be read back is not a backup.

Two things this script used to get wrong, both of which made it look like it
was protecting the run when it was not:

- It backed up `order_journal.db` and claimed that file also held the
  decision journal. It does not: `decision_journal.py` writes a separate
  `decision_journal.db`. The record of *why* the agent held or traded, which
  is half of what the weekly report reads, was never in a backup. Sources are
  now discovered from the data directory rather than hardcoded, so a journal
  added later is picked up without anyone remembering to add it here.

- It wrote to `<repo>/backups`, which is not a mounted volume in any
  containerised deployment. Every backup was destroyed by the next redeploy.
  The default is now inside the data directory, which IS the mounted volume.

Note what this does and does not protect. Backups living beside the journals
survive a redeploy, a bad migration and an accidental deletion, which are the
common cases. They do not survive losing the volume or the server. For that,
copy the backup directory off the machine as well.
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

BACKUP_DIRNAME = 'backups'
STAMP_FORMAT = '%Y%m%d_%H%M%S'

# Non-database state, resolved against the data directory actually in use.
# These used to be module-level constants built from the global DATA_DIR,
# which meant --data-dir was honoured for the journals and silently ignored
# for everything else: pointing it at an empty directory still "succeeded"
# by picking up the repository's own users.json.

def extra_files(data_dir: str) -> dict:
    """Loose files worth keeping, as {label: absolute path}."""
    return {
        # Lives beside the code, not with the journals.
        'users.json': os.path.join(str(PROJECT_ROOT), 'users.json'),
        'paper_trading_state.json': os.path.join(data_dir, 'paper_trading_state.json'),
        'strategy_params.json': os.path.join(data_dir, 'strategy_params.json'),
        'event_calendar.json': os.path.join(data_dir, 'event_calendar.json'),
    }


def extra_dirs(data_dir: str) -> dict:
    """Directories copied wholesale, as {label: absolute path}."""
    return {
        # Run manifests. Without them a restored run has no start date, and
        # the weekly report cannot tell "nothing in three weeks" from
        # "started an hour ago".
        'paper_runs': os.path.join(data_dir, 'paper_runs'),
    }


def default_dest() -> str:
    """Inside the data directory, which deployments mount as a volume."""
    return os.path.join(str(DATA_DIR), BACKUP_DIRNAME)


def discover_databases(data_dir: str) -> dict:
    """Every SQLite journal in the data directory, keyed by filename.

    Discovered rather than listed, so a journal added later is backed up
    without anyone having to remember this file exists.
    """
    out = {}
    if not os.path.isdir(data_dir):
        return out
    for name in sorted(os.listdir(data_dir)):
        if name.endswith('.db'):
            out[name] = os.path.join(data_dir, name)
    return out


def backup_sqlite(src_path: str, dest_path: str) -> None:
    """Consistent snapshot via SQLite's online backup API."""
    src = sqlite3.connect(f'file:{src_path}?mode=ro', uri=True)
    try:
        dest = sqlite3.connect(dest_path)
        try:
            src.backup(dest)
        finally:
            dest.close()
    finally:
        src.close()


def verify_sqlite(path: str) -> bool:
    """Read the copy back. Catches a snapshot that wrote but is unusable."""
    try:
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        try:
            result = conn.execute('PRAGMA integrity_check').fetchone()
            return bool(result) and result[0] == 'ok'
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def run_backup(keep_days: int, dest_root: str, data_dir: str) -> dict:
    now = datetime.now(timezone.utc)
    out_dir = os.path.join(dest_root, now.strftime(STAMP_FORMAT))
    os.makedirs(out_dir, exist_ok=True)

    manifest = {'timestamp_utc': now.isoformat(), 'files': [],
                'databases': [], 'skipped': [], 'failed': []}

    for name, src_path in discover_databases(data_dir).items():
        dest_path = os.path.join(out_dir, name)
        try:
            backup_sqlite(src_path, dest_path)
        except sqlite3.Error as e:
            manifest['failed'].append({'name': name, 'error': str(e)})
            print(f"  FAILED: {name}: {e}")
            continue
        if verify_sqlite(dest_path):
            manifest['files'].append(name)
            manifest['databases'].append(name)
            print(f"  backed up and verified: {name}")
        else:
            manifest['failed'].append({'name': name, 'error': 'integrity check failed'})
            print(f"  FAILED integrity check: {name}")

    for label, src_path in extra_files(data_dir).items():
        if not os.path.exists(src_path):
            manifest['skipped'].append(label)
            continue
        shutil.copy2(src_path, os.path.join(out_dir, label))
        manifest['files'].append(label)
        print(f"  backed up: {label}")

    for label, src_path in extra_dirs(data_dir).items():
        if not os.path.isdir(src_path):
            manifest['skipped'].append(label)
            continue
        shutil.copytree(src_path, os.path.join(out_dir, label), dirs_exist_ok=True)
        manifest['files'].append(f'{label}/')
        print(f"  backed up: {label}/")

    manifest['path'] = out_dir
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    prune_old(keep_days, dest_root)
    return manifest


def prune_old(keep_days: int, dest_root: str) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    if not os.path.isdir(dest_root):
        return 0
    pruned = 0
    for entry in os.listdir(dest_root):
        entry_path = os.path.join(dest_root, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            stamp = datetime.strptime(entry, STAMP_FORMAT).replace(tzinfo=timezone.utc)
        except ValueError:
            continue  # not one of ours; leave it alone
        if stamp < cutoff:
            shutil.rmtree(entry_path, ignore_errors=True)
            pruned += 1
            print(f"  pruned old backup: {entry}")
    return pruned


def newest_backup(dest_root: str):
    if not os.path.isdir(dest_root):
        return None
    stamps = []
    for entry in os.listdir(dest_root):
        try:
            datetime.strptime(entry, STAMP_FORMAT)
        except ValueError:
            continue
        if os.path.isdir(os.path.join(dest_root, entry)):
            stamps.append(entry)
    return os.path.join(dest_root, max(stamps)) if stamps else None


def verify_backup(path: str) -> bool:
    """Every database in a backup reads back cleanly."""
    ok = True
    for name in sorted(os.listdir(path)):
        if not name.endswith('.db'):
            continue
        good = verify_sqlite(os.path.join(path, name))
        print(f"  {'ok  ' if good else 'BAD '} {name}")
        ok = ok and good
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--keep-days', type=int, default=14,
                        help='Delete backups older than this many days (default: 14)')
    parser.add_argument('--dest', default=None,
                        help=f'Where backups go (default: {default_dest()})')
    parser.add_argument('--data-dir', default=str(DATA_DIR),
                        help='Directory holding the journals')
    parser.add_argument('--verify-only', action='store_true',
                        help='Check the newest backup instead of taking one')
    args = parser.parse_args()

    dest_root = args.dest or default_dest()

    if args.verify_only:
        newest = newest_backup(dest_root)
        if not newest:
            print(f'No backups found in {dest_root}', file=sys.stderr)
            return 1
        print(f'Verifying {newest}')
        return 0 if verify_backup(newest) else 1

    os.makedirs(dest_root, exist_ok=True)
    print(f'Backing up {args.data_dir} -> {dest_root} (retention: {args.keep_days}d)')
    manifest = run_backup(args.keep_days, dest_root, args.data_dir)

    if manifest['failed']:
        print(f"\n{len(manifest['failed'])} file(s) failed. The backup is "
              f"incomplete; do not rely on it.", file=sys.stderr)
        return 1
    if not manifest['databases']:
        print('\nNothing was backed up: no journals found. Check --data-dir '
              'points at the directory holding the .db files.', file=sys.stderr)
        return 1
    print(f"\nDone: {manifest['path']} ({len(manifest['files'])} item(s))")
    return 0


if __name__ == '__main__':
    sys.exit(main())
