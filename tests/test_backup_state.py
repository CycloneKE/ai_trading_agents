"""Backups that actually contain the run.

Two defects made the previous version look like it was protecting the agent
while it was not, and both are the kind that only surface when you need the
backup:

- It backed up order_journal.db and asserted in a comment that the same file
  held the decision journal. It does not; they are separate files. The record
  of why the agent acted was never backed up.
- It wrote to <repo>/backups, which is not a mounted volume, so every backup
  was destroyed by the next container redeploy.
"""
import importlib.util
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    'backup_state', os.path.join(_ROOT, 'scripts', 'backup_state.py'))
backup = importlib.util.module_from_spec(_SPEC)
sys.modules['backup_state'] = backup
_SPEC.loader.exec_module(backup)


def _make_db(path, rows=3):
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)')
    conn.executemany('INSERT INTO t (v) VALUES (?)', [(f'row{i}',) for i in range(rows)])
    conn.commit()
    return conn


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / 'data'
    d.mkdir()
    for name in ('order_journal.db', 'decision_journal.db', 'audit_journal.db',
                 'risk_state.db'):
        _make_db(str(d / name)).close()
    (d / 'paper_runs').mkdir()
    (d / 'paper_runs' / 'run_1.json').write_text('{"started_at": "2026-01-01T00:00:00Z"}')
    return d


# ---------------------------------------------------- the two real defects

def test_every_journal_is_backed_up_not_just_the_order_one(data_dir, tmp_path):
    """The decision journal is a separate file and was silently omitted."""
    dest = tmp_path / 'backups'
    manifest = backup.run_backup(14, str(dest), str(data_dir))
    for name in ('order_journal.db', 'decision_journal.db', 'audit_journal.db',
                 'risk_state.db'):
        assert name in manifest['files'], f'{name} missing from the backup'


def test_a_journal_added_later_is_picked_up_without_code_changes(data_dir, tmp_path):
    """Sources are discovered, not hardcoded, so this cannot rot again."""
    _make_db(str(data_dir / 'brand_new_journal.db')).close()
    manifest = backup.run_backup(14, str(tmp_path / 'b'), str(data_dir))
    assert 'brand_new_journal.db' in manifest['files']


def test_the_default_destination_is_inside_the_mounted_data_volume():
    """<repo>/backups is destroyed by every container redeploy."""
    from src.utils.paths import DATA_DIR
    assert backup.default_dest().startswith(str(DATA_DIR))
    assert os.path.basename(backup.default_dest()) == 'backups'


# --------------------------------------------------------- what it captures

def test_run_manifests_are_included(data_dir, tmp_path):
    """Without them a restored run has no recorded start date."""
    manifest = backup.run_backup(14, str(tmp_path / 'b'), str(data_dir))
    assert 'paper_runs/' in manifest['files']
    assert os.path.exists(os.path.join(manifest['path'], 'paper_runs', 'run_1.json'))


def test_the_backup_carries_the_actual_rows(data_dir, tmp_path):
    manifest = backup.run_backup(14, str(tmp_path / 'b'), str(data_dir))
    copy = os.path.join(manifest['path'], 'order_journal.db')
    conn = sqlite3.connect(copy)
    try:
        assert conn.execute('SELECT COUNT(*) FROM t').fetchone()[0] == 3
    finally:
        conn.close()


def test_a_manifest_is_written(data_dir, tmp_path):
    manifest = backup.run_backup(14, str(tmp_path / 'b'), str(data_dir))
    with open(os.path.join(manifest['path'], 'manifest.json'), encoding='utf-8') as f:
        on_disk = json.load(f)
    assert on_disk['files'] == manifest['files']
    assert on_disk['timestamp_utc']


# ----------------------------------------------------- safe against a live agent

def test_backing_up_while_a_writer_holds_the_database_open(data_dir, tmp_path):
    """The agent never stops for a backup, so this is the normal case."""
    live = _make_db(str(data_dir / 'order_journal.db'), rows=0)
    live.execute("INSERT INTO t (v) VALUES ('written_while_open')")
    live.commit()
    try:
        manifest = backup.run_backup(14, str(tmp_path / 'b'), str(data_dir))
    finally:
        live.close()
    copy = os.path.join(manifest['path'], 'order_journal.db')
    conn = sqlite3.connect(copy)
    try:
        rows = [r[0] for r in conn.execute('SELECT v FROM t').fetchall()]
    finally:
        conn.close()
    assert 'written_while_open' in rows, (
        'WAL data was lost; the copy missed committed rows')


# ----------------------------------------------------------- integrity check

def test_a_corrupt_copy_is_reported_not_counted_as_backed_up(tmp_path):
    corrupt = tmp_path / 'broken.db'
    corrupt.write_bytes(b'this is not a database')
    assert backup.verify_sqlite(str(corrupt)) is False


def test_a_good_copy_verifies(data_dir):
    assert backup.verify_sqlite(str(data_dir / 'order_journal.db')) is True


# ------------------------------------------------------------------ pruning

def test_pruning_removes_only_expired_backups(tmp_path):
    root = tmp_path / 'backups'
    root.mkdir()
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=40)).strftime(backup.STAMP_FORMAT)
    recent = (now - timedelta(days=2)).strftime(backup.STAMP_FORMAT)
    for name in (old, recent):
        (root / name).mkdir()
    pruned = backup.prune_old(14, str(root))
    assert pruned == 1
    assert not (root / old).exists()
    assert (root / recent).exists()


def test_pruning_leaves_unrecognised_directories_alone(tmp_path):
    """Never delete something we did not create."""
    root = tmp_path / 'backups'
    root.mkdir()
    (root / 'someone_elses_data').mkdir()
    backup.prune_old(0, str(root))
    assert (root / 'someone_elses_data').exists()


# --------------------------------------------------------------- exit codes

def test_an_empty_data_directory_fails_rather_than_reporting_success(tmp_path, monkeypatch, capsys):
    """A daily job that silently backs up nothing is worse than no job."""
    empty = tmp_path / 'empty'
    empty.mkdir()
    monkeypatch.setattr(sys, 'argv',
                        ['backup_state.py', '--data-dir', str(empty),
                         '--dest', str(tmp_path / 'b')])
    assert backup.main() == 1
    assert 'Nothing was backed up' in capsys.readouterr().err


def test_verify_only_on_a_missing_destination_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv',
                        ['backup_state.py', '--verify-only',
                         '--dest', str(tmp_path / 'nope')])
    assert backup.main() == 1


def test_a_successful_run_exits_zero(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv',
                        ['backup_state.py', '--data-dir', str(data_dir),
                         '--dest', str(tmp_path / 'b')])
    assert backup.main() == 0
