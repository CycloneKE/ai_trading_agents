#!/usr/bin/env python3
"""Gate a paper run on whether it will actually produce evidence.

A ninety-day paper run that silently does nothing costs ninety days and
teaches nothing, and this codebase has three documented ways of doing
exactly that: strategies falling back to MockStrategy, the API failing to
import so nobody can see the run, and the data layer serving synthetic
fallback prices the agent correctly refuses to trade.

This refuses to start a run that would produce no evidence, and writes a
manifest recording what was started, when, and against which commit. The
manifest is what lets scripts/paper_run_report.py say "no trades in three
weeks" rather than "no trades ever", which are different problems.

Usage:
    # Check only. Exits 1 if the run would be pointless.
    python scripts/start_paper_run.py --csv-dir data/history

    # Check against live vendors and the broker, then start the agent.
    python scripts/start_paper_run.py --csv-dir data/history \
        --probe-data --probe-broker --days 90 --start

    # Check and record the run start, but launch the agent yourself
    # (Docker, systemd, or a terminal you keep open).
    python scripts/start_paper_run.py --days 90 --write-manifest

The checks that need the network (--probe-data, --probe-broker) are opt-in,
so the gate is still useful offline. What they cannot confirm they report as
a warning rather than a pass: an unanswered question is not a good answer.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.run_readiness import BLOCK, assess_readiness  # noqa: E402
from src.utils.config_validator import load_config  # noqa: E402
from src.utils.paths import DATA_DIR  # noqa: E402

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('start_paper_run')

RUNS_DIR = DATA_DIR / 'paper_runs'

TICK = '[ok]'
CROSS = '[BLOCK]'
WARNM = '[warn]'


# ------------------------------------------------------------------- probes

def probe_market_data(cfg, seconds=45):
    """Start the real data layer briefly and see what it returns.

    Returns {symbol: {'source': ..., 'close': ...}} so the readiness check can
    tell a real vendor quote from the synthetic fallback. Returns None if the
    data layer could not be started at all, which the caller reports as an
    unprobed warning rather than a pass.
    """
    try:
        from src.agent.data_manager import DataManager
    except Exception as e:
        logger.warning(f'DataManager unavailable: {e}')
        return None

    dm = None
    try:
        dm = DataManager(cfg.get('data_manager', {}))
        dm.start()
        deadline = time.time() + seconds
        sample = {}
        while time.time() < deadline:
            time.sleep(2)
            latest = dm.get_latest_data('market_data') or {}
            for source, payload in latest.items():
                for symbol, bar in (payload.get('data') or {}).items():
                    sample[str(symbol).upper()] = {
                        'source': source,
                        'close': (bar or {}).get('close'),
                    }
            if sample:
                break
        return sample or {}
    except Exception as e:
        logger.warning(f'Market data probe failed: {e}')
        return None
    finally:
        if dm is not None:
            try:
                dm.stop()
            except Exception:
                pass


def probe_broker(cfg):
    """Connect the configured brokers and hand back the primary one."""
    try:
        from src.agent.broker_manager import BrokerManager
    except Exception as e:
        logger.warning(f'BrokerManager unavailable: {e}')
        return None
    try:
        mgr = BrokerManager(cfg)
        mgr.connect_all()
        broker = getattr(mgr, 'primary_broker', None)
        if broker is None:
            for b in (getattr(mgr, 'brokers', {}) or {}).values():
                if getattr(b, 'is_connected', False):
                    return b
        return broker
    except Exception as e:
        logger.warning(f'Broker probe failed: {e}')
        return None


def load_bars(csv_dir, cfg):
    """Historical bars for the signal-path check, restricted to the universe."""
    if not csv_dir:
        return None
    from src.utils.price_files import load_price_dir
    dm = cfg.get('data_manager', {})
    universe = {s.upper() for s in
                (dm.get('symbols', []) or []) + (dm.get('nse_symbols', []) or [])}
    try:
        bars = load_price_dir(csv_dir, symbols=universe or None)
    except FileNotFoundError as e:
        logger.warning(str(e))
        return None
    if not bars:
        logger.warning(f'No usable CSVs in {csv_dir} for the configured universe')
    return bars or None


# ------------------------------------------------------------------ manifest

def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def write_manifest(cfg, readiness, args, path=None):
    """The run's birth certificate.

    Without a recorded start time, a report cannot distinguish a run that has
    traded nothing from a run that started an hour ago. Without a recorded
    commit, it cannot tell whether the code changed underneath it.
    """
    dm = cfg.get('data_manager', {})
    os.makedirs(RUNS_DIR, exist_ok=True)
    started = datetime.now(timezone.utc)
    manifest = {
        'started_at': started.isoformat(),
        'planned_days': args.days,
        'git_commit': git_commit(),
        'config_path': os.path.abspath(args.config),
        'universe': {
            'loop_symbols': sorted({s.upper() for s in dm.get('symbols', []) or []}),
            'slow_pass_symbols': sorted({s.upper() for s in dm.get('nse_symbols', []) or []}),
        },
        'ensemble_method': cfg.get('strategy_manager', {}).get('ensemble_method'),
        'brokers_enabled': sorted(
            n for n, b in (cfg.get('brokers') or {}).items()
            if b.get('enabled', n == 'paper_broker')),
        'started_with_blockers': [c.name for c in readiness.blockers],
        'warnings_at_start': [c.name for c in readiness.warnings],
        'checks': [{'level': c.level, 'name': c.name, 'ok': c.ok,
                    'detail': c.detail} for c in readiness.checks],
    }
    path = path or str(RUNS_DIR / f"run_{started.strftime('%Y%m%dT%H%M%SZ')}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return path


# -------------------------------------------------------------------- output

def render(readiness):
    lines = ['', 'Paper run readiness', '=' * 60]
    for c in readiness.checks:
        if c.ok:
            mark = TICK
        else:
            mark = CROSS if c.level == BLOCK else WARNM
        lines.append(f'{mark:9} {c.name}')
        if c.detail:
            lines.append(f'{"":9} {c.detail}')
    lines.append('=' * 60)
    if readiness.ready:
        lines.append(f'READY. {len(readiness.warnings)} warning(s) above do not '
                     f'stop the run but shape how to read its results.')
    else:
        lines.append(f'NOT READY. {len(readiness.blockers)} blocker(s). '
                     f'Starting now would produce a run with no evidence in it.')
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(
        description='Check whether a paper run would produce evidence, then start it.')
    p.add_argument('--config', default='config/config.json')
    p.add_argument('--csv-dir', help='historical bars, to test the signal path')
    p.add_argument('--probe-data', action='store_true',
                   help='start the data layer and check vendors return real prices')
    p.add_argument('--probe-broker', action='store_true',
                   help='connect the broker and check it accepts orders')
    p.add_argument('--probe-seconds', type=int, default=45,
                   help='how long to wait for the first vendor quotes')
    p.add_argument('--days', type=int, default=90, help='planned run length')
    p.add_argument('--start', action='store_true',
                   help='launch the agent if the checks pass')
    p.add_argument('--force', action='store_true',
                   help='start despite blockers; recorded in the manifest')
    p.add_argument('--manifest', help='where to write the run manifest')
    p.add_argument('--no-manifest', action='store_true')
    p.add_argument('--write-manifest', action='store_true',
                   help='record the run start without launching the agent, '
                        'for when you run it under Docker, systemd or by hand')
    p.add_argument('--json', action='store_true', help='machine-readable output')
    p.add_argument('--skip-api-check', action='store_true',
                   help='skip the api_server import check')
    args = p.parse_args()

    cfg = load_config(args.config)
    if not cfg:
        print(f'Could not load {args.config}', file=sys.stderr)
        return 2

    bars = load_bars(args.csv_dir, cfg)
    sample = probe_market_data(cfg, args.probe_seconds) if args.probe_data else None
    broker = probe_broker(cfg) if args.probe_broker else None

    readiness = assess_readiness(cfg, broker=broker, data_sample=sample,
                                 bars=bars,
                                 check_api_import=not args.skip_api_check)

    if args.json:
        print(json.dumps({
            'ready': readiness.ready,
            'checks': [{'level': c.level, 'name': c.name, 'ok': c.ok,
                        'detail': c.detail} for c in readiness.checks],
        }, indent=2))
    else:
        print(render(readiness))

    if not readiness.ready and not args.force:
        if not args.json:
            print('Fix the blockers above, or pass --force to start anyway.\n'
                  'Forcing is recorded in the manifest, so a later report can '
                  'say the run was started knowing it might produce nothing.')
        return 1

    if not (args.start or args.write_manifest):
        return 0

    if not args.no_manifest:
        path = write_manifest(cfg, readiness, args, args.manifest)
        if not args.json:
            print(f'Run manifest: {path}')

    if not args.start:
        if not args.json:
            print('\nManifest written. Start the agent however you normally '
                  'do; scripts/paper_run_report.py will date the run from '
                  'this manifest.')
        return 0

    if not args.json:
        print(f'\nStarting the agent for a planned {args.days} days.\n'
              f'Check progress with: python scripts/paper_run_report.py\n')
    os.execv(sys.executable,
             [sys.executable, 'main.py', '--config', args.config])
    return 0  # unreachable; execv replaces the process


if __name__ == '__main__':
    sys.exit(main())
