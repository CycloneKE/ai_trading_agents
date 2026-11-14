#!/usr/bin/env python3
"""Make a review pack: the backtests and the scorecard in one zip file.

Runs, one after the other, and keeps going if a step fails:

1. scripts/fetch_history.py   daily prices for the US-account symbols
                              (skipped with --skip-download)
2. scripts/run_backtest.py    the technical strategies (ensemble) on their
                              own list of symbols
3. scripts/run_backtest.py    the momentum sleeve (--strategy sleeve) on its
                              funds, with the leave-one-out check
4. scripts/agent_scorecard.py the paper run against its targets

Everything lands in data/reports/review_pack_<date>/ and a zip of it: under
data/, so in a Docker deployment it is on the data volume and survives a
redeploy. Send the zip (or SUMMARY.md and the reports inside) for review, or
run with --print to show them on screen for copying, which is the easy way
from a container terminal. The pack holds reports only: no keys, no
journals, no account numbers.

Usage:
    python scripts/review_pack.py
    python scripts/review_pack.py --print                # also show the reports on screen
    python scripts/review_pack.py --skip-download        # prices already in data/history
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def steps(cfg, pack, history, skip_download):
    """(name, command, what it is) for each step."""
    from src.agent.cost_model import classify
    py = sys.executable
    technical = [s for s in (cfg.get('data_manager') or {}).get('symbols') or [] if classify(s, cfg) != 'nse']
    sleeve = list((cfg.get('momentum_sleeve') or {}).get('universe') or [])
    out = []
    if not skip_download:
        out.append(('download', [py, 'scripts/fetch_history.py', '--out', history],
                    'Daily prices from Yahoo Finance'))
    out.append(('backtest_strategies',
                [py, 'scripts/run_backtest.py', '--csv-dir', history, '--symbols', ','.join(technical),
                 '--out', os.path.join(pack, 'backtest_strategies')],
                'The technical strategies on their own symbols'))
    if sleeve:
        out.append(('backtest_momentum_sleeve',
                    [py, 'scripts/run_backtest.py', '--csv-dir', history, '--symbols', ','.join(sleeve),
                     '--strategy', 'sleeve', '--leave-one-out',
                     '--out', os.path.join(pack, 'backtest_momentum_sleeve')],
                    'The 12-month momentum sleeve on its funds'))
    out.append(('scorecard', [py, 'scripts/agent_scorecard.py', '--out', os.path.join(pack, 'scorecard.md')],
                'The paper run against its targets'))
    return out


def main(argv=None, run=subprocess.run) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config/config.json')
    ap.add_argument('--history', default='data/history')
    ap.add_argument('--out', default='data/reports')
    ap.add_argument('--skip-download', action='store_true')
    ap.add_argument('--print', dest='show', action='store_true',
                    help='print the summary and the reports, to copy from a terminal')
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    with open(args.config) as f:
        cfg = json.load(f)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')
    pack = os.path.join(args.out, f'review_pack_{stamp}')
    logs = os.path.join(pack, 'logs')
    os.makedirs(logs, exist_ok=True)

    results = []
    for name, cmd, what in steps(cfg, pack, args.history, args.skip_download):
        print(f"\n== {what} ...", flush=True)
        with open(os.path.join(logs, f'{name}.log'), 'w') as log:
            try:
                code = run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
            except Exception as e:
                log.write(f"could not run: {e}\n")
                code = -1
        results.append((name, what, code))
        print(f"   {'done' if code == 0 else f'FAILED (exit {code}); see logs/{name}.log'}")

    lines = [f"# Review pack, {stamp} UTC", '',
             '| Step | What | Result |', '|---|---|---|']
    lines += [f"| {n} | {w} | {'done' if c == 0 else f'failed (exit {c}), see logs/{n}.log'} |"
              for n, w, c in results]
    lines += ['', 'Reports: backtest_strategies/report.md, backtest_momentum_sleeve/report.md, scorecard.md.',
              'Logs for every step are in logs/.']
    with open(os.path.join(pack, 'SUMMARY.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    archive = shutil.make_archive(pack, 'zip', root_dir=args.out, base_dir=os.path.basename(pack))
    if args.show:
        for rel in ('SUMMARY.md', 'backtest_strategies/report.md', 'backtest_momentum_sleeve/report.md',
                    'scorecard.md'):
            path = os.path.join(pack, rel)
            if os.path.exists(path):
                print(f"\n{'=' * 20} {rel} {'=' * 20}\n")
                with open(path) as f:
                    print(f.read())
    failed = [n for n, _, c in results if c != 0]
    print(f"\nReview pack: {archive}")
    if failed:
        print(f"Some steps failed: {', '.join(failed)}. The pack still holds what worked and the logs.")
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
