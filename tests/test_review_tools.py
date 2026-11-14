"""The tools for reviewing the agent on the operator's own server:
scripts/fetch_history.py, the sleeve backtest (run_backtest --strategy
sleeve) and scripts/review_pack.py.

Before: the backtest needed hand-made CSV files, and its rotation mode
counted momentum in bars, so on daily prices the momentum sleeve's
"12-month" lookback was twelve days: a backtest of something the agent does
not trade.
"""
import csv
import importlib.util
import json
import os
import zipfile
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'scripts', f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config():
    with open(os.path.join(ROOT, 'config', 'config.json')) as f:
        return json.load(f)


# ------------------------------------------------------------ fetch_history

def test_it_downloads_every_us_account_symbol_and_no_nse_stock():
    fh = _script('fetch_history')
    cfg = _config()
    syms = fh.symbols_from_config(cfg)
    assert 'SPY' in syms and syms.count('SPY') == 1
    assert set(cfg['core_satellite']['core_symbols']) <= set(syms)
    assert set(cfg['momentum_sleeve']['universe']) <= set(syms)
    assert not set(cfg['data_manager'].get('nse_symbols') or []) & set(syms)
    mixed = {**cfg, 'data_manager': {**cfg['data_manager'], 'symbols': ['SCOM', 'AAPL', 'aapl']}}
    got = fh.symbols_from_config(mixed)
    assert 'SCOM' not in got and got.count('AAPL') == 1                  # Yahoo's SCOM is not Safaricom


def test_it_writes_one_csv_per_symbol_and_one_failure_stops_nothing(tmp_path):
    fh = _script('fetch_history')

    def get(sym, years):
        if sym == 'BAD':
            raise RuntimeError('rate limited')
        return [('2026-01-02', 10.0, 11.0, 9.5, 10.5, 1000.0), ('2026-01-05', 10.5, 10.9, 10.1, 10.8, 900.0)]
    done = fh.fetch(['AAPL', 'BAD', 'MSFT'], str(tmp_path), years=2, get=get)
    assert done == {'AAPL': 2, 'BAD': 0, 'MSFT': 2}
    with open(tmp_path / 'AAPL.csv') as f:
        rows = list(csv.DictReader(f))
    assert rows[0] == {'date': '2026-01-02', 'open': '10.0', 'high': '11.0', 'low': '9.5', 'close': '10.5',
                       'volume': '1000'}
    assert not (tmp_path / 'BAD.csv').exists()


# ------------------------------------------------------------ sleeve backtest

def _days(n, start=date(2022, 1, 3)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_the_sleeve_backtest_ranks_on_months_and_rebalances_monthly():
    """A fund that fell for a year but jumped in its last fortnight is not a
    12-month leader. Counting daily bars as months picked it."""
    rb = _script('run_backtest')
    from src.agent.momentum_sleeve import config_from_dict
    days = _days(400)
    n = len(days)
    steady = [100 * 1.001 ** i for i in range(n)]                       # rose all year
    jumper = [100 * 0.999 ** i for i in range(n - 10)] + \
             [100 * 0.999 ** (n - 10) * 1.02 ** k for k in range(1, 11)]   # fell, then a late jump
    falling = [100 * 0.998 ** i for i in range(n)]
    data = {s: pd.DataFrame({'close': c}, index=pd.to_datetime(days))
            for s, c in (('STEADY', steady), ('JUMPER', jumper), ('FALLING', falling))}
    cfg = {'momentum_sleeve': {'enabled': True, 'universe': list(data),
                               'rotation': {'top_n': 1, 'lookback': 12, 'skip_recent': 1}}}
    sleeve = config_from_dict(cfg['momentum_sleeve'])
    engine = SimpleNamespace(cash=100000.0)
    fn = rb.build_rotation_fn(cfg, engine, data, sleeve=sleeve)
    bought, rebalance_months = [], []
    for i, day in enumerate(days):
        ts = pd.Timestamp(day)
        prices = {s: float(df['close'].iloc[i]) for s, df in data.items()}
        orders = fn(ts, prices, {'portfolio_value': 100000.0, 'positions': {}})
        if orders:
            rebalance_months.append((ts.year, ts.month))
            bought += [o.symbol for o in orders]
    assert set(bought) == {'STEADY'}                                     # never the late jumper
    assert len(rebalance_months) == len(set(rebalance_months))           # at most once a month
    first = days[sleeve.bars_needed - 1]
    assert rebalance_months[0] >= (first.year, first.month)              # only once a year of history exists
    assert fn.stats['stops'] == 0


# ------------------------------------------------------------ review pack

def test_the_review_pack_runs_every_step_records_failures_and_zips_the_result(tmp_path):
    rp = _script('review_pack')
    ran = []

    def run(cmd, stdout=None, stderr=None, cwd=None):
        ran.append(cmd[1])
        if 'agent_scorecard.py' in cmd[1]:
            return SimpleNamespace(returncode=2)                         # a failing step does not stop the rest
        out = cmd[cmd.index('--out') + 1]
        os.makedirs(out if not out.endswith('.md') else os.path.dirname(out), exist_ok=True)
        return SimpleNamespace(returncode=0)

    code = rp.main(['--out', str(tmp_path), '--history', str(tmp_path / 'h')], run=run)
    assert code == 1
    assert ran == ['scripts/fetch_history.py', 'scripts/run_backtest.py', 'scripts/run_backtest.py',
                   'scripts/agent_scorecard.py']
    packs = [p for p in os.listdir(tmp_path) if p.endswith('.zip')]
    assert len(packs) == 1
    with zipfile.ZipFile(tmp_path / packs[0]) as z:
        summary = z.read(next(n for n in z.namelist() if n.endswith('SUMMARY.md'))).decode()
    assert '| scorecard | The paper run against its targets | failed (exit 2)' in summary
    assert '| backtest_momentum_sleeve | The 12-month momentum sleeve on its funds | done |' in summary


def test_the_pack_backtests_each_book_on_its_own_symbols():
    rp = _script('review_pack')
    cfg = _config()
    by_name = {name: cmd for name, cmd, _ in rp.steps(cfg, 'pack', 'hist', skip_download=True)}
    assert 'download' not in by_name
    tech = by_name['backtest_strategies'][by_name['backtest_strategies'].index('--symbols') + 1].split(',')
    assert 'AAPL' in tech and not set(cfg['momentum_sleeve']['universe']) & set(tech)
    sleeve_cmd = by_name['backtest_momentum_sleeve']
    assert sleeve_cmd[sleeve_cmd.index('--strategy') + 1] == 'sleeve' and '--leave-one-out' in sleeve_cmd
