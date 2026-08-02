"""Grid-search regime parameters on the FIRST HALF of history only.

The second half is the out-of-sample window and is evaluated exactly once, in
scripts/run_strategy_backtest.py, after these parameters are frozen. Looking
at it here would contaminate the only honest number this project has.

The grid is deliberately coarse: a fine search over four parameters is easy
to overfit, and the difference between 0.35 and 0.36 is noise at this sample
size.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.strategy_backtest import run_backtest  # noqa: E402

HISTORY = 'data/backtest_history.json'
CONFIG = 'config/config.json'
OUT = 'data/regime_calibration.json'

WINDOWS = [10, 20, 30, 40]
TREND = [0.30, 0.35, 0.40, 0.45]
RANGE = [0.10, 0.15, 0.20, 0.25]


def main():
    history = json.load(open(HISTORY, encoding='utf-8'))
    config = json.load(open(CONFIG, encoding='utf-8'))
    history = {s: c for s, c in history.items() if c and len(c) > 60}

    n = min(len(c) for c in history.values())
    first_half = {s: c[: n // 2] for s, c in history.items()}
    print(f'calibrating on bars 0..{n // 2} of {n} — second half NOT touched')

    results = []
    for window in WINDOWS:
        for trend in TREND:
            for rng in RANGE:
                if rng >= trend:
                    continue          # dead band must be non-empty
                cfg = json.loads(json.dumps(config))
                cfg['ensemble_method'] = 'regime_routed'
                cfg['regime'] = {'window': window, 'trend_threshold': trend,
                                 'range_threshold': rng}
                res = run_backtest(first_half, cfg, initial_capital=100_000.0)
                row = {'window': window, 'trend_threshold': trend,
                       'range_threshold': rng, 'sharpe': round(res.sharpe, 3),
                       'total_return': round(res.total_return, 4),
                       'trades': len(res.trades),
                       'max_drawdown': round(res.max_drawdown, 4)}
                results.append(row)
                print(f"  w={window:2d} t={trend:.2f} r={rng:.2f} -> "
                      f"sharpe {row['sharpe']:6.2f}  trades {row['trades']:4d}")

    # Require enough trades that the Sharpe means something. A 3-trade
    # parameter set with a great ratio is noise, not a strategy.
    viable = [r for r in results if r['trades'] >= 50]
    if not viable:
        print('\nNo parameter set produced 50+ trades on the first half.')
        print('Report this rather than lowering the bar.')
        json.dump({'grid': results, 'best': None}, open(OUT, 'w'), indent=2)
        return 1

    best = max(viable, key=lambda r: r['sharpe'])
    print(f"\nBEST (in-sample): window={best['window']} "
          f"trend={best['trend_threshold']} range={best['range_threshold']} "
          f"sharpe={best['sharpe']} trades={best['trades']}")
    json.dump({'grid': results, 'best': best}, open(OUT, 'w'), indent=2)
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
