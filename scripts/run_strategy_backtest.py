"""Backtest the live strategy against real history and report honestly.

    python scripts/run_strategy_backtest.py

Reads data/backtest_history.json (daily closes per symbol, oldest first) and
runs src.agent.strategy_backtest over it with the project's own config —
same StrategyManager, same ExposureGuard, same sizing, costs on.

Prints an in-sample run, a walk-forward split, and the SPY buy-and-hold
benchmark. Writes data/strategy_backtest_results.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.strategy_backtest import (  # noqa: E402
    buy_and_hold_return, run_backtest,
)

HISTORY = 'data/backtest_history.json'
OUT = 'data/strategy_backtest_results.json'
CONFIG = 'config/config.json'
CAPITAL = 100_000.0


def correlation(a, b):
    """Pearson correlation of two equal-length return series."""
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def daily_returns(values):
    return [(values[i] - values[i - 1]) / values[i - 1]
            for i in range(1, len(values)) if values[i - 1]]


ACCEPTANCE = {
    'min_sharpe': 0.5, 'max_abs_correlation': 0.3,
    'max_drawdown': 0.15, 'min_trades': 100,
}


def verdict(summary, baseline_sharpe):
    """Every criterion from the spec, reported individually."""
    checks = {
        'sharpe > 0.5': summary['sharpe'] > ACCEPTANCE['min_sharpe'],
        '|SPY correlation| < 0.3':
            abs(summary['correlation_to_spy']) < ACCEPTANCE['max_abs_correlation'],
        'max drawdown < 15%': summary['max_drawdown'] < ACCEPTANCE['max_drawdown'],
        'trades > 100': summary['trades'] > ACCEPTANCE['min_trades'],
        'beats baseline sharpe': summary['sharpe'] > baseline_sharpe,
    }
    return checks, all(checks.values())


def summarise(name, res, benchmark, spy_returns):
    excess = res.total_return - benchmark
    return {
        'window': name,
        'bars': res.bars,
        'trades': len(res.trades),
        'final_equity': round(res.final_equity, 2),
        'total_return': round(res.total_return, 4),
        'benchmark_return': round(benchmark, 4),
        'excess_return': round(excess, 4),
        'sharpe': round(res.sharpe, 3),
        'max_drawdown': round(res.max_drawdown, 4),
        'win_rate': round(res.win_rate, 3),
        'wins': res.wins,
        'losses': res.losses,
        'commission_paid': round(res.commission_paid, 2),
        'slippage_paid': round(res.slippage_paid, 2),
        'total_costs': round(res.commission_paid + res.slippage_paid, 2),
        'exposure_blocks': res.exposure_blocks,
        'cooldown_blocks': res.cooldown_blocks,
        'stop_exits': sum(1 for t in res.trades if t.reason == 'stop_loss'),
        'trailing_exits': sum(1 for t in res.trades if t.reason == 'trailing_stop'),
        'correlation_to_spy': round(
            correlation(daily_returns([p['equity'] for p in res.equity_curve]),
                        spy_returns), 3),
    }


def show(s):
    print(f"\n--- {s['window']} ({s['bars']} bars) ---")
    print(f"  strategy return   {s['total_return']:>9.2%}"
          f"   final ${s['final_equity']:,.0f}")
    print(f"  SPY buy & hold    {s['benchmark_return']:>9.2%}")
    print(f"  EXCESS            {s['excess_return']:>9.2%}")
    print(f"  sharpe            {s['sharpe']:>9.2f}"
          f"   max drawdown {s['max_drawdown']:.2%}")
    print(f"  trades            {s['trades']:>9,}"
          f"   win rate {s['win_rate']:.1%} ({s['wins']}W/{s['losses']}L)")
    print(f"  costs paid        ${s['total_costs']:>8,.0f}"
          f"   ({s['commission_paid']:,.0f} commission"
          f" + {s['slippage_paid']:,.0f} slippage)")
    print(f"  risk exits        {s['stop_exits']} stop-loss,"
          f" {s['trailing_exits']} trailing")
    print(f"  gate blocks       {s['exposure_blocks']} exposure,"
          f" {s['cooldown_blocks']} cooldown")


def main():
    if not os.path.exists(HISTORY):
        print(f"missing {HISTORY} — fetch daily closes first")
        return 1

    history = json.load(open(HISTORY, encoding='utf-8'))
    config = json.load(open(CONFIG, encoding='utf-8'))
    history = {s: c for s, c in history.items() if c and len(c) > 60}
    spy = history.get('SPY', [])

    print(f"symbols: {', '.join(sorted(history))}")
    print(f"bars per symbol: {min(len(c) for c in history.values())}")
    print(f"costs: {config['trading']['commission']:.2%} commission, "
          f"{config['trading']['slippage']:.2%} slippage")

    results = []

    full = run_backtest(history, config, initial_capital=CAPITAL)
    results.append(summarise('full period (in-sample)', full,
                             buy_and_hold_return(spy), daily_returns(spy)))

    # Walk-forward: the second half is data the parameters were never chosen
    # against. Nothing here is fitted, but the split still guards against a
    # regime-specific fluke.
    n = min(len(c) for c in history.values())
    mid = n // 2
    for label, sl in (('first half', slice(0, mid)), ('second half (out-of-sample)',
                                                      slice(mid, n))):
        window = {s: c[sl] for s, c in history.items()}
        res = run_backtest(window, config, initial_capital=CAPITAL)
        results.append(summarise(label, res, buy_and_hold_return(spy[sl]),
                                 daily_returns(spy[sl])))

    for s in results:
        show(s)

    # A/B on the out-of-sample half, with parameters frozen by Task 5.
    n = min(len(c) for c in history.values())
    oos = {s: c[n // 2:] for s, c in history.items()}
    oos_spy = daily_returns(spy[n // 2:])

    baseline_cfg = json.loads(json.dumps(config))
    baseline_cfg['ensemble_method'] = 'adaptive_confidence'
    baseline = summarise('OOS baseline (current ensemble)',
                         run_backtest(oos, baseline_cfg, initial_capital=CAPITAL),
                         buy_and_hold_return(spy[n // 2:]), oos_spy)

    routed_cfg = json.loads(json.dumps(config))
    routed_cfg['ensemble_method'] = 'regime_routed'
    routed = summarise('OOS regime_routed',
                       run_backtest(oos, routed_cfg, initial_capital=CAPITAL),
                       buy_and_hold_return(spy[n // 2:]), oos_spy)

    show(baseline)
    show(routed)

    checks, passed = verdict(routed, baseline['sharpe'])
    print('\n--- ACCEPTANCE GATE (out-of-sample) ---')
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  VERDICT: {'PASS — cleared for live config' if passed else 'FAIL — do not ship'}")

    results.extend([baseline, routed])
    json.dump({'results': results, 'acceptance': checks, 'passed': passed},
              open(OUT, 'w'), indent=2)
    print(f"\nwrote {OUT}")
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
