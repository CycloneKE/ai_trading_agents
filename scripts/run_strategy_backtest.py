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


def summarise(name, res, benchmark):
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
                             buy_and_hold_return(spy)))

    # Walk-forward: the second half is data the parameters were never chosen
    # against. Nothing here is fitted, but the split still guards against a
    # regime-specific fluke.
    n = min(len(c) for c in history.values())
    mid = n // 2
    for label, sl in (('first half', slice(0, mid)), ('second half (out-of-sample)',
                                                      slice(mid, n))):
        window = {s: c[sl] for s, c in history.items()}
        res = run_backtest(window, config, initial_capital=CAPITAL)
        results.append(summarise(label, res, buy_and_hold_return(spy[sl])))

    for s in results:
        show(s)

    json.dump({'results': results}, open(OUT, 'w'), indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
