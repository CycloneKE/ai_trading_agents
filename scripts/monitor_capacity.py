#!/usr/bin/env python3
"""Measure whether the agent can carry its universe, and what breaks first.

Adding symbols is cheap to type and not cheap to run. Every symbol costs
wall time and API quota on each cycle, and the failure modes are quiet:
past the LLM quota the orchestrator's cooldown skips validation, so trades
still execute but stop being checked; past the vendor rate limit the data
manager falls back to synthetic prices, which the agent correctly refuses
to trade on, so the universe simply stops trading. Neither raises an error
you would notice.

This measures the per-symbol cost on real code, then projects where the
cycle budget and the quota run out.

Usage:
    python scripts/monitor_capacity.py                    # measure and project
    python scripts/monitor_capacity.py --symbols 50       # ask about a size
    python scripts/monitor_capacity.py --measure-only
    python scripts/monitor_capacity.py --json capacity.json
"""

import argparse
import json
import logging
import os
import resource
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.capacity import (DEFAULT_PROVIDER_LIMITS, assess,  # noqa: E402
                                projection)

logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')


def _synthetic_walk(n, seed=7):
    """A seeded random walk, not a repeating pattern.

    An oscillation with a short period reads as a textbook range, which the
    regime filter answers by silencing the trend strategies. That produces a
    signal rate far below anything real and makes the quota ceiling look
    several times more generous than it is.
    """
    import random
    rng = random.Random(seed)
    price, out = 100.0, []
    for _ in range(n):
        price *= 1.0 + rng.gauss(0.0003, 0.018)
        out.append(max(1.0, price))
    return out


def measure_per_symbol(cfg, samples=200, warmup=60, prices=None,
                       source='synthetic'):
    """Time one symbol's signal generation, using the real StrategyManager.

    Times the agent's own work only. Vendor latency is deliberately excluded:
    it varies with the network and would swamp the per-symbol cost this is
    trying to isolate. Real cycles are therefore slower than this suggests.
    """
    from src.agent.strategy_manager import MockStrategy, StrategyManager

    manager = StrategyManager(cfg)
    mocks = [n for n, s in manager.strategies.items() if isinstance(s, MockStrategy)]
    if mocks:
        print(f"WARNING: {mocks} loaded as MockStrategy, which does almost no "
              f"work. The timing below will be far too optimistic.\n")

    series = list(prices) if prices else _synthetic_walk(warmup + samples)
    if len(series) < warmup + 20:
        raise SystemExit(f"Need at least {warmup + 20} bars to measure; "
                         f"got {len(series)}.")
    warm, measured_bars = series[:warmup], series[warmup:]

    # Warm the buffers so the measurement covers the steady state, not the
    # cheap early cycles before lookback_period is reached.
    for price in warm:
        manager.generate_signals({'symbol': 'BENCH', 'price': price,
                                  'close': price, 'open': price})

    durations, signals = [], 0
    for price in measured_bars:
        start = time.perf_counter()
        out = manager.generate_signals({'symbol': 'BENCH', 'price': price,
                                        'close': price, 'open': price})
        durations.append(time.perf_counter() - start)
        if (out or {}).get('action', 'hold') != 'hold':
            signals += 1

    return {
        'median_seconds': statistics.median(durations),
        'mean_seconds': statistics.mean(durations),
        'p95_seconds': sorted(durations)[int(0.95 * len(durations)) - 1],
        'samples': len(durations),
        'observed_signal_rate': signals / len(durations) if durations else 0.0,
        'signal_rate_source': source,
        'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config/config.json')
    ap.add_argument('--symbols', type=int, default=None,
                    help='universe size to assess (default: what config holds)')
    ap.add_argument('--samples', type=int, default=200)
    ap.add_argument('--csv-dir',
                    help='measure the signal rate on real bars from here rather '
                         'than a synthetic walk. Strongly preferred: the signal '
                         'rate sets the LLM quota ceiling, and a synthetic '
                         'series gives a misleading one.')
    ap.add_argument('--measure-symbol',
                    help='which symbol in --csv-dir to measure (default: the '
                         'one with the most bars)')
    ap.add_argument('--per-symbol-seconds', type=float, default=None,
                    help='skip measurement and use this figure')
    ap.add_argument('--signal-rate', type=float, default=None,
                    help='share of symbols producing a signal per cycle '
                         '(default: measured)')
    ap.add_argument('--llm-provider', default=None,
                    help='provider whose quota to check (default: config '
                         'primary_llm_provider)')
    ap.add_argument('--measure-only', action='store_true')
    ap.add_argument('--strict', action='store_true',
                    help='exit non-zero on ANY finding, not just a cycle that '
                         'overruns its budget. Use this when running on a '
                         'schedule: an LLM quota overrun leaves the cycle well '
                         'inside its budget, so without this the job reports '
                         'success while validation silently switches off.')
    ap.add_argument('--json', help='write the full report here')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    dm = cfg.get('data_manager', {})
    configured = len({s.upper() for s in
                      (dm.get('symbols', []) + dm.get('nse_symbols', [])
                       + dm.get('crypto_symbols', []))})
    symbols = args.symbols if args.symbols is not None else configured
    budget = float(cfg.get('trading_loop_interval', 60))

    provider = args.llm_provider or cfg.get('primary_llm_provider', 'gemini')
    limits = {**DEFAULT_PROVIDER_LIMITS,
              **(cfg.get('capacity', {}).get('provider_limits', {}))}
    llm_limit = limits.get(provider, DEFAULT_PROVIDER_LIMITS['gemini'])
    llm_enabled = cfg.get('llm_enabled', True)

    measured = None
    if args.per_symbol_seconds is None:
        prices, source = None, 'synthetic'
        if args.csv_dir:
            from src.utils.price_files import load_price_dir
            data = load_price_dir(args.csv_dir)
            if not data:
                raise SystemExit(f"No usable price files in {args.csv_dir}")
            sym = (args.measure_symbol or '').upper() or max(data, key=lambda k: len(data[k]))
            if sym not in data:
                raise SystemExit(f"{sym} not found in {args.csv_dir}")
            prices = data[sym]['close'].tolist()
            source = f"real bars ({sym}, {len(prices)})"
        print(f"Measuring per-symbol cost (real StrategyManager, {source})...\n")
        measured = measure_per_symbol(cfg, samples=args.samples,
                                      prices=prices, source=source)
        per_symbol = measured['median_seconds']
        print(f"  median      {per_symbol * 1000:8.2f} ms per symbol")
        print(f"  mean        {measured['mean_seconds'] * 1000:8.2f} ms")
        print(f"  p95         {measured['p95_seconds'] * 1000:8.2f} ms")
        print(f"  signal rate {measured['observed_signal_rate']:8.1%} of cycles "
              f"({measured['signal_rate_source']})")
        print(f"  peak RSS    {measured['peak_rss_mb']:8.0f} MB\n")
    else:
        per_symbol = args.per_symbol_seconds

    signal_rate = (args.signal_rate if args.signal_rate is not None
                   else (measured or {}).get('observed_signal_rate', 0.4))

    if (args.signal_rate is None and measured
            and measured['signal_rate_source'] == 'synthetic'):
        print("NOTE: the signal rate above came from a synthetic series, and it "
              "is what sets the LLM quota ceiling below. On real GOOG daily bars "
              "this system signalled on about 43% of cycles. Re-run with "
              "--csv-dir, or pass --signal-rate, before trusting that ceiling.\n")

    if args.measure_only:
        return 0

    report = assess(symbols, per_symbol, cycle_budget_seconds=budget,
                    signal_rate=signal_rate, llm_limit_per_minute=llm_limit,
                    llm_enabled=llm_enabled)

    print(f"Universe of {symbols} symbols against a {budget:.0f}s cycle, "
          f"LLM provider '{provider}' at {llm_limit}/min:\n")
    print(f"  estimated cycle   {report.cycle_seconds:.1f}s")
    print(f"  headroom          {report.headroom_seconds:+.1f}s")
    print(f"  LLM calls/cycle   {report.detail['llm_calls_per_cycle']:.0f}")
    print(f"  ceiling by time   {report.max_symbols_by_time} symbols")
    print(f"  ceiling by quota  {report.max_symbols_by_llm} symbols")
    print(f"  binds first       {report.binding_constraint}\n")

    for f in report.findings:
        print(f"  ! {f}\n")
    if not report.findings:
        print("  No capacity concerns at this size.\n")

    rows = projection(per_symbol, 0.0, budget, signal_rate, llm_limit)
    print(f"{'symbols':>9}{'cycle s':>10}{'headroom':>10}{'LLM/cycle':>11}  status")
    for r in rows:
        flags = []
        if not r['fits']:
            flags.append('over cycle budget')
        if not r['llm_ok']:
            flags.append('over LLM quota')
        marker = ' <- current' if r['symbols'] == symbols else ''
        print(f"{r['symbols']:>9}{r['cycle_seconds']:>10.1f}"
              f"{r['headroom_seconds']:>+10.1f}{r['llm_calls']:>11.0f}  "
              f"{', '.join(flags) or 'ok'}{marker}")

    basis = (measured or {}).get('signal_rate_source', 'supplied figures')
    print(f"\nMeasured on this machine from {basis}, with no vendor latency, so "
          f"treat the time ceiling as an upper bound: real cycles also wait on "
          f"market data and broker calls. The quota ceiling is the one that "
          f"bites first and it is not an upper bound.")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or '.', exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump({'measured': measured, 'symbols': symbols,
                       'budget_seconds': budget, 'provider': provider,
                       'llm_limit_per_minute': llm_limit,
                       'signal_rate': signal_rate,
                       'report': report.__dict__, 'projection': rows},
                      f, indent=2, default=str)
        print(f"\nWritten to {args.json}")

    if not report.within_budget:
        return 2
    if args.strict and report.findings:
        print(f"\n{len(report.findings)} finding(s) above. Failing because "
              f"--strict was given.", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
