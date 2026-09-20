#!/usr/bin/env python3
"""Run the configured strategy ensemble over historical bars and report performance.

Drives the *real* StrategyManager (the same momentum / mean-reversion / RSI
strategies the live agent runs) through the *real* BacktestEngine, applying the
config's position-size cap, stop-loss and trailing stop. Nothing is simulated
that the live loop does differently, with two exceptions, both stated in the
report it writes: there is no LLM validation layer (it is non-deterministic and
rate-limited) and no news/sentiment input.

Usage:
    python scripts/run_backtest.py --csv-dir data/history --out reports/bt
    python scripts/run_backtest.py --csv-dir data/history --symbols SCOM,EQTY \
        --capital 500000 --benchmark SCOM

CSV format: one <SYMBOL>.csv per symbol, with a date column and at least a
close column. open/high/low/volume are used when present.
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.backtest_engine import BacktestEngine, Order, OrderSide, OrderType  # noqa: E402
from src.agent.strategy_manager import StrategyManager, MockStrategy  # noqa: E402

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('backtest')

TRADING_DAYS = 252


# ---------------------------------------------------------------- data loading

def load_csv_dir(path, symbols=None):
    """Load <SYMBOL>.csv files into {symbol: DataFrame(indexed by date)}."""
    out = {}
    if not os.path.isdir(path):
        raise SystemExit(f"--csv-dir not found: {path}")
    for fn in sorted(os.listdir(path)):
        if not fn.lower().endswith('.csv'):
            continue
        sym = os.path.splitext(fn)[0].upper()
        if symbols and sym not in symbols:
            continue
        df = pd.read_csv(os.path.join(path, fn))
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in ('date', 'timestamp', 'time', 'datetime')
                         if c in df.columns), None)
        if date_col is None:
            logger.warning("%s: no date column, skipped", fn)
            continue
        if 'close' not in df.columns:
            logger.warning("%s: no close column, skipped", fn)
            continue
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        df = df[~df.index.duplicated(keep='last')]
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        df = df[df['close'] > 0]
        if len(df) < 60:
            logger.warning("%s: only %d usable bars, skipped", fn, len(df))
            continue
        out[sym] = df
    if not out:
        raise SystemExit(f"No usable CSVs in {path}")
    return out


# ------------------------------------------------------------------- analytics

def _max_drawdown(equity):
    """Peak-to-trough decline as a positive fraction, plus its duration."""
    eq = np.asarray(equity, dtype=float)
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks
    trough = int(np.argmin(dd))
    peak = int(np.argmax(eq[:trough + 1])) if trough > 0 else 0
    recovery = None
    for i in range(trough, len(eq)):
        if eq[i] >= eq[peak]:
            recovery = i
            break
    return abs(float(dd.min())), peak, trough, recovery


def _round_trips(trades):
    """FIFO-match buys against sells into closed round trips with net P&L.

    The engine's own trade_win_rate reads a `pnl` attribute that Trade never
    sets, so it is always 0.0. This computes it properly, net of the
    commission and slippage actually charged on both legs.
    """
    lots, trips = {}, []
    for t in trades:
        q, price = float(t.quantity), float(t.price)
        cost_per_share = (t.commission / q) if q else 0.0
        if t.side == OrderSide.BUY:
            lots.setdefault(t.symbol, deque()).append(
                {'qty': q, 'price': price, 'fee': cost_per_share, 'ts': t.timestamp})
        else:
            book = lots.setdefault(t.symbol, deque())
            remaining = q
            while remaining > 1e-9 and book:
                lot = book[0]
                take = min(remaining, lot['qty'])
                gross = (price - lot['price']) * take
                fees = (cost_per_share + lot['fee']) * take
                trips.append({
                    'symbol': t.symbol, 'qty': take,
                    'entry': lot['price'], 'exit': price,
                    'pnl': gross - fees,
                    'entry_ts': lot['ts'], 'exit_ts': t.timestamp,
                    'bars_held': (t.timestamp - lot['ts']).days,
                })
                lot['qty'] -= take
                remaining -= take
                if lot['qty'] <= 1e-9:
                    book.popleft()
    return trips


def analyse(equity_series, trades, initial_capital, periods_per_year=TRADING_DAYS,
            risk_free=0.0):
    """Institutional metric set computed from the equity curve and round trips."""
    eq = pd.Series(equity_series).astype(float)
    rets = eq.pct_change().dropna()
    n = len(eq)
    years = max(n / periods_per_year, 1e-9)

    total_return = float(eq.iloc[-1] / initial_capital - 1.0)
    cagr = float((eq.iloc[-1] / initial_capital) ** (1 / years) - 1.0) if eq.iloc[-1] > 0 else -1.0
    vol = float(rets.std(ddof=1) * math.sqrt(periods_per_year)) if len(rets) > 1 else 0.0

    excess = rets - risk_free / periods_per_year
    sharpe = float(excess.mean() / rets.std(ddof=1) * math.sqrt(periods_per_year)) \
        if len(rets) > 1 and rets.std(ddof=1) > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(excess.mean() / downside.std(ddof=1) * math.sqrt(periods_per_year)) \
        if len(downside) > 1 and downside.std(ddof=1) > 0 else 0.0

    mdd, peak_i, trough_i, rec_i = _max_drawdown(eq.to_numpy())
    calmar = float(cagr / mdd) if mdd > 0 else 0.0

    trips = _round_trips(trades)
    wins = [t for t in trips if t['pnl'] > 0]
    losses = [t for t in trips if t['pnl'] <= 0]
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))

    return {
        'periods': n,
        'years': round(years, 2),
        'initial_capital': initial_capital,
        'final_equity': float(eq.iloc[-1]),
        'total_return': total_return,
        'cagr': cagr,
        'annual_volatility': vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': mdd,
        'max_drawdown_peak_idx': peak_i,
        'max_drawdown_trough_idx': trough_i,
        'max_drawdown_recovered': rec_i is not None,
        'calmar_ratio': calmar,
        'best_period': float(rets.max()) if len(rets) else 0.0,
        'worst_period': float(rets.min()) if len(rets) else 0.0,
        'positive_periods_pct': float((rets > 0).mean()) if len(rets) else 0.0,
        'executed_trades': len(trades),
        'round_trips': len(trips),
        'trade_win_rate': (len(wins) / len(trips)) if trips else 0.0,
        'avg_win': (gross_win / len(wins)) if wins else 0.0,
        'avg_loss': (-gross_loss / len(losses)) if losses else 0.0,
        'profit_factor': (gross_win / gross_loss) if gross_loss > 0 else float('inf') if gross_win > 0 else 0.0,
        'avg_hold_days': float(np.mean([t['bars_held'] for t in trips])) if trips else 0.0,
        '_round_trips': trips,
    }


def buy_and_hold(data, symbols, initial_capital):
    """Equal-weight buy-and-hold on the same bars, as the benchmark."""
    idx = sorted(set().union(*[set(data[s].index) for s in symbols]))
    per = initial_capital / len(symbols)
    units = {}
    for s in symbols:
        first = data[s]['close'].iloc[0]
        units[s] = per / first if first > 0 else 0.0
    curve = []
    last = {s: data[s]['close'].iloc[0] for s in symbols}
    for ts in idx:
        for s in symbols:
            if ts in data[s].index:
                last[s] = float(data[s].loc[ts, 'close'])
        curve.append(sum(units[s] * last[s] for s in symbols))
    return curve


# -------------------------------------------------------------------- strategy

def build_strategy_fn(cfg, manager, engine, stop_loss_pct, trailing_pct,
                      max_position_pct, min_confidence):
    """Bar-by-bar signal -> order translation, mirroring the live loop's rules."""
    peak_price = {}
    stats = {'signals': 0, 'stops': 0, 'blocked_cash': 0}

    def strategy(timestamp, prices, portfolio):
        orders = []
        pv = portfolio.get('portfolio_value', 0.0) or 0.0
        held = portfolio.get('positions', {})

        # 1. Stop-loss and trailing-stop enforcement, before any new entry.
        for sym, pos in list(held.items()):
            if sym not in prices or pos.get('quantity', 0) <= 0:
                continue
            price = float(prices[sym])
            engine_pos = engine.positions.get(sym)
            entry = float(getattr(engine_pos, 'avg_price', 0.0) or 0.0)
            peak_price[sym] = max(peak_price.get(sym, price), price)
            hit_stop = entry > 0 and price <= entry * (1 - stop_loss_pct)
            hit_trail = price <= peak_price[sym] * (1 - trailing_pct)
            if hit_stop or hit_trail:
                orders.append(Order(symbol=sym, side=OrderSide.SELL,
                                    order_type=OrderType.MARKET,
                                    quantity=pos['quantity'], timestamp=timestamp))
                peak_price.pop(sym, None)
                stats['stops'] += 1

        stopped = {o.symbol for o in orders}

        # 2. Strategy signals.
        for sym, price in prices.items():
            if sym in stopped:
                continue
            price = float(price)
            sig = manager.generate_signals({'symbol': sym, 'price': price,
                                            'close': price, 'timestamp': timestamp})
            action = (sig or {}).get('action', 'hold')
            conf = float((sig or {}).get('confidence', 0.0) or 0.0)
            if action == 'hold' or conf < min_confidence:
                continue
            stats['signals'] += 1

            pos = held.get(sym, {})
            qty_held = float(pos.get('quantity', 0.0) or 0.0)

            if action == 'buy':
                if qty_held > 0:
                    continue  # already long; this book does not pyramid
                target_value = pv * max_position_pct * min(max(conf, 0.0), 1.0)
                qty = math.floor(target_value / price) if price > 0 else 0
                if qty <= 0:
                    continue
                if qty * price > engine.cash:
                    qty = math.floor(engine.cash * 0.98 / price)
                    stats['blocked_cash'] += 1
                if qty <= 0:
                    continue
                orders.append(Order(symbol=sym, side=OrderSide.BUY,
                                    order_type=OrderType.MARKET,
                                    quantity=qty, timestamp=timestamp))
                peak_price[sym] = price
            elif action == 'sell' and qty_held > 0:
                orders.append(Order(symbol=sym, side=OrderSide.SELL,
                                    order_type=OrderType.MARKET,
                                    quantity=qty_held, timestamp=timestamp))
                peak_price.pop(sym, None)
        return orders

    strategy.stats = stats
    return strategy


# ------------------------------------------------------------------- reporting

def pct(x):
    return f"{x * 100:.2f}%"


def write_report(path, meta, strat, bench, strategy_stats, warnings_list):
    def row(label, a, b, fmt=pct):
        return f"| {label} | {fmt(a)} | {fmt(b)} |"

    lines = [
        "# Backtest Report",
        "",
        f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Run parameters",
        "",
        f"- Symbols: {', '.join(meta['symbols'])}",
        f"- Period: {meta['start']} to {meta['end']} ({strat['periods']} bars, {strat['years']} years)",
        f"- Initial capital: {meta['capital']:,.2f}",
        f"- Commission: {meta['commission']:.4%} per trade (min {meta['min_commission']})",
        f"- Slippage: {meta['slippage']:.4%} ({meta['slippage_model']} model)",
        f"- Max position size: {pct(meta['max_position_pct'])} of portfolio",
        f"- Stop loss: {pct(meta['stop_loss_pct'])}; trailing stop: {pct(meta['trailing_pct'])}",
        f"- Minimum signal confidence to act: {meta['min_confidence']:.2f}",
        "",
        "## Headline results",
        "",
        "| Metric | Strategy | Buy & hold |",
        "|---|---|---|",
        row("Total return", strat['total_return'], bench['total_return']),
        row("CAGR", strat['cagr'], bench['cagr']),
        row("Annualised volatility", strat['annual_volatility'], bench['annual_volatility']),
        row("Max drawdown", strat['max_drawdown'], bench['max_drawdown']),
        f"| Sharpe ratio | {strat['sharpe_ratio']:.2f} | {bench['sharpe_ratio']:.2f} |",
        f"| Sortino ratio | {strat['sortino_ratio']:.2f} | {bench['sortino_ratio']:.2f} |",
        f"| Calmar ratio | {strat['calmar_ratio']:.2f} | {bench['calmar_ratio']:.2f} |",
        f"| Final equity | {strat['final_equity']:,.2f} | {bench['final_equity']:,.2f} |",
        "",
        "## Trade statistics",
        "",
        f"- Orders executed: {strat['executed_trades']}",
        f"- Closed round trips: {strat['round_trips']}",
        f"- Trade win rate: {pct(strat['trade_win_rate'])}",
        f"- Average win: {strat['avg_win']:,.2f}",
        f"- Average loss: {strat['avg_loss']:,.2f}",
        f"- Profit factor: {strat['profit_factor']:.2f}",
        f"- Average holding period: {strat['avg_hold_days']:.1f} days",
        f"- Signals acted on: {strategy_stats['signals']}",
        f"- Stop-loss / trailing exits: {strategy_stats['stops']}",
        f"- Total commission paid: {meta['total_commission']:,.2f}",
        f"- Total slippage paid: {meta['total_slippage']:,.2f}",
        "",
        "## Risk detail",
        "",
        f"- Worst single bar: {pct(strat['worst_period'])}",
        f"- Best single bar: {pct(strat['best_period'])}",
        f"- Positive bars: {pct(strat['positive_periods_pct'])}",
        f"- Max drawdown recovered within the test window: "
        f"{'yes' if strat['max_drawdown_recovered'] else 'NO'}",
        "",
        "## Scope and limitations",
        "",
        "Read this before quoting any number above.",
        "",
    ]
    lines += [f"- {w}" for w in warnings_list]
    lines += [
        "",
        "A backtest is a hypothesis, not a track record. Past results do not",
        "predict future returns. Forward paper trading over a comparable period",
        "is the only honest validation before committing capital.",
        "",
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv-dir', required=True, help='directory of <SYMBOL>.csv files')
    ap.add_argument('--config', default='config/config.json')
    ap.add_argument('--symbols', help='comma-separated subset to test')
    ap.add_argument('--capital', type=float, default=None)
    ap.add_argument('--start'); ap.add_argument('--end')
    ap.add_argument('--min-confidence', type=float, default=0.0,
                    help='ignore signals weaker than this (live loop applies its own LLM gate instead)')
    ap.add_argument('--periods-per-year', type=int, default=TRADING_DAYS,
                    help='252 for daily bars, 52 weekly, 12 monthly')
    ap.add_argument('--out', default='reports/backtest')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    symbols = {s.strip().upper() for s in args.symbols.split(',')} if args.symbols else None
    data = load_csv_dir(args.csv_dir, symbols)
    syms = sorted(data)

    start = pd.to_datetime(args.start) if args.start else None
    end = pd.to_datetime(args.end) if args.end else None
    if start or end:
        for s in syms:
            df = data[s]
            if start is not None:
                df = df[df.index >= start]
            if end is not None:
                df = df[df.index <= end]
            data[s] = df
        syms = [s for s in syms if len(data[s]) >= 60]
        if not syms:
            raise SystemExit("No symbol has >= 60 bars in the requested window")
        data = {s: data[s] for s in syms}

    trading = cfg.get('trading', {})
    limits = cfg.get('risk_limits', {})
    capital = args.capital if args.capital is not None else trading.get('initial_capital', 100000)

    engine_cfg = {
        'initial_capital': capital,
        'commission_rate': trading.get('commission', 0.001),
        'min_commission': 1.0,
        'slippage_rate': trading.get('slippage', 0.0005),
        'slippage_model': 'linear',
        'market_impact_model': 'sqrt',
    }
    engine = BacktestEngine(engine_cfg)
    engine.add_market_data(data)

    manager = StrategyManager(cfg)
    mocks = [n for n, s in manager.strategies.items() if isinstance(s, MockStrategy)]
    if mocks:
        raise SystemExit(
            f"Refusing to run: {mocks} loaded as MockStrategy (always 'hold'), so the "
            f"result would be a flat line, not a backtest. Install the strategy "
            f"dependencies and retry.")

    strat_fn = build_strategy_fn(
        cfg, manager, engine,
        stop_loss_pct=limits.get('stop_loss_pct', 0.05),
        trailing_pct=limits.get('trailing_stop_pct', 0.03),
        max_position_pct=limits.get('max_position_size', 0.05),
        min_confidence=args.min_confidence)

    print(f"Running {', '.join(syms)} over "
          f"{min(data[s].index.min() for s in syms).date()} -> "
          f"{max(data[s].index.max() for s in syms).date()} ...")
    results = engine.run_backtest(strat_fn)
    if not results:
        raise SystemExit("Backtest produced no results; check logs.")

    equity = [p['portfolio_value'] for p in engine.portfolio_history]
    strat = analyse(equity, engine.trades, capital, args.periods_per_year)
    bench_curve = buy_and_hold(data, syms, capital)
    bench = analyse(bench_curve, [], capital, args.periods_per_year)

    warnings_list = [
        "No LLM validation layer: the live agent sends every candidate trade to an "
        "LLM that can veto it. That layer is non-deterministic and rate-limited, so it "
        "is excluded here. Live results will differ.",
        "No news or sentiment input, and no sector-specialist context.",
        "Long only. No shorting, no leverage, no options.",
        "Orders fill at the same bar's close plus modelled slippage. Intraday "
        "stop fills are therefore approximated at close, which flatters stop-loss exits.",
        "Survivorship: only symbols with usable history are included.",
        f"Costs modelled: {engine_cfg['commission_rate']:.3%} commission and "
        f"{engine_cfg['slippage_rate']:.3%} slippage per side. Verify these against "
        f"your own broker's schedule; NSE commissions are materially higher than US ones.",
    ]

    os.makedirs(args.out, exist_ok=True)
    meta = {
        'symbols': syms,
        'start': str(min(data[s].index.min() for s in syms).date()),
        'end': str(max(data[s].index.max() for s in syms).date()),
        'capital': capital,
        'commission': engine_cfg['commission_rate'],
        'min_commission': engine_cfg['min_commission'],
        'slippage': engine_cfg['slippage_rate'],
        'slippage_model': engine_cfg['slippage_model'],
        'max_position_pct': limits.get('max_position_size', 0.05),
        'stop_loss_pct': limits.get('stop_loss_pct', 0.05),
        'trailing_pct': limits.get('trailing_stop_pct', 0.03),
        'min_confidence': args.min_confidence,
        'total_commission': engine.total_commission,
        'total_slippage': engine.total_slippage,
    }

    write_report(os.path.join(args.out, 'report.md'), meta, strat, bench,
                 strat_fn.stats, warnings_list)

    dates = [p['timestamp'] for p in engine.portfolio_history]
    pd.DataFrame({'date': dates, 'strategy_equity': equity,
                  'buy_hold_equity': bench_curve}).to_csv(
        os.path.join(args.out, 'equity_curve.csv'), index=False)

    if strat['_round_trips']:
        pd.DataFrame(strat['_round_trips']).to_csv(
            os.path.join(args.out, 'round_trips.csv'), index=False)

    dump = {k: v for k, v in strat.items() if not k.startswith('_')}
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump({'meta': meta, 'strategy': dump,
                   'benchmark': {k: v for k, v in bench.items() if not k.startswith('_')},
                   'limitations': warnings_list}, f, indent=2, default=str)

    print(f"\n{'':<26}{'Strategy':>14}{'Buy & hold':>14}")
    for label, key, is_pct in [
            ('Total return', 'total_return', True), ('CAGR', 'cagr', True),
            ('Annualised volatility', 'annual_volatility', True),
            ('Max drawdown', 'max_drawdown', True),
            ('Sharpe ratio', 'sharpe_ratio', False),
            ('Sortino ratio', 'sortino_ratio', False),
            ('Calmar ratio', 'calmar_ratio', False),
            ('Final equity', 'final_equity', False)]:
        fa = pct(strat[key]) if is_pct else f"{strat[key]:,.2f}"
        fb = pct(bench[key]) if is_pct else f"{bench[key]:,.2f}"
        print(f"{label:<26}{fa:>14}{fb:>14}")
    print(f"\nRound trips: {strat['round_trips']}  "
          f"win rate: {pct(strat['trade_win_rate'])}  "
          f"profit factor: {strat['profit_factor']:.2f}")
    print(f"Report written to {os.path.join(args.out, 'report.md')}")


if __name__ == '__main__':
    main()
