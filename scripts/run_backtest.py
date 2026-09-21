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
        'risk_free_rate': risk_free,
        # The number that actually matters: a Kenyan T-bill is the no-effort
        # alternative, so anything at or below it is worse than doing nothing.
        'excess_cagr': cagr - risk_free,
        'beats_risk_free': bool(cagr > risk_free),
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
                      max_position_pct, min_confidence, vol=None, data=None):
    """Bar-by-bar signal -> order translation, mirroring the live loop's rules."""
    from src.agent.position_sizing import volatility_scaled_value
    from src.agent.volatility import stop_distances
    peak_price = {}
    stats = {'signals': 0, 'stops': 0, 'blocked_cash': 0, 'atr_sized': 0}
    limits = cfg.get('risk_limits', {})
    risk_per_trade = cfg.get('trading', {}).get('risk_per_trade', 0.005)
    stop_mult = limits.get('stop_loss_atr_mult', 2.5)

    def strategy(timestamp, prices, portfolio):
        orders = []
        pv = portfolio.get('portfolio_value', 0.0) or 0.0
        held = portfolio.get('positions', {})

        # Feed this bar to the volatility tracker before it is consulted, so
        # stops and sizing reflect current conditions rather than last bar's.
        if vol is not None and data:
            for sym, price in prices.items():
                bar = data.get(sym)
                if bar is None or timestamp not in bar.index:
                    vol.update(sym, price)
                    continue
                row = bar.loc[timestamp]
                vol.update(sym, price,
                           row.get('high') if hasattr(row, 'get') else None,
                           row.get('low') if hasattr(row, 'get') else None)

        # 1. Stop-loss and trailing-stop enforcement, before any new entry.
        for sym, pos in list(held.items()):
            if sym not in prices or pos.get('quantity', 0) <= 0:
                continue
            price = float(prices[sym])
            engine_pos = engine.positions.get(sym)
            entry = float(getattr(engine_pos, 'avg_price', 0.0) or 0.0)
            peak_price[sym] = max(peak_price.get(sym, price), price)
            d = stop_distances(cfg, vol.atr_pct(sym) if vol else None)
            hit_stop = entry > 0 and price <= entry * (1 - d['stop_loss_pct'])
            hit_trail = price <= peak_price[sym] * (1 - d['trailing_stop_pct'])
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
                atr_pct = vol.atr_pct(sym) if vol else None
                if atr_pct:
                    stats['atr_sized'] += 1
                target_value = volatility_scaled_value(
                    pv, atr_pct, confidence=conf,
                    risk_per_trade=risk_per_trade,
                    stop_atr_mult=stop_mult,
                    max_position_pct=max_position_pct)
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


def build_rotation_fn(cfg, engine, data, rebalance_every=1):
    """Cross-sectional momentum rotation: rank the universe, hold the leaders,
    rebalance on a fixed schedule.

    Unlike the ensemble path this sees every symbol at once, which is the
    whole point: it is a relative-strength ranking, not a per-symbol verdict.
    """
    from src.agent.momentum_rotation import config_from_dict, rebalance_orders, target_weights

    rot_cfg = config_from_dict(cfg.get('momentum_rotation', {}))
    history = {}
    stats = {'signals': 0, 'stops': 0, 'blocked_cash': 0, 'atr_sized': 0,
             'rebalances': 0, 'periods_in_cash': 0}
    state = {'bar': 0}

    def strategy(timestamp, prices, portfolio):
        state['bar'] += 1
        for sym, price in prices.items():
            history.setdefault(sym, []).append(float(price))

        if (state['bar'] - 1) % max(1, rebalance_every) != 0:
            return []

        target = target_weights(history, rot_cfg)
        if not target:
            stats['periods_in_cash'] += 1

        pv = portfolio.get('portfolio_value', 0.0) or 0.0
        held = portfolio.get('positions', {})
        current = {sym: (pos.get('market_value', 0.0) / pv if pv > 0 else 0.0)
                   for sym, pos in held.items() if pos.get('quantity', 0)}

        deltas = rebalance_orders(current, target, pv, prices)
        if not deltas:
            return []
        stats['rebalances'] += 1
        stats['signals'] += len(deltas)

        # Sells first: they free the cash the buys need, and the engine
        # rejects a buy it cannot fund rather than borrowing.
        orders = []
        budget = engine.cash
        for sym, notional in sorted(deltas.items(), key=lambda kv: kv[1]):
            price = prices.get(sym)
            if not price or price <= 0:
                continue
            if notional < 0:
                qty = math.floor(min(abs(notional) / price,
                                     held.get(sym, {}).get('quantity', 0.0)))
                if qty > 0:
                    orders.append(Order(symbol=sym, side=OrderSide.SELL,
                                        order_type=OrderType.MARKET,
                                        quantity=qty, timestamp=timestamp))
                    # Proceeds fund the buys below. Discounted because the
                    # fill price and commission are not known until execution.
                    budget += qty * price * 0.98
            else:
                # Each buy draws from the running budget rather than the
                # opening balance; sizing them all against the same figure
                # is what made every order after the first unfundable.
                spend = min(notional, budget * 0.98)
                qty = math.floor(spend / price) if spend > 0 else 0
                if qty > 0:
                    orders.append(Order(symbol=sym, side=OrderSide.BUY,
                                        order_type=OrderType.MARKET,
                                        quantity=qty, timestamp=timestamp))
                    budget -= qty * price
                    stats['blocked_cash'] += 1 if spend < notional else 0
        return orders

    strategy.stats = stats
    return strategy


# ------------------------------------------------------------------- reporting

def pct(x):
    return f"{x * 100:.2f}%"


def verdict_lines(meta, strat):
    """Separate what the trading earned from what idle cash earned.

    A mostly-uninvested strategy inherits the risk-free rate through its cash
    balance and can clear the hurdle while its trades lose money. Reporting
    only the headline would read as success. This states the split.
    """
    interest = float(meta.get('interest_earned', 0.0))
    total_profit = strat['final_equity'] - meta['capital']
    trading_pnl = total_profit - interest

    lines = [
        f"- Total profit: {total_profit:,.2f}",
        f"- Of which interest on idle cash: {interest:,.2f}",
        f"- **Of which trading: {trading_pnl:,.2f}**",
        "",
    ]
    if trading_pnl <= 0:
        lines.append(
            f"**The trading lost money.** Every shilling of profit came from "
            f"interest on cash the strategy never deployed. Holding Treasury "
            f"bills and placing no trades at all would have returned more, "
            f"without the {strat['round_trips']} round trips.")
    elif interest > 0 and trading_pnl < 0.25 * abs(total_profit):
        lines.append(
            f"**Most of the return is interest, not skill.** Trading "
            f"contributed {trading_pnl / total_profit:.0%} of the profit; "
            f"the rest is the risk-free rate on undeployed cash.")
    elif strat['beats_risk_free']:
        lines.append(
            f"The strategy returned {pct(strat['cagr'])} a year against a "
            f"{pct(strat['risk_free_rate'])} hurdle, an excess of "
            f"{pct(strat['excess_cagr'])}, with trading contributing "
            f"{trading_pnl:,.2f}.")
    else:
        lines.append(
            f"The strategy returned {pct(strat['cagr'])} a year against a "
            f"{pct(strat['risk_free_rate'])} hurdle. **It does not clear it**, "
            f"so the capital is better left in a Treasury bill.")
    return lines


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
        f"- Risk-free hurdle: {pct(meta['risk_free_rate'])} a year",
        "",
        "Transaction costs applied, per side:",
        "",
        "| Market | Symbols | Commission | Slippage | Round trip | Verified |",
        "|---|---|---|---|---|---|",
    ] + meta['cost_rows'] + [
        "",
        f"- Max position size: {pct(meta['max_position_pct'])} of portfolio",
        f"- Stop loss: {pct(meta['stop_loss_pct'])}; trailing stop: {pct(meta['trailing_pct'])}",
        f"- Minimum signal confidence to act: {meta['min_confidence']:.2f}",
        "",
        "## Verdict",
        "",
    ] + verdict_lines(meta, strat) + [
        "",
        "## Headline results",
        "",
        "| Metric | Strategy | Buy & hold |",
        "|---|---|---|",
        row("Total return", strat['total_return'], bench['total_return']),
        row("CAGR", strat['cagr'], bench['cagr']),
        row("Excess over risk-free", strat['excess_cagr'], bench['excess_cagr']),
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
        f"- Entries sized from ATR (rest fell back to the flat cap): "
        f"{strategy_stats.get('atr_sized', 0)}",
    ] + ([f"- Rebalances: {strategy_stats['rebalances']}",
          f"- Periods held fully in cash: {strategy_stats['periods_in_cash']}"]
         if 'rebalances' in strategy_stats else []) + [
        f"- Total commission paid: {meta['total_commission']:,.2f}",
        f"- Interest earned on idle cash: {meta['interest_earned']:,.2f}",
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


# -------------------------------------------------------------- robustness

def leave_one_out(args, cfg, data, syms, capital, risk_free):
    """Re-run the strategy once per symbol, with that symbol removed.

    A backtest whose result depends on a single name is measuring that name,
    not the rule. On a five-symbol US universe over 2000-2010 the rotation
    returned 16.1% a year, beating buy-and-hold; dropping AAPL took it to
    6.7% against a 8.7% benchmark, and dropping AAPL and AMZN made it lose
    money. Nothing in the headline said so.
    """
    results = []
    for dropped in syms:
        kept = [s for s in syms if s != dropped]
        subset = {s: data[s] for s in kept}
        try:
            sub_engine = _build_engine(cfg, capital, risk_free, args)
            sub_engine.add_market_data(subset)
            fn = build_rotation_fn(cfg, sub_engine, subset,
                                   rebalance_every=args.rebalance_every) \
                if args.strategy == 'rotation' else None
            if fn is None:
                return []          # only meaningful for the rotation path today
            sub_engine.run_backtest(fn)
            eq = [p['portfolio_value'] for p in sub_engine.portfolio_history]
            a = analyse(eq, sub_engine.trades, capital, args.periods_per_year, risk_free)
            b = analyse(buy_and_hold(subset, kept, capital), [], capital,
                        args.periods_per_year, risk_free)
            results.append({
                'dropped': dropped,
                'cagr': a['cagr'],
                'benchmark_cagr': b['cagr'],
                'beats_benchmark': a['cagr'] > b['cagr'],
                'trading_pnl': a['final_equity'] - capital
                               - getattr(sub_engine, 'interest_earned', 0.0),
                'profit_factor': a['profit_factor'],
            })
        except Exception as e:
            logger.warning(f"Leave-one-out failed for {dropped}: {e}")
    return results


def format_leave_one_out(results, full_cagr):
    if not results:
        return ""
    lines = ["Leave-one-out robustness (full universe CAGR "
             f"{pct(full_cagr)}):", "",
             f"  {'dropped':<10}{'CAGR':>9}{'vs B&H':>10}{'trading P&L':>15}{'PF':>7}"]
    for r in sorted(results, key=lambda r: r['cagr']):
        flag = "" if r['beats_benchmark'] else "   <- loses to buy & hold"
        lines.append(f"  {r['dropped']:<10}{pct(r['cagr']):>9}"
                     f"{pct(r['benchmark_cagr']):>10}{r['trading_pnl']:>15,.0f}"
                     f"{r['profit_factor']:>7.2f}{flag}")
    worst = min(results, key=lambda r: r['cagr'])
    losers = [r for r in results if not r['beats_benchmark']]
    lines += ["", f"  Worst case drops CAGR to {pct(worst['cagr'])} "
                  f"(removing {worst['dropped']})."]
    if losers:
        lines.append(f"  {len(losers)} of {len(results)} subsets LOSE to buy & hold. "
                     f"The headline rests on specific names, not the rule.")
    else:
        lines.append("  Every subset still beats buy & hold, so the result does "
                     "not rest on any single name.")
    return "\n".join(lines)


def _build_engine(cfg, capital, risk_free, args):
    trading = cfg.get('trading', {})
    return BacktestEngine({
        'initial_capital': capital,
        'commission_rate': trading.get('commission', 0.001),
        'min_commission': 0.0,
        'slippage_rate': trading.get('slippage', 0.0005),
        'slippage_model': 'linear',
        'market_impact_model': 'sqrt',
        'cost_config': cfg,
        'cash_yield': risk_free,
        'periods_per_year': args.periods_per_year,
    })


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
    ap.add_argument('--risk-free-rate', type=float, default=None,
                    help='annual risk-free rate as a decimal, e.g. 0.10 for 10%%. '
                         'Defaults to config analytics.risk_free_rate. For a Kenyan '
                         'investor this is the 91-day T-bill: a strategy that cannot '
                         'beat it after costs is not worth running.')
    ap.add_argument('--strategy', choices=['ensemble', 'rotation'], default='ensemble',
                    help="'ensemble' drives the live StrategyManager bar by bar; "
                         "'rotation' runs cross-sectional momentum, ranking the "
                         "whole universe and rebalancing on a schedule.")
    ap.add_argument('--rebalance-every', type=int, default=1,
                    help='bars between rotation rebalances (1 for monthly bars, '
                         '~21 for daily). Ignored for --strategy ensemble.')
    ap.add_argument('--leave-one-out', action='store_true',
                    help='after the main run, re-run once per symbol with that '
                         'symbol removed, and report the spread. A result that '
                         'collapses when one name is dropped rests on that name, '
                         'not on the rule.')
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
    risk_free = args.risk_free_rate if args.risk_free_rate is not None \
        else cfg.get('analytics', {}).get('risk_free_rate', 0.0)
    capital = args.capital if args.capital is not None else trading.get('initial_capital', 100000)

    engine_cfg = {
        'initial_capital': capital,
        # Flat fallbacks, used only for a symbol no market claims.
        'commission_rate': trading.get('commission', 0.001),
        'min_commission': 0.0,
        'slippage_rate': trading.get('slippage', 0.0005),
        'slippage_model': 'linear',
        'market_impact_model': 'sqrt',
        # Real costs are resolved per market from here. An NSE round trip is
        # several percent against Alpaca's zero on US equities; pricing both
        # at one rate is what made earlier NSE numbers meaningless.
        'cost_config': cfg,
        # Idle cash earns the hurdle rate, so a strategy that sits out of the
        # market is compared fairly against simply holding Treasury bills.
        'cash_yield': risk_free,
        'periods_per_year': args.periods_per_year,
    }
    engine = BacktestEngine(engine_cfg)
    engine.add_market_data(data)

    if args.strategy == 'ensemble':
        manager = StrategyManager(cfg)
        mocks = [n for n, s in manager.strategies.items() if isinstance(s, MockStrategy)]
        if mocks:
            raise SystemExit(
                f"Refusing to run: {mocks} loaded as MockStrategy (always 'hold'), so the "
                f"result would be a flat line, not a backtest. Install the strategy "
                f"dependencies and retry.")
    else:
        manager = None

    # Feed a volatility tracker from the same bars, so stops and sizing use
    # each symbol's own ATR rather than one flat percentage for all of them.
    from src.agent.volatility import VolatilityTracker
    vol = VolatilityTracker(length=limits.get('atr_length', 14))

    if args.strategy == 'rotation':
        strat_fn = build_rotation_fn(cfg, engine, data,
                                     rebalance_every=args.rebalance_every)
    else:
        strat_fn = build_strategy_fn(
            cfg, manager, engine,
            stop_loss_pct=limits.get('stop_loss_pct', 0.05),
            trailing_pct=limits.get('trailing_stop_pct', 0.03),
            max_position_pct=limits.get('max_position_size', 0.05),
            min_confidence=args.min_confidence, vol=vol, data=data)

    print(f"[{args.strategy}] Running {', '.join(syms)} over "
          f"{min(data[s].index.min() for s in syms).date()} -> "
          f"{max(data[s].index.max() for s in syms).date()} ...")
    results = engine.run_backtest(strat_fn)
    if not results:
        raise SystemExit("Backtest produced no results; check logs.")

    equity = [p['portfolio_value'] for p in engine.portfolio_history]
    strat = analyse(equity, engine.trades, capital, args.periods_per_year, risk_free)
    bench_curve = buy_and_hold(data, syms, capital)
    bench = analyse(bench_curve, [], capital, args.periods_per_year, risk_free)

    warnings_list = [
        "No LLM validation layer: the live agent sends every candidate trade to an "
        "LLM that can veto it. That layer is non-deterministic and rate-limited, so it "
        "is excluded here. Live results will differ.",
        "No news or sentiment input, and no sector-specialist context.",
        "Long only. No shorting, no leverage, no options.",
        "Idle cash accrues at the risk-free rate, so a strategy that stays out of "
        "the market is not scored as earning zero. Set --risk-free-rate 0 to disable.",
        "Orders fill at the same bar's close plus modelled slippage. Intraday "
        "stop fills are therefore approximated at close, which flatters stop-loss exits.",
        "Survivorship: only symbols with usable history are included.",
        f"Costs modelled: {engine_cfg['commission_rate']:.3%} commission and "
        f"{engine_cfg['slippage_rate']:.3%} slippage per side. Verify these against "
        f"your own broker's schedule; NSE commissions are materially higher than US ones.",
    ]

    # Per-market cost table, plus a loud warning for any market still priced
    # from a placeholder schedule.
    from src.agent.cost_model import classify, costs_for, round_trip_pct, unverified_markets
    by_market = {}
    for sym in syms:
        by_market.setdefault(classify(sym, cfg), []).append(sym)
    cost_rows = []
    for market, market_syms in sorted(by_market.items()):
        c = costs_for(market_syms[0], cfg)
        shown = ', '.join(market_syms[:4]) + (' ...' if len(market_syms) > 4 else '')
        cost_rows.append(
            f"| {market} | {shown} | {c.get('commission_pct', 0):.3%} | "
            f"{c.get('slippage_pct', 0):.3%} | {round_trip_pct(market_syms[0], cfg):.2%} | "
            f"{'yes' if c.get('verified') else '**NO**'} |")

    unverified = {m: note for m, note in unverified_markets(cfg).items() if m in by_market}
    for market, note in unverified.items():
        warnings_list.insert(0, f"**Costs for `{market}` are unverified.** {note}")

    os.makedirs(args.out, exist_ok=True)
    meta = {
        'symbols': syms,
        'risk_free_rate': risk_free,
        'cost_rows': cost_rows,
        'unverified_markets': unverified,
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
        'interest_earned': getattr(engine, 'interest_earned', 0.0),
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
    print(f"{'Excess over risk-free':<26}{pct(strat['excess_cagr']):>14}"
          f"{pct(bench['excess_cagr']):>14}")
    print(f"\nRound trips: {strat['round_trips']}  "
          f"win rate: {pct(strat['trade_win_rate'])}  "
          f"profit factor: {strat['profit_factor']:.2f}")
    interest = meta['interest_earned']
    trading_pnl = strat['final_equity'] - capital - interest
    print(f"\nProfit split: {trading_pnl:,.2f} from trading, "
          f"{interest:,.2f} from interest on idle cash")
    if trading_pnl <= 0:
        print("Verdict: THE TRADING LOST MONEY. All profit is interest on "
              "undeployed cash; holding T-bills and trading nothing beats this.")
    else:
        verdict = "CLEARS" if strat['beats_risk_free'] else "DOES NOT CLEAR"
        print(f"Verdict: {verdict} the {pct(risk_free)} risk-free hurdle.")
    for market in unverified:
        print(f"WARNING: costs for '{market}' are a placeholder, not a verified "
              f"broker schedule. Treat these numbers as provisional.")
    if args.leave_one_out and len(syms) > 2:
        loo = leave_one_out(args, cfg, data, syms, capital, risk_free)
        print("\n" + format_leave_one_out(loo, strat['cagr']))
        with open(os.path.join(args.out, 'leave_one_out.json'), 'w') as f:
            json.dump(loo, f, indent=2, default=str)

    print(f"\nReport written to {os.path.join(args.out, 'report.md')}")


if __name__ == '__main__':
    main()
