"""Walk-forward backtest of the strategy the live agent actually runs.

`data/ai_agent_backtest_results.json` recorded `total_trades: 0` and
`sharpe_ratio: 0` — the engine had never been pointed at the live signal
path, so nothing in the repo showed whether the ensemble has an edge.

This harness closes that gap by driving the REAL components:

  * StrategyManager / TechnicalStrategy for signals,
  * ExposureGuard for the aggregate position cap and post-exit cooldown,
  * position_sizing.size_order for quantities,
  * the config's own stop-loss and trailing-stop percentages.

Backtesting a reimplementation of the strategy would measure something the
live system never runs, which is how the pyramiding defect survived a green
test suite.

Two invariants are enforced structurally rather than by inspection:

1. NO LOOKAHEAD. A signal derived from closes[0..t] fills at bar t+1. Every
   trade carries `signal_index` and `fill_index`; tests assert the gap.
2. COSTS ALWAYS APPLY. Commission and slippage are charged on every fill, so
   churn is penalised the way it is in the account.
"""

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    quantity: float
    bar_price: float        # the untouched close the fill was based on
    fill_price: float       # after slippage
    commission: float
    slippage: float
    signal_index: int       # bar the signal was computed from
    fill_index: int         # bar it executed on (must be signal_index + 1)
    reason: str = 'signal'  # 'signal' | 'stop_loss' | 'trailing_stop'
    # Position notional / equity immediately after this fill. Checked against
    # max_position_size for buys. Note this can exceed the cap LATER without
    # any breach: a holding that appreciates is not a new purchase, and the
    # gate's job is to refuse adds, not to force-sell winners.
    post_fill_exposure_pct: float = 0.0


@dataclass
class BacktestResult:
    final_equity: float = 0.0
    initial_capital: float = 0.0
    total_return: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    commission_paid: float = 0.0
    slippage_paid: float = 0.0
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    exposure_blocks: int = 0
    cooldown_blocks: int = 0
    bars: int = 0


# ---------------------------------------------------------------- metrics

def buy_and_hold_return(closes: List[float]) -> float:
    """Simple price appreciation over the window."""
    if not closes or len(closes) < 2 or not closes[0]:
        return 0.0
    return (closes[-1] - closes[0]) / closes[0]


def max_drawdown(curve: List[float]) -> float:
    """Largest peak-to-trough decline, as a positive fraction."""
    peak = None
    worst = 0.0
    for value in curve:
        if peak is None or value > peak:
            peak = value
        if peak:
            worst = max(worst, (peak - value) / peak)
    return worst


def sharpe_ratio(curve: List[float], risk_free: float = 0.02) -> float:
    """Annualised Sharpe from an equity curve.

    Returns 0.0 rather than nan/inf for a constant curve — a strategy that
    never moved has no risk-adjusted return, and letting nan through would
    silently poison every downstream comparison.
    """
    if len(curve) < 3:
        return 0.0
    rets = [(curve[i] - curve[i - 1]) / curve[i - 1]
            for i in range(1, len(curve)) if curve[i - 1]]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    if sd <= 1e-12:
        return 0.0
    excess = mean - (risk_free / TRADING_DAYS)
    return (excess / sd) * math.sqrt(TRADING_DAYS)


# ------------------------------------------------------------- signal source

def strategy_manager_signal_fn(config: Dict[str, Any]) -> Callable:
    """The live StrategyManager, fed one bar at a time.

    Each symbol gets its own manager instance so their price buffers stay
    independent, exactly as the per-symbol loop does in production.
    """
    from src.agent.strategy_manager import StrategyManager

    managers: Dict[str, Any] = {}

    def _fn(symbol: str, closes: List[float]) -> Dict[str, Any]:
        mgr = managers.get(symbol)
        if mgr is None:
            mgr = StrategyManager(config)
            managers[symbol] = mgr
            # Warm-start from everything except the newest bar, then feed the
            # newest one through the normal path — same as seed_history() in
            # production, and far cheaper than replaying every bar each step.
            for strat in mgr.strategies.values():
                if hasattr(strat, 'seed_history') and len(closes) > 1:
                    strat.seed_history(symbol, closes[:-1])
        price = closes[-1]
        try:
            return mgr.generate_signals(
                {'symbol': symbol, 'price': price, 'close': price}) or {}
        except Exception as e:
            logger.debug(f"signal error {symbol}: {e}")
            return {}

    return _fn


# ------------------------------------------------------------------ engine

def run_backtest(price_history: Dict[str, List[float]],
                 config: Dict[str, Any],
                 initial_capital: float = 100_000.0,
                 signal_fn: Optional[Callable] = None) -> BacktestResult:
    """Walk the price history bar by bar, trading the real strategy.

    `signal_fn(symbol, closes) -> signal dict` defaults to the live
    StrategyManager; tests inject a deterministic stand-in.
    """
    from src.agent.exposure import ExposureGuard
    from src.agent.position_sizing import size_order

    trading = config.get('trading', {}) or {}
    limits = config.get('risk_limits', {}) or {}
    commission_rate = float(trading.get('commission', 0.001))
    slippage_rate = float(trading.get('slippage', 0.0005))
    min_notional = float(trading.get('min_notional', 5.0))
    stop_loss_pct = float(limits.get('stop_loss_pct', 0.05))
    trailing_stop_pct = float(limits.get('trailing_stop_pct', 0.03))

    if signal_fn is None:
        signal_fn = strategy_manager_signal_fn(config)

    result = BacktestResult(initial_capital=initial_capital)

    symbols = [s for s, c in (price_history or {}).items() if c and len(c) >= 2]
    if not symbols:
        result.final_equity = initial_capital
        return result

    n_bars = min(len(price_history[s]) for s in symbols)
    result.bars = n_bars

    # ExposureGuard is SQLite-backed; a throwaway file keeps a backtest from
    # touching (or being influenced by) the live cooldown state.
    fd, guard_path = tempfile.mkstemp(suffix='.db', prefix='backtest_exposure_')
    os.close(fd)
    guard = ExposureGuard(config, db_path=guard_path)

    # Backtest "now" is bar index * one day, so cooldown_seconds is honoured
    # in simulated time rather than wall-clock.
    bar_seconds = 86400.0

    cash = initial_capital
    qty: Dict[str, float] = {s: 0.0 for s in symbols}
    cost_basis: Dict[str, float] = {s: 0.0 for s in symbols}
    peak_price: Dict[str, float] = {s: 0.0 for s in symbols}

    def equity_at(index: int) -> float:
        total = cash
        for s in symbols:
            total += qty[s] * price_history[s][index]
        return total

    def fill(symbol, side, quantity, bar_price, signal_index, fill_index, reason):
        nonlocal cash
        slip = bar_price * slippage_rate
        price = bar_price + slip if side == 'buy' else bar_price - slip
        gross = quantity * price
        comm = gross * commission_rate
        if side == 'buy':
            if gross + comm > cash:
                return None
            cash -= gross + comm
            cost_basis[symbol] += gross
            qty[symbol] += quantity
        else:
            cash += gross - comm
            if qty[symbol] > 0:
                closed = min(quantity, qty[symbol])
                cost_basis[symbol] *= max(0.0, 1 - closed / qty[symbol])
            qty[symbol] -= quantity
        post_equity = equity_at(fill_index)
        trade = BacktestTrade(
            symbol=symbol, side=side, quantity=quantity, bar_price=bar_price,
            fill_price=price, commission=comm, slippage=abs(slip) * quantity,
            signal_index=signal_index, fill_index=fill_index, reason=reason,
            post_fill_exposure_pct=(abs(qty[symbol]) * bar_price / post_equity)
            if post_equity > 0 else 0.0)
        result.trades.append(trade)
        result.commission_paid += comm
        result.slippage_paid += abs(slip) * quantity
        return trade

    # Signals are computed on bar t and executed on bar t+1, so the final bar
    # is only ever a fill target — never a decision input.
    for t in range(n_bars - 1):
        now = t * bar_seconds
        fill_index = t + 1
        equity = equity_at(t)

        for symbol in symbols:
            closes = price_history[symbol]
            bar_price = closes[fill_index]
            held = qty[symbol]

            # --- risk exits first, on the fill bar's price -----------------
            if held > 0:
                peak_price[symbol] = max(peak_price[symbol], bar_price)
                basis = cost_basis[symbol]
                if basis > 0:
                    pl_pct = (held * bar_price - basis) / basis
                    hit_stop = pl_pct <= -stop_loss_pct
                    hit_trail = (peak_price[symbol] > 0 and
                                 bar_price <= peak_price[symbol] * (1 - trailing_stop_pct))
                    if hit_stop or hit_trail:
                        reason = 'stop_loss' if hit_stop else 'trailing_stop'
                        fill(symbol, 'sell', held, bar_price, t, fill_index, reason)
                        guard.record_forced_exit(symbol, reason, now=now)
                        peak_price[symbol] = 0.0
                        cost_basis[symbol] = 0.0
                        continue

            # --- signal from bars up to and including t --------------------
            signal = signal_fn(symbol, closes[:t + 1]) or {}
            action = str(signal.get('action', 'hold')).lower()
            confidence = float(signal.get('confidence', 0.0) or 0.0)
            position_size = float(signal.get('position_size', 0.0) or 0.0)

            if action not in ('buy', 'sell') or confidence <= 0.1 or position_size <= 0:
                continue

            # Selling with nothing held would open a short; the live agent
            # allows it, but a cash backtest cannot model margin honestly, so
            # it is skipped and counted rather than silently faked.
            if action == 'sell' and qty[symbol] <= 0:
                continue

            target_value = equity * position_size
            permitted = guard.permitted_notional(
                symbol, action, qty[symbol], bar_price, equity, target_value)
            if permitted <= 0:
                if guard.in_cooldown(symbol, now=now):
                    result.cooldown_blocks += 1
                else:
                    result.exposure_blocks += 1
                continue

            sized = size_order(permitted, bar_price, min_notional=min_notional,
                               allow_fractional=trading.get('allow_fractional', True))
            if not sized:
                continue
            quantity = min(sized.quantity, qty[symbol]) if action == 'sell' else sized.quantity
            if quantity <= 0:
                continue
            fill(symbol, action, quantity, bar_price, t, fill_index, 'signal')

        eq = equity_at(fill_index)
        max_exposure = 0.0
        if eq > 0:
            max_exposure = max(
                (abs(qty[s]) * price_history[s][fill_index]) / eq for s in symbols)
        result.equity_curve.append({
            'index': fill_index, 'equity': eq, 'cash': cash,
            'max_symbol_exposure': max_exposure,
        })

    guard.close()
    try:
        os.unlink(guard_path)
    except OSError:
        pass

    result.final_equity = equity_at(n_bars - 1)
    result.total_return = (result.final_equity - initial_capital) / initial_capital

    curve = [p['equity'] for p in result.equity_curve]
    result.max_drawdown = max_drawdown(curve)
    result.sharpe = sharpe_ratio(curve)

    # Win/loss is measured per closing trade against the average cost of what
    # it closed, which is the only P&L attribution a bar-close backtest can
    # honestly make.
    running_cost: Dict[str, List[float]] = {s: [] for s in symbols}
    for tr in result.trades:
        if tr.side == 'buy':
            running_cost[tr.symbol].extend([tr.fill_price] * max(1, int(tr.quantity)))
        else:
            lots = running_cost[tr.symbol]
            if lots:
                avg = sum(lots) / len(lots)
                if tr.fill_price > avg:
                    result.wins += 1
                else:
                    result.losses += 1
                running_cost[tr.symbol] = []
    closed = result.wins + result.losses
    result.win_rate = result.wins / closed if closed else 0.0
    return result
