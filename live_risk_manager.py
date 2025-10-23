"""
Live trading risk management.
"""

import logging
from typing import Dict, Any
import time
from math import floor

logger = logging.getLogger(__name__)

# Prometheus metrics (optional dependency). If prometheus_client isn't installed,
# provide no-op stand-ins so code can run in environments without the package.
try:
    from prometheus_client import Counter, Gauge

    portfolio_drawdown_gauge = Gauge(
        'live_portfolio_drawdown', 'Current portfolio drawdown fraction')
    stop_loss_trigger_counter = Counter(
        'live_portfolio_stop_loss_triggers_total', 'Number of times portfolio stop loss triggered')
    kelly_fraction_gauge = Gauge(
        'live_kelly_fraction', 'Most recent computed kelly fraction')
    trade_cooldown_counter = Counter(
        'live_trade_cooldown_events_total', 'Number of trade cooldown vetoes')
    can_trade_gauge = Gauge(
        'live_can_trade', '1 if symbol allowed to trade, 0 if in cooldown')

except Exception:
    # Define no-op replacements
    class _NoOp:
        def __call__(self, *a, **k):
            return None

        def set(self, *a, **k):
            return None

        def inc(self, *a, **k):
            return None

    portfolio_drawdown_gauge = _NoOp()
    stop_loss_trigger_counter = _NoOp()
    kelly_fraction_gauge = _NoOp()
    trade_cooldown_counter = _NoOp()
    can_trade_gauge = _NoOp()

class LiveRiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.max_position_size = config.get('max_position_size', 0.05)  # 5% max per position
        self.max_daily_loss = config.get('max_daily_loss', 0.02)  # 2% max daily loss
        self.max_total_exposure = config.get('max_total_exposure', 0.8)  # 80% max exposure
        # Portfolio-level stop-loss (fraction of portfolio equity) - e.g., 0.1 = stop trading if drawdown >10%
        self.portfolio_stop_loss = config.get('portfolio_stop_loss', 0.15)
        # Kelly sizing controls
        self.kelly_shrink = config.get('kelly_shrink', 0.5)  # shrink the Kelly fraction to be conservative
        self.kelly_cap = config.get('kelly_cap', 0.2)  # maximum fraction of portfolio to risk per Kelly
        # Anti-overtrade / cooldown settings (seconds)
        self.trade_cooldown_seconds = config.get('trade_cooldown_seconds', 60 * 5)  # default 5 minutes
        # Track last exit timestamp per symbol
        self._last_exit_ts: Dict[str, float] = {}
        
    def check_position_size(self, symbol: str, quantity: int, price: float, portfolio_value: float) -> bool:
        """Check if position size is within limits."""
        position_value = quantity * price
        position_ratio = position_value / portfolio_value
        
        if position_ratio > self.max_position_size:
            logger.warning(f"Position size too large: {position_ratio:.2%} > {self.max_position_size:.2%}")
            return False
        return True
    
    def check_daily_loss(self, current_pnl: float, portfolio_value: float) -> bool:
        """Check if daily loss exceeds limit."""
        loss_ratio = abs(current_pnl) / portfolio_value if current_pnl < 0 else 0
        
        if loss_ratio > self.max_daily_loss:
            logger.warning(f"Daily loss limit exceeded: {loss_ratio:.2%} > {self.max_daily_loss:.2%}")
            return False
        return True

    def check_portfolio_stop_loss(self, peak_portfolio_value: float, current_portfolio_value: float) -> bool:
        """Check portfolio-level stop-loss based on peak-to-current drawdown.

        Returns False if drawdown exceeds configured `portfolio_stop_loss` (stop trading / escalate).
        """
        if peak_portfolio_value <= 0:
            return True
        drawdown = (peak_portfolio_value - current_portfolio_value) / peak_portfolio_value
        if drawdown > self.portfolio_stop_loss:
            logger.warning(f"Portfolio stop-loss triggered: drawdown {drawdown:.2%} > {self.portfolio_stop_loss:.2%}")
            # metrics
            try:
                portfolio_drawdown_gauge.set(drawdown)
            except Exception:
                pass
            try:
                stop_loss_trigger_counter.inc()
            except Exception:
                pass
            return False
        else:
            try:
                portfolio_drawdown_gauge.set(drawdown)
            except Exception:
                pass
        return True

    def kelly_fraction(self, win_rate: float, win_loss_ratio: float) -> float:
        """Compute a (shrunk and capped) Kelly fraction.

        Kelly formula: f* = p - (1-p)/b where b = win_loss_ratio (win avg payoff / loss avg)
        We then shrink and cap the fraction for live trading safety.
        """
        if win_loss_ratio <= 0:
            try:
                kelly_fraction_gauge.set(0.0)
            except Exception:
                pass
            return 0.0
        p = max(0.0, min(1.0, win_rate))
        b = win_loss_ratio
        f = p - (1 - p) / b
        # shrink and cap
        f_shrunk = f * float(self.kelly_shrink)
        f_capped = max(0.0, min(self.kelly_cap, f_shrunk))
        try:
            kelly_fraction_gauge.set(f_capped)
        except Exception:
            pass
        return f_capped

    def adaptive_position_size(self, target_risk_fraction: float, volatility: float, portfolio_value: float) -> float:
        """Return position size (dollar amount) aiming for a target fraction of portfolio risk.

        Simple heuristic: position_size = (target_risk_fraction * portfolio_value) / (volatility + 1e-9)
        This is intentionally simple; replace with more advanced sizing if needed.
        """
        if volatility <= 0:
            # fallback to a conservative flat size
            val = portfolio_value * min(self.max_position_size, target_risk_fraction)
            try:
                # expose as gauge as fraction of portfolio
                kelly_fraction_gauge.set(val / (portfolio_value or 1.0))
            except Exception:
                pass
            return val
        size = (target_risk_fraction * portfolio_value) / volatility
        # cap to max position size
        cap = portfolio_value * self.max_position_size
        return min(size, cap)

    def record_exit(self, symbol: str, timestamp: float = None) -> None:
        """Record an exit time for a symbol to enforce cooldowns."""
        ts = timestamp if timestamp is not None else time.time()
        self._last_exit_ts[symbol] = float(ts)

    def can_trade(self, symbol: str, timestamp: float = None) -> bool:
        """Return True if the symbol is allowed to be traded (not in cooldown)."""
        ts = timestamp if timestamp is not None else time.time()
        last = self._last_exit_ts.get(symbol)
        if last is None:
            try:
                can_trade_gauge.set(1)
            except Exception:
                pass
            return True
        if (ts - last) < float(self.trade_cooldown_seconds):
            logger.info(f"Symbol {symbol} in cooldown: {ts-last:.1f}s < {self.trade_cooldown_seconds}s")
            try:
                trade_cooldown_counter.inc()
            except Exception:
                pass
            try:
                can_trade_gauge.set(0)
            except Exception:
                pass
            return False
        try:
            can_trade_gauge.set(1)
        except Exception:
            pass
        return True
        return True