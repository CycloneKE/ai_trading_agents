"""Account-level loss limits: a daily loss limit and a drawdown limit.

Why this exists: the settings promised both (risk_management.max_daily_loss,
risk_limits.max_drawdown) and neither worked. Nothing read the daily limit,
and the risk manager reported drawdown as a hard-coded 0.0, so the only
protection was each position's own stop, the operator's halt button and the
frozen-loop watchdog. A bad day across many positions had no circuit breaker.

This measures both from the primary account's equity, which the trading loop
already reads every cycle:

- Daily loss: how far equity is below the first reading of the UTC day.
  Crypto trades around the clock, so the day is the UTC calendar day.
- Drawdown: how far equity is below its highest reading since the peak was
  last reset.

While either is at or past its limit, new buys are refused (adds included).
Sells and stops are never refused: a loss limit that stopped the agent
getting out would make the loss worse. A daily-loss breach clears on its own
at the next UTC day. A drawdown breach clears when equity recovers inside the
limit, or when the operator resets the peak (/api/risk/loss-limits/reset),
which makes the current equity the new reference.

The day's opening equity and the peak are saved to data/loss_limits.json, so
a restart neither forgets the peak nor treats a mid-day restart as the start
of the day. Deposits and withdrawals are not told apart from gains and
losses; after a deposit or withdrawal, reset the peak.

Scope: the primary broker's account (US shares and crypto, core funds
included). The forex paper book and the NSE paper account keep their own
books and are not covered here.
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_MAX_DAILY_LOSS = 0.03
DEFAULT_MAX_DRAWDOWN = 0.10
_EPS = 1e-9


def limits_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    """(max_daily_loss, max_drawdown) as fractions; 0 turns a limit off.

    The daily limit lives in risk_management, the drawdown limit in
    risk_limits (where the order-time risk check already reads it), falling
    back to risk_management's copy.
    """
    rm = (config or {}).get('risk_management') or {}
    rl = (config or {}).get('risk_limits') or {}
    daily = rm.get('max_daily_loss', DEFAULT_MAX_DAILY_LOSS)
    drawdown = rl.get('max_drawdown', rm.get('max_drawdown', DEFAULT_MAX_DRAWDOWN))
    return {'max_daily_loss': max(0.0, float(daily or 0.0)),
            'max_drawdown': max(0.0, float(drawdown or 0.0))}


class LossLimits:
    """Tracks one account's daily loss and drawdown and says when to stop buying."""

    def __init__(self, config: Dict[str, Any], path: Optional[str] = None):
        limits = limits_from_config(config)
        self.max_daily_loss = limits['max_daily_loss']
        self.max_drawdown = limits['max_drawdown']
        self.path = str(path or DATA_DIR / 'loss_limits.json')
        self._lock = threading.Lock()
        self.state: Dict[str, Any] = self._load()
        self.equity: Optional[float] = None

    # ------------------------------------------------------------ storage

    def _load(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.path):
                with open(self.path, encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            logger.warning(f"Could not read saved loss-limit state ({self.path}): {e}")
        return {}

    def _save(self) -> None:
        tmp = self.path + '.tmp'
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:
            logger.warning(f"Could not save loss-limit state ({self.path}): {e}")

    # ------------------------------------------------------------ readings

    def update(self, equity: Any, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Take the account's latest equity; returns the status after it.

        A missing or non-positive reading is ignored (a broker having a bad
        minute must not look like a 100% loss).
        """
        try:
            equity = float(equity)
        except (TypeError, ValueError):
            return self.status()
        if equity <= 0:
            return self.status()
        now = now or datetime.now(timezone.utc)
        day = now.date().isoformat()
        with self._lock:
            changed = False
            if self.state.get('day') != day or not self.state.get('day_start'):
                self.state['day'] = day
                self.state['day_start'] = equity
                changed = True
            if equity > float(self.state.get('peak') or 0.0):
                self.state['peak'] = equity
                self.state['peak_at'] = now.isoformat()
                changed = True
            self.equity = equity
            if changed:
                self._save()
        return self.status()

    def reset_peak(self, by: str = 'an operator', now: Optional[datetime] = None) -> Dict[str, Any]:
        """Make the latest equity the drawdown reference (clears a drawdown breach)."""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            if self.equity:
                self.state['peak'] = self.equity
            self.state['peak_at'] = now.isoformat()
            self.state['peak_reset_by'] = by
            self._save()
        logger.warning(f"Drawdown reference reset to {self.state.get('peak')} by {by}")
        return self.status()

    # ------------------------------------------------------------ verdicts

    def daily_loss(self) -> float:
        start = float(self.state.get('day_start') or 0.0)
        if not self.equity or start <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / start)

    def drawdown(self) -> float:
        peak = float(self.state.get('peak') or 0.0)
        if not self.equity or peak <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / peak)

    def breach(self) -> Optional[str]:
        """'daily_loss' or 'drawdown' while that limit is reached, else None."""
        # A hair of tolerance: 1 - 90000/100000 is 0.0999... in floating point,
        # and a fall that lands exactly on a limit must count as reaching it.
        if self.max_daily_loss and self.daily_loss() >= self.max_daily_loss - _EPS:
            return 'daily_loss'
        if self.max_drawdown and self.drawdown() >= self.max_drawdown - _EPS:
            return 'drawdown'
        return None

    def blocks_buys(self) -> bool:
        return self.breach() is not None

    def status(self) -> Dict[str, Any]:
        return {
            'equity': self.equity,
            'day': self.state.get('day'),
            'day_start_equity': self.state.get('day_start'),
            'peak_equity': self.state.get('peak'),
            'peak_at': self.state.get('peak_at'),
            'daily_loss': round(self.daily_loss(), 6),
            'drawdown': round(self.drawdown(), 6),
            'max_daily_loss': self.max_daily_loss,
            'max_drawdown': self.max_drawdown,
            'breach': self.breach(),
            'buys_blocked': self.blocks_buys(),
        }
