"""Portfolio-level cash deployment controller.

Deploys idle cash into the *best* ideas rather than spraying it across the
book: when the portfolio sits below its deployment target and a signal has
real conviction, this scales that position's risk budget up. Weak signals
are never boosted.

This used to return a *notional cap* (2% to 5% of equity), which the trading
loop then used as the position size outright. That made it a second,
competing position sizer alongside src/agent/position_sizing.py, and the two
disagreed: a 2% notional "risk cap" alongside a 5% position cap is not a risk
figure at all, and neither one looked at volatility.

It now returns a *multiplier on the risk budget*, which composes with
volatility-scaled sizing instead of competing with it:

    risk fraction = trading.risk_per_trade * risk_multiplier(...)
    notional      = equity * risk fraction / stop distance
    notional      = min(notional, equity * risk_limits.max_position_size)

So conviction and spare cash change how much is *risked*, while the stop
distance decides how large a position that risk buys, and the position cap
remains a genuine concentration ceiling.
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULT_MAX_MULTIPLIER = 2.5


class CashDeploymentPolicy:
    def __init__(self, config: Dict[str, Any]):
        cp = config.get('cash_policy', {})
        self.enabled = cp.get('enabled', False)
        self.target = cp.get('target_deployment', 0.80)
        self.min_conf = cp.get('min_confidence_to_boost', 0.55)
        self.max_multiplier = self._resolve_multiplier(cp)

    @staticmethod
    def _resolve_multiplier(cp: Dict[str, Any]) -> float:
        """How far conviction may stretch the risk budget, at most.

        Prefers the explicit setting. Falls back to the ratio of the legacy
        notional caps, so an operator's existing choice of "how much more
        aggressive when flush" carries over unchanged (the shipped 5%/2%
        becomes 2.5x) rather than being silently reset to a default.
        """
        explicit = cp.get('max_risk_multiplier')
        if explicit:
            return max(1.0, float(explicit))

        base = cp.get('base_risk_per_trade')
        ceiling = cp.get('max_risk_per_trade')
        if base and ceiling and float(base) > 0:
            derived = max(1.0, float(ceiling) / float(base))
            logger.warning(
                "cash_policy.base_risk_per_trade/max_risk_per_trade are legacy "
                "notional caps and no longer size positions. Derived "
                "max_risk_multiplier=%.2f from their ratio. Set "
                "cash_policy.max_risk_multiplier explicitly to silence this.",
                derived)
            return derived

        return DEFAULT_MAX_MULTIPLIER

    def risk_multiplier(self, deployed_pct: float, confidence: float) -> float:
        """1.0 normally; up to max_multiplier when under-deployed and confident.

        Never below 1.0: this deploys spare cash, it does not shrink positions.
        Cutting risk is the job of the stop distance and the position cap.
        """
        if not self.enabled or confidence < self.min_conf:
            return 1.0
        gap = max(0.0, self.target - deployed_pct)
        if gap <= 0 or self.target <= 0:
            return 1.0
        # An empty book with confidence 1.0 reaches the ceiling.
        boost = (self.max_multiplier - 1.0) * (gap / self.target) * confidence
        return round(min(self.max_multiplier, 1.0 + boost), 4)
