"""Portfolio-level cash deployment controller.

The per-trade risk cap floats between base (2%) and ceiling (5%) in
proportion to (a) how far current deployment sits below target and
(b) signal conviction. Weak signals never get boosted, so idle cash is
deployed into the *best* ideas, not sprayed across the book.
"""
from typing import Any, Dict


class CashDeploymentPolicy:
    def __init__(self, config: Dict[str, Any]):
        cp = config.get('cash_policy', {})
        self.enabled = cp.get('enabled', False)
        self.target = cp.get('target_deployment', 0.80)
        self.base = cp.get('base_risk_per_trade', 0.02)
        self.ceiling = cp.get('max_risk_per_trade', 0.05)
        self.min_conf = cp.get('min_confidence_to_boost', 0.55)

    def risk_cap(self, deployed_pct: float, confidence: float) -> float:
        if not self.enabled or confidence < self.min_conf:
            return self.base
        gap = max(0.0, self.target - deployed_pct)
        if gap <= 0:
            return self.base
        # gap=target (empty book) with confidence=1.0 reaches the ceiling.
        boost = (self.ceiling - self.base) * (gap / self.target) * confidence
        return round(min(self.ceiling, self.base + boost), 4)
