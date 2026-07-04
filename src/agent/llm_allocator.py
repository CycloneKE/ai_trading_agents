"""LLM-driven strategy weight allocator with hard deterministic guardrails.

The LLM *proposes* per-strategy ensemble weights from realized attribution
data; deterministic code *disposes*: unknown strategies are dropped, weights
are clamped to [0, 2], per-rebalance drift is capped, and an empty/degenerate
proposal leaves current weights untouched. The LLM can therefore tilt the
ensemble but can never disable risk controls, exceed bounds, or brick the
strategy mix — and with no API key configured the allocator is a no-op.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

WEIGHT_MIN = 0.0
WEIGHT_MAX = 2.0
MAX_STEP = 0.5          # max weight change per rebalance (gradual re-tilting)
DEFAULT_INTERVAL_HOURS = 6

_SYSTEM_PROMPT = (
    "You are the capital allocator of an automated trading system.\n"
    "Given realized per-strategy performance, propose ensemble weights that "
    "tilt capital toward strategies with better risk-adjusted results.\n"
    "Rules: weights are floats between 0.0 and 2.0; 1.0 is neutral; do not "
    "zero out every strategy; prefer gradual changes.\n"
    'Return ONLY raw JSON: {"<strategy_name>": <weight>, ...} with exactly '
    "the strategy names given. No markdown, no commentary."
)


class LLMAllocator:
    def __init__(self, llm_orchestrator, strategy_manager, order_journal,
                 config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.llm = llm_orchestrator
        self.strategy_manager = strategy_manager
        self.order_journal = order_journal
        self.enabled = cfg.get('enabled', True)
        self.interval_seconds = float(cfg.get('interval_hours', DEFAULT_INTERVAL_HOURS)) * 3600
        self.last_rebalance = 0.0
        self.last_proposal: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------

    def maybe_rebalance(self) -> Optional[Dict[str, float]]:
        """Called from the trading loop; self rate-limits to the configured
        interval. Returns applied weights when a rebalance happened."""
        if not self.enabled or not self.llm or not getattr(self.llm, 'enabled', False):
            return None
        now = time.time()
        if now - self.last_rebalance < self.interval_seconds:
            return None
        self.last_rebalance = now  # even on failure: don't hammer the LLM

        current = dict(self.strategy_manager.strategy_weights)
        if not current:
            return None

        context = self._build_context(current)
        proposal = self.llm.propose_json(_SYSTEM_PROMPT, context)
        weights = self.apply_guardrails(proposal, current)
        if weights is None:
            logger.info("LLM allocator: no usable proposal; weights unchanged")
            return None

        self.strategy_manager.strategy_weights.update(weights)
        self.last_proposal = weights
        logger.warning(f"LLM allocator applied weights: {weights} (was {current})")
        return weights

    # ------------------------------------------------------------------

    def _build_context(self, current: Dict[str, float]) -> str:
        attribution = {}
        try:
            if self.order_journal:
                from src.agent.strategy_attribution import compute_attribution
                attribution = compute_attribution(self.order_journal.filled_orders())
        except Exception as e:
            logger.debug(f"Allocator attribution unavailable: {e}")
        return (
            f"Strategies and current weights: {json.dumps(current)}\n"
            f"Realized attribution (P&L, win rate, open inventory): "
            f"{json.dumps(attribution, default=str)}\n"
            "Propose new weights now."
        )

    # ------------------------------------------------------------------

    @staticmethod
    def apply_guardrails(proposal: Optional[Dict[str, Any]],
                         current: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Deterministic gate between the LLM and the ensemble.

        Returns sanitized weights, or None when the proposal is unusable
        (keep current weights).
        """
        if not proposal or not isinstance(proposal, dict):
            return None

        sanitized: Dict[str, float] = {}
        for name, cur in current.items():
            raw = proposal.get(name)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue  # missing/garbage for this strategy: keep current
            value = max(WEIGHT_MIN, min(WEIGHT_MAX, value))
            # Cap drift so one bad proposal can't flip the book overnight.
            value = max(cur - MAX_STEP, min(cur + MAX_STEP, value))
            sanitized[name] = round(value, 3)

        if not sanitized:
            return None
        # Never let the whole ensemble go dark.
        merged = {**current, **sanitized}
        if sum(merged.values()) <= 0:
            return None
        return sanitized
