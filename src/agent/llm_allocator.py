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

from src.agent.guardrails import enough_evidence, step_toward

logger = logging.getLogger(__name__)

WEIGHT_MIN = 0.0
WEIGHT_MAX = 2.0
MAX_STEP = 0.5          # max weight change per rebalance (gradual re-tilting)
DEFAULT_INTERVAL_HOURS = 6
# Evidence first (guardrails.py): no tilt at all before this many trades
# have closed, and a strategy's own weight moves only once it has closed
# this many. Before, the AI re-weighted every six hours with no results.
MIN_CLOSED_TRADES = 10
MIN_STRATEGY_TRADES = 5

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
        self.min_closed_trades = int(cfg.get('min_closed_trades', MIN_CLOSED_TRADES))
        self.min_strategy_trades = int(cfg.get('min_strategy_trades', MIN_STRATEGY_TRADES))

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

        attribution = self._attribution()
        closed = sum(int((a or {}).get('closed_trades') or 0) for a in attribution.values())
        if not enough_evidence(closed, self.min_closed_trades):
            logger.info(f"LLM allocator: {closed} closed trades, waiting for "
                        f"{self.min_closed_trades}; weights unchanged")
            return None

        context = self._build_context(current, attribution)
        proposal = self.llm.propose_json(_SYSTEM_PROMPT, context)
        weights = self.apply_guardrails(proposal, current, attribution, self.min_strategy_trades)
        if weights is None:
            logger.info("LLM allocator: no usable proposal; weights unchanged")
            return None

        self.strategy_manager.strategy_weights.update(weights)
        self.last_proposal = weights
        logger.warning(f"LLM allocator applied weights: {weights} (was {current})")
        return weights

    # ------------------------------------------------------------------

    def _attribution(self) -> Dict[str, Any]:
        try:
            if self.order_journal:
                from src.agent.strategy_attribution import compute_attribution
                return compute_attribution(self.order_journal.filled_orders()) or {}
        except Exception as e:
            logger.debug(f"Allocator attribution unavailable: {e}")
        return {}

    def _build_context(self, current: Dict[str, float], attribution: Dict[str, Any]) -> str:
        return (
            f"Strategies and current weights: {json.dumps(current)}\n"
            f"Realized attribution (P&L, win rate, open inventory): "
            f"{json.dumps(attribution, default=str)}\n"
            "Propose new weights now."
        )

    # ------------------------------------------------------------------

    @staticmethod
    def apply_guardrails(proposal: Optional[Dict[str, Any]],
                         current: Dict[str, float],
                         attribution: Optional[Dict[str, Any]] = None,
                         min_strategy_trades: int = 0) -> Optional[Dict[str, float]]:
        """Deterministic gate between the LLM and the ensemble.

        Returns sanitized weights, or None when the proposal is unusable
        (keep current weights). With `attribution`, a strategy that has
        closed fewer than `min_strategy_trades` trades keeps its weight.
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
            if value != value:
                continue  # NaN
            if attribution is not None and not enough_evidence(
                    (attribution.get(name) or {}).get('closed_trades') or 0, min_strategy_trades):
                continue  # too few of its own trades to judge it by
            # Bounded, and drift capped so one bad proposal can't flip the book.
            sanitized[name] = round(step_toward(cur, value, WEIGHT_MIN, WEIGHT_MAX, MAX_STEP), 3)

        if not sanitized:
            return None
        # Never let the whole ensemble go dark.
        merged = {**current, **sanitized}
        if sum(merged.values()) <= 0:
            return None
        return sanitized
