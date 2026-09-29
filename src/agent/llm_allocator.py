"""LLM strategy weight allocator with hard deterministic guardrails.

OFF by default (config `llm_allocator.enabled`). The deterministic evidence
tilt (guardrails.evidence_tilt, applied in the ensemble) already re-weights
strategies from the same realised results this allocator is shown, and it
is recomputed from the order journal, so a redeploy cannot lose it. The AI
sees nothing the tilt does not, so it adds cost and noise; it is kept for
whoever wants an AI's judgment layered on top.

The LLM *proposes* per-strategy weights; deterministic code *disposes*:

- Scale. The ensemble's weights are normalised to sum to 1 (about 0.17 each
  for six strategies), but this allocator used to hand the AI those numbers
  and tell it "1.0 is neutral, 0 to 2": a proposal of 0.3, meant as a cut,
  was a raise, and 0.0 switched a strategy off. Weights are now shown and
  proposed relative to neutral (their mean is 1.0), and converted back.
- Bounds. Relative weights stay within 0.5 to 1.5 (the tilt's range) and
  move by at most 0.25 a rebalance. Unknown strategies are dropped, and an
  empty or degenerate proposal leaves the weights untouched.
- Evidence. Nothing before `min_closed_trades` (10) have closed in all, a
  strategy moves only once it has closed `min_strategy_trades` (5), and the
  AI is asked again only after `min_new_trades` (5) more have closed:
  results do not change between asks otherwise.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from src.agent.guardrails import enough_evidence, step_toward

logger = logging.getLogger(__name__)

WEIGHT_MIN = 0.5          # relative to neutral (1.0 = the strategies' mean weight)
WEIGHT_MAX = 1.5
MAX_STEP = 0.25           # max change per rebalance (gradual re-tilting)
DEFAULT_INTERVAL_HOURS = 24
MIN_CLOSED_TRADES = 10
MIN_STRATEGY_TRADES = 5
MIN_NEW_TRADES = 5

_SYSTEM_PROMPT = (
    "You allocate weight between the strategies of an automated trading system.\n"
    "Weights are relative: 1.0 is neutral, and each must stay between 0.5 and 1.5. Given each "
    "strategy's realised results, propose weights that lean toward the ones with convincing "
    "records. A strategy with few trades has not earned a change: leave it at its current weight. "
    "Prefer small changes.\n"
    'Return ONLY raw JSON: {"<strategy_name>": <weight>, ...} with exactly the strategy names given.'
)


class LLMAllocator:
    def __init__(self, llm_orchestrator, strategy_manager, order_journal,
                 config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.llm = llm_orchestrator
        self.strategy_manager = strategy_manager
        self.order_journal = order_journal
        self.enabled = cfg.get('enabled', False)
        self.interval_seconds = float(cfg.get('interval_hours', DEFAULT_INTERVAL_HOURS)) * 3600
        self.last_rebalance = 0.0
        self.last_closed = 0
        self.last_proposal: Optional[Dict[str, float]] = None
        self.min_closed_trades = int(cfg.get('min_closed_trades', MIN_CLOSED_TRADES))
        self.min_strategy_trades = int(cfg.get('min_strategy_trades', MIN_STRATEGY_TRADES))
        self.min_new_trades = int(cfg.get('min_new_trades', MIN_NEW_TRADES))

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
        if not enough_evidence(closed - self.last_closed, self.min_new_trades):
            logger.info(f"LLM allocator: {closed - self.last_closed} new closed trades since the last "
                        f"ask, waiting for {self.min_new_trades}; weights unchanged")
            return None
        self.last_closed = closed

        mean = sum(current.values()) / len(current)
        if mean <= 0:
            return None
        relative = {n: w / mean for n, w in current.items()}
        proposal = self.llm.propose_json(_SYSTEM_PROMPT, self._build_context(relative, attribution))
        chosen = self.apply_guardrails(proposal, relative, attribution, self.min_strategy_trades) or {}
        if not chosen:
            logger.info("LLM allocator: no usable proposal; weights unchanged")
            return None

        # Back to the ensemble's scale, keeping its overall level (its mean).
        merged = {**relative, **chosen}
        level = sum(merged.values()) / len(merged)
        applied = {n: round(merged[n] / level * mean, 6) for n in chosen}
        self.strategy_manager.strategy_weights.update(applied)
        self.last_proposal = applied
        logger.warning(f"LLM allocator applied relative weights {chosen} (were "
                       f"{ {n: round(relative[n], 3) for n in chosen} })")
        return applied

    # ------------------------------------------------------------------

    def _attribution(self) -> Dict[str, Any]:
        try:
            if self.order_journal:
                from src.agent.strategy_attribution import compute_attribution
                return compute_attribution(self.order_journal.filled_orders()) or {}
        except Exception as e:
            logger.debug(f"Allocator attribution unavailable: {e}")
        return {}

    def _build_context(self, relative: Dict[str, float], attribution: Dict[str, Any]) -> str:
        results = {n: {'closed_trades': a.get('closed_trades'), 'win_rate': a.get('win_rate'),
                       'mean_trade_return': (round(sum(a.get('trade_returns') or []) /
                                                   len(a['trade_returns']), 4)
                                             if a.get('trade_returns') else None)}
                   for n, a in attribution.items() if n in relative}
        return (f"Current relative weights: {json.dumps({n: round(w, 3) for n, w in relative.items()})}\n"
                f"Realised results per strategy: {json.dumps(results)}\n"
                "Propose the new relative weights now.")

    # ------------------------------------------------------------------

    @staticmethod
    def apply_guardrails(proposal: Optional[Dict[str, Any]],
                         current: Dict[str, float],
                         attribution: Optional[Dict[str, Any]] = None,
                         min_strategy_trades: int = 0) -> Optional[Dict[str, float]]:
        """Deterministic gate between the LLM and the ensemble, on the
        relative scale (1.0 neutral).

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

        return sanitized or None
