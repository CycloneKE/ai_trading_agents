"""Guardrails for everything the agent changes about itself.

Several parts of the agent correct their own behaviour from results. Each
one follows the same rules, and each rule is enforced in code, not left to
an AI's good sense:

1. Bounded. A setting moves only within a fixed range.
2. Gradual. One adjustment moves a setting by a limited step.
3. Evidence first. Nothing is adjusted until enough real trades have closed
   to judge by; before that the current settings stand.
4. Never more aggressive after losses. Falling behind a goal may make the
   agent more careful, never bigger or quicker to trade.
5. Frozen while trading is halted.
6. Recorded, so the operator can see what changed and why.

Where each rule lives:

| Mechanism | What it adjusts | Guardrails |
|---|---|---|
| strategy_tuner.py | strategy settings | fixed bounds, one step, held-out evidence, weekly, frozen when halted |
| strategy_manager rebalance | strategy weights from results | 0.1 floor, 0.3 smoothing, weekly |
| strategy_manager adaptive_confidence | each vote's weight | evidence_tilt: none under 5 closed trades, then 0.5 to 1.5 by t-statistic, recomputed from the journal |
| llm_allocator.py (AI, off by default) | strategy weights | relative to neutral, 0.5 to 1.5, step 0.25, new-evidence gate, frozen when halted |
| adaptive_integration.py | position size, risk tolerance | reduce-only, floors, capped at max_position_size |
| self_assessment.py (AI) | nothing: advice only | evidence gate, checked suggestions |
| llm_orchestrator.validate_trade (AI) | a trade signal | bound_verdict: confirm, weaken or veto only |
| nse_screener AI review | the NSE short list | removals only, holdings kept, a reason required |
| sleeve llm_veto (AI) | a dividend-sleeve buy | veto only |
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ACTIONS = ('buy', 'sell', 'hold')


def _number(value: Any) -> Optional[float]:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if n == n else None           # NaN is no number


def bound_verdict(signal: Dict[str, Any], verdict: Any) -> Dict[str, Any]:
    """The AI trade review's verdict, held to what a reviewer may do.

    The AI may confirm a signal, weaken it (lower confidence or size) or
    veto it (hold). It may not reverse it (a sell the strategies did not
    make, or a buy out of a sell) or strengthen it (more confidence or a
    bigger size than the strategies proposed): confidence feeds the order
    size, so a raised confidence was a bigger order on the AI's say-so.
    A reversal counts as a veto. A missing or unreadable field keeps the
    signal's own value, as an AI outage does. The signal's other fields
    (price, per-strategy votes) are kept.
    """
    if not isinstance(signal, dict):
        return signal
    if not isinstance(verdict, dict) or verdict is signal:
        return signal
    proposed = str(signal.get('action') or 'hold').lower()
    out = {**signal, **verdict}
    notes = []

    action = str(verdict.get('action') or '').lower()
    if action not in ACTIONS:
        action = proposed
    elif action not in (proposed, 'hold'):
        notes.append(f"the AI reversed a {proposed} into a {action}; a reversal counts as a veto")
        action = 'hold'
    out['action'] = action

    for key in ('confidence', 'position_size'):
        mine, theirs = _number(signal.get(key)), _number(verdict.get(key))
        if theirs is None:
            if mine is not None:
                out[key] = mine
            continue
        theirs = max(0.0, theirs)
        if mine is not None and theirs > mine:
            notes.append(f"the AI raised {key.replace('_', ' ')} from {mine:.2f} to {theirs:.2f}; "
                         f"kept at {mine:.2f}")
            theirs = mine
        out[key] = theirs

    if notes:
        note = '; '.join(notes)
        out['guardrail'] = note
        out['reasoning'] = f"{verdict.get('reasoning') or ''} [Guardrail: {note}]".strip()
        logger.info(f"AI verdict bounded for {signal.get('symbol', '?')}: {note}")
    return out


def step_toward(current: float, proposed: float, low: float, high: float, max_step: float) -> float:
    """`proposed`, kept within [low, high] and within max_step of current."""
    value = max(low, min(high, float(proposed)))
    return max(current - max_step, min(current + max_step, value))


def enough_evidence(closed_trades: int, minimum: int) -> bool:
    """Whether enough trades have closed to adjust anything by."""
    try:
        return int(closed_trades) >= int(minimum)
    except (TypeError, ValueError):
        return False


TILT_LOW, TILT_HIGH = 0.5, 1.5
MIN_TILT_TRADES = 5


def evidence_tilt(closed_trades: Any, t_stat: Any, min_trades: int = MIN_TILT_TRADES) -> float:
    """How much a strategy's vote is scaled by its own realised results.

    `t_stat` is the mean of its per-trade returns over their standard error
    (strategy_manager's `sharpe_ratio` field: mean / std * sqrt(n)), so it
    already grows with the number of trades: a lucky handful cannot look
    as convincing as a long record. The tilt is 1.0 (no change) until the
    strategy has `min_trades` closed trades, then 1 + t/4, held to 0.5 to
    1.5: a t of 2 (about 95% confidence) earns the largest boost, and a
    strategy that is losing significantly is halved, never switched off.
    It is symmetric (before, losers were never reduced) and it is
    recomputed from the order journal, so a redeploy cannot lose it.
    """
    try:
        n, t = int(closed_trades or 0), float(t_stat or 0.0)
    except (TypeError, ValueError):
        return 1.0
    if n < int(min_trades) or t != t:
        return 1.0
    return max(TILT_LOW, min(TILT_HIGH, 1.0 + t / 4.0))
