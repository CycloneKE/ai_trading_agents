"""How many symbols the agent can actually carry.

Adding a symbol is a one-line config change, which makes it easy to forget
that each one costs time and quota on every cycle. Three limits bite, in
this order:

1. LLM quota. Every candidate trade is sent to an LLM for validation. Free
   Gemini tiers allow on the order of 10 requests a minute. At a 60-second
   cycle, more than about 10 symbols signalling at once exceeds that, and
   the orchestrator's cooldown then silently skips validation for the rest
   of the cycle. Trades still happen; they just stop being checked.

2. Cycle time. The loop targets `trading_loop_interval` seconds. When
   per-symbol work exceeds that budget, cycles overlap or drift, and price
   data is stale by the time an order is placed.

3. Vendor rate limits. Market data providers cap requests per minute; going
   over means gaps, which the fallback generator fills with synthetic prices
   that the agent correctly refuses to trade on. The universe silently stops
   trading rather than erroring.

This module turns those into arithmetic, so the answer to "can I add ten
NSE symbols" is a number rather than a shrug.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Published free-tier ceilings, requests per minute. Override in config under
# `capacity.provider_limits` when on a paid plan.
DEFAULT_PROVIDER_LIMITS = {
    'gemini': 10,
    'openrouter': 20,
    'alpha_vantage': 5,
    'finnhub': 60,
    'fmp': 10,
}


@dataclass
class CapacityReport:
    symbols: int
    cycle_seconds: float
    budget_seconds: float
    headroom_seconds: float
    max_symbols_by_time: Optional[int]
    max_symbols_by_llm: Optional[int]
    binding_constraint: str
    findings: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def within_budget(self) -> bool:
        return self.headroom_seconds >= 0


def llm_requests_per_cycle(symbols: int, signal_rate: float) -> float:
    """Validation calls a cycle, given how often a symbol produces a signal.

    `signal_rate` is the share of symbols that emit a non-hold action on a
    typical cycle. Measured on the GOOG backtest after the regime filter
    landed, roughly 0.43 of bars produced a signal; before it, 0.05.
    """
    return max(0.0, symbols) * max(0.0, min(1.0, signal_rate))


def max_symbols_for_llm(limit_per_minute: int, cycle_seconds: float,
                        signal_rate: float) -> Optional[int]:
    """Largest universe whose validation calls stay inside the quota."""
    if signal_rate <= 0 or limit_per_minute <= 0 or cycle_seconds <= 0:
        return None
    calls_allowed = limit_per_minute * (cycle_seconds / 60.0)
    return max(0, int(calls_allowed / signal_rate))


def assess(symbols: int,
           per_symbol_seconds: float,
           fixed_overhead_seconds: float = 0.0,
           cycle_budget_seconds: float = 60.0,
           signal_rate: float = 0.4,
           llm_limit_per_minute: int = DEFAULT_PROVIDER_LIMITS['gemini'],
           llm_enabled: bool = True) -> CapacityReport:
    """Whether this universe fits in the cycle, and what breaks first."""
    cycle = fixed_overhead_seconds + symbols * max(0.0, per_symbol_seconds)
    headroom = cycle_budget_seconds - cycle

    by_time = None
    if per_symbol_seconds > 0:
        by_time = max(0, int((cycle_budget_seconds - fixed_overhead_seconds)
                             / per_symbol_seconds))

    by_llm = (max_symbols_for_llm(llm_limit_per_minute, cycle_budget_seconds,
                                  signal_rate) if llm_enabled else None)

    candidates = {k: v for k, v in
                  (('cycle time', by_time), ('LLM quota', by_llm))
                  if v is not None}
    binding = min(candidates, key=candidates.get) if candidates else 'none measured'

    findings: List[str] = []
    if headroom < 0:
        findings.append(
            f"Cycle takes {cycle:.1f}s against a {cycle_budget_seconds:.0f}s budget. "
            f"Cycles will overlap or drift and prices will be stale at order time.")
    elif headroom < 0.25 * cycle_budget_seconds:
        findings.append(
            f"Only {headroom:.1f}s of headroom in a {cycle_budget_seconds:.0f}s "
            f"cycle. A slow vendor response would push it over.")

    if llm_enabled and by_llm is not None:
        calls = llm_requests_per_cycle(symbols, signal_rate)
        allowed = llm_limit_per_minute * (cycle_budget_seconds / 60.0)
        if calls > allowed:
            findings.append(
                f"About {calls:.0f} LLM validation calls a cycle against "
                f"{allowed:.0f} allowed. Past the limit the orchestrator's "
                f"cooldown skips validation, so trades still execute but stop "
                f"being checked. That is a silent loss of a safety layer, not "
                f"an error you will see.")
        elif calls > 0.8 * allowed:
            findings.append(
                f"LLM validation at {calls:.0f} of {allowed:.0f} calls a cycle, "
                f"within 20% of the quota.")

    return CapacityReport(
        symbols=symbols, cycle_seconds=cycle,
        budget_seconds=cycle_budget_seconds, headroom_seconds=headroom,
        max_symbols_by_time=by_time, max_symbols_by_llm=by_llm,
        binding_constraint=binding, findings=findings,
        detail={'per_symbol_seconds': per_symbol_seconds,
                'fixed_overhead_seconds': fixed_overhead_seconds,
                'signal_rate': signal_rate,
                'llm_limit_per_minute': llm_limit_per_minute,
                'llm_calls_per_cycle': llm_requests_per_cycle(symbols, signal_rate)},
    )


def projection(per_symbol_seconds: float, fixed_overhead_seconds: float,
               cycle_budget_seconds: float, signal_rate: float,
               llm_limit_per_minute: int,
               counts: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """A row per universe size, so the cliff is visible before you reach it."""
    counts = counts or [5, 10, 15, 20, 30, 40, 50, 75, 100]
    rows = []
    for n in counts:
        r = assess(n, per_symbol_seconds, fixed_overhead_seconds,
                   cycle_budget_seconds, signal_rate, llm_limit_per_minute)
        rows.append({
            'symbols': n,
            'cycle_seconds': round(r.cycle_seconds, 2),
            'headroom_seconds': round(r.headroom_seconds, 2),
            'llm_calls': round(r.detail['llm_calls_per_cycle'], 1),
            'fits': r.within_budget,
            'llm_ok': r.detail['llm_calls_per_cycle'] <=
                      llm_limit_per_minute * (cycle_budget_seconds / 60.0),
        })
    return rows
