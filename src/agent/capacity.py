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
    max_burst_symbols: Optional[int] = None
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
           llm_enabled: bool = True,
           burst_symbols: int = 0,
           burst_interval_seconds: float = 1800.0) -> CapacityReport:
    """Whether this universe fits in the cycle, and what breaks first.

    `symbols` is the set walked every cycle. `burst_symbols` is a second set
    walked on a slower schedule, `burst_interval_seconds` apart: the NSE pass
    is one, running every 30 minutes during Nairobi market hours rather than
    every 60-second cycle.

    The distinction matters and is easy to get wrong. Counting both sets as
    one per-cycle load overstates sustained demand several times over. What
    the slower set actually creates is a periodic spike, and a spike is the
    thing to size against, because the quota is per minute and the burst
    lands inside one of them.
    """
    cycle = fixed_overhead_seconds + symbols * max(0.0, per_symbol_seconds)
    headroom = cycle_budget_seconds - cycle

    by_time = None
    if per_symbol_seconds > 0:
        by_time = max(0, int((cycle_budget_seconds - fixed_overhead_seconds)
                             / per_symbol_seconds))

    by_llm = (max_symbols_for_llm(llm_limit_per_minute, cycle_budget_seconds,
                                  signal_rate) if llm_enabled else None)
    # Headroom the per-cycle set leaves for a burst, in symbols.
    burst_headroom = None
    if llm_enabled and signal_rate > 0:
        spare = llm_limit_per_minute - (
            llm_requests_per_cycle(symbols, signal_rate)
            * (60.0 / cycle_budget_seconds if cycle_budget_seconds > 0 else 0.0))
        burst_headroom = max(0, int(spare / signal_rate))

    candidates = {k: v for k, v in
                  (('cycle time', by_time), ('LLM quota', by_llm))
                  if v is not None}
    binding = min(candidates, key=candidates.get) if candidates else 'none measured'

    # Sustained demand from the per-cycle set, then the worst minute, when a
    # burst lands on top of an ordinary cycle.
    per_cycle_calls = llm_requests_per_cycle(symbols, signal_rate)
    cycles_per_minute = (60.0 / cycle_budget_seconds) if cycle_budget_seconds > 0 else 0.0
    sustained_per_minute = per_cycle_calls * cycles_per_minute
    burst_calls = llm_requests_per_cycle(burst_symbols, signal_rate)
    peak_per_minute = sustained_per_minute + burst_calls

    findings: List[str] = []
    if headroom < 0:
        findings.append(
            f"Cycle takes {cycle:.1f}s against a {cycle_budget_seconds:.0f}s budget. "
            f"Cycles will overlap or drift and prices will be stale at order time.")
    elif headroom < 0.25 * cycle_budget_seconds:
        findings.append(
            f"Only {headroom:.1f}s of headroom in a {cycle_budget_seconds:.0f}s "
            f"cycle. A slow vendor response would push it over.")

    if llm_enabled:
        limit = llm_limit_per_minute
        if sustained_per_minute > limit:
            findings.append(
                f"Sustained LLM demand is about {sustained_per_minute:.0f} calls a "
                f"minute against {limit} allowed. Past the limit the "
                f"orchestrator's cooldown skips validation, so trades still "
                f"execute but stop being checked. That is a silent loss of a "
                f"safety layer, not an error you will see.")
        elif sustained_per_minute > 0.8 * limit:
            findings.append(
                f"Sustained LLM demand at {sustained_per_minute:.0f} of {limit} "
                f"calls a minute, within 20% of the quota.")

        if burst_symbols and peak_per_minute > limit:
            findings.append(
                f"The {burst_symbols}-symbol burst every "
                f"{burst_interval_seconds / 60:.0f} minutes pushes that minute to "
                f"about {peak_per_minute:.0f} calls against {limit} allowed. "
                f"Sustained demand is fine at {sustained_per_minute:.0f}/min; only "
                f"the burst minute overruns, so staggering the burst fixes this "
                f"without trimming the universe.")

    return CapacityReport(
        symbols=symbols, cycle_seconds=cycle,
        budget_seconds=cycle_budget_seconds, headroom_seconds=headroom,
        max_symbols_by_time=by_time, max_symbols_by_llm=by_llm,
        binding_constraint=binding, max_burst_symbols=burst_headroom,
        findings=findings,
        detail={'per_symbol_seconds': per_symbol_seconds,
                'fixed_overhead_seconds': fixed_overhead_seconds,
                'signal_rate': signal_rate,
                'llm_limit_per_minute': llm_limit_per_minute,
                'llm_calls_per_cycle': per_cycle_calls,
                'sustained_calls_per_minute': sustained_per_minute,
                'burst_symbols': burst_symbols,
                'burst_calls': burst_calls,
                'peak_calls_per_minute': peak_per_minute},
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
