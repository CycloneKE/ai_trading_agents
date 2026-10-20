"""Deterministic yield + quality scoring for NSE dividend sleeve candidates.

Pure function: no network calls, no LLM. Every ranking is reproducible from
a Fundamentals record and the configured weights/thresholds — this is the
one place in the sleeve where the buy decision's numbers come from.
"""
from dataclasses import dataclass
from typing import List, Optional

from src.agent.sleeve.fundamentals_store import Fundamentals


@dataclass
class ScoringConfig:
    yield_cap_pct: float = 12.0
    yield_weight: float = 0.5
    quality_weight: float = 0.5
    payout_ratio_min: float = 0.30
    payout_ratio_max: float = 0.70
    years_paid_target: int = 5
    min_avg_daily_volume: int = 50_000
    stale_days: int = 400


@dataclass
class ScoredCandidate:
    symbol: str
    combined_score: float
    yield_score: float
    quality_score: float
    stale: bool


def score_symbol(f: Fundamentals, cfg: ScoringConfig) -> Optional[ScoredCandidate]:
    """Returns None when the symbol fails a hard gate (illiquid, no/negative
    yield) rather than assigning a low score — a gated-out name never
    appears in the ranked list at all."""
    if f.avg_daily_volume < cfg.min_avg_daily_volume:
        return None
    if f.yield_ttm_pct <= 0:
        return None

    yield_capped = min(f.yield_ttm_pct, cfg.yield_cap_pct)
    yield_score = round(yield_capped / cfg.yield_cap_pct, 4)

    payout_ok = (f.payout_ratio is not None and
                cfg.payout_ratio_min <= f.payout_ratio <= cfg.payout_ratio_max)
    consistency_score = min(f.years_consecutive_paid / cfg.years_paid_target, 1.0)
    trend_ok = 1.0 if f.eps_trend != 'negative' else 0.0
    quality_score = round(
        ((1.0 if payout_ok else 0.0) + consistency_score + trend_ok) / 3, 4)

    combined = round(cfg.yield_weight * yield_score + cfg.quality_weight * quality_score, 4)
    return ScoredCandidate(
        symbol=f.symbol, combined_score=combined,
        yield_score=yield_score, quality_score=quality_score,
        stale=f.is_stale(cfg.stale_days),
    )


def rank_candidates(fundamentals: List[Fundamentals], cfg: ScoringConfig,
                    top_n: int) -> List[ScoredCandidate]:
    scored = [c for c in (score_symbol(f, cfg) for f in fundamentals) if c is not None]
    scored.sort(key=lambda c: c.combined_score, reverse=True)
    return scored[:top_n]
