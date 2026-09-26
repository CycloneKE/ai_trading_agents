"""Deterministic yield + quality scoring for NSE dividend sleeve candidates.

Pure function: no network calls, no LLM. Every ranking is reproducible from
a Fundamentals record and the configured weights/thresholds — this is the
one place in the sleeve where the buy decision's numbers come from.
"""
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from src.agent.sleeve.fundamentals_store import Fundamentals
from src.agent.sleeve.screens import (APPROACHING, JUST_AFTER,
                                      apply_concentration_limit,
                                      assess_sustainability, ex_date_timing)


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
    # Sustainability screen. A high yield on thin cover is a cut being
    # priced in; buying it for the headline number buys the cut.
    require_sustainable: bool = True
    min_dividend_cover: float = 1.2  # reported below this; gated below 1.0
    yield_trap_pct: float = 12.0
    min_years_paid: int = 3
    # At most this many holdings from one sector. Kenyan banks dominate the
    # high-yield list, so an unconstrained top-N is a bet on one credit cycle.
    max_per_sector: int = 2
    # Ex-dividend timing tilt. The price drops by roughly the dividend on the
    # ex-date, so buying just before pays for a payout you then receive and
    # are taxed on; just after buys the same business cheaper.
    ex_date_tilt: float = 0.10
    # Value: earnings yield (earnings per share / price, the inverse of the
    # P/E). Cheap, profitable companies score higher. 0 leaves it out.
    value_weight: float = 0.0
    earnings_yield_cap_pct: float = 25.0


def earnings_yield_pct(f: Fundamentals) -> Optional[float]:
    """Earnings per share over price, in percent, or None if unknown.

    The price is the one the dividend yield was measured at: price =
    dividend / yield, so earnings / price = yield x earnings / dividend.
    """
    if not f.dividend_per_share_kes or f.dividend_per_share_kes <= 0 or f.yield_ttm_pct <= 0 \
            or f.eps_kes is None:
        return None
    return f.yield_ttm_pct * f.eps_kes / f.dividend_per_share_kes


@dataclass
class ScoredCandidate:
    symbol: str
    combined_score: float
    yield_score: float
    quality_score: float
    stale: bool
    ex_date_timing: str = 'unknown'
    dividend_cover: Optional[float] = None
    sustainability: str = 'not assessed'
    value_score: float = 0.0
    earnings_yield_pct: Optional[float] = None


def score_symbol(f: Fundamentals, cfg: ScoringConfig,
                 today: Optional[date] = None) -> Optional[ScoredCandidate]:
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

    verdict = assess_sustainability(
        eps=f.eps_kes, dividend_per_share=f.dividend_per_share_kes,
        yield_pct=f.yield_ttm_pct,
        years_consecutive_paid=f.years_consecutive_paid,
        eps_trend=f.eps_trend,
        min_cover=cfg.min_dividend_cover,
        yield_trap_pct=cfg.yield_trap_pct,
        min_years_paid=cfg.min_years_paid)
    if cfg.require_sustainable and verdict.is_disqualified:
        # Gate only on affordability: negative earnings, a payout exceeding
        # earnings, or a yield high enough that the market is pricing in a
        # cut. Weak payment history and falling earnings already lower the
        # quality score above, and gating on them too would exclude tight
        # but genuine payers.
        return None

    ey = earnings_yield_pct(f)
    value_score = round(min(max(ey or 0.0, 0.0), cfg.earnings_yield_cap_pct)
                        / cfg.earnings_yield_cap_pct, 4) if cfg.earnings_yield_cap_pct > 0 else 0.0
    combined = (cfg.yield_weight * yield_score + cfg.quality_weight * quality_score
                + cfg.value_weight * value_score)

    # Timing tilt, applied after the fundamental score so it can reorder
    # near-equals without ever promoting a weaker business.
    timing = ex_date_timing(getattr(f, 'ex_dividend_date', None), today)
    if cfg.ex_date_tilt:
        if timing == JUST_AFTER:
            combined *= (1.0 + cfg.ex_date_tilt)
        elif timing == APPROACHING:
            combined *= (1.0 - cfg.ex_date_tilt)

    return ScoredCandidate(
        symbol=f.symbol, combined_score=round(combined, 4),
        yield_score=yield_score, quality_score=quality_score,
        stale=f.is_stale(cfg.stale_days),
        ex_date_timing=timing,
        dividend_cover=verdict.cover,
        sustainability=verdict.summary,
        value_score=value_score,
        earnings_yield_pct=round(ey, 2) if ey is not None else None,
    )


def rank_candidates(fundamentals: List[Fundamentals], cfg: ScoringConfig,
                    top_n: int, sector_lookup: Any = None,
                    today: Optional[date] = None) -> List[ScoredCandidate]:
    """Rank, then apply the sector cap, then take the top N.

    Order matters: capping before the cut means a displaced bank frees its
    slot for the next-best name from another sector, rather than the sleeve
    simply holding fewer positions.
    """
    scored = [c for c in (score_symbol(f, cfg, today) for f in fundamentals)
              if c is not None]
    scored.sort(key=lambda c: c.combined_score, reverse=True)
    scored = apply_concentration_limit(scored, sector_lookup,
                                       max_per_sector=cfg.max_per_sector)
    return scored[:top_n]
