from src.agent.sleeve.dividend_scorer import ScoringConfig, rank_candidates, score_symbol
from src.agent.sleeve.fundamentals_store import Fundamentals


def make_fundamentals(**overrides):
    base = dict(symbol='SCOM', yield_ttm_pct=6.5, dividend_per_share_kes=2.3,
               eps_kes=2.4, payout_ratio=0.48, years_consecutive_paid=8,
               eps_trend='positive', avg_daily_volume=100_000,
               last_updated='2026-07-01')
    base.update(overrides)
    return Fundamentals(**base)


def test_liquidity_gate_excludes_illiquid_symbol():
    cfg = ScoringConfig(min_avg_daily_volume=50_000)
    f = make_fundamentals(avg_daily_volume=1_000)
    assert score_symbol(f, cfg) is None


def test_zero_yield_excluded():
    cfg = ScoringConfig()
    f = make_fundamentals(yield_ttm_pct=0.0)
    assert score_symbol(f, cfg) is None


def test_yield_above_cap_is_capped_not_boosted():
    cfg = ScoringConfig(yield_cap_pct=12.0)
    normal = score_symbol(make_fundamentals(yield_ttm_pct=10.0), cfg)
    trap = score_symbol(make_fundamentals(yield_ttm_pct=30.0), cfg)
    assert trap.yield_score == 1.0  # capped at 12%, same as any yield >= cap
    assert normal.yield_score < trap.yield_score


def test_payout_ratio_outside_band_lowers_quality_score():
    cfg = ScoringConfig(payout_ratio_min=0.30, payout_ratio_max=0.70)
    healthy = score_symbol(make_fundamentals(payout_ratio=0.5), cfg)
    unhealthy = score_symbol(make_fundamentals(payout_ratio=0.95), cfg)
    assert healthy.quality_score > unhealthy.quality_score


def test_negative_eps_trend_lowers_quality_score():
    cfg = ScoringConfig()
    positive = score_symbol(make_fundamentals(eps_trend='positive'), cfg)
    negative = score_symbol(make_fundamentals(eps_trend='negative'), cfg)
    assert positive.quality_score > negative.quality_score


def test_stale_flag_propagates():
    cfg = ScoringConfig(stale_days=1)
    stale = score_symbol(make_fundamentals(last_updated='2020-01-01'), cfg)
    assert stale.stale is True


def test_rank_candidates_orders_by_combined_score_desc():
    cfg = ScoringConfig()
    strong = make_fundamentals(symbol='SCOM', yield_ttm_pct=8.0, years_consecutive_paid=10)
    weak = make_fundamentals(symbol='EQTY', yield_ttm_pct=3.0, years_consecutive_paid=1)
    ranked = rank_candidates([weak, strong], cfg, top_n=5)
    assert [c.symbol for c in ranked] == ['SCOM', 'EQTY']


def test_rank_candidates_respects_top_n():
    cfg = ScoringConfig(min_avg_daily_volume=0)
    many = [make_fundamentals(symbol=f'SYM{i}', yield_ttm_pct=1.0 + i) for i in range(10)]
    ranked = rank_candidates(many, cfg, top_n=3)
    assert len(ranked) == 3


def test_rank_candidates_excludes_gated_symbols():
    cfg = ScoringConfig(min_avg_daily_volume=50_000)
    ok = make_fundamentals(symbol='SCOM', avg_daily_volume=100_000)
    illiquid = make_fundamentals(symbol='EQTY', avg_daily_volume=100)
    ranked = rank_candidates([ok, illiquid], cfg, top_n=5)
    assert [c.symbol for c in ranked] == ['SCOM']
