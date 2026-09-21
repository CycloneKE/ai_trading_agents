"""Deciding whether a symbol belongs in the trading universe.

The question a screen has to answer is not "does this look like a good
company" but "can this be traded profitably by this system, at this size,
through this broker". Three things usually decide it, and only the first is
obvious:

1. Cost. An NSE round trip runs several percent. A symbol whose typical
   move over the holding period is smaller than the cost of getting in and
   out cannot be traded at a profit no matter how good the signal. This is
   the check that disqualifies most NSE candidates and it is the one people
   skip.

2. Liquidity. A position that cannot be exited at the modelled price is not
   the position the backtest measured.

3. Diversification. A symbol that moves with something already held adds
   risk without adding opportunity; it doubles a bet rather than spreading
   one.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.agent.cost_model import classify, costs_for, round_trip_pct

logger = logging.getLogger(__name__)

ADD = 'ADD'
MARGINAL = 'MARGINAL'
REJECT = 'REJECT'


@dataclass
class ScreenResult:
    symbol: str
    market: str
    verdict: str
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict in (ADD, MARGINAL)


def _ann_factor(periods_per_year: int) -> float:
    return math.sqrt(max(1, periods_per_year))


def cost_hurdle(df: pd.DataFrame, symbol: str, config: Dict[str, Any],
                holding_periods: int = 20) -> Dict[str, Any]:
    """Compare the cost of a round trip against the move it must clear.

    `holding_periods` is how long a position is expected to be held, in
    bars. The comparison is against the *median* absolute move over that
    span rather than the mean, because a handful of large moves should not
    make a symbol look tradeable when most trades will see a typical one.
    """
    closes = df['close']
    if len(closes) <= holding_periods:
        return {'insufficient_history': True}

    fwd = (closes.shift(-holding_periods) / closes - 1.0).dropna()
    typical_move = float(fwd.abs().median())
    upside_move = float(fwd.abs().quantile(0.75))
    rt = round_trip_pct(symbol, config)

    return {
        'round_trip_pct': rt,
        'holding_periods': holding_periods,
        'typical_abs_move': typical_move,
        'upper_quartile_move': upside_move,
        # How many times the cost the typical move is. Below 1 the average
        # trade cannot pay for itself; below about 3 the edge has to be
        # exceptional to survive.
        'move_to_cost_ratio': (typical_move / rt) if rt > 0 else float('inf'),
    }


def correlation_with_universe(candidate: pd.Series,
                              universe: Dict[str, pd.DataFrame],
                              min_overlap: int = 40) -> Dict[str, float]:
    """Return correlation of the candidate's returns with each existing name.

    Computed on overlapping dates only, and skipped where the overlap is too
    short to mean anything.
    """
    cand_ret = candidate.pct_change().dropna()
    out = {}
    for sym, df in (universe or {}).items():
        other = df['close'].pct_change().dropna()
        joined = pd.concat([cand_ret, other], axis=1, join='inner').dropna()
        if len(joined) < min_overlap:
            continue
        c = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        if c == c:                      # NaN-safe
            out[sym] = float(c)
    return out


def screen(symbol: str, df: pd.DataFrame, config: Dict[str, Any],
           universe: Optional[Dict[str, pd.DataFrame]] = None,
           periods_per_year: int = 252,
           holding_periods: int = 20,
           min_move_to_cost: float = 3.0,
           max_correlation: float = 0.85,
           min_avg_volume: Optional[float] = None) -> ScreenResult:
    """Full screen for one candidate. Verdict plus the numbers behind it."""
    market = classify(symbol, config)
    reasons: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, Any] = {}

    closes = df['close']
    rets = closes.pct_change().dropna()
    metrics['bars'] = int(len(closes))
    metrics['start'] = str(df.index.min().date())
    metrics['end'] = str(df.index.max().date())
    metrics['total_return'] = float(closes.iloc[-1] / closes.iloc[0] - 1.0)
    metrics['annual_volatility'] = (float(rets.std(ddof=1) * _ann_factor(periods_per_year))
                                    if len(rets) > 1 else 0.0)

    # --- 1. cost ----------------------------------------------------------
    hurdle = cost_hurdle(df, symbol, config, holding_periods)
    metrics['cost'] = hurdle
    if hurdle.get('insufficient_history'):
        reasons.append(f"fewer than {holding_periods} bars of forward history "
                       f"to measure a typical move against")
    else:
        ratio = hurdle['move_to_cost_ratio']
        rt = hurdle['round_trip_pct']
        if ratio < 1.0:
            reasons.append(
                f"typical {holding_periods}-bar move of "
                f"{hurdle['typical_abs_move']:.2%} is smaller than the "
                f"{rt:.2%} round-trip cost: the average trade cannot pay for itself")
        elif ratio < min_move_to_cost:
            warnings.append(
                f"typical move is only {ratio:.1f}x the {rt:.2%} round trip "
                f"(want {min_move_to_cost:.0f}x): the edge must be exceptional")

    # --- 2. liquidity -----------------------------------------------------
    if 'volume' in df.columns:
        vol = pd.to_numeric(df['volume'], errors='coerce').dropna()
        avg_vol = float(vol.tail(60).mean()) if len(vol) else 0.0
        metrics['avg_volume'] = avg_vol
        if min_avg_volume and avg_vol < min_avg_volume:
            reasons.append(f"average volume {avg_vol:,.0f} below the "
                           f"{min_avg_volume:,.0f} minimum")
        elif avg_vol == 0:
            warnings.append("volume column is present but all zero, so liquidity "
                            "could not be checked")
    else:
        warnings.append("no volume data, so liquidity could not be checked")

    # --- 3. diversification ----------------------------------------------
    corrs = correlation_with_universe(closes, universe or {})
    if corrs:
        worst_sym = max(corrs, key=lambda s: abs(corrs[s]))
        metrics['max_correlation'] = corrs[worst_sym]
        metrics['max_correlation_with'] = worst_sym
        metrics['correlations'] = corrs
        if abs(corrs[worst_sym]) >= max_correlation:
            warnings.append(
                f"{corrs[worst_sym]:.2f} correlated with {worst_sym}: this "
                f"doubles an existing bet rather than spreading one")

    # --- 4. data quality --------------------------------------------------
    from src.utils.price_files import data_quality
    quality = data_quality(df)
    metrics['quality'] = {k: (str(v) if hasattr(v, 'isoformat') else v)
                          for k, v in quality.items()}
    warnings.extend(quality['issues'])

    # --- verdict ----------------------------------------------------------
    if reasons:
        verdict = REJECT
    elif warnings:
        verdict = MARGINAL
    else:
        verdict = ADD

    # An unverified cost schedule undermines the headline check above, so it
    # caps the verdict rather than merely appearing in the notes.
    cost_cfg = costs_for(symbol, config)
    if not cost_cfg.get('verified', False) and verdict == ADD:
        verdict = MARGINAL
        warnings.append(
            f"costs for '{market}' are an unverified placeholder, so the "
            f"cost check above is provisional")

    return ScreenResult(symbol=symbol, market=market, verdict=verdict,
                        reasons=reasons, warnings=warnings, metrics=metrics)


def format_result(r: ScreenResult) -> str:
    m = r.metrics
    lines = [f"{r.symbol}  [{r.market}]  ->  {r.verdict}", ""]
    if 'bars' in m:
        lines.append(f"  history      {m['bars']} bars, {m['start']} to {m['end']}")
        lines.append(f"  total return {m['total_return']:+.1%}")
        lines.append(f"  volatility   {m['annual_volatility']:.1%} annualised")
    c = m.get('cost') or {}
    if 'round_trip_pct' in c:
        lines.append(f"  round trip   {c['round_trip_pct']:.2%}")
        lines.append(f"  typical move {c['typical_abs_move']:.2%} over "
                     f"{c['holding_periods']} bars "
                     f"({c['move_to_cost_ratio']:.1f}x the cost)")
    if 'avg_volume' in m:
        lines.append(f"  avg volume   {m['avg_volume']:,.0f}")
    if 'max_correlation' in m:
        lines.append(f"  max corr     {m['max_correlation']:+.2f} "
                     f"with {m['max_correlation_with']}")
    for reason in r.reasons:
        lines.append(f"  REJECT: {reason}")
    for w in r.warnings:
        lines.append(f"  warning: {w}")
    return "\n".join(lines)
