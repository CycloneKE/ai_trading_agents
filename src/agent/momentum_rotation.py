"""Cross-sectional momentum rotation.

Distinct from the existing `momentum` technical strategy, which is
*time-series* momentum: it asks "is this symbol going up?" one symbol at a
time. This asks "which of these symbols is going up the most relative to the
others?", holds the leaders, and rebalances on a fixed schedule.

The cross-sectional form is the one with decades of out-of-sample evidence
behind it, and it suits this account's constraints: it rebalances monthly
rather than every 60 seconds, so on a zero-commission US broker the cost
drag is negligible, and it needs no forecast, only a ranking.

The lookback deliberately skips the most recent period ("12-1"). Short-term
returns tend to reverse, so including the latest month dilutes the signal
with noise that is about to mean-revert.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class RotationConfig:
    lookback: int = 12          # periods of history the ranking looks back over
    skip_recent: int = 1        # periods nearest the present to exclude
    top_n: int = 3              # how many leaders to hold
    min_momentum: float = 0.0   # a leader must still be rising to be held
    absolute_filter: bool = True  # drop names below min_momentum even if top-ranked
    max_weight: float = 0.5     # cap on any single position's share of the sleeve

    @property
    def required_history(self) -> int:
        """Bars needed before a ranking can be formed."""
        return self.lookback + 1


@dataclass
class Ranked:
    symbol: str
    momentum: float
    rank: int
    weight: float = 0.0


@dataclass
class RotationState:
    """What the strategy currently holds, as target weights by symbol."""
    weights: Dict[str, float] = field(default_factory=dict)
    last_rebalance: Optional[Any] = None


def momentum_score(prices: Sequence[float], cfg: RotationConfig) -> Optional[float]:
    """Total return over the lookback window, excluding the most recent
    `skip_recent` periods. None when there is not enough history.

    With lookback=12 and skip_recent=1 on monthly bars this is the classic
    12-1: the return from 13 months ago to 1 month ago.
    """
    prices = [p for p in (prices or []) if p and p > 0]
    need = cfg.lookback + cfg.skip_recent
    if len(prices) < need + 1:
        return None
    end = -(cfg.skip_recent + 1)          # -1 when skipping one period
    start = end - cfg.lookback
    if -start > len(prices):
        return None
    past, recent = prices[start], prices[end]
    if past <= 0:
        return None
    return (recent / past) - 1.0


def rank_universe(prices_by_symbol: Dict[str, Sequence[float]],
                  cfg: RotationConfig) -> List[Ranked]:
    """Rank every symbol with enough history, strongest first.

    Symbols lacking history are omitted rather than ranked last: a name with
    no track record is not the same as a name with a bad one, and treating
    them alike would systematically short new listings.
    """
    scored = []
    for symbol, prices in (prices_by_symbol or {}).items():
        m = momentum_score(prices, cfg)
        if m is not None:
            scored.append((symbol, m))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [Ranked(symbol=s, momentum=m, rank=i + 1)
            for i, (s, m) in enumerate(scored)]


def select(prices_by_symbol: Dict[str, Sequence[float]],
           cfg: RotationConfig) -> List[Ranked]:
    """The names to hold this period, with target weights.

    Applies the absolute filter before sizing: in a market where everything
    is falling, the "best" name is still falling, and holding it because it
    ranks first is how a relative-strength rule turns into a losing long.
    Filtered out, the sleeve simply holds less, or nothing.
    """
    ranked = rank_universe(prices_by_symbol, cfg)
    if cfg.absolute_filter:
        ranked = [r for r in ranked if r.momentum > cfg.min_momentum]
    chosen = ranked[:max(0, cfg.top_n)]
    if not chosen:
        return []

    weight = 1.0 / len(chosen)
    if cfg.max_weight > 0:
        weight = min(weight, cfg.max_weight)
    for r in chosen:
        r.weight = weight
    return chosen


def target_weights(prices_by_symbol: Dict[str, Sequence[float]],
                   cfg: RotationConfig) -> Dict[str, float]:
    """Symbol -> target share of the sleeve. Empty means hold cash."""
    return {r.symbol: r.weight for r in select(prices_by_symbol, cfg)}


def rebalance_orders(current_weights: Dict[str, float],
                     target: Dict[str, float],
                     portfolio_value: float,
                     prices: Dict[str, float],
                     min_trade_fraction: float = 0.005) -> Dict[str, float]:
    """Notional deltas per symbol to move from current to target.

    Positive buys, negative sells. Deltas below `min_trade_fraction` of the
    portfolio are dropped: rebalancing a position by a fraction of a percent
    pays spread and commission for no meaningful change in exposure, and on
    a monthly schedule that churn compounds.
    """
    if portfolio_value <= 0:
        return {}
    orders = {}
    for symbol in set(current_weights) | set(target):
        delta_w = target.get(symbol, 0.0) - current_weights.get(symbol, 0.0)
        if abs(delta_w) < min_trade_fraction:
            continue
        price = prices.get(symbol)
        if not price or price <= 0:
            # Cannot price it, so cannot trade it. Leaving the position
            # untouched is safer than guessing at a stale price.
            logger.debug(f"Skipping rebalance of {symbol}: no usable price")
            continue
        orders[symbol] = delta_w * portfolio_value
    return orders


def config_from_dict(d: Optional[Dict[str, Any]]) -> RotationConfig:
    d = d or {}
    return RotationConfig(
        lookback=max(1, int(d.get('lookback', 12))),
        skip_recent=max(0, int(d.get('skip_recent', 1))),
        top_n=max(0, int(d.get('top_n', 3))),
        min_momentum=float(d.get('min_momentum', 0.0)),
        absolute_filter=bool(d.get('absolute_filter', True)),
        max_weight=float(d.get('max_weight', 0.5)),
    )
