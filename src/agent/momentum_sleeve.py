"""The momentum sleeve: 12-month relative strength, run live (US account).

Why this exists: of the approaches in this codebase, momentum over a year
has the strongest published record (time-series momentum, Moskowitz, Ooi
and Pedersen; a century of evidence, Hurst, Ooi and Pedersen), and
momentum_rotation.py implemented it, with the volatility caution its critics
raise built in as an absolute filter. Only the backtest script ever used it.
The live agent traded fast technical rules, whose record on US shares after
costs is the weakest (Bessembinder and Chan), on ten closely related names.

This runs it as a sleeve: a share of the strategies' money (`share` of the
active share), rebalanced on the first US session of each month, on its own
universe of liquid funds chosen so the books never overlap:

- not the funds the technical strategies trade (data_manager.symbols), or a
  technical sell signal would sell a sleeve holding;
- not the core's funds (core_satellite.core_symbols).

Each month it ranks the universe by 12-1 momentum (the return from about
thirteen months ago to one month ago, in 21-trading-day steps), holds the
top `top_n` that are still rising, equal weight, and holds cash for the rest.
Positions carry no stop-loss: the monthly ranking and its absolute filter
are the exit, as the research tested it (a daily stop would turn it into a
different strategy, as it did crypto's trend). It never borrows, and while
the account's daily loss or drawdown limit is reached it buys nothing.
"""
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.agent.momentum_rotation import RotationConfig, config_from_dict as rotation_config

logger = logging.getLogger(__name__)

STRATEGY = 'momentum_rotation'
BLOCK = 21  # trading days in one "month" step

# Liquid funds outside the technical list and the core: the remaining US
# sectors, small caps, developed and emerging markets, long Treasuries and
# broad commodities, so the ranking chooses among things that move differently.
DEFAULT_UNIVERSE = ('XLK', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC',
                    'IWM', 'EFA', 'EEM', 'TLT', 'DBC')


@dataclass
class SleeveConfig:
    enabled: bool = False
    share: float = 0.3                 # of the strategies' (active) money
    universe: Tuple[str, ...] = DEFAULT_UNIVERSE
    rotation: RotationConfig = field(default_factory=RotationConfig)
    min_trade_usd: float = 100.0

    @property
    def bars_needed(self) -> int:
        """Daily bars for the monthly ranking, plus a step to spare."""
        return (self.rotation.lookback + self.rotation.skip_recent + 2) * BLOCK


def config_from_dict(cfg: Optional[Dict[str, Any]], technical: Iterable[str] = (),
                     core: Iterable[str] = ()) -> SleeveConfig:
    """Read `momentum_sleeve`; funds another book trades are left out."""
    cfg = cfg or {}
    taken = {s.upper() for s in technical} | {s.upper() for s in core}
    universe = tuple(s.upper() for s in cfg.get('universe', DEFAULT_UNIVERSE))
    clash = [s for s in universe if s in taken]
    if clash:
        logger.warning(f"Momentum sleeve: {clash} are traded by another book; left out of the sleeve "
                       f"so no two books share a position")
        universe = tuple(s for s in universe if s not in taken)
    share = min(max(float(cfg.get('share', 0.3)), 0.0), 1.0)
    rot = rotation_config(cfg.get('rotation'))
    return SleeveConfig(enabled=bool(cfg.get('enabled', False)) and bool(universe) and share > 0,
                        share=share, universe=universe, rotation=rot,
                        min_trade_usd=float(cfg.get('min_trade_usd', 100.0)))


def monthly_closes(daily: Sequence[float], block: int = BLOCK) -> List[float]:
    """Every `block`-th close counting back from the latest, oldest first:
    month-sized steps from daily bars."""
    closes = [c for c in (daily or []) if c and c > 0]
    return closes[::-1][::block][::-1]


def due(last_rebalance: Optional[str], today: date) -> Optional[str]:
    """'build' before the first rebalance, 'monthly' on the first session of
    a new month, else None."""
    if not last_rebalance:
        return 'build'
    last = date.fromisoformat(last_rebalance[:10])
    return 'monthly' if (today.year, today.month) != (last.year, last.month) else None


def targets(daily_by_symbol: Dict[str, Sequence[float]], cfg: SleeveConfig) -> Dict[str, Any]:
    """This month's weights, and the ranking behind them (for the record)."""
    from src.agent.momentum_rotation import rank_universe, target_weights
    monthly = {s: monthly_closes(c) for s, c in daily_by_symbol.items()}
    ranking = [{'symbol': r.symbol, 'momentum': round(r.momentum, 4), 'rank': r.rank}
               for r in rank_universe(monthly, cfg.rotation)]
    return {'weights': target_weights(monthly, cfg.rotation), 'ranking': ranking}
