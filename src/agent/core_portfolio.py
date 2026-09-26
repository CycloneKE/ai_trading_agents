"""The core of the core-satellite split (US account).

Most of the evidence on active trading says the same thing: after costs,
few strategies beat simply holding an index. So part of the account is not
traded at all. It holds low-cost index funds, and the agent's strategies
trade the rest, the "satellite". The gap analysis set the satellite at 50
to 60% of the money; the shipped setting is 55%.

The core holds a stock fund, a Treasury bond fund and a gold fund, weighted
so each carries about the same risk: a fund that swings twice as much gets
half the money (inverse-volatility weighting). It is rebalanced once a month,
or sooner if a fund drifts well away from its target weight, so it trades a
handful of times a year.

The core's funds are deliberately not in the active universe (VOO rather
than SPY, which the strategies trade), so the two books never share a
position: the strategies never sell a core holding, and the core never
undoes a strategy's trade. Any core fund that is also in the active universe
is dropped from the core with a warning.
"""
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# S&P 500, 7-10 year US Treasuries, gold: three low-cost funds that tend to
# move differently from each other.
DEFAULT_SYMBOLS = ('VOO', 'IEF', 'IAU')
ACTIVE_SHARE_DEFAULT = 0.55
STRATEGY = 'core'


@dataclass
class CoreConfig:
    enabled: bool = False
    active_share: float = ACTIVE_SHARE_DEFAULT  # share of equity the strategies trade
    symbols: Tuple[str, ...] = DEFAULT_SYMBOLS
    vol_lookback: int = 60        # daily returns used to measure each fund's swings
    max_weight: float = 0.6       # no fund above this share of the core
    drift_band: float = 0.05      # rebalance early when a weight is this far off target
    min_trade_usd: float = 100.0  # smaller adjustments are not worth an order

    @property
    def core_share(self) -> float:
        return 1.0 - self.active_share


def config_from_dict(cfg: Optional[Dict[str, Any]],
                     active_universe: Iterable[str] = ()) -> CoreConfig:
    """Read `core_satellite` from config; out-of-range values are clamped."""
    cfg = cfg or {}
    active = {s.upper() for s in active_universe}
    symbols = tuple(s.upper() for s in cfg.get('core_symbols', DEFAULT_SYMBOLS))
    clash = [s for s in symbols if s in active]
    if clash:
        logger.warning(f"Core funds {clash} are also traded by the strategies; "
                       f"leaving them out of the core so the two books never share a position")
        symbols = tuple(s for s in symbols if s not in active)
    share = float(cfg.get('active_share', ACTIVE_SHARE_DEFAULT))
    return CoreConfig(
        enabled=bool(cfg.get('enabled', False)) and bool(symbols),
        active_share=min(max(share, 0.0), 1.0),
        symbols=symbols,
        vol_lookback=int(cfg.get('vol_lookback', 60)),
        max_weight=float(cfg.get('max_weight', 0.6)),
        drift_band=float(cfg.get('drift_band', 0.05)),
        min_trade_usd=float(cfg.get('min_trade_usd', 100.0)),
    )


def active_equity(equity: float, cfg: CoreConfig) -> float:
    """What the strategies size their trades from: their share of equity."""
    return equity * cfg.active_share if cfg.enabled else equity


def realised_vol(closes: Sequence[float], lookback: int) -> Optional[float]:
    """Annualised volatility of daily log returns over the last `lookback`
    returns, or None with fewer than 20 returns to go on."""
    px = [float(c) for c in closes if c and c > 0]
    rets = [math.log(b / a) for a, b in zip(px, px[1:])][-lookback:]
    if len(rets) < 20:
        return None
    vol = pstdev(rets) * math.sqrt(252)
    return vol if vol > 0 else None


def _cap(weights: Dict[str, float], cap: float) -> Dict[str, float]:
    """Cap each weight and share the excess among the rest, pro rata."""
    w = dict(weights)
    if cap * len(w) < 1:  # the cap cannot hold; equal weight is the closest
        return {s: 1 / len(w) for s in w}
    for _ in range(len(w)):
        over = {s: v for s, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        free = {s: v for s, v in w.items() if s not in over}
        for s in over:
            w[s] = cap
        total_free = sum(free.values())
        for s, v in free.items():
            w[s] = v + excess * v / total_free
    return w


def inverse_vol_weights(closes_by_symbol: Dict[str, Sequence[float]],
                        lookback: int = 60, max_weight: float = 0.6) -> Dict[str, float]:
    """Target weights with about equal risk from each fund.

    A fund without enough history to measure is given the average of the
    others' volatility, so a new listing is held rather than left out. With
    no history for any fund the core is equal-weighted.
    """
    symbols = list(closes_by_symbol)
    if not symbols:
        return {}
    vols = {s: realised_vol(closes_by_symbol[s], lookback) for s in symbols}
    known = [v for v in vols.values() if v]
    if not known:
        return {s: 1 / len(symbols) for s in symbols}
    fill = sum(known) / len(known)
    inv = {s: 1 / (vols[s] or fill) for s in symbols}
    total = sum(inv.values())
    return _cap({s: v / total for s, v in inv.items()}, max_weight)


def current_weights(values: Dict[str, float]) -> Dict[str, float]:
    total = sum(v for v in values.values() if v > 0)
    return {s: (v / total if total else 0.0) for s, v in values.items()}


def rebalance_reason(last_rebalance: Optional[str], today: date,
                     held_values: Dict[str, float], targets: Dict[str, float],
                     drift_band: float) -> Optional[str]:
    """Why the core should trade today, or None.

    'build' when it has never been bought, 'monthly' on the first session of
    a new month, 'drift' when a fund's weight is more than `drift_band` from
    its target.
    """
    if not last_rebalance or not any(v > 0 for v in held_values.values()):
        return 'build'
    last = date.fromisoformat(last_rebalance[:10])
    if (today.year, today.month) != (last.year, last.month):
        return 'monthly'
    now = current_weights({s: held_values.get(s, 0.0) for s in targets})
    if any(abs(now.get(s, 0.0) - w) > drift_band for s, w in targets.items()):
        return 'drift'
    return None


def plan_orders(targets: Dict[str, float], core_value: float,
                held: Dict[str, float], prices: Dict[str, float],
                min_trade_usd: float = 100.0) -> List[Dict[str, Any]]:
    """Orders that move the core to its targets, sells first.

    `held` is the quantity of each fund held; a fund held but no longer a
    target is sold. A sell never exceeds what is held. Adjustments under
    `min_trade_usd` are skipped. Returns dicts with symbol, side, notional
    and, for sells, quantity.
    """
    orders = []
    for sym in sorted(set(targets) | set(held)):
        price = prices.get(sym)
        if not price or price <= 0:
            continue
        qty = float(held.get(sym, 0.0) or 0.0)
        diff = core_value * targets.get(sym, 0.0) - qty * price
        if abs(diff) < min_trade_usd:
            continue
        if diff < 0:
            sell_qty = min(qty, round(-diff / price, 6))
            if sell_qty > 0:
                orders.append({'symbol': sym, 'side': 'sell', 'quantity': sell_qty,
                               'notional': round(sell_qty * price, 2)})
        else:
            orders.append({'symbol': sym, 'side': 'buy', 'notional': round(diff, 2)})
    return sorted(orders, key=lambda o: o['side'] != 'sell')


def fit_to_cash(orders: List[Dict[str, Any]], cash: float,
                headroom: float = 0.02) -> List[Dict[str, Any]]:
    """Scale buys down to the cash on hand plus what the sells raise.

    Alpaca paper accounts allow margin, so an order larger than the cash
    would quietly borrow. The core never does: when the strategies already
    hold more than their share, the core is built from what is left and
    catches up at later rebalances. `headroom` allows for prices moving
    between the plan and the fill.
    """
    raised = sum(o['notional'] for o in orders if o['side'] == 'sell')
    budget = max(cash + raised, 0.0) * (1 - headroom)
    wanted = sum(o['notional'] for o in orders if o['side'] == 'buy')
    if wanted <= budget:
        return orders
    scale = budget / wanted if wanted else 0.0
    out = []
    for o in orders:
        if o['side'] == 'buy':
            o = {**o, 'notional': round(o['notional'] * scale, 2)}
            if o['notional'] <= 0:
                continue
        out.append(o)
    return out


def us_session_open(now: Optional[datetime] = None) -> bool:
    """Regular US session, 09:30-16:00 New York time, Monday to Friday.
    Holidays are not modelled; an order on one simply waits for the open."""
    from zoneinfo import ZoneInfo
    et = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo('America/New_York'))
    if et.weekday() >= 5:
        return False
    minutes = et.hour * 60 + et.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


class CoreState:
    """When the core last rebalanced and to what, kept in a small JSON file
    so a restart does not rebalance again."""

    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {}
        try:
            with open(path, encoding='utf-8') as f:
                self.data = json.load(f) or {}
        except (OSError, ValueError):
            self.data = {}

    @property
    def last_rebalance(self) -> Optional[str]:
        return self.data.get('last_rebalance')

    def save(self, **fields: Any) -> None:
        self.data.update(fields)
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)
