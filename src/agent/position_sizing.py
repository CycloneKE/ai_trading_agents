"""Dollar-notional position sizing with fractional-share support.

Sizing is done in dollars, then converted to (possibly fractional) share
quantity, so every strategy works identically on a $2,000 account and a
$200,000 one. Broker constraints encoded here:

- Alpaca fractional orders must be DAY time-in-force (GTC fractional is
  rejected), and fractional is not available in extended hours.
- Sub-minimum notionals are skipped: fee/spread drag makes them noise.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MIN_NOTIONAL = 5.0      # skip orders worth less than this
FRACTIONAL_DECIMALS = 4          # Alpaca accepts up to 9; 4 is plenty


@dataclass
class SizedOrder:
    quantity: float
    time_in_force: str  # 'day' when fractional (broker requirement), else caller's choice
    is_fractional: bool
    notional: float


def size_order(target_value: float, price: float,
               min_notional: float = DEFAULT_MIN_NOTIONAL,
               allow_fractional: bool = True,
               preferred_tif: str = 'day') -> Optional[SizedOrder]:
    """Convert a dollar target into an executable share quantity.

    Returns None when the order should be skipped (bad inputs or below the
    minimum notional).
    """
    if not price or price <= 0 or not target_value or target_value <= 0:
        return None
    if target_value < min_notional:
        logger.info(
            f"Skipping order: target ${target_value:.2f} below min notional ${min_notional:.2f}")
        return None

    quantity = target_value / price

    if not allow_fractional:
        quantity = float(int(quantity))
        if quantity < 1:
            logger.info(
                f"Skipping order: <1 whole share (${target_value:.2f} at ${price:.2f}) "
                f"and fractional disabled")
            return None
        return SizedOrder(quantity=quantity, time_in_force=preferred_tif,
                          is_fractional=False, notional=quantity * price)

    quantity = round(quantity, FRACTIONAL_DECIMALS)
    if quantity <= 0:
        return None
    is_fractional = quantity != int(quantity)
    # Fractional orders must be DAY per Alpaca; whole-share keeps caller's TIF.
    tif = 'day' if is_fractional else preferred_tif
    return SizedOrder(quantity=quantity, time_in_force=tif,
                      is_fractional=is_fractional, notional=round(quantity * price, 2))


# --- volatility-scaled sizing ------------------------------------------------

# Fraction of equity risked per position if the stop is hit.
#
# This interacts with max_position_pct, and the two must be chosen together.
# Notional is equity * risk_per_trade / (stop_atr_mult * atr_pct), so with
# 2.5x ATR stops and a 5% position cap the cap binds for anything quieter
# than about 4% ATR. Below that crossover every position is the same 5% and
# volatility scaling does nothing; above it, size falls as volatility rises.
#
# The shipped config previously held both a 5% position cap and a 2%
# cash_policy.base_risk_per_trade. Those cannot both be active: 2% risk with
# a 2.5x ATR stop on a 2.6%-ATR instrument implies a 31% position, six times
# the cap. 0.5% keeps the cap as a genuine concentration ceiling rather than
# the thing that decides every size.
DEFAULT_RISK_PER_TRADE = 0.005
DEFAULT_TARGET_VOL = 0.02          # the ATR% a "normal" position is sized against


def volatility_scaled_value(equity: float, atr_pct: Optional[float],
                            confidence: float = 1.0,
                            risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
                            stop_atr_mult: float = 2.5,
                            max_position_pct: float = 0.05,
                            min_position_pct: float = 0.0,
                            target_vol: float = DEFAULT_TARGET_VOL) -> float:
    """Notional to allocate so each position risks a comparable amount.

    A flat 5% of equity in every name means a position's contribution to
    portfolio risk is decided by whatever volatility that instrument happens
    to have: 5% in a crypto pair moving 6% a day is several times the risk of
    5% in an NSE bank moving 1%. Sizing off the stop distance equalises that.

    The position is sized so that being stopped out costs approximately
    `risk_per_trade` of equity:

        stop distance = stop_atr_mult * atr_pct
        notional      = equity * risk_per_trade / stop distance

    then scaled by `confidence` and capped at `max_position_pct`. Without an
    ATR reading it falls back to `max_position_pct * confidence`, the previous
    behaviour, so a cold start still trades.
    """
    if not equity or equity <= 0:
        return 0.0
    confidence = max(0.0, min(1.0, float(confidence or 0.0)))
    if confidence <= 0:
        return 0.0

    cap = equity * max_position_pct
    if not atr_pct or atr_pct <= 0:
        return cap * confidence

    stop_distance = max(1e-6, stop_atr_mult * atr_pct)
    notional = equity * risk_per_trade / stop_distance
    notional *= confidence

    floor = equity * min_position_pct
    return max(floor, min(cap, notional))


def risk_contribution(notional: float, equity: float, atr_pct: Optional[float],
                      stop_atr_mult: float = 2.5) -> float:
    """Fraction of equity at risk if this position is stopped out.

    The figure `volatility_scaled_value` targets, exposed so callers can
    report or assert on it.
    """
    if not equity or equity <= 0 or not atr_pct or atr_pct <= 0:
        return 0.0
    return (notional * stop_atr_mult * atr_pct) / equity
