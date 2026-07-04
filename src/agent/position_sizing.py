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
