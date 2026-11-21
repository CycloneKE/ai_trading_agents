"""Crypto position caps: size each coin by how much it swings, trim weekly.

Why this exists: the agent sizes a crypto position once, at entry, and never
again. crypto_trend holds a coin for months while it trends, so a winner can
grow many times over: in the agent's own backtest a 2015 Bitcoin position
grew about 26 times and was most of the account when the 2018 crash took it
down 42%. After such a gain the emergency stop, set from the entry price, is
far below the market, so only the trend exit protects the gain.

The fix, the way trend-following funds size (Hurst, Ooi and Pedersen 2017:
each position targets the same volatility):

- a coin's cap is its usual maximum (crypto_trend's max_position_size of the
  strategies' money) scaled by vol_target / its yearly volatility over the
  last vol_days days, never above the maximum: a coin swinging 80% a year
  gets 50/80 of the room of one swinging 50%;
- a buy is never larger than the cap;
- once a week, a held coin worth more than its cap by more than trim_buffer
  is sold back down to the cap. Never up: the trend rule's buy is the only
  way into a coin.

scripts/crypto_study.py's harness, replaying this on BTC, ETH and SOL from
2015 to 2026 (each coin an equal share, 0.35% costs a side), cut the worst
fall from 65% to 43% and raised the Sharpe ratio from 1.19 to 1.27, while the
yearly return fell from 74% to 51%: a smaller, steadier book.

Pure functions only; TradingAgent._run_crypto_trim_cycle places the trims.
"""
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional, Sequence

STRATEGY = 'crypto_trim'
DAYS = 365          # crypto trades every day


@dataclass
class CryptoRiskConfig:
    enabled: bool = False
    vol_target: float = 0.50
    vol_days: int = 60
    trim_buffer: float = 0.20
    min_trade_usd: float = 25.0


def config_from_dict(cfg: Optional[Dict[str, Any]]) -> CryptoRiskConfig:
    cfg = cfg or {}
    return CryptoRiskConfig(
        enabled=bool(cfg.get('enabled', False)),
        vol_target=max(float(cfg.get('vol_target', 0.50)), 0.0),
        vol_days=max(int(cfg.get('vol_days', 60)), 5),
        trim_buffer=max(float(cfg.get('trim_buffer', 0.20)), 0.0),
        min_trade_usd=max(float(cfg.get('min_trade_usd', 25.0)), 0.0))


def annual_vol(closes: Sequence[float], days: int) -> Optional[float]:
    """Yearly volatility of the last `days` daily returns, or None with too
    little history."""
    px = [float(c) for c in (closes or []) if c and c > 0]
    if len(px) <= days:
        return None
    rets = [px[i] / px[i - 1] - 1.0 for i in range(len(px) - days, len(px))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
    vol = math.sqrt(var) * math.sqrt(DAYS)
    return vol if vol > 0 else None


def cap_value(max_value: float, vol: Optional[float], cfg: CryptoRiskConfig) -> float:
    """The most a coin may be worth: `max_value` scaled down for a coin that
    swings more than vol_target. Without a volatility reading, max_value."""
    if not vol or not cfg.vol_target:
        return max_value
    return max_value * min(cfg.vol_target / vol, 1.0)


def trim_quantity(value: float, price: float, cap: float, cfg: CryptoRiskConfig) -> float:
    """Units to sell to bring a holding back to its cap: none while it is
    within trim_buffer of the cap, or when the trim would be tiny."""
    if price <= 0 or cap <= 0 or value <= cap * (1.0 + cfg.trim_buffer):
        return 0.0
    excess = value - cap
    if excess < cfg.min_trade_usd:
        return 0.0
    return round(excess / price, 8)


def due(last_trim: Optional[str], today: date) -> bool:
    """Once per ISO week."""
    if not last_trim:
        return True
    return date.fromisoformat(last_trim[:10]).isocalendar()[:2] != today.isocalendar()[:2]
