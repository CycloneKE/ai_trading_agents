"""The pullback sleeve: buy short dips in an S&P 500 fund (US account).

Why this exists: scripts/rule_research.py tested well-documented rules on
twenty years of fund prices (December 2007 to November 2026), each at its
published settings, after costs. The two-day RSI pullback rule (Connors and
Alvarez, Short Term Trading Strategies That Work, 2008) had a Sharpe ratio
of 0.50 over cash, the same in both halves, 0.35 to 0.50 when acted on one
to five days late, and a worst fall of 15% against 53% for holding the
index, while invested only about 11% of the time. Because it is out of the
market most days, it adds a different kind of return beside the other books.

The rule, unchanged:

- buy when the close is above its 200-day average (the market is in an
  uptrend) and the 2-day RSI is below 10 (it has just had a sharp dip);
- sell when the close rises above its 5-day average (the dip has bounced).

It is decided once a day, in the last minutes of the US session
(`decide_after`, New York time), using today's live price as the close, so
the order fills close to the closing price the research traded at. It holds
its own fund (IVV by default), never one another book trades, with `share`
of the strategies' (active) money, all in when in. No stop-loss: the 5-day
exit is the rule's own, as tested. While a loss limit is reached it buys
nothing; its sells always go.

Pure functions only; orders are placed by TradingAgent._run_pullback_cycle.
"""
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

STRATEGY = 'rsi2_pullback'


@dataclass
class PullbackConfig:
    enabled: bool = False
    symbol: str = 'IVV'
    share: float = 0.15            # of the strategies' (active) money
    trend_days: int = 200
    exit_days: int = 5
    rsi_days: int = 2
    rsi_entry: float = 10.0
    decide_after: str = '15:45'    # New York time, until the 16:00 close
    min_trade_usd: float = 100.0

    @property
    def bars_needed(self) -> int:
        return self.trend_days + 60


def config_from_dict(cfg: Optional[Dict[str, Any]], taken: Iterable[str] = ()) -> PullbackConfig:
    """Read `pullback_sleeve`. Off if its fund is traded by another book."""
    cfg = cfg or {}
    symbol = str(cfg.get('symbol', 'IVV')).upper()
    clash = symbol in {s.upper() for s in taken}
    if clash:
        logger.warning(f"Pullback sleeve: {symbol} is traded by another book; the sleeve stays off "
                       f"so no two books share a position")
    share = min(max(float(cfg.get('share', 0.15)), 0.0), 1.0)
    return PullbackConfig(
        enabled=bool(cfg.get('enabled', False)) and not clash and share > 0,
        symbol=symbol, share=share,
        trend_days=int(cfg.get('trend_days', 200)), exit_days=int(cfg.get('exit_days', 5)),
        rsi_days=int(cfg.get('rsi_days', 2)), rsi_entry=float(cfg.get('rsi_entry', 10.0)),
        decide_after=str(cfg.get('decide_after', '15:45')),
        min_trade_usd=float(cfg.get('min_trade_usd', 100.0)))


def rsi(closes: Sequence[float], days: int) -> Optional[float]:
    """Wilder's RSI of the last close: average gains against average losses,
    each smoothed with weight 1/days (the same as pandas ewm(alpha=1/days,
    adjust=False), which scripts/rule_research.py tested)."""
    if len(closes) < days + 1:
        return None
    alpha = 1.0 / days
    up = down = None
    for a, b in zip(closes, closes[1:]):
        gain, loss = max(b - a, 0.0), max(a - b, 0.0)
        up = gain if up is None else (1 - alpha) * up + alpha * gain
        down = loss if down is None else (1 - alpha) * down + alpha * loss
    if not down:
        return 100.0 if up else 50.0
    return 100.0 - 100.0 / (1.0 + up / down)


def decide(closes: Sequence[float], holding: bool, cfg: PullbackConfig) -> Dict[str, Any]:
    """Today's action from closes ending with today's: 'buy', 'sell' or
    None, and the readings behind it."""
    px = [float(c) for c in closes if c and c > 0]
    out: Dict[str, Any] = {'action': None, 'close': px[-1] if px else None}
    if len(px) < max(cfg.trend_days, cfg.exit_days, cfg.rsi_days + 1):
        out['why'] = 'not enough history'
        return out
    close = px[-1]
    trend = sum(px[-cfg.trend_days:]) / cfg.trend_days
    exit_avg = sum(px[-cfg.exit_days:]) / cfg.exit_days
    reading = rsi(px[-(cfg.trend_days + 60):], cfg.rsi_days)
    out.update({'sma_trend': round(trend, 4), 'sma_exit': round(exit_avg, 4),
                'rsi': round(reading, 2) if reading is not None else None})
    if holding:
        if close > exit_avg:
            out['action'] = 'sell'
    elif close > trend and reading is not None and reading < cfg.rsi_entry:
        out['action'] = 'buy'
    return out


def in_window(now: datetime, cfg: PullbackConfig) -> bool:
    """A weekday between decide_after and the 16:00 close, New York time."""
    from zoneinfo import ZoneInfo
    et = now.astimezone(ZoneInfo('America/New_York'))
    if et.weekday() >= 5:
        return False
    hh, mm = (int(x) for x in cfg.decide_after.split(':'))
    minutes = et.hour * 60 + et.minute
    return hh * 60 + mm <= minutes < 16 * 60


def new_york_date(now: datetime) -> date:
    from zoneinfo import ZoneInfo
    return now.astimezone(ZoneInfo('America/New_York')).date()


def decided_today(last_decision: Optional[str], now: datetime) -> bool:
    return bool(last_decision) and last_decision[:10] == new_york_date(now).isoformat()


def replay_positions(closes: Sequence[float], cfg: PullbackConfig) -> list:
    """1.0 on each day the rule holds the fund (decided at that day's close),
    else 0.0: the live decision run over history, for the research script."""
    held, out = False, []
    window = cfg.trend_days + 60
    for t in range(len(closes)):
        action = decide(closes[max(0, t + 1 - window):t + 1], held, cfg)['action']
        if action == 'buy':
            held = True
        elif action == 'sell':
            held = False
        out.append(1.0 if held else 0.0)
    return out
