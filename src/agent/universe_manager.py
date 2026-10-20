"""What should the agent be trading?

`universe_scout` proposes anything that moved 3% today, with no screening for
whether the name is tradeable or whether it duplicates exposure the book
already has. With SPY, QQQ and DIA all tracked, eleven symbols carry roughly
seven independent bets — the wrong shape for an uncorrelated-return mandate.

Everything here produces PROPOSALS. Nothing auto-applies; every change goes
to the operator as an escalation.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_MIN_DOLLAR_VOLUME = 1_000_000.0
DEFAULT_MIN_PRICE = 5.0
DEFAULT_MAX_CORRELATION = 0.9

MIN_CORRELATION_BARS = 30


def is_screenable(closes: Optional[List[float]]) -> bool:
    """True when a price series can actually be correlation-screened.

    A series containing NaN or Infinity CANNOT be screened, and that must
    mean REJECT, not admit. Note that returning a neutral 0.0 correlation is
    not a safe fallback either: 0.0 reads as "uncorrelated" and admits the
    candidate just as readily as a NaN does. The only safe answer for data
    we cannot evaluate is to refuse it — the same treatment insufficient
    history already gets.
    """
    if not closes:
        return False
    finite = [c for c in closes if isinstance(c, (int, float))
              and not isinstance(c, bool) and math.isfinite(c)]
    if len(finite) != len(closes):
        return False
    return len(_returns(closes)) >= MIN_CORRELATION_BARS


def _returns(closes: List[float]) -> List[float]:
    """Daily returns. Bars at a zero price are skipped rather than dividing
    by zero; a non-finite result is dropped so it can never reach the
    correlation arithmetic."""
    out = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if not prev:
            continue
        try:
            r = (closes[i] - prev) / prev
        except (TypeError, ZeroDivisionError):
            continue
        if math.isfinite(r):
            out.append(r)
    return out


def correlation(a: List[float], b: List[float]) -> float:
    """Pearson correlation of two price series' daily returns.

    Returns 0.0 for short or zero-variance input rather than nan — a nan
    would silently compare False against every threshold and wave the
    candidate through.
    """
    ra, rb = _returns(a or []), _returns(b or [])
    n = min(len(ra), len(rb))
    if n < MIN_CORRELATION_BARS:
        return 0.0
    ra, rb = ra[-n:], rb[-n:]
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    if not math.isfinite(va) or not math.isfinite(vb) or va <= 0 or vb <= 0:
        return 0.0
    result = cov / (va * vb)
    # Belt and braces: callers compare this against a threshold, and every
    # comparison against NaN is False — a NaN here would wave a candidate
    # through the screen it was meant to fail.
    return result if math.isfinite(result) else 0.0


def propose_additions(tracked: Set[str], candidates: List[Dict[str, Any]],
                      history: Dict[str, List[float]],
                      cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Screen raw scout candidates down to ones worth an operator's attention."""
    section = (cfg or {}).get('universe') or {}
    min_volume = float(section.get('min_avg_dollar_volume', DEFAULT_MIN_DOLLAR_VOLUME))
    min_price = float(section.get('min_price', DEFAULT_MIN_PRICE))
    max_corr = float(section.get('max_correlation', DEFAULT_MAX_CORRELATION))

    out = []
    for cand in candidates or []:
        # One malformed row must not poison the batch. universe_scout feeds
        # this directly, so a single bad candidate would otherwise block
        # screening of every other candidate in the same call.
        try:
            screened = _screen_one(cand, tracked, history,
                                   min_volume, min_price, max_corr)
        except Exception as e:
            logger.warning(f"Universe: rejecting malformed candidate {cand!r}: {e}")
            continue
        if screened:
            out.append(screened)
    return out


def _screen_one(cand: Any, tracked: Set[str], history: Dict[str, List[float]],
                min_volume: float, min_price: float,
                max_corr: float) -> Optional[Dict[str, str]]:
    """Screen a single candidate. Returns the proposal, or None to reject."""
    if not isinstance(cand, dict):
        logger.info(f"Universe: rejecting non-dict candidate {cand!r}")
        return None

    symbol = (cand.get('symbol') or '').upper()
    if not symbol or symbol in tracked:
        return None

    volume = _finite_number(cand.get('avg_dollar_volume'))
    if volume is None or volume < min_volume:
        logger.info(f"Universe: rejecting {symbol} — illiquid or unreadable "
                    f"volume ({cand.get('avg_dollar_volume')!r})")
        return None

    price = _finite_number(cand.get('price'))
    if price is None or price < min_price:
        logger.info(f"Universe: rejecting {symbol} — price "
                    f"{cand.get('price')!r} below ${min_price}")
        return None

    closes = history.get(symbol)
    if not is_screenable(closes):
        logger.info(f"Universe: rejecting {symbol} — history is too short or "
                    f"contains non-finite values, so it cannot be screened")
        return None

    worst = None
    for held in tracked:
        other = history.get(held)
        if not other:
            continue
        if not is_screenable(other):
            # We cannot prove this candidate is uncorrelated with a holding
            # whose own series is unusable, so refuse rather than guess.
            logger.info(f"Universe: rejecting {symbol} — cannot screen against "
                        f"{held}, whose history is unusable")
            return None
        rho = correlation(closes, other)
        if worst is None or abs(rho) > abs(worst[1]):
            worst = (held, rho)
    if worst and abs(worst[1]) > max_corr:
        logger.info(f"Universe: rejecting {symbol} — {worst[1]:+.2f} correlated "
                    f"with {worst[0]}, duplicates existing exposure")
        return None

    return {'symbol': symbol,
            'reason': cand.get('reason', 'scout candidate'),
            'market': cand.get('market', 'us')}


DEFAULT_MAX_CONSECUTIVE_LOSSES = 4
DEFAULT_MAX_UNDEFINED_DAYS = 30


def propose_removals(tracked: Set[str], stats: Dict[str, Dict[str, Any]],
                     cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Symbols that have earned their way out of the universe.

    One proposal per symbol, first matching reason wins — an operator does
    not need four escalations telling them the same name is bad.

    A symbol with no stats entry is KEPT. Absence of evidence is not
    evidence of failure, and silently dropping unmeasured names would
    shrink the universe by accident.

    Each stat is coerced with `_finite_number` before being judged. A NaN
    or Infinity in `consecutive_losses`, `undefined_regime_days`,
    `avg_dollar_volume` or `attribution` is unreadable, not evidence of
    failure — comparisons against NaN are always False, so trusting the raw
    value would silently and permanently mask a real breach. An unreadable
    stat is treated the same as a missing one: no evidence, so KEEP.
    """
    section = (cfg or {}).get('universe') or {}
    min_volume = float(section.get('min_avg_dollar_volume', DEFAULT_MIN_DOLLAR_VOLUME))
    max_losses = int(section.get('max_consecutive_losses', DEFAULT_MAX_CONSECUTIVE_LOSSES))
    max_undefined = int(section.get('max_undefined_days', DEFAULT_MAX_UNDEFINED_DAYS))

    out = []
    stats = stats if isinstance(stats, dict) else {}
    for symbol in sorted(tracked or set()):
        s = stats.get(symbol)
        # A missing entry, or one that isn't a dict, is "no evidence" — the
        # same guard the additions side already has, so one corrupt journal
        # row cannot crash the whole removal pass.
        if not isinstance(s, dict) or not s:
            if s:
                logger.warning(
                    f"Universe: ignoring unreadable stats for {symbol}: {s!r}")
            continue

        losses = _finite_number(s.get('consecutive_losses'))
        undefined_days = _finite_number(s.get('undefined_regime_days'))
        volume = _finite_number(s.get('avg_dollar_volume'))
        attribution = _finite_number(s.get('attribution'))

        if losses is not None and losses > max_losses:
            reason = f'{losses:g} consecutive losing round trips (limit {max_losses})'
        elif undefined_days is not None and undefined_days > max_undefined:
            reason = (f'regime undefined for {undefined_days:g} days — '
                      f'the router can never trade it')
        elif volume is not None and volume < min_volume:
            reason = f'average dollar volume ${volume:,.0f} below ${min_volume:,.0f}'
        elif attribution is not None and attribution < 0:
            reason = f'negative attribution ${attribution:,.0f} over the review window'
        else:
            continue

        logger.info(f"Universe: proposing removal of {symbol} — {reason}")
        out.append({'symbol': symbol, 'reason': reason})
    return out


def _finite_number(value: Any) -> Optional[float]:
    """Coerce to a finite float, or None. NaN and Infinity are None: they
    pass every `<` comparison silently instead of failing loudly."""
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


MARKET_CONFIG_KEY = {'us': 'symbols', 'kenyan': 'nse_symbols'}


def effective_universe(config: Dict[str, Any], watchlist: List[Dict[str, Any]],
                       market: str = 'us') -> List[str]:
    """What the trading loop should actually iterate over.

    config symbols + operator-approved additions - paused/removed names.
    Previously the loop read config alone, so an approved symbol was written
    to the watchlist and never traded.
    """
    key = MARKET_CONFIG_KEY.get(market, 'symbols')
    universe = list((config.get('data_manager') or {}).get(key) or [])

    for row in watchlist or []:
        symbol = (row.get('symbol') or '').upper()
        if not symbol or row.get('market', 'us') != market:
            continue
        status = row.get('status', 'active')
        if status == 'active':
            if symbol not in universe:
                universe.append(symbol)
        elif symbol in universe:
            universe.remove(symbol)

    seen, out = set(), []
    for symbol in universe:
        if symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def persist_universe(config: Dict[str, Any], symbol: str, market: str,
                     overlay_path: Optional[str] = None) -> None:
    """Add an approved symbol to the right list AND to the on-disk overlay.

    Without the overlay the approval is in-memory only and the next restart
    silently reverts it.
    """
    from src.agent.self_assessment import DEFAULT_OVERLAY_PATH, _write_overlay_key

    symbol = (symbol or '').upper()
    if not symbol:
        return
    key = MARKET_CONFIG_KEY.get(market, 'symbols')
    target = config.setdefault('data_manager', {}).setdefault(key, [])
    if symbol not in target:
        target.append(symbol)
    _write_overlay_key(f'data_manager.{key}', list(target),
                       overlay_path or DEFAULT_OVERLAY_PATH)
    logger.warning(f"Universe: {symbol} added to {key} and persisted")

