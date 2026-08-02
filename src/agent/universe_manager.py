"""What should the agent be trading?

`universe_scout` proposes anything that moved 3% today, with no screening for
whether the name is tradeable or whether it duplicates exposure the book
already has. With SPY, QQQ and DIA all tracked, eleven symbols carry roughly
seven independent bets — the wrong shape for an uncorrelated-return mandate.

Everything here produces PROPOSALS. Nothing auto-applies; every change goes
to the operator as an escalation.
"""

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_MIN_DOLLAR_VOLUME = 1_000_000.0
DEFAULT_MIN_PRICE = 5.0
DEFAULT_MAX_CORRELATION = 0.9

MIN_CORRELATION_BARS = 30


def _returns(closes: List[float]) -> List[float]:
    return [(closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes)) if closes[i - 1]]


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
    if va <= 0 or vb <= 0:
        return 0.0
    return cov / (va * vb)


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
        symbol = (cand.get('symbol') or '').upper()
        if not symbol or symbol in tracked:
            continue

        if float(cand.get('avg_dollar_volume', 0) or 0) < min_volume:
            logger.info(f"Universe: rejecting {symbol} — illiquid "
                        f"(${cand.get('avg_dollar_volume', 0):,.0f}/day)")
            continue

        if float(cand.get('price', 0) or 0) < min_price:
            logger.info(f"Universe: rejecting {symbol} — price "
                        f"${cand.get('price', 0)} below ${min_price}")
            continue

        closes = history.get(symbol)
        if not closes or len(_returns(closes)) < MIN_CORRELATION_BARS:
            logger.info(f"Universe: rejecting {symbol} — not enough history to screen")
            continue

        worst = None
        for held in tracked:
            other = history.get(held)
            if not other:
                continue
            rho = correlation(closes, other)
            if worst is None or abs(rho) > abs(worst[1]):
                worst = (held, rho)
        if worst and abs(worst[1]) > max_corr:
            logger.info(f"Universe: rejecting {symbol} — {worst[1]:+.2f} correlated "
                        f"with {worst[0]}, duplicates existing exposure")
            continue

        out.append({'symbol': symbol,
                    'reason': cand.get('reason', 'scout candidate'),
                    'market': cand.get('market', 'us')})
    return out
