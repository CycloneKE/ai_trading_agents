"""Market regime classification for strategy routing.

Momentum and mean-reversion measure the same thing with opposite sign. A
diagnostic over 13,145 symbol-bars found they contradict each other on 44.3%
of bars and agree on 0.6%, so averaging them cancelled a live signal 62.7% of
the time and pinned surviving confidence at ~0.5.

The fix is to run exactly one of them, chosen by regime. This module decides
which, using Kaufman's efficiency ratio: the net distance travelled divided
by the total distance walked. High means price went somewhere directly
(trending); low means it chopped (ranging).

Chosen over ADX or the Hurst exponent because it needs no library, has one
parameter, and measures the routing criterion directly rather than by proxy.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

TRENDING = 'trending'
RANGING = 'ranging'
UNDEFINED = 'undefined'

DEFAULT_WINDOW = 20
DEFAULT_TREND_THRESHOLD = 0.35
DEFAULT_RANGE_THRESHOLD = 0.20

CONVICTION_FLOOR = 0.7


def _cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    section = (cfg or {}).get('regime') or {}
    return {
        'window': int(section.get('window', DEFAULT_WINDOW)),
        'trend': float(section.get('trend_threshold', DEFAULT_TREND_THRESHOLD)),
        'range': float(section.get('range_threshold', DEFAULT_RANGE_THRESHOLD)),
    }


def _segment_and_path(closes: Optional[List[float]], window: int) -> Tuple[List[float], float]:
    """The trailing `window`-bar segment and the total distance walked in it.

    Single source of truth for "what counts as a usable window" (non-empty
    history, positive window, enough bars, all-finite prices) and "what
    counts as flat" (zero path). Returns `([], 0.0)` whenever the window
    can't be evaluated — never raises, because this runs inside the
    per-bar signal path.

    A NaN or Infinity anywhere in the segment makes every comparison
    against it False, so a plain `path <= 0` check would miss it and let
    a contaminated `net / path` (NaN) slip through `min(1.0, ...)`, which
    returns 1.0 in CPython -- the most confident possible verdict from
    the least trustworthy input. Reject it here, once, for every caller.
    """
    if not closes or window < 1 or len(closes) < window + 1:
        return [], 0.0
    segment = closes[-(window + 1):]
    if not all(math.isfinite(price) for price in segment):
        return [], 0.0
    path = sum(abs(segment[i] - segment[i - 1]) for i in range(1, len(segment)))
    return segment, path


def efficiency_ratio(closes: List[float], window: int) -> float:
    """Net move over `window` bars divided by the total path walked.

    Returns 0.0 for insufficient history or a flat series — never raises,
    because this runs inside the per-bar signal path.
    """
    segment, path = _segment_and_path(closes, window)
    if not segment or path <= 0:
        return 0.0
    net = abs(segment[-1] - segment[0])
    return min(1.0, net / path)


def classify(closes: List[float], previous: Optional[str],
             cfg: Optional[Dict[str, Any]] = None) -> str:
    """Current regime, with a dead band and hysteresis.

    Between the two thresholds the prior regime is held. Without that, a
    series hovering near a boundary would flip the router every other bar and
    pay the round-trip cost each time.
    """
    c = _cfg(cfg)
    window = c['window']
    segment, path = _segment_and_path(closes, window)
    if not segment or path <= 0:
        return UNDEFINED          # bad window, insufficient history, or flat series
    er = efficiency_ratio(closes, window)
    if er >= c['trend']:
        return TRENDING
    if er <= c['range']:
        return RANGING
    return previous if previous in (TRENDING, RANGING) else UNDEFINED


def conviction(closes: List[float], regime: str,
               cfg: Optional[Dict[str, Any]] = None) -> float:
    """How firmly the series sits inside its regime, in [0.7, 1.0].

    Scales position size so a marginal regime trades smaller than an obvious
    one. Never returns 0 — a routed signal that passed classification should
    still trade, just less.
    """
    c = _cfg(cfg)
    er = efficiency_ratio(closes, c['window'])
    if regime == TRENDING:
        span = max(1e-9, 1.0 - c['trend'])
        depth = (er - c['trend']) / span
    elif regime == RANGING:
        span = max(1e-9, c['range'])
        depth = (c['range'] - er) / span
    else:
        depth = 0.0
    depth = max(0.0, min(1.0, depth))
    return CONVICTION_FLOOR + (1.0 - CONVICTION_FLOOR) * depth
