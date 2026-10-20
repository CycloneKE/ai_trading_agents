"""Dashboard view for symbols the risk layer is currently refusing to enter.

A symbol inside a post-exit cooldown places no orders, which on a dashboard
is indistinguishable from a symbol with no signal. Making the block visible
turns "why isn't it trading TSLA?" from a log-archaeology exercise into a
glance.
"""

from typing import Any, Dict, Optional

EMPTY: Dict[str, Any] = {'active': [], 'count': 0, 'cooldown_seconds': 0}


def _label(seconds: float) -> str:
    """Compact human duration: '15m 0s', or just '25s' under a minute."""
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f'{minutes}m {secs}s' if minutes else f'{secs}s'


def build_cooldown_view(guard, now: Optional[float] = None) -> Dict[str, Any]:
    """Serialisable view of every active re-entry cooldown.

    `guard` may be None — the agent builds its ExposureGuard lazily, and the
    dashboard must render before the first trading cycle rather than 500.
    """
    if guard is None:
        return dict(EMPTY)

    entries = guard.active_cooldowns(now=now)
    return {
        'active': [
            {
                'symbol': e['symbol'],
                'reason': e['reason'],
                'seconds_remaining': round(e['seconds_remaining'], 1),
                'remaining_label': _label(e['seconds_remaining']),
            }
            for e in entries
        ],
        'count': len(entries),
        'cooldown_seconds': int(getattr(guard, 'cooldown_seconds', 0) or 0),
    }
