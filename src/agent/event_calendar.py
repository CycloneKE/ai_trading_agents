"""Event calendar: known risk windows scale position sizing down.

Elections, rate decisions and similar events repeat with too few samples
to *predict* returns, but plenty of evidence of elevated volatility — so
the platform treats them as risk regimes: inside a window, position
sizing is scaled by the window's multiplier (0.0–1.0). Deterministic,
auditable, and overridable via data/event_calendar.json.
"""

import json
import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

CALENDAR_FILE = str(DATA_DIR / 'event_calendar.json')

# Defaults ship with the code; data/event_calendar.json (same shape) is
# merged on top. days_before/days_after define the window; multiplier
# scales position sizing inside it.
DEFAULT_EVENTS = [
    {"name": "Kenya general election", "date": "2027-08-09",
     "days_before": 45, "days_after": 21, "multiplier": 0.5,
     "markets": ["NSE"]},
    {"name": "US midterm election", "date": "2026-11-03",
     "days_before": 14, "days_after": 7, "multiplier": 0.7,
     "markets": ["US"]},
    # FOMC decision dates (2026 H2 schedule)
    {"name": "FOMC decision", "date": "2026-07-29", "days_before": 2,
     "days_after": 1, "multiplier": 0.7, "markets": ["US"]},
    {"name": "FOMC decision", "date": "2026-09-16", "days_before": 2,
     "days_after": 1, "multiplier": 0.7, "markets": ["US"]},
    {"name": "FOMC decision", "date": "2026-11-04", "days_before": 2,
     "days_after": 1, "multiplier": 0.7, "markets": ["US"]},
    {"name": "FOMC decision", "date": "2026-12-16", "days_before": 2,
     "days_after": 1, "multiplier": 0.7, "markets": ["US"]},
]


class EventCalendar:
    def __init__(self, events: Optional[List[Dict[str, Any]]] = None):
        self.events = list(events) if events is not None else list(DEFAULT_EVENTS)
        try:
            if os.path.exists(CALENDAR_FILE):
                with open(CALENDAR_FILE) as f:
                    self.events.extend(json.load(f))
                logger.info(f"Loaded extra events from {CALENDAR_FILE}")
        except Exception as e:
            logger.warning(f"Could not read {CALENDAR_FILE}: {e}")

    def active_windows(self, on: Optional[date] = None,
                       market: str = 'US') -> List[Dict[str, Any]]:
        """Events whose risk window covers ``on`` for the given market."""
        on = on or datetime.now().date()
        out = []
        for ev in self.events:
            try:
                if market not in ev.get('markets', ['US']):
                    continue
                ev_date = datetime.strptime(ev['date'], '%Y-%m-%d').date()
                delta = (ev_date - on).days
                if -int(ev.get('days_after', 0)) <= delta <= int(ev.get('days_before', 0)):
                    out.append({**ev, 'days_until': delta})
            except Exception as e:
                logger.warning(f"Bad calendar entry {ev}: {e}")
        return out

    def risk_multiplier(self, on: Optional[date] = None,
                        market: str = 'US') -> float:
        """Sizing multiplier: most-cautious (lowest) active window wins;
        1.0 when no window is active. Never below 0."""
        windows = self.active_windows(on, market)
        if not windows:
            return 1.0
        m = min(max(0.0, float(w.get('multiplier', 1.0))) for w in windows)
        names = ', '.join(w['name'] for w in windows)
        logger.info(f"Event window active ({names}): sizing x{m}")
        return m
