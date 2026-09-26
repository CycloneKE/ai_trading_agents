"""A hard monthly cap on what the paid AI tier may spend.

Every paid call is priced from the token counts the API returns and added to
this month's total, kept in data/ai_spend.json so a restart does not reset
it. Before a call, its worst case (the prompt plus the most it may write) is
checked against what is left; once the cap would be passed, the paid tier is
skipped until the next calendar month (UTC) and the free models take over.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# US dollars per million tokens (input, output): Anthropic's first-party API
# rates. A cache read costs a tenth of the input rate and a cache write
# 1.25 times it. Configurable, since prices change: ai_budget.prices.
PRICES_PER_MTOK = {
    'claude-sonnet-5': (2.00, 10.00),
    'claude-haiku-4-5': (1.00, 5.00),
    'claude-opus-5': (5.00, 25.00),
}
MONTHLY_CAP_DEFAULT = 20.0


def cost_usd(model: str, input_tokens: int = 0, output_tokens: int = 0,
             cache_read_tokens: int = 0, cache_write_tokens: int = 0,
             prices: Optional[Dict[str, Any]] = None) -> float:
    """What a call cost. An unknown model is priced at the dearest known
    rate, so the cap can only err on the safe side."""
    table = {**PRICES_PER_MTOK, **{k: tuple(v) for k, v in (prices or {}).items()}}
    rate_in, rate_out = table.get(model, max(table.values(), key=lambda r: r[1]))
    return (input_tokens * rate_in + output_tokens * rate_out
            + cache_read_tokens * rate_in * 0.1
            + cache_write_tokens * rate_in * 1.25) / 1_000_000


class AiBudget:
    def __init__(self, path: Path, monthly_cap_usd: float = MONTHLY_CAP_DEFAULT,
                 prices: Optional[Dict[str, Any]] = None):
        self.path = Path(path)
        self.cap = max(float(monthly_cap_usd or 0.0), 0.0)
        self.prices = prices or {}
        self._lock = threading.Lock()
        self._warned_month: Optional[str] = None
        try:
            self._data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self._data = {}

    @staticmethod
    def month(now: Optional[datetime] = None) -> str:
        return (now or datetime.now(timezone.utc)).strftime('%Y-%m')

    def spent(self, now: Optional[datetime] = None) -> float:
        with self._lock:
            return float((self._data.get(self.month(now)) or {}).get('spent_usd', 0.0))

    def remaining(self, now: Optional[datetime] = None) -> float:
        return max(self.cap - self.spent(now), 0.0)

    def allows(self, worst_case_usd: float, now: Optional[datetime] = None) -> bool:
        """Whether a call costing at most `worst_case_usd` fits this month."""
        ok = self.spent(now) + worst_case_usd <= self.cap
        month = self.month(now)
        if not ok and self._warned_month != month:
            self._warned_month = month
            logger.warning(f"AI budget: ${self.spent(now):.2f} of ${self.cap:.2f} spent in "
                           f"{month}; the paid tier is off until next month, free models only")
        return ok

    def worst_case(self, model: str, prompt_chars: int, max_tokens: int) -> float:
        # About four characters a token in English; three errs on the high side.
        return cost_usd(model, input_tokens=prompt_chars // 3 + 1, output_tokens=max_tokens,
                        prices=self.prices)

    def record(self, model: str, usage: Any, purpose: str = '',
               now: Optional[datetime] = None) -> float:
        """Add a call's cost from the response's usage; returns the cost."""
        get = (lambda k: int(getattr(usage, k, 0) or 0)) if not isinstance(usage, dict) \
            else (lambda k: int(usage.get(k) or 0))
        cost = cost_usd(model, get('input_tokens'), get('output_tokens'),
                        get('cache_read_input_tokens'), get('cache_creation_input_tokens'),
                        prices=self.prices)
        month = self.month(now)
        with self._lock:
            m = self._data.setdefault(month, {'spent_usd': 0.0, 'calls': 0, 'by_model': {}})
            m['spent_usd'] = round(m['spent_usd'] + cost, 6)
            m['calls'] += 1
            key = f"{model}:{purpose}" if purpose else model
            m['by_model'][key] = round(m['by_model'].get(key, 0.0) + cost, 6)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix('.tmp')
                tmp.write_text(json.dumps(self._data, indent=2))
                tmp.replace(self.path)
            except OSError as e:
                logger.error(f"AI budget: could not save spending: {e}")
        return cost

    def summary(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        month = self.month(now)
        with self._lock:
            m = dict(self._data.get(month) or {})
        spent = float(m.get('spent_usd', 0.0))
        return {'month': month, 'cap_usd': self.cap, 'spent_usd': round(spent, 4),
                'remaining_usd': round(max(self.cap - spent, 0.0), 4),
                'calls': int(m.get('calls', 0)), 'by_model': m.get('by_model', {})}
