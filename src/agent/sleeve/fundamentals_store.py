"""Merges scraped afx fundamentals, operator-maintained dividend history
(config/nse_dividends.json), and locally stored NSE price history into
per-symbol scoring inputs for the dividend sleeve.

A symbol with no operator entry, or with no usable dividend yield after
merging, returns None from `get()` — the caller excludes it from ranking
rather than scoring it on guessed data.
"""
import json
import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.connectors.nse_scraper import load_csv, scrape_afx_company_page
from src.utils.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_DIVIDENDS_PATH = PROJECT_ROOT / 'config' / 'nse_dividends.json'


@dataclass
class Fundamentals:
    symbol: str
    yield_ttm_pct: float
    dividend_per_share_kes: float
    eps_kes: float
    payout_ratio: Optional[float]
    years_consecutive_paid: int
    eps_trend: str  # 'positive' | 'flat' | 'negative'
    avg_daily_volume: int
    last_updated: str  # ISO date/datetime string

    def is_stale(self, stale_days: int) -> bool:
        try:
            updated = datetime.fromisoformat(self.last_updated)
        except (ValueError, TypeError):
            return True
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - updated).days
        return age_days > stale_days


class FundamentalsStore:
    def __init__(self, dividends_path: Path = DEFAULT_DIVIDENDS_PATH,
                volume_window: int = 20):
        self.dividends_path = Path(dividends_path)
        self.volume_window = volume_window

    def _load_operator_data(self) -> Dict[str, Dict[str, Any]]:
        if not self.dividends_path.exists():
            return {}
        try:
            with open(self.dividends_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read {self.dividends_path}: {e}")
            return {}

    def _avg_daily_volume(self, symbol: str) -> int:
        bars = load_csv(symbol)
        if not bars:
            return 0
        recent = bars[-self.volume_window:]
        volumes = [int(float(b.get('volume', 0) or 0)) for b in recent]
        return int(statistics.mean(volumes)) if volumes else 0

    @staticmethod
    def _prefer_scraped(scraped_val: Any, operator_val: Any) -> Any:
        """Prefer scraped value over operator value, including legitimate zeros."""
        return scraped_val if scraped_val is not None else operator_val

    def get(self, symbol: str) -> Optional[Fundamentals]:
        operator = self._load_operator_data().get(symbol.upper())
        if not operator:
            return None
        scraped = scrape_afx_company_page(symbol) or {}
        eps = self._prefer_scraped(scraped.get('eps'), operator.get('eps_kes', 0.0))
        dps = self._prefer_scraped(scraped.get('dividend_per_share'), operator.get('dividend_per_share_kes', 0.0))
        yield_pct = self._prefer_scraped(scraped.get('dividend_yield_pct'), operator.get('yield_ttm_pct', 0.0))
        if not yield_pct or not dps:
            return None
        payout_ratio = round(dps / eps, 4) if eps > 0 else operator.get('payout_ratio_override')
        return Fundamentals(
            symbol=symbol.upper(),
            yield_ttm_pct=float(yield_pct),
            dividend_per_share_kes=float(dps),
            eps_kes=float(eps),
            payout_ratio=payout_ratio,
            years_consecutive_paid=int(operator.get('years_consecutive_paid', 0)),
            eps_trend=operator.get('eps_trend', 'flat'),
            avg_daily_volume=self._avg_daily_volume(symbol),
            last_updated=operator.get('last_updated', datetime.now(timezone.utc).isoformat()),
        )

    def get_all(self, symbols: List[str]) -> List[Fundamentals]:
        out = []
        for s in symbols:
            try:
                f = self.get(s)
            except Exception as e:
                logger.warning(f"Sleeve: excluding {s} from ranking ({e})")
                continue
            if f:
                out.append(f)
            else:
                logger.debug(f"Sleeve: excluding {s} from ranking (no usable fundamentals)")
        return out
