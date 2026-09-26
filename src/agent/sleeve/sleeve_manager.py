"""Monthly accumulation cycle for the NSE dividend-led long-term sleeve.

Ties fundamentals, deterministic scoring, the LLM veto, and the dividend
ledger together to turn scored candidates into buy-only order tickets in
the existing NSE order-ticket queue. Never sells or trims — this sleeve
only ever deploys new capital (fixed % of NSE capital + swept dividends)
across the current top-ranked candidates.
"""
import logging
import math
import sqlite3
import threading
from datetime import date
from typing import Any, Dict, List, Optional

from src.agent.sleeve.dividend_scorer import ScoringConfig, rank_candidates
from src.agent.sleeve.llm_veto import check_candidate
from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sleeve_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _scoring_config_from_dict(d: Dict[str, Any]) -> ScoringConfig:
    """Builds the ScoringConfig used inside the (gate-protected) monthly
    cycle. `years_paid_target` and `yield_cap_pct` are divisors in
    dividend_scorer.py's scoring math — clamp them here so a config typo
    (e.g. an operator setting either to 0) can't ZeroDivisionError partway
    through a cycle after the month gate has already been checked."""
    years_paid_target = max(1, d.get('years_paid_target', 5))
    yield_cap_pct = d.get('yield_cap_pct', 12.0)
    if yield_cap_pct <= 0:
        yield_cap_pct = 0.1
    return ScoringConfig(
        require_sustainable=d.get('require_sustainable', True),
        min_dividend_cover=d.get('min_dividend_cover', 1.2),
        yield_trap_pct=d.get('yield_trap_pct', 12.0),
        min_years_paid=d.get('min_years_paid', 3),
        max_per_sector=d.get('max_per_sector', 2),
        ex_date_tilt=d.get('ex_date_tilt', 0.10),
        yield_cap_pct=yield_cap_pct,
        yield_weight=d.get('yield_weight', 0.5),
        quality_weight=d.get('quality_weight', 0.5),
        payout_ratio_min=d.get('payout_ratio_min', 0.30),
        payout_ratio_max=d.get('payout_ratio_max', 0.70),
        years_paid_target=years_paid_target,
        min_avg_daily_volume=d.get('min_avg_daily_volume', 50_000),
        stale_days=d.get('stale_days', 400),
        value_weight=d.get('value_weight', 0.0),
        earnings_yield_cap_pct=max(d.get('earnings_yield_cap_pct', 25.0), 0.1),
    )


def tbill_switch_reason(candidates, tbill_rate_pct: float,
                        withholding: float = 0.15) -> Optional[str]:
    """Why the sleeve should buy Treasury bills instead of shares, or None.

    When the typical (median) earnings yield of the stocks it can buy is
    below what a T-bill pays after its 15% withholding tax, shares are
    dear: the sleeve skips the month and the money stays in bills. Too few
    stocks with the data to judge means no switch.
    """
    from statistics import median
    from src.agent.sleeve.dividend_scorer import earnings_yield_pct
    yields = [y for y in (earnings_yield_pct(f) for f in candidates) if y is not None]
    if len(yields) < 3:
        return None
    net_bill = tbill_rate_pct * (1 - withholding)
    typical = median(yields)
    if typical < net_bill:
        return (f"median earnings yield {typical:.1f}% is below the T-bill's "
                f"{net_bill:.1f}% after tax; holding bills this month")
    return None


class SleeveManager:
    def __init__(self, config: Dict[str, Any], nse_order_queue, fundamentals_store,
                dividend_ledger, llm_orchestrator=None,
                db_path: str = str(DATA_DIR / 'sleeve.db')):
        sc = config.get('sleeve') if isinstance(config.get('sleeve'), dict) else config
        self.enabled = sc.get('enabled', False)
        self.nse_capital_kes = sc.get('nse_capital_kes', 0.0)
        self.capital_split_pct = sc.get('capital_split_pct', 0.0)
        self.top_n = sc.get('top_n', 5)
        self.weighting_mode = sc.get('weighting_mode', 'equal')
        self.universe = sc.get('universe', [])
        self.scoring_cfg = _scoring_config_from_dict(sc.get('scoring', {}))
        # Sector map for the concentration cap. Kenyan banks dominate the
        # high-yield list, so without this the sleeve becomes a single bet on
        # one credit cycle. Unmapped symbols are never capped.
        self.sector_map = sc.get('sectors', {}) or {}
        # Earnings yield versus T-bills (tbill_switch_reason). The bill rate
        # is the benchmarks' one unless the sleeve sets its own.
        ts = sc.get('tbill_switch', {}) or {}
        self.tbill_switch = bool(ts.get('enabled', False))
        from src.agent.benchmarks import TBILL_RATE_DEFAULT, TBILL_TAX_DEFAULT
        bench = config.get('benchmarks', {}) if isinstance(config.get('benchmarks'), dict) else {}
        self.tbill_rate_pct = 100 * float(ts.get('tbill_rate_pct') or
                                          bench.get('tbill_rate_pct', TBILL_RATE_DEFAULT))
        self.tbill_withholding = float(bench.get('tbill_withholding_pct', TBILL_TAX_DEFAULT))

        self.nse_order_queue = nse_order_queue
        self.fundamentals_store = fundamentals_store
        self.dividend_ledger = dividend_ledger
        self.llm_orchestrator = llm_orchestrator
        self._db_path = db_path
        self.db_path = db_path

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _get_state(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM sleeve_state WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None

    def _set_state(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sleeve_state (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
            self._conn.commit()

    def current_holdings(self) -> Dict[str, Dict[str, Any]]:
        """Sleeve's own holdings, isolated from the trading book by the
        `book='long_term'` tag every ticket this manager creates carries."""
        return self.nse_order_queue.positions(book='long_term')

    def _target_weights(self, ranked) -> Dict[str, float]:
        if not ranked:
            return {}
        if self.weighting_mode == 'score':
            total = sum(c.combined_score for c in ranked)
            if total <= 0:
                return {}
            return {c.symbol: c.combined_score / total for c in ranked}
        weight = 1.0 / len(ranked)
        return {c.symbol: weight for c in ranked}

    def run_monthly_cycle(self, quotes: Dict[str, float],
                          today: Optional[date] = None) -> List[Dict[str, Any]]:
        """Runs once per calendar month. `quotes` maps symbol -> current KES
        price (the caller supplies these from the NSE connector so this
        module has no direct network dependency)."""
        if not self.enabled:
            return []
        today = today or date.today()
        month_key = today.strftime('%Y-%m')
        if self._get_state('last_cycle_month') == month_key:
            return []

        # A data outage on the month's first tick should retry next tick, not
        # burn the month gate — this path does no scraping so it cannot cause
        # a retry storm the way a failure further in (see below) can.
        if not quotes:
            logger.warning("Sleeve: no NSE quotes available; deferring cycle")
            return []

        try:
            candidates = self.fundamentals_store.get_all(self.universe)
            if self.tbill_switch:
                why = tbill_switch_reason(candidates, self.tbill_rate_pct, self.tbill_withholding)
                if why:
                    logger.info(f"Sleeve: {why}")
                    self._set_state('last_skip_reason', why)
                    self._set_state('last_cycle_month', month_key)
                    return []
            ranked = rank_candidates(candidates, self.scoring_cfg, self.top_n,
                                     sector_lookup=self.sector_map)
            ranked = [c for c in ranked if quotes.get(c.symbol, 0) > 0]
            weights = self._target_weights(ranked)

            new_capital_kes = (self.nse_capital_kes * self.capital_split_pct +
                               self.dividend_ledger.unswept_cash_kes())

            results: List[Dict[str, Any]] = []
            for candidate in ranked:
                symbol = candidate.symbol
                price = quotes[symbol]
                allocation_kes = new_capital_kes * weights.get(symbol, 0)
                quantity = int(math.floor(allocation_kes / price))
                if quantity <= 0:
                    continue

                veto = check_candidate(self.llm_orchestrator, symbol)
                tags = []
                if candidate.stale:
                    tags.append('STALE_DATA')
                if veto.flag:
                    tags.append('VETO_FLAG')
                if not veto.available:
                    tags.append('VETO_UNAVAILABLE')
                rationale = (f"Sleeve accumulation: yield_score={candidate.yield_score}, "
                            f"quality_score={candidate.quality_score}, "
                            f"value_score={candidate.value_score}, "
                            f"combined={candidate.combined_score}")
                if tags:
                    rationale += " | " + ", ".join(tags)

                ticket_id = self.nse_order_queue.create_ticket(
                    symbol, 'buy', quantity, suggested_limit_price=price,
                    rationale=rationale, ensemble_confidence=candidate.combined_score,
                    llm_reasoning=veto.reason, book='long_term')
                if ticket_id is None:
                    logger.debug(
                        "Sleeve: skipping %s — identical pending long_term ticket already exists",
                        symbol)
                    continue

                results.append({
                    'symbol': symbol, 'quantity': quantity, 'price': price,
                    'ticket_id': ticket_id, 'combined_score': candidate.combined_score,
                    'veto_flag': veto.flag, 'veto_available': veto.available,
                    'veto_reason': veto.reason, 'stale_data': candidate.stale,
                })

            # Only mark dividend cash "swept" if it was actually deployed this
            # cycle — otherwise a zero-ticket month would silently lose that
            # cash instead of rolling it into next month's capital.
            if results:
                self.dividend_ledger.mark_swept()
        except Exception as e:
            # A failed month must degrade to "skipped month", never a retry
            # storm: without the month gate below, the trading loop would
            # retry this cycle every ~60s all month, each retry re-running
            # live network scrapes against a cycle that's already broken.
            logger.error(f"Sleeve: monthly cycle failed, skipping this month: {e}")
            self._set_state('last_cycle_month', month_key)
            return []

        self._set_state('last_cycle_month', month_key)
        return results

    def close(self):
        with self._lock:
            self._conn.close()
