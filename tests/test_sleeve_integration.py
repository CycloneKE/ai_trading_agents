"""End-to-end: a full monthly cycle against a fixture universe, asserting
the right tickets land in the REAL NseOrderQueue with correct tags, sizes,
and isolation from a trading-book ticket for the same symbol."""
import os
import tempfile
from datetime import date

import pytest

from src.agent.nse_order_queue import NseOrderQueue
from src.agent.sleeve.fundamentals_store import Fundamentals
from src.agent.sleeve.sleeve_manager import SleeveManager


class FakeFundamentalsStore:
    def __init__(self, records):
        self._records = {r.symbol: r for r in records}

    def get_all(self, symbols):
        return [self._records[s] for s in symbols if s in self._records]


class FakeDividendLedger:
    def unswept_cash_kes(self):
        return 0.0

    def mark_swept(self):
        return 0


class FakeOrchestrator:
    enabled = True

    def propose_json(self, system_prompt, user_prompt, model_override=None):
        return {'flag': False, 'reason': ''}


@pytest.fixture
def queue():
    path = os.path.join(tempfile.mkdtemp(), 'nse_test.db')
    q = NseOrderQueue(path)
    yield q
    q.close()


def test_monthly_cycle_lands_tagged_tickets_in_real_queue(queue):
    records = [
        Fundamentals('SCOM', yield_ttm_pct=6.5, dividend_per_share_kes=2.3, eps_kes=2.4,
                    payout_ratio=0.48, years_consecutive_paid=8, eps_trend='positive',
                    avg_daily_volume=100_000, last_updated=date.today().isoformat()),
    ]
    config = {
        'sleeve': {
            'enabled': True, 'nse_capital_kes': 100000, 'capital_split_pct': 0.5,
            'top_n': 5, 'weighting_mode': 'equal', 'universe': ['SCOM'],
            'scoring': {'min_avg_daily_volume': 1000},
        }
    }
    mgr = SleeveManager(config, queue, FakeFundamentalsStore(records),
                        FakeDividendLedger(), FakeOrchestrator(),
                        db_path=os.path.join(tempfile.mkdtemp(), 'sleeve.db'))

    results = mgr.run_monthly_cycle({'SCOM': 20.0})
    assert len(results) == 1

    pending = queue.get_pending()
    sleeve_tickets = [t for t in pending if t['book'] == 'long_term']
    assert len(sleeve_tickets) == 1
    assert sleeve_tickets[0]['symbol'] == 'SCOM'
    assert sleeve_tickets[0]['quantity'] > 0

    # A trading-book ticket for the same symbol/side is a separate ticket,
    # not deduped against the sleeve one (book-scoped duplicate check).
    trading_id = queue.create_ticket('SCOM', 'buy', 50, book='trading')
    assert trading_id is not None
    mgr.close()
