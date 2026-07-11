import os
import tempfile
from datetime import date

import pytest

from src.agent.sleeve.fundamentals_store import Fundamentals
from src.agent.sleeve.sleeve_manager import SleeveManager


class FakeQueue:
    def __init__(self):
        self.tickets = []

    def create_ticket(self, symbol, side, quantity, suggested_limit_price=None,
                      rationale='', ensemble_confidence=None, llm_reasoning='', book='trading'):
        tid = len(self.tickets) + 1
        self.tickets.append({'id': tid, 'symbol': symbol, 'side': side,
                             'quantity': quantity, 'book': book})
        return tid

    def positions(self, book=None):
        return {}


class FakeFundamentalsStore:
    def __init__(self, records):
        self._records = {r.symbol: r for r in records}

    def get_all(self, symbols):
        return [self._records[s] for s in symbols if s in self._records]


class FakeDividendLedger:
    def __init__(self, unswept=0.0):
        self._unswept = unswept
        self.swept_calls = 0

    def unswept_cash_kes(self):
        return self._unswept

    def mark_swept(self):
        self.swept_calls += 1
        return 1


class FakeOrchestrator:
    enabled = True

    def propose_json(self, system_prompt, user_prompt, model_override=None):
        return {'flag': False, 'reason': ''}


def make_config(top_n=2):
    return {
        'sleeve': {
            'enabled': True, 'nse_capital_kes': 100000, 'capital_split_pct': 0.5,
            'top_n': top_n, 'weighting_mode': 'equal', 'universe': ['SCOM', 'EQTY'],
            'scoring': {'min_avg_daily_volume': 1000},
        }
    }


def make_fundamentals():
    return [
        Fundamentals('SCOM', yield_ttm_pct=6.5, dividend_per_share_kes=2.3, eps_kes=2.4,
                    payout_ratio=0.48, years_consecutive_paid=8, eps_trend='positive',
                    avg_daily_volume=100_000, last_updated=date.today().isoformat()),
        Fundamentals('EQTY', yield_ttm_pct=5.0, dividend_per_share_kes=4.0, eps_kes=8.0,
                    payout_ratio=0.50, years_consecutive_paid=6, eps_trend='positive',
                    avg_daily_volume=80_000, last_updated=date.today().isoformat()),
    ]


@pytest.fixture
def db_path():
    return os.path.join(tempfile.mkdtemp(), 'sleeve_test.db')


def make_manager(db_path, config=None, dividend_ledger=None, orchestrator=None,
                 fundamentals=None, queue=None):
    return SleeveManager(
        config or make_config(), queue or FakeQueue(),
        FakeFundamentalsStore(fundamentals if fundamentals is not None else make_fundamentals()),
        dividend_ledger or FakeDividendLedger(), orchestrator or FakeOrchestrator(),
        db_path=db_path)


def test_generates_tickets_for_ranked_candidates(db_path):
    mgr = make_manager(db_path)
    results = mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0})
    assert {r['symbol'] for r in results} == {'SCOM', 'EQTY'}
    assert all(r['quantity'] > 0 for r in results)
    mgr.close()


def test_second_call_same_month_is_noop(db_path):
    mgr = make_manager(db_path)
    first = mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0}, today=date(2026, 7, 1))
    second = mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0}, today=date(2026, 7, 15))
    assert len(first) == 2
    assert second == []
    mgr.close()


def test_new_month_runs_again(db_path):
    mgr = make_manager(db_path)
    mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0}, today=date(2026, 7, 1))
    second = mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0}, today=date(2026, 8, 1))
    assert len(second) == 2
    mgr.close()


def test_missing_quote_excludes_symbol(db_path):
    mgr = make_manager(db_path)
    results = mgr.run_monthly_cycle({'SCOM': 20.0})  # no EQTY quote
    assert {r['symbol'] for r in results} == {'SCOM'}
    mgr.close()


def test_dividend_cash_increases_deployed_quantity(db_path):
    ledger = FakeDividendLedger(unswept=50000.0)
    with_div = make_manager(db_path, dividend_ledger=ledger)
    without_div = make_manager(os.path.join(tempfile.mkdtemp(), 'baseline.db'))

    with_results = with_div.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0})
    without_results = without_div.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0})

    with_scom = next(r for r in with_results if r['symbol'] == 'SCOM')['quantity']
    without_scom = next(r for r in without_results if r['symbol'] == 'SCOM')['quantity']
    assert with_scom > without_scom
    assert ledger.swept_calls == 1
    with_div.close()
    without_div.close()


def test_no_sweep_when_zero_tickets_generated(db_path):
    ledger = FakeDividendLedger(unswept=50000.0)
    mgr = make_manager(db_path, dividend_ledger=ledger, fundamentals=[])
    results = mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0})
    assert results == []
    assert ledger.swept_calls == 0
    mgr.close()


def test_veto_flag_surfaced_on_ticket_result(db_path):
    class FlaggingOrchestrator:
        enabled = True

        def propose_json(self, system_prompt, user_prompt, model_override=None):
            return {'flag': True, 'reason': 'profit warning issued'}

    mgr = make_manager(db_path, orchestrator=FlaggingOrchestrator())
    results = mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0})
    assert all(r['veto_flag'] is True for r in results)
    assert all(r['veto_reason'] == 'profit warning issued' for r in results)
    mgr.close()


def test_disabled_sleeve_returns_no_tickets(db_path):
    cfg = make_config()
    cfg['sleeve']['enabled'] = False
    mgr = make_manager(db_path, config=cfg)
    assert mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0}) == []
    mgr.close()
