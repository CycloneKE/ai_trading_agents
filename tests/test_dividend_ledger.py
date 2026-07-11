import os
import tempfile

import pytest

from src.agent.sleeve.dividend_ledger import DividendLedger


@pytest.fixture
def ledger():
    path = os.path.join(tempfile.mkdtemp(), 'sleeve_test.db')
    l = DividendLedger(path)
    yield l
    l.close()


def test_record_and_unswept_cash(ledger):
    ledger.record_dividend('SCOM', 100.0)
    ledger.record_dividend('EQTY', 50.0)
    assert ledger.unswept_cash_kes() == 150.0


def test_mark_swept_zeroes_unswept_cash(ledger):
    ledger.record_dividend('SCOM', 100.0)
    swept = ledger.mark_swept()
    assert swept == 1
    assert ledger.unswept_cash_kes() == 0.0


def test_declared_status_excluded_from_unswept_cash(ledger):
    ledger.record_dividend('SCOM', 100.0, status='declared')
    assert ledger.unswept_cash_kes() == 0.0


def test_history_filters_by_symbol(ledger):
    ledger.record_dividend('SCOM', 100.0)
    ledger.record_dividend('EQTY', 50.0)
    hist = ledger.history(symbol='SCOM')
    assert len(hist) == 1
    assert hist[0]['symbol'] == 'SCOM'


def test_history_returns_all_without_symbol_filter(ledger):
    ledger.record_dividend('SCOM', 100.0)
    ledger.record_dividend('EQTY', 50.0)
    assert len(ledger.history()) == 2
