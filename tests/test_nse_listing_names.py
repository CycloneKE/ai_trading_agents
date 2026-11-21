"""The NSE list: Homeboyz trades as HBE, the operator's sheet's names are
known, and funds, REITs and preference shares are kept out of the screen."""
from datetime import date, timedelta

from src.agent.nse_screener import metrics_for
from src.connectors import nse_universe as u

CFG = {'stale_after_days': 10, 'min_history_days': 50, 'min_avg_value_kes': 0}


def test_homeboyz_is_named_under_its_current_ticker():
    assert u.name_of('HBE') == 'Homeboyz Entertainment'
    assert u.sector_of('HBE') == 'Commercial & Services'
    assert 'HBER' not in u.LISTED


def test_the_sheets_tickers_have_names_and_sectors():
    for sym in ('ALP', 'ARM', 'DCON', 'FMLY', 'GLD', 'HBE', 'KPC', 'KPLC-P4', 'KPLC-P7',
                'LAPR', 'MSC', 'SMWF', 'TRFC', 'QMRT'):
        assert u.sector_of(sym) != u.UNCLASSIFIED, sym
        assert u.valid_ticker(sym), sym


def test_funds_reits_and_preference_shares_are_not_ordinary_shares():
    for sym in ('GLD', 'SMWF', 'ALP', 'LAPR', 'TRFC', 'KPLC-P4', 'KPLC-P7'):
        assert not u.is_ordinary_share(sym), sym
    for sym in ('SCOM', 'KPLC', 'FMLY', 'HBE', 'SOMETHINGNEW'):
        assert u.is_ordinary_share(sym), sym


def _rows(n=150, price=100.0):
    start = date(2026, 1, 1)
    return [{'date': (start + timedelta(days=i)).isoformat(), 'close': price * (1 + i / 1000),
             'volume': 100000, 'change_pct': 0.1} for i in range(n)]


def test_the_screener_leaves_out_a_fund_however_good_its_numbers():
    today = date(2026, 1, 1) + timedelta(days=150)
    assert metrics_for('SCOM', _rows(), CFG, today).eligible
    gld = metrics_for('GLD', _rows(), CFG, today)
    assert not gld.eligible and 'not an ordinary share' in gld.reason
