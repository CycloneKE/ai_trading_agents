# tests/test_nse_hours.py
from datetime import datetime
from src.connectors.nse_connector import NSEConnector, EAT_OFFSET


def _at(h, m, weekday_date='2026-07-10'):  # 2026-07-10 is a Friday
    return datetime.fromisoformat(f'{weekday_date}T{h:02d}:{m:02d}:00').replace(tzinfo=EAT_OFFSET)


def test_preopen_is_not_tradeable():
    nse = NSEConnector.__new__(NSEConnector)  # skip network/db init
    assert nse.market_phase(_at(9, 15)) == 'preopen'
    assert nse.is_market_open(_at(9, 15)) is False


def test_continuous_session_bounds():
    nse = NSEConnector.__new__(NSEConnector)
    assert nse.market_phase(_at(9, 30)) == 'open'
    assert nse.market_phase(_at(14, 59)) == 'open'
    assert nse.market_phase(_at(15, 0)) == 'closed'


def test_weekend_closed():
    nse = NSEConnector.__new__(NSEConnector)
    sat = datetime.fromisoformat('2026-07-11T10:00:00').replace(tzinfo=EAT_OFFSET)
    assert nse.market_phase(sat) == 'closed'


def test_public_holiday_closed_even_during_trading_hours():
    nse = NSEConnector.__new__(NSEConnector)
    nse._holidays = {'2026-06-01'}  # Madaraka Day 2026 falls on a Monday
    holiday = datetime.fromisoformat('2026-06-01T11:00:00').replace(tzinfo=EAT_OFFSET)
    assert nse.market_phase(holiday) == 'closed'
    assert nse.is_market_open(holiday) is False


def test_non_holiday_weekday_unaffected_by_holiday_set():
    nse = NSEConnector.__new__(NSEConnector)
    nse._holidays = {'2026-12-25'}  # unrelated holiday configured
    assert nse.market_phase(_at(11, 0)) == 'open'  # 2026-07-10, not in the set


def test_missing_holidays_attr_does_not_crash():
    # Instances built without __init__ (e.g. other bypass fixtures) shouldn't
    # AttributeError just because _holidays was never set.
    nse = NSEConnector.__new__(NSEConnector)
    assert nse.market_phase(_at(11, 0)) == 'open'
