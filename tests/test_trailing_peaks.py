"""Trailing-stop peaks survive a restart (realtime_risk_manager). They were
kept in memory only, so every redeploy reset each position's peak to the
current price and moved its trailing stop down."""
from types import SimpleNamespace

import pytest

import src.utils.paths as paths
from src.agent.realtime_risk_manager import RealTimeRiskManager


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA_DIR', tmp_path)
    return tmp_path


def pos(symbol, price, avg=100.0, qty=10.0):
    return SimpleNamespace(symbol=symbol, quantity=qty, avg_entry_price=avg, current_price=price,
                           market_value=qty * price)


def test_a_restart_remembers_how_far_a_position_had_run(data_dir):
    before = RealTimeRiskManager({}, None)
    for price in (100.0, 112.0, 120.0, 118.0):
        before.sync_broker_positions([pos('NVDA', price)])
    assert before.positions['NVDA']['high_watermark'] == 120.0
    after = RealTimeRiskManager({}, None)                                 # a redeploy
    after.sync_broker_positions([pos('NVDA', 110.0)])
    assert after.positions['NVDA']['high_watermark'] == 120.0
    [stop] = after.check_trailing_stops(0.08)                             # 110 is 8.3% below 120
    assert stop['symbol'] == 'NVDA'


def test_a_new_holding_of_the_same_symbol_starts_afresh(data_dir):
    first = RealTimeRiskManager({}, None)
    first.sync_broker_positions([pos('TSLA', 300.0, avg=250.0)])
    again = RealTimeRiskManager({}, None)
    again.sync_broker_positions([pos('TSLA', 200.0, avg=195.0)])          # sold, later bought back
    assert again.positions['TSLA']['high_watermark'] == 200.0


def test_an_empty_answer_from_the_broker_does_not_wipe_the_peaks(data_dir):
    risk = RealTimeRiskManager({}, None)
    risk.sync_broker_positions([pos('XLV', 150.0, avg=140.0)])
    risk.sync_broker_positions([])                                        # a bad minute at the broker
    later = RealTimeRiskManager({}, None)
    later.sync_broker_positions([pos('XLV', 145.0, avg=140.0)])
    assert later.positions['XLV']['high_watermark'] == 150.0
