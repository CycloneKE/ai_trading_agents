"""Volatility tracking and volatility-relative stop distances."""
import pandas as pd
import pytest

from src.agent.indicators import atr, atr_from_closes, rsi
from src.agent.volatility import VolatilityTracker, stop_distances


CFG = {'risk_limits': {
    'stop_loss_pct': 0.05, 'trailing_stop_pct': 0.03,
    'stop_loss_atr_mult': 2.5, 'trailing_stop_atr_mult': 3.0,
    'min_stop_pct': 0.02, 'max_stop_pct': 0.25,
}}


def test_rsi_matches_wilders_published_example():
    # New Concepts in Technical Trading Systems, the worked RSI table.
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
              46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41]
    s = rsi(pd.Series(closes), 14)
    assert round(float(s.iloc[14]), 2) == 70.46
    assert round(float(s.iloc[15]), 2) == 66.25
    assert s.iloc[:14].isna().all()


def test_atr_is_positive_and_seeded_after_length_bars():
    highs = [10 + i * 0.1 + 0.5 for i in range(40)]
    lows = [10 + i * 0.1 - 0.5 for i in range(40)]
    closes = [10 + i * 0.1 for i in range(40)]
    a = atr(highs, lows, closes, 14)
    assert a.iloc[:14].isna().all()
    assert float(a.iloc[14]) > 0
    assert (a.dropna() > 0).all()


def test_atr_needs_more_bars_than_its_length():
    assert atr([1] * 5, [1] * 5, [1] * 5, 14).isna().all()


def test_close_only_atr_understates_true_atr():
    # Documented behaviour: without high/low the true range collapses to the
    # close-to-close move, so stops derived from it are tighter, never looser.
    highs = [10 + (i % 3) for i in range(60)]
    lows = [8 + (i % 3) for i in range(60)]
    closes = [9 + (i % 3) for i in range(60)]
    full = float(atr(highs, lows, closes, 14).iloc[-1])
    close_only = float(atr_from_closes(closes, 14).iloc[-1])
    assert close_only < full


def test_tracker_returns_none_until_enough_history():
    t = VolatilityTracker(length=14)
    for i in range(10):
        t.update('X', 100 + i)
    assert t.atr_pct('X') is None
    assert t.bars('X') == 10


def test_tracker_produces_atr_pct_with_enough_history():
    t = VolatilityTracker(length=14)
    for i in range(40):
        p = 100 + (i % 5)
        t.update('X', p, p + 1, p - 1)
    pct = t.atr_pct('X')
    assert pct is not None and 0 < pct < 1


def test_tracker_ignores_bad_prices():
    t = VolatilityTracker(length=14)
    t.update('X', 0)
    t.update('X', -5)
    t.update('X', None)
    t.update('X', 'abc')
    assert t.bars('X') == 0


def test_inverted_high_low_degrades_to_close_only_instead_of_corrupting():
    t = VolatilityTracker(length=14)
    for i in range(40):
        p = 100 + (i % 5)
        t.update('X', p, high=p - 2, low=p + 2)   # inverted on purpose
    assert t.atr_pct('X') is not None             # still usable, not NaN


def test_warm_start_seeds_without_live_bars():
    t = VolatilityTracker(length=14)
    closes = [100 + (i % 7) for i in range(50)]
    assert t.warm_start('X', closes) == 50
    assert t.atr_pct('X') is not None


def test_stops_fall_back_to_fixed_without_atr():
    d = stop_distances(CFG, None)
    assert d['source'] == 'fixed'
    assert d['stop_loss_pct'] == 0.05
    assert d['trailing_stop_pct'] == 0.03


def test_stops_scale_with_atr():
    d = stop_distances(CFG, 0.04)          # 4% ATR
    assert d['source'] == 'atr'
    assert d['stop_loss_pct'] == pytest.approx(0.10)    # 2.5x
    assert d['trailing_stop_pct'] == pytest.approx(0.12)  # 3.0x


def test_stops_are_clamped_at_both_ends():
    quiet = stop_distances(CFG, 0.0005)    # would be 0.125%, below the floor
    assert quiet['stop_loss_pct'] == 0.02
    wild = stop_distances(CFG, 0.50)       # would be 125%, above the ceiling
    assert wild['stop_loss_pct'] == 0.25
    assert wild['trailing_stop_pct'] == 0.25


def test_goog_atr_gives_a_stop_wider_than_the_old_fixed_three_percent():
    """The regression this whole change exists to prevent."""
    t = VolatilityTracker(length=14)
    # ~2.6% ATR, the measured GOOG daily average.
    for i in range(60):
        p = 100 + (i % 4)
        t.update('GOOG', p, p * 1.013, p * 0.987)
    d = stop_distances(CFG, t.atr_pct('GOOG'))
    assert d['trailing_stop_pct'] > 0.03


# --- live-loop wiring -------------------------------------------------------

class _StubAgent:
    """Just enough of TradingAgent to exercise the stop resolver."""
    from src.agent.main import TradingAgent
    _stop_distances_for = TradingAgent._stop_distances_for

    def __init__(self, config, volatility=None):
        self.config = config
        self.volatility = volatility


def test_agent_resolver_falls_back_when_no_tracker():
    a = _StubAgent(CFG, volatility=None)
    d = a._stop_distances_for('AAPL', 0.05, 0.03)
    assert d == {'stop_loss_pct': 0.05, 'trailing_stop_pct': 0.03, 'source': 'fixed'}


def test_agent_resolver_falls_back_when_symbol_has_no_history():
    a = _StubAgent(CFG, volatility=VolatilityTracker(length=14))
    d = a._stop_distances_for('UNSEEN', 0.05, 0.03)
    assert d['source'] == 'fixed'


def test_agent_resolver_uses_atr_once_history_exists():
    t = VolatilityTracker(length=14)
    for i in range(60):
        p = 100 + (i % 4)
        t.update('AAPL', p, p * 1.02, p * 0.98)
    a = _StubAgent(CFG, volatility=t)
    d = a._stop_distances_for('AAPL', 0.05, 0.03)
    assert d['source'] == 'atr'
    assert d['trailing_stop_pct'] > 0.03      # wider than the old flat stop


def test_agent_resolver_survives_a_broken_tracker():
    class Boom:
        def atr_pct(self, symbol):
            raise RuntimeError("feed exploded")
    a = _StubAgent(CFG, volatility=Boom())
    d = a._stop_distances_for('AAPL', 0.05, 0.03)
    assert d['source'] == 'fixed'             # degrades, never leaves a position unstopped


def test_risk_manager_accepts_a_per_symbol_resolver():
    from src.agent.realtime_risk_manager import RealTimeRiskManager
    rm = RealTimeRiskManager.__new__(RealTimeRiskManager)
    rm.positions = {
        'TIGHT': {'quantity': 10, 'market_value': 900.0, 'high_watermark': 100.0},
        'WIDE': {'quantity': 10, 'market_value': 900.0, 'high_watermark': 100.0},
    }
    # Price is 90 against a 100 watermark, a 10% drawdown: trips a 5% stop,
    # survives a 20% one. Both positions are identical, so any difference
    # comes from the resolver being consulted per symbol.
    fired = {a['symbol'] for a in rm.check_trailing_stops(
        lambda sym: 0.05 if sym == 'TIGHT' else 0.20)}
    assert fired == {'TIGHT'}


def test_risk_manager_still_accepts_a_plain_float():
    from src.agent.realtime_risk_manager import RealTimeRiskManager
    rm = RealTimeRiskManager.__new__(RealTimeRiskManager)
    rm.positions = {'X': {'quantity': 10, 'market_value': 900.0, 'high_watermark': 100.0}}
    assert len(rm.check_trailing_stops(0.05)) == 1
    assert len(rm.check_trailing_stops(0.20)) == 0
