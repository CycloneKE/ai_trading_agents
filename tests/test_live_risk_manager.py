import time
import pytest
from live_risk_manager import LiveRiskManager


def test_portfolio_stop_loss_trigger():
    cfg = {'portfolio_stop_loss': 0.1}
    rm = LiveRiskManager(cfg)
    peak = 100000.0
    current_safe = 95000.0
    current_bad = 85000.0

    assert rm.check_portfolio_stop_loss(peak, current_safe) is True
    assert rm.check_portfolio_stop_loss(peak, current_bad) is False


def test_kelly_fraction_and_caps():
    cfg = {'kelly_shrink': 0.5, 'kelly_cap': 0.2}
    rm = LiveRiskManager(cfg)

    # decent edge-case win rate and win/loss
    f = rm.kelly_fraction(0.6, 1.5)
    assert 0.0 <= f <= 0.2

    # if win_loss_ratio <= 0 -> 0
    assert rm.kelly_fraction(0.5, 0) == 0.0


def test_adaptive_position_size_caps():
    cfg = {'max_position_size': 0.05}
    rm = LiveRiskManager(cfg)
    pv = 100000.0
    vol = 0.02
    size = rm.adaptive_position_size(0.03, vol, pv)
    # capped by max position size
    assert size <= pv * cfg['max_position_size']


def test_cooldown_behavior():
    cfg = {'trade_cooldown_seconds': 2}
    rm = LiveRiskManager(cfg)
    sym = 'AAPL'
    now = time.time()
    rm.record_exit(sym, timestamp=now)
    # immediately after exit, cannot trade
    assert rm.can_trade(sym, timestamp=now + 0.5) is False
    # after cooldown, can trade
    assert rm.can_trade(sym, timestamp=now + 3) is True