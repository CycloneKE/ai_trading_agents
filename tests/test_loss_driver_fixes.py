"""The four mechanisms that made the agent lose money, and their fixes.

1. Sells were outvoted by "hold" (no opinion) votes, so positions were
   closed almost only by their stops.
2. Crypto's 200-day trend holding was sold by an ordinary-pullback trailing
   stop and bought back a day later by the still-valid trend signal.
3. The volatile-chop regime left RSI as the sole voter, which bought every
   sell-off at its first oversold reading.
4. The fast strategies traded NSE shares, where a round trip costs ~3.9%.
"""
import json
from types import SimpleNamespace

import pytest

from src.agent.realtime_risk_manager import RealTimeRiskManager
from src.agent.regime import RANGING, TRENDING_UP, VOLATILE_CHOP, filter_signals, is_fast
from src.agent.strategy_manager import StrategyManager
from src.agent.volatility import risk_limits_for, stop_distances


def _cfg(**over):
    with open('config/config.json') as f:
        cfg = json.load(f)
    cfg.update(over)
    return cfg


def _sig(action, conf):
    return {'action': action, 'confidence': conf, 'position_size': 0.05 * conf}


def _blend(manager, signals, regime, symbol='AAPL'):
    kept = filter_signals(signals, regime)
    return manager._adaptive_confidence_ensemble(kept, symbol, {'open': 100.0, 'close': 101.0})


@pytest.fixture(scope='module')
def manager():
    return StrategyManager(_cfg())


# ------------------------------------------------------- 1. exits get through

def test_a_mean_reversion_sell_is_no_longer_outvoted_by_a_hold(manager):
    """It needed 0.95 confidence before; the RSI 'hold' diluted it."""
    out = _blend(manager, {'mean_reversion': _sig('sell', 0.6),
                           'rsi_strategy': _sig('hold', 0.0)}, RANGING)
    assert out['action'] == 'sell'
    assert out['confidence'] == pytest.approx(0.6)


def test_an_overbought_rsi_sell_can_take_profit_once_momentum_fades(manager):
    """It could never pass in an uptrend before: 0.40 at most."""
    out = _blend(manager, {'momentum': _sig('hold', 0.0),
                           'rsi_strategy': _sig('sell', 0.5)}, TRENDING_UP)
    assert out['action'] == 'sell'


def test_a_stronger_sell_never_reports_a_weaker_confidence(manager):
    """Confidence decides a full exit (>= 0.75) against a half trim, so it
    must rise with the sell, not drop where the old diluted rule took over."""
    confs = [_blend(manager, {'mean_reversion': _sig('sell', c),
                              'rsi_strategy': _sig('hold', 0.0)}, RANGING)['confidence']
             for c in (0.5, 0.7, 0.85, 0.95, 1.0)]
    assert confs == sorted(confs)
    assert confs[3] == pytest.approx(0.95)


def test_a_sell_still_loses_to_a_stronger_buy(manager):
    """Momentum still buying outweighs an overbought RSI: no exit."""
    out = _blend(manager, {'momentum': _sig('buy', 0.6),
                           'rsi_strategy': _sig('sell', 0.6)}, TRENDING_UP)
    assert out['action'] == 'hold'


def test_a_weak_sell_is_still_ignored(manager):
    out = _blend(manager, {'mean_reversion': _sig('sell', 0.3),
                           'rsi_strategy': _sig('hold', 0.0)}, RANGING)
    assert out['action'] == 'hold'


def test_entries_keep_the_strict_rule(manager):
    """A lone buy is still diluted by the others' holds: entries need agreement."""
    out = _blend(manager, {'mean_reversion': _sig('buy', 0.6),
                           'rsi_strategy': _sig('hold', 0.0)}, RANGING)
    assert out['action'] == 'hold'


def test_the_old_exit_rule_can_be_restored():
    old = StrategyManager(_cfg(exit_ignores_hold_votes=False))
    out = _blend(old, {'mean_reversion': _sig('sell', 0.6),
                       'rsi_strategy': _sig('hold', 0.0)}, RANGING)
    assert out['action'] == 'hold'


# ------------------------------------------------ 2. crypto: no pullback stop

def test_crypto_has_no_trailing_stop_and_a_wide_emergency_stop():
    cfg = _cfg()
    d = stop_distances(cfg, 0.035, 'crypto')            # a coin moving 3.5% a day
    assert d['trailing_stop_pct'] == 0.0
    assert d['stop_loss_pct'] == pytest.approx(0.21)    # 6 x ATR
    assert stop_distances(cfg, 0.01, 'crypto')['stop_loss_pct'] == pytest.approx(0.20)   # floor
    assert stop_distances(cfg, 0.10, 'crypto')['stop_loss_pct'] == pytest.approx(0.30)   # cap
    cold = stop_distances(cfg, None, 'crypto')          # no ATR yet
    assert cold == {'stop_loss_pct': 0.2, 'trailing_stop_pct': 0.0, 'source': 'fixed'}


def test_other_markets_keep_the_shared_stops():
    cfg = _cfg()
    assert stop_distances(cfg, 0.02, 'us_equity') == stop_distances(cfg, 0.02)
    assert stop_distances(cfg, 0.02, 'us_equity')['trailing_stop_pct'] == pytest.approx(0.06)
    assert risk_limits_for(cfg, 'us_equity') is cfg['risk_limits']


def test_a_twelve_percent_crypto_pullback_no_longer_sells(tmp_path):
    """The old 3x ATR trailing stop (10.5%) sold this dip; the trend signal
    then bought it back. Now only the trend's own exit or the emergency stop
    can close it."""
    from src.agent.main import TradingAgent
    agent = SimpleNamespace(config=_cfg(), volatility=SimpleNamespace(atr_pct=lambda s: 0.035))
    rm = RealTimeRiskManager.__new__(RealTimeRiskManager)
    rm.positions = {'BTC-USD': {'quantity': 1.0, 'market_value': 88000.0,
                                'high_watermark': 100000.0, 'avg_price': 90000.0},
                    'AAPL': {'quantity': 10.0, 'market_value': 1860.0,
                             'high_watermark': 200.0, 'avg_price': 190.0}}
    trail = lambda sym: TradingAgent._stop_distances_for(agent, sym, 0.05, 0.03)['trailing_stop_pct']
    hits = {h['symbol'] for h in rm.check_trailing_stops(trail)}
    assert 'BTC-USD' not in hits
    agent.volatility = SimpleNamespace(atr_pct=lambda s: 0.02)
    assert 'AAPL' in {h['symbol'] for h in rm.check_trailing_stops(trail)}   # 6% trail, down 7%


def test_crypto_is_sized_off_its_wider_stop():
    """A wider stop with the same position would risk far more than
    risk_per_trade; the live loop sizes off this market's own multiple."""
    from src.agent.position_sizing import volatility_scaled_value
    cfg = _cfg()
    mult = float(risk_limits_for(cfg, 'crypto')['stop_loss_atr_mult'])
    value = volatility_scaled_value(100000, 0.035, confidence=1.0, risk_per_trade=0.005,
                                    stop_atr_mult=mult, max_position_pct=0.05)
    assert value == pytest.approx(100000 * 0.005 / (6.0 * 0.035))   # about 2.4% of equity
    assert value * 6.0 * 0.035 == pytest.approx(500)                # a stop-out costs 0.5%


# --------------------------------------------- 3. volatile chop: no new buys

def _chopping(manager, symbol):
    wild = [100.0 + (6.0 if i % 2 else -6.0) for i in range(80)]
    manager.regime_detector.warm_start(symbol, wild)
    assert manager.regime_detector.regime(symbol) == VOLATILE_CHOP


def test_rsi_no_longer_buys_a_sell_off_on_its_own():
    m = StrategyManager(_cfg())
    _chopping(m, 'TSLA')
    out = m._combine_signals({'momentum': _sig('hold', 0.0), 'mean_reversion': _sig('hold', 0.0),
                              'rsi_strategy': _sig('buy', 0.5)}, {'symbol': 'TSLA'})
    assert out['action'] == 'hold'
    assert out['regime'] == VOLATILE_CHOP


def test_in_chop_a_holding_can_still_be_sold():
    m = StrategyManager(_cfg())
    _chopping(m, 'TSLA')
    out = m._combine_signals({'momentum': _sig('sell', 0.7), 'mean_reversion': _sig('hold', 0.0),
                              'rsi_strategy': _sig('buy', 0.6)}, {'symbol': 'TSLA'})
    assert out['action'] == 'sell'


def test_slow_strategies_are_not_affected_by_chop():
    kept = filter_signals({'crypto_trend': _sig('buy', 0.8)}, VOLATILE_CHOP)
    assert kept == {'crypto_trend': _sig('buy', 0.8)}
    assert not is_fast('crypto_trend') and not is_fast('nse_rotation')
    assert all(is_fast(n) for n in ('momentum', 'mean_reversion', 'rsi_strategy'))


# ------------------------------------------------- 4. NSE: rotation only

def test_nse_shares_hear_only_the_rotation(manager):
    voters = {n for n in manager.strategies if manager.in_scope(n, 'SCOM')}
    assert voters == {'nse_rotation'}
    assert {n for n in manager.strategies if manager.in_scope(n, 'AAPL')} >= {
        'momentum', 'mean_reversion', 'rsi_strategy'}
