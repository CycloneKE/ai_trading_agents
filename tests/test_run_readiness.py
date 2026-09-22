"""The gate that decides whether a paper run is worth starting.

Each test here corresponds to a way this system has silently produced
nothing. The point of the gate is not that it passes; it is that it fails
loudly in exactly these cases.
"""
import pandas as pd
import pytest

from src.agent.run_readiness import (BLOCK, WARN, Readiness, assess_readiness,
                                     check_broker, check_costs, check_journals,
                                     check_market_data, check_risk_settings,
                                     check_signal_path, check_strategies)


def _cfg(**over):
    cfg = {
        'data_manager': {'symbols': ['AAPL', 'MSFT'], 'nse_symbols': ['SCOM']},
        'brokers': {'paper_broker': {'type': 'paper', 'enabled': True}},
        'risk_limits': {'trailing_stop_atr_mult': 3.0},
        'strategy_manager': {'ensemble_method': 'adaptive_confidence'},
        'trading_loop_interval': 60,
    }
    cfg.update(over)
    return cfg


def _named(r, name):
    return next(c for c in r.checks if c.name == name)


# ------------------------------------------------------------- the container

def test_a_blocker_makes_the_run_not_ready_but_a_warning_does_not():
    r = Readiness()
    r.add(WARN, 'soft', False, 'shapes interpretation')
    assert r.ready
    r.add(BLOCK, 'hard', False, 'stops the run')
    assert not r.ready
    assert [c.name for c in r.blockers] == ['hard']
    assert [c.name for c in r.warnings] == ['soft']


# -------------------------------------------------------- the silent-nothings

def test_mock_strategies_block_the_run(monkeypatch):
    """The failure that cost the most: the agent starts, reports healthy, and
    holds forever because every strategy is a MockStrategy."""
    import src.agent.strategy_manager as sm

    class _AllMocks:
        def __init__(self, cfg):
            self.strategies = {'momentum': sm.MockStrategy('momentum', {}),
                               'rsi_strategy': sm.MockStrategy('rsi_strategy', {})}

    monkeypatch.setattr(sm, 'StrategyManager', _AllMocks)
    r = Readiness()
    check_strategies(_cfg(), r)
    c = _named(r, 'strategies are real, not mocks')
    assert not c.ok and c.level == BLOCK
    assert 'no trades at all' in c.detail


def test_no_strategies_at_all_blocks(monkeypatch):
    import src.agent.strategy_manager as sm

    class _Empty:
        def __init__(self, cfg):
            self.strategies = {}

    monkeypatch.setattr(sm, 'StrategyManager', _Empty)
    r = Readiness()
    check_strategies(_cfg(), r)
    c = _named(r, 'strategies loaded')
    assert not c.ok and c.level == BLOCK


def test_fallback_only_data_blocks_the_run():
    r = Readiness()
    check_market_data(_cfg(data_manager={'use_fallback_only': True}), r)
    c = _named(r, 'real market data')
    assert not c.ok and c.level == BLOCK
    assert 'refuse to trade' in c.detail


def test_every_symbol_on_fallback_prices_blocks_the_run():
    r = Readiness()
    sample = {'AAPL': {'source': 'fallback'}, 'MSFT': {'source': 'fallback'}}
    check_market_data(_cfg(), r, sample)
    c = _named(r, 'real market data')
    assert not c.ok and c.level == BLOCK


def test_one_real_price_among_fallbacks_is_not_a_blocker():
    """A partial vendor outage degrades the run; it does not make it pointless."""
    r = Readiness()
    sample = {'AAPL': {'source': 'real_data'}, 'MSFT': {'source': 'fallback'}}
    check_market_data(_cfg(), r, sample)
    assert _named(r, 'real market data').ok


def test_unprobed_data_is_a_warning_not_a_pass():
    r = Readiness()
    check_market_data(_cfg(), r, None)
    c = _named(r, 'real market data')
    assert not c.ok and c.level == WARN


# ---------------------------------------------------------------- the broker

def test_a_live_money_broker_blocks_a_paper_run():
    cfg = _cfg(brokers={'alpaca_broker': {'enabled': True, 'paper': False}})
    r = Readiness()
    check_broker(cfg, r)
    c = _named(r, 'all enabled brokers are paper mode')
    assert not c.ok and c.level == BLOCK
    assert 'real money' in c.detail


def test_a_disabled_live_broker_does_not_block():
    cfg = _cfg(brokers={'alpaca_broker': {'enabled': False, 'paper': False},
                        'paper_broker': {'enabled': True}})
    r = Readiness()
    check_broker(cfg, r)
    assert _named(r, 'all enabled brokers are paper mode').ok


def test_a_disconnected_broker_blocks_because_orders_cannot_fill():
    class Disconnected:
        is_connected = False

    r = Readiness()
    check_broker(_cfg(), r, Disconnected())
    c = _named(r, 'broker connected')
    assert not c.ok and c.level == BLOCK


def test_a_broker_whose_connector_failed_to_import_blocks():
    """The gate used to certify the config's intent, never what was built."""
    cfg = _cfg(brokers={'ibkr_broker': {'type': 'interactive_brokers',
                                        'enabled': True, 'paper': True}})
    r = Readiness()
    check_broker(cfg, r)
    c = _named(r, 'every enabled broker can be created')
    assert not c.ok and c.level == BLOCK
    assert 'ibkr_broker' in c.detail


def test_a_primary_broker_that_was_never_built_blocks():
    """Exactly the production state: Alpaca primary, connector unimportable,
    orders quietly filled by the internal simulator, gate said READY."""
    cfg = _cfg(brokers={'paper_broker': {'type': 'paper', 'enabled': True},
                        'ghost': {'type': 'interactive_brokers',
                                  'enabled': True, 'primary': True}})
    r = Readiness()
    check_broker(cfg, r)
    c = _named(r, 'primary broker is the configured one')
    assert not c.ok and c.level == BLOCK
    assert 'different venue' in c.detail


def test_a_live_broker_without_an_enabled_flag_still_blocks():
    """The gate defaulted any broker not named 'paper_broker' to disabled and
    skipped it, which would have waved a live-money broker through."""
    cfg = _cfg(brokers={'ibkr': {'type': 'alpaca', 'paper': False}})
    r = Readiness()
    check_broker(cfg, r)
    assert not _named(r, 'all enabled brokers are paper mode').ok


def test_the_real_config_passes_the_broker_checks():
    import json
    cfg = json.load(open('config/config.json'))
    r = Readiness()
    check_broker(cfg, r)
    for name in ('all enabled brokers are paper mode',
                 'every enabled broker can be created',
                 'primary broker is the configured one'):
        assert _named(r, name).ok, f"{name}: {_named(r, name).detail}"


# ----------------------------------------------------------- the signal path

class _Holds:
    """A strategy manager stand-in that never produces an actionable signal."""

    strategies = {'always_hold': object()}

    def __init__(self, cfg):
        pass

    def generate_signals(self, data):
        return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0}


class _Buys:
    strategies = {'always_buy': object()}

    def __init__(self, cfg):
        pass

    def generate_signals(self, data):
        return {'action': 'buy', 'confidence': 0.8, 'position_size': 0.04}


class _ConfidentButUnsized:
    """Confident and directional, yet sized to nothing.

    The live loop needs all three of action, confidence and position_size, so
    this configuration places no orders while looking perfectly healthy in
    every other check.
    """

    strategies = {'unsized': object()}

    def __init__(self, cfg):
        pass

    def generate_signals(self, data):
        return {'action': 'buy', 'confidence': 0.9, 'position_size': 0.0}


def _bars():
    idx = pd.date_range('2024-01-01', periods=30, freq='D')
    return {'AAPL': pd.DataFrame({'close': [100 + i for i in range(30)]}, index=idx)}


def test_strategies_that_never_act_block_the_run(monkeypatch):
    import src.agent.strategy_manager as sm
    monkeypatch.setattr(sm, 'StrategyManager', _Holds)
    r = Readiness()
    check_signal_path(_cfg(), r, _bars())
    c = _named(r, 'strategies produce actionable signals')
    assert not c.ok and c.level == BLOCK
    assert 'hold for the entire run' in c.detail


def test_confident_signals_sized_to_zero_also_block(monkeypatch):
    import src.agent.strategy_manager as sm
    monkeypatch.setattr(sm, 'StrategyManager', _ConfidentButUnsized)
    r = Readiness()
    check_signal_path(_cfg(), r, _bars())
    c = _named(r, 'strategies produce actionable signals')
    assert not c.ok and c.level == BLOCK


def test_actionable_signals_pass_and_report_the_rate(monkeypatch):
    import src.agent.strategy_manager as sm
    monkeypatch.setattr(sm, 'StrategyManager', _Buys)
    r = Readiness()
    check_signal_path(_cfg(), r, _bars())
    c = _named(r, 'strategies produce actionable signals')
    assert c.ok
    assert '30 of 30' in c.detail


def test_no_bars_is_a_warning_not_a_silent_pass():
    r = Readiness()
    check_signal_path(_cfg(), r, None)
    c = _named(r, 'strategies produce actionable signals')
    assert not c.ok and c.level == WARN


# ------------------------------------------------------------ the soft checks

def test_unwritable_journal_directory_blocks(tmp_path):
    blocked = tmp_path / 'file_not_a_dir'
    blocked.write_text('x')
    r = Readiness()
    check_journals(r, str(blocked / 'nested'))
    c = _named(r, 'journal directory writable')
    assert not c.ok and c.level == BLOCK


def test_writable_journal_directory_passes(tmp_path):
    r = Readiness()
    check_journals(r, str(tmp_path / 'journals'))
    assert _named(r, 'journal directory writable').ok


def test_placeholder_fees_warn_but_do_not_block():
    r = Readiness()
    check_costs(_cfg(), r)
    c = _named(r, 'transaction costs verified')
    assert c.level == WARN


def test_fixed_percentage_stops_warn():
    r = Readiness()
    check_risk_settings(_cfg(risk_limits={'trailing_stop_pct': 0.03}), r)
    c = _named(r, 'stops scale with volatility')
    assert not c.ok and c.level == WARN


# --------------------------------------------------------------- end to end

def test_assess_readiness_runs_every_check_without_the_network(tmp_path):
    r = assess_readiness(_cfg(), data_dir=str(tmp_path), check_api_import=False)
    names = {c.name for c in r.checks}
    for expected in ('strategies produce actionable signals', 'real market data',
                     'all enabled brokers are paper mode', 'cycle fits its budget',
                     'journal directory writable', 'transaction costs verified',
                     'stops scale with volatility'):
        assert expected in names


def test_unprobed_preconditions_never_masquerade_as_passes(tmp_path):
    """Nothing that was not actually checked may report OK."""
    r = assess_readiness(_cfg(), data_dir=str(tmp_path), check_api_import=False)
    warned = {c.name for c in r.warnings}
    assert 'real market data' in warned
    assert 'broker connected' in warned
    assert 'strategies produce actionable signals' in warned


# ------------------------------------------------------- the liveness probe

def test_disabled_monitoring_warns_because_nothing_reports_liveness():
    from src.agent.run_readiness import check_health_endpoint
    r = Readiness()
    check_health_endpoint(_cfg(monitoring={'enabled': False}), r)
    c = _named(r, 'liveness endpoint enabled')
    assert not c.ok and c.level == WARN
    assert 'died on day three' in c.detail


def test_enabled_monitoring_passes():
    from src.agent.run_readiness import check_health_endpoint
    r = Readiness()
    check_health_endpoint(_cfg(monitoring={'enabled': True, 'port': 8080}), r)
    assert _named(r, 'liveness endpoint enabled').ok
