import json
import tempfile
import os

from src.agent import main as main_module


def test_main_propagates_top_level_use_fallback_only(monkeypatch):
    """Ensure that if top-level use_fallback_only is set, main propagates it into DataManager config."""
    # Create a temp config file with top-level use_fallback_only
    cfg = {
        # Provide a non-empty connectors dict because validate_config requires it
        'data_manager': {'symbols': ['AAPL'], 'connectors': {'yahoo_finance': {'enabled': False}}},
        'use_fallback_only': True,
        'monitoring': {'enabled': False},
        # validate_config requires a non-empty strategies section with a 'type'
        'strategies': {'momentum': {'type': 'technical', 'enabled': True, 'weight': 1.0}},
        'risk_management': {},
        'risk_limits': {},
        'brokers': {'paper_broker': {'type': 'paper', 'initial_cash': 100000}},
        'trading': {'initial_capital': 100000},
        'security': {},
        'logging': {}
    }

    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    with open(path, 'w') as f:
        json.dump(cfg, f)

    captured = {}

    # Stub out many heavy components to make TradingAgent init cheap
    class Dummy:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(main_module, 'DatabaseManager', Dummy)
    monkeypatch.setattr(main_module, 'RealTimeDataFeed', Dummy)
    monkeypatch.setattr(main_module, 'StrategyManager', Dummy)
    monkeypatch.setattr(main_module, 'BrokerManager', Dummy)
    monkeypatch.setattr(main_module, 'OrderExecutionEngine', Dummy)
    monkeypatch.setattr(main_module, 'RealTimeRiskManager', Dummy)
    monkeypatch.setattr(main_module, 'PerformanceAnalytics', Dummy)
    monkeypatch.setattr(main_module, 'SecureConfigManager', lambda *a, **k: Dummy())

    # Replace get_monitoring_service to return None (or a dummy with start method)
    monkeypatch.setattr(main_module, 'get_monitoring_service', lambda cfg: None)

    # Intercept DataManager creation to capture the config passed in
    def fake_datamanager(dm_cfg):
        captured['dm_cfg'] = dm_cfg
        return Dummy()

    monkeypatch.setattr(main_module, 'DataManager', fake_datamanager)

    # Bypass secret validation which exits if env vars are missing
    monkeypatch.setattr(main_module.TradingAgent, '_validate_secrets', lambda self: None)

    try:
        agent = main_module.TradingAgent(path)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

    # Assert DataManager was given a config with use_fallback_only=True
    assert 'dm_cfg' in captured
    assert captured['dm_cfg'].get('use_fallback_only') is True
