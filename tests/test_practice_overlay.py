"""Tuned strategy settings applied at start-up (data/strategy_params.json).

The weekly strategy tuner keeps its adopted settings under "params"; the
practice session used to write a flat {strategy: settings} file there and is
now report-only, but an old file is still read. Only the settings the tuner
owns for a strategy's family are applied.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.strategy_manager import StrategyManager

CONFIG = {
    'ensemble_method': 'weighted_average',
    'strategies': {
        'momentum': {'type': 'technical', 'enabled': True, 'weight': 1.0,
                     'rsi_oversold': 30, 'lookback_period': 50},
    },
}


def test_the_tuners_settings_apply_at_start_up(tmp_path, monkeypatch):
    # strategy_manager reads the file from the repo-anchored DATA_DIR
    # (not the CWD), so point DATA_DIR at tmp_path.
    monkeypatch.setattr('src.agent.strategy_manager.DATA_DIR', tmp_path)
    with open(tmp_path / 'strategy_params.json', 'w') as f:
        json.dump({'params': {'momentum': {'threshold': 0.03}}, 'last_run': 0, 'log': []}, f)

    mgr = StrategyManager(CONFIG)
    strat = mgr.strategies['momentum']
    assert strat.momentum_threshold == 0.03   # tuned value wins
    assert strat.lookback_period == 50        # untouched knobs keep config values


def test_an_old_flat_file_applies_only_the_strategys_own_settings(tmp_path, monkeypatch):
    """The practice session wrote RSI levels and lookback_period onto every
    technical strategy, momentum included, which does not use RSI levels."""
    monkeypatch.setattr('src.agent.strategy_manager.DATA_DIR', tmp_path)
    with open(tmp_path / 'strategy_params.json', 'w') as f:
        json.dump({'momentum': {'threshold': 0.025, 'rsi_oversold': 35, 'lookback_period': 30}}, f)

    strat = StrategyManager(CONFIG).strategies['momentum']
    assert strat.momentum_threshold == 0.025
    assert strat.rsi_oversold == 30 and strat.lookback_period == 50


def test_missing_or_corrupt_overlay_is_harmless(tmp_path, monkeypatch):
    monkeypatch.setattr('src.agent.strategy_manager.DATA_DIR', tmp_path)
    mgr = StrategyManager(CONFIG)          # no overlay file at all
    assert mgr.strategies['momentum'].rsi_oversold == 30

    with open(tmp_path / 'strategy_params.json', 'w') as f:
        f.write('{not valid json')
    mgr2 = StrategyManager(CONFIG)         # corrupt overlay: config wins
    assert mgr2.strategies['momentum'].rsi_oversold == 30
