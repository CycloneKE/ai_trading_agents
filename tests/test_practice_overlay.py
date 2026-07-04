"""Tests for the practiced-parameter overlay in StrategyManager."""
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


def test_overlay_applies_promoted_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs('data', exist_ok=True)
    with open(os.path.join('data', 'strategy_params.json'), 'w') as f:
        json.dump({'momentum': {'rsi_oversold': 35}}, f)

    mgr = StrategyManager(CONFIG)
    strat = mgr.strategies['momentum']
    assert strat.rsi_oversold == 35        # promoted value wins
    assert strat.lookback_period == 50     # untouched knobs keep config values


def test_missing_or_corrupt_overlay_is_harmless(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mgr = StrategyManager(CONFIG)          # no overlay file at all
    assert mgr.strategies['momentum'].rsi_oversold == 30

    os.makedirs('data', exist_ok=True)
    with open(os.path.join('data', 'strategy_params.json'), 'w') as f:
        f.write('{not valid json')
    mgr2 = StrategyManager(CONFIG)         # corrupt overlay: config wins
    assert mgr2.strategies['momentum'].rsi_oversold == 30
