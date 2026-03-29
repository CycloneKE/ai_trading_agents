import importlib
import pytest

import data_manager as dm_module
from data_manager import DataManager


def test_data_manager_skips_connectors_when_fallback_only(monkeypatch):
    """When use_fallback_only is True, DataManager should not initialize external connectors."""
    # Ensure fallback generator is not relied on for this test
    monkeypatch.setattr(dm_module, 'FALLBACK_AVAILABLE', False)
    monkeypatch.setattr(dm_module, 'FallbackDataGenerator', None)

    cfg = {
        'symbols': ['AAPL'],
        'connectors': {},
        'use_fallback_only': True
    }

    dm = DataManager(cfg)

    # The connectors dict should be empty because we forced fallback-only mode
    assert isinstance(dm.connectors, dict)
    assert dm.connectors == {}
