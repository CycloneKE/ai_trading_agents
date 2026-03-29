import data_manager as dm_module
from data_manager import DataManager


def test_data_manager_initializes_fallback_generator_when_available(monkeypatch):
    """When FALLBACK_AVAILABLE is True and a FallbackDataGenerator is provided,
    DataManager should set self.fallback_generator to an instance of it.
    """
    class DummyFallback:
        def __init__(self):
            self.called = True

    # Simulate fallback availability
    monkeypatch.setattr(dm_module, 'FALLBACK_AVAILABLE', True)
    monkeypatch.setattr(dm_module, 'FallbackDataGenerator', DummyFallback)

    cfg = {
        'symbols': ['AAPL'],
        'connectors': {},
        # Use fallback-only to avoid initializing external connectors during test
        'use_fallback_only': True
    }

    dm = DataManager(cfg)

    assert hasattr(dm, 'fallback_generator')
    assert dm.fallback_generator is not None
    assert isinstance(dm.fallback_generator, DummyFallback)
