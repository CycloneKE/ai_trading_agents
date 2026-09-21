"""The sentiment analyser's backends must be optional.

`src/api/api_server.py` imports `FinancialSentimentAnalyzer` at module scope,
so any hard dependency in that module takes the REST API and the whole
dashboard down with it. That is what happened: `torch` was imported
unconditionally for a FinBERT path that is itself optional, and textblob,
vaderSentiment and nltk were not declared in requirements.txt at all. The
agent reported it as "Flask not available", which is not where anyone would
look.
"""
import builtins
import importlib
import sys

import pytest


def _reimport_without(monkeypatch, blocked):
    """Reimport the sentiment module with `blocked` modules unimportable."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in blocked or name.split('.')[0] in blocked:
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    for mod in list(sys.modules):
        if mod.startswith('src.agent.sentiment_analyzer'):
            del sys.modules[mod]
    return importlib.import_module('src.agent.sentiment_analyzer')


@pytest.mark.parametrize('blocked', [
    {'torch'},
    {'transformers'},
    {'textblob'},
    {'vaderSentiment'},
    {'nltk'},
    {'torch', 'transformers', 'textblob', 'vaderSentiment', 'nltk'},
])
def test_module_imports_without_each_optional_backend(monkeypatch, blocked):
    mod = _reimport_without(monkeypatch, blocked)
    assert mod.FinancialSentimentAnalyzer is not None


def test_analyzer_constructs_and_scores_with_no_backends(monkeypatch):
    """With every backend gone it must still answer, neutrally, not raise."""
    mod = _reimport_without(
        monkeypatch, {'torch', 'transformers', 'textblob', 'vaderSentiment', 'nltk'})
    analyzer = mod.FinancialSentimentAnalyzer({})
    assert analyzer.use_finbert is False
    assert analyzer.use_vader is False
    assert analyzer.use_textblob is False
    result = analyzer.analyze_sentiment('Safaricom raises its dividend')
    assert isinstance(result, dict)


def test_device_is_none_rather_than_exploding_without_torch(monkeypatch):
    mod = _reimport_without(monkeypatch, {'torch'})
    assert mod.TORCH_AVAILABLE is False
    assert mod.FinancialSentimentAnalyzer({}).device is None


def test_config_cannot_enable_a_backend_that_is_not_installed(monkeypatch):
    """Asking for FinBERT without torch must not re-enable the broken path."""
    mod = _reimport_without(monkeypatch, {'torch', 'transformers'})
    analyzer = mod.FinancialSentimentAnalyzer({'use_finbert': True})
    assert analyzer.use_finbert is False


def test_api_server_imports_which_is_what_actually_broke():
    """The regression: this import failing disabled the entire dashboard."""
    import src.api.api_server as api
    assert api.TradingAPI is not None
