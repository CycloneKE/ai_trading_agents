# tests/test_llm_provider_resilience.py
"""Provider-ordering, fallback, and 429-cooldown behavior of the LLM
orchestrator's _complete() helper. Providers are stubbed — no network."""
import time

import pytest
import requests

from src.agent.llm_orchestrator import LLMOrchestrator


def _make(primary='gemini', monkeypatch=None):
    orch = LLMOrchestrator({'primary_llm_provider': primary, 'llm_cooldown_seconds': 60})
    # Pretend both keys are set so both providers are usable.
    orch.gemini_api_key = 'g'
    orch.openrouter_api_key = 'o'
    return orch


def _rate_limit_error():
    resp = requests.Response()
    resp.status_code = 429
    err = requests.HTTPError('429')
    err.response = resp
    return err


def test_primary_success_returns_immediately(monkeypatch):
    orch = _make('gemini')
    calls = []
    monkeypatch.setattr(orch, '_call_gemini', lambda s, u, f, model_override=None: (calls.append('gemini'), {'ok': 'gemini'})[1])
    monkeypatch.setattr(orch, '_call_openrouter', lambda *a, **k: (_ for _ in ()).throw(AssertionError('should not be called')))
    out = orch._complete('s', 'u', {'base': True})
    assert out == {'ok': 'gemini'}
    assert calls == ['gemini']  # secondary never touched


def test_429_on_primary_falls_through_to_secondary(monkeypatch):
    orch = _make('gemini')
    monkeypatch.setattr(orch, '_call_gemini', lambda *a, **k: (_ for _ in ()).throw(_rate_limit_error()))
    monkeypatch.setattr(orch, '_call_openrouter', lambda s, u, f, model_override=None: {'ok': 'openrouter'})
    out = orch._complete('s', 'u', {'base': True})
    assert out == {'ok': 'openrouter'}
    # gemini is now on cooldown
    assert orch._cooldown_until.get('gemini', 0) > time.time()


def test_provider_on_cooldown_is_skipped(monkeypatch):
    orch = _make('gemini')
    orch._cooldown_until['gemini'] = time.time() + 60
    gemini_called = []
    monkeypatch.setattr(orch, '_call_gemini', lambda *a, **k: (gemini_called.append(1), {})[1])
    monkeypatch.setattr(orch, '_call_openrouter', lambda s, u, f, model_override=None: {'ok': 'openrouter'})
    out = orch._complete('s', 'u', {'base': True})
    assert out == {'ok': 'openrouter'}
    assert gemini_called == []  # skipped without a call


def test_all_providers_down_returns_fallback(monkeypatch):
    orch = _make('gemini')
    monkeypatch.setattr(orch, '_call_gemini', lambda *a, **k: (_ for _ in ()).throw(_rate_limit_error()))
    monkeypatch.setattr(orch, '_call_openrouter', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    base = {'base': True}
    assert orch._complete('s', 'u', base) is base


def test_cooldown_expires(monkeypatch):
    orch = _make('gemini')
    orch.cooldown_seconds = 0  # immediate expiry
    monkeypatch.setattr(orch, '_call_gemini', lambda *a, **k: (_ for _ in ()).throw(_rate_limit_error()))
    monkeypatch.setattr(orch, '_call_openrouter', lambda s, u, f, model_override=None: {'ok': 'openrouter'})
    orch._complete('s', 'u', {})           # trips gemini cooldown (0s)
    time.sleep(0.01)
    # gemini should be usable again now
    monkeypatch.setattr(orch, '_call_gemini', lambda s, u, f, model_override=None: {'ok': 'gemini-back'})
    assert orch._complete('s', 'u', {}) == {'ok': 'gemini-back'}


def test_provider_order_respects_primary_and_keys():
    orch = _make('openrouter')
    assert orch._provider_order() == ['openrouter', 'gemini']
    orch.gemini_api_key = None
    assert orch._provider_order() == ['openrouter']
