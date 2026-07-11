# tests/test_llm_cache.py
from src.agent.llm_orchestrator import LLMOrchestrator


def _orch(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    return LLMOrchestrator({'primary_llm_provider': 'gemini', 'llm_cache_ttl': 900})


def test_second_identical_call_is_served_from_cache(monkeypatch):
    orch = _orch(monkeypatch)
    calls = {'n': 0}

    def fake_gemini(sys_p, usr_p, fb, model_override=None):
        calls['n'] += 1
        return {'action': 'buy', 'confidence': 0.7, 'position_size': 0.03, 'reasoning': 'ok'}

    monkeypatch.setattr(orch, '_call_gemini', fake_gemini)
    sig = {'action': 'buy', 'confidence': 0.71}
    r1 = orch.validate_trade('SCOM', sig, {'close': 34.0})
    r2 = orch.validate_trade('SCOM', sig, {'close': 34.1})
    assert calls['n'] == 1
    assert r1['action'] == r2['action'] == 'buy'


def test_different_action_bypasses_cache(monkeypatch):
    orch = _orch(monkeypatch)
    calls = {'n': 0}
    monkeypatch.setattr(orch, '_call_gemini',
                        lambda *a, **k: calls.__setitem__('n', calls['n'] + 1) or {'action': 'hold', 'confidence': 0.5, 'position_size': 0, 'reasoning': 'x'})
    orch.validate_trade('SCOM', {'action': 'buy', 'confidence': 0.7}, {'close': 34.0})
    orch.validate_trade('SCOM', {'action': 'sell', 'confidence': 0.7}, {'close': 34.0})
    assert calls['n'] == 2


def test_fallback_verdict_is_not_cached(monkeypatch):
    # Simulate an all-providers-down outage: _complete returns the exact
    # `strategy_signal` object it was handed (same identity), which is what
    # happens when every provider fails/cools-down or JSON decoding fails.
    # That un-validated base signal must NOT be cached, so a real LLM
    # validation is retried on the very next call once a provider recovers.
    orch = _orch(monkeypatch)
    calls = {'n': 0}

    def fake_complete(sysp, usrp, fallback, model_override=None):
        calls['n'] += 1
        return fallback  # same object identity as strategy_signal

    monkeypatch.setattr(orch, '_complete', fake_complete)
    sig = {'action': 'buy', 'confidence': 0.71}
    r1 = orch.validate_trade('SCOM', sig, {'close': 34.0})
    r2 = orch.validate_trade('SCOM', sig, {'close': 34.1})

    assert calls['n'] == 2  # NOT served from cache on the second call
    assert r1 is sig and r2 is sig
    conf_bucket = round(sig['confidence'] * 10)
    cache_key = f"SCOM:{sig['action']}:{conf_bucket}"
    assert cache_key not in orch._verdict_cache
