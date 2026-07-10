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
