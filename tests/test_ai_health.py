"""The AI services' health, as the dashboard shows it
(llm_orchestrator.provider_health, describe_error, the Gemini fallback)."""
import json
from types import SimpleNamespace

import pytest
import requests

from src.agent import llm_orchestrator as lo


def _reply(code, body):
    r = SimpleNamespace(status_code=code, json=lambda: body)

    def raise_for_status():
        if code >= 400:
            raise requests.HTTPError(f"{code} Client Error for url: https://x/?key=SECRET123", response=r)
    r.raise_for_status = raise_for_status
    return r


OK = {'candidates': [{'content': {'parts': [{'text': json.dumps({'action': 'buy', 'confidence': 0.8})}]}}]}


@pytest.fixture
def gemini_only(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'SECRET123')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    return lo.LLMOrchestrator({'primary_llm_provider': 'gemini', 'gemini_model': 'gemini-2.0-flash'})


def test_a_retired_gemini_model_gives_way_to_a_current_one(gemini_only, monkeypatch):
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append((url, headers))
        if 'gemini-2.0-flash' in url:
            return _reply(404, {'error': {'message': 'models/gemini-2.0-flash is not found'}})
        return _reply(200, OK)
    monkeypatch.setattr(lo.requests, 'post', post)
    out = gemini_only.propose_json('sys', 'user')
    assert out['action'] == 'buy'
    assert gemini_only.gemini_model == 'gemini-2.5-flash'
    assert all('SECRET123' not in url for url, _ in calls)             # the key is not in the address
    assert calls[0][1]['x-goog-api-key'] == 'SECRET123'
    [g] = gemini_only.provider_health()
    assert g['status'] == 'ok' and g['model'] == 'gemini-2.5-flash'


def test_a_failing_provider_says_why_without_the_key(gemini_only, monkeypatch):
    monkeypatch.setattr(lo.requests, 'post', lambda *a, **k: _reply(
        429, {'error': {'message': 'Resource has been exhausted (e.g. check quota).'}}))
    assert gemini_only.propose_json('sys', 'user') is None
    [g] = gemini_only.provider_health()
    assert g['status'] == 'failing' and g['failures'] == 1
    assert g['last_error'] == 'HTTP 429: Resource has been exhausted (e.g. check quota).'
    assert 'resets daily' in g['advice']
    refused = lo.describe_error(requests.HTTPError('403 for url https://x/?key=SECRET123',
                                                   response=SimpleNamespace(status_code=403, json=lambda: {})))
    assert 'SECRET123' not in refused and refused.startswith('HTTP 403')
    assert 'GEMINI_API_KEY' in lo.advice('gemini', refused)


def test_the_picture_reader_says_why_it_could_not_read(gemini_only, monkeypatch, tmp_path):
    from src.agent import daily_whispers as dw
    monkeypatch.setattr(lo.requests, 'post', lambda *a, **k: _reply(
        403, {'error': {'message': 'API key not valid. Please pass a valid API key.'}}))
    img = tmp_path / 'w.jpg'
    img.write_bytes(b'\xff\xd8')
    with pytest.raises(ValueError) as e:
        dw.read(str(img), gemini_only, last_close=lambda s, d: None)
    assert 'Gemini: HTTP 403: API key not valid' in str(e.value)


def test_text_only_ai_is_named_as_the_reason(monkeypatch, tmp_path):
    from src.agent import daily_whispers as dw
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    llm = lo.LLMOrchestrator({'primary_llm_provider': 'openrouter'})
    img = tmp_path / 'w.jpg'
    img.write_bytes(b'\xff\xd8')
    with pytest.raises(ValueError, match='reads text only. Add GEMINI_API_KEY'):
        dw.read(str(img), llm, last_close=lambda s, d: None)
    assert [p['provider'] for p in llm.provider_health()] == ['openrouter']
