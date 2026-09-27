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
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
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
    assert gemini_only.gemini_model == 'gemini-flash-lite-latest'       # the larger free allowance
    assert all('SECRET123' not in url for url, _ in calls)             # the key is not in the address
    assert calls[0][1]['x-goog-api-key'] == 'SECRET123'
    [g] = gemini_only.provider_health()
    assert g['status'] == 'ok' and g['model'] == 'gemini-flash-lite-latest'


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
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    llm = lo.LLMOrchestrator({'primary_llm_provider': 'openrouter'})
    img = tmp_path / 'w.jpg'
    img.write_bytes(b'\xff\xd8')
    with pytest.raises(ValueError, match='reads text only. Add GROQ_API_KEY or GEMINI_API_KEY'):
        dw.read(str(img), llm, last_close=lambda s, d: None)
    assert [p['provider'] for p in llm.provider_health()] == ['openrouter']


def test_a_spent_quota_is_left_alone_for_a_while(gemini_only, monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return _reply(429, {'error': {'message': 'You exceeded your current quota, please check your plan '
                                                 'and billing details. For more information on this error, '
                                                 'head to: https://ai.google.dev/gemini-api/docs/rate-limits.'}})
    monkeypatch.setattr(lo.requests, 'post', post)
    gemini_only.propose_json('sys', 'user')
    gemini_only.propose_json('sys', 'user')
    assert len(calls) == 1                                                 # not asked again straight away
    [g] = gemini_only.provider_health()
    assert g['last_error'] == 'HTTP 429: You exceeded your current quota, please check your plan and billing details.'
    left = gemini_only._cooldown_until['gemini'] - lo.time.time()
    assert 1700 < left <= lo.QUOTA_COOLDOWN_SECONDS


def test_a_withdrawn_free_openrouter_model_is_replaced_by_one_offered_now(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    llm = lo.LLMOrchestrator({'primary_llm_provider': 'openrouter',
                              'swarm': {'agents': {'synthesizer': 'meta-llama/llama-3.3-70b-instruct:free'}}})
    listing = {'data': [
        {'id': 'meta-llama/llama-3.3-70b-instruct', 'pricing': {'prompt': '0.0000001', 'completion': '0.0000003'}},
        {'id': 'qwen/qwen3-32b:free', 'pricing': {'prompt': '0', 'completion': '0'}, 'context_length': 40000,
         'supported_parameters': ['response_format']},
        {'id': 'meta-llama/llama-4-maverick:free', 'pricing': {'prompt': '0', 'completion': '0'},
         'context_length': 128000, 'supported_parameters': ['response_format', 'tools']},
        {'id': 'meta-llama/llama-3.2-3b-instruct:free', 'pricing': {'prompt': '0', 'completion': '0'},
         'context_length': 8000, 'supported_parameters': ['tools']},                # no JSON mode
    ]}
    sent = []

    def post(url, headers=None, json=None, timeout=None):
        sent.append(json['model'])
        if json['model'].endswith('llama-3.3-70b-instruct:free'):
            return _reply(404, {'error': {'message': 'This model is unavailable for free.'}})
        return _reply(200, {'choices': [{'message': {'content': '{"action": "buy"}'}}]})
    monkeypatch.setattr(lo.requests, 'post', post)
    monkeypatch.setattr(lo.requests, 'get', lambda *a, **k: _reply(200, listing))
    assert llm.propose_json('sys', 'user') == {'action': 'buy'}
    assert sent == ['meta-llama/llama-3.3-70b-instruct:free', 'meta-llama/llama-4-maverick:free']
    assert llm.propose_json('sys', 'user') == {'action': 'buy'}
    assert sent[-1] == 'meta-llama/llama-4-maverick:free'                  # remembered, no second 404
    [o] = llm.provider_health()
    assert o['status'] == 'ok' and o['model'] == 'meta-llama/llama-4-maverick:free'


# ------------------------------------------------------------------ Groq

def _groq_reply(content):
    return _reply(200, {'choices': [{'message': {'content': content}}]})


GONE = {'error': {'message': 'The model `x` does not exist or you do not have access to it.',
                  'code': 'model_not_found'}}


@pytest.fixture
def groq_first(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'GSK_SECRET')
    monkeypatch.setenv('GEMINI_API_KEY', 'g')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    return lo.LLMOrchestrator({'primary_llm_provider': 'groq'})


def test_groq_reviews_first_when_it_is_the_main_provider(groq_first, monkeypatch):
    sent = []

    def post(url, headers=None, json=None, timeout=None):
        sent.append((url, headers, json))
        return _groq_reply('{"action": "buy", "confidence": 0.7}')
    monkeypatch.setattr(lo.requests, 'post', post)
    assert groq_first._provider_order() == ['groq', 'gemini']
    assert groq_first.propose_json('sys', 'user') == {'action': 'buy', 'confidence': 0.7}
    [(url, headers, body)] = sent
    assert url == lo.GROQ_CHAT_URL and headers['Authorization'] == 'Bearer GSK_SECRET'
    assert body['model'] == 'openai/gpt-oss-120b' and body['response_format'] == {'type': 'json_object'}
    assert body['reasoning_effort'] == 'low'
    health = {p['provider']: p for p in groq_first.provider_health()}
    assert health['groq']['status'] == 'ok' and health['groq']['reads_images']
    assert health['gemini']['status'] == 'unused'


def test_a_refused_reasoning_setting_is_dropped_not_fatal(groq_first, monkeypatch):
    sent = []

    def post(url, headers=None, json=None, timeout=None):
        sent.append(dict(json))
        if 'reasoning_effort' in json:
            return _reply(400, {'error': {'message': '`reasoning_effort` is not supported with this model'}})
        return _groq_reply('{"action": "hold"}')
    monkeypatch.setattr(lo.requests, 'post', post)
    assert groq_first.propose_json('sys', 'user') == {'action': 'hold'}
    assert len(sent) == 2 and 'reasoning_effort' not in sent[1]


def test_when_groq_is_rate_limited_gemini_answers(groq_first, monkeypatch):
    def post(url, headers=None, json=None, timeout=None):
        if url == lo.GROQ_CHAT_URL:
            return _reply(429, {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` '
                                                     'on tokens per day (TPD). Please try again in 12m3s.'}})
        return _reply(200, OK)
    monkeypatch.setattr(lo.requests, 'post', post)
    assert groq_first.propose_json('sys', 'user')['action'] == 'buy'
    health = {p['provider']: p for p in groq_first.provider_health()}
    assert health['groq']['status'] == 'failing' and 'Groq' in health['groq']['advice']
    assert 'GSK_SECRET' not in health['groq']['last_error']
    assert health['gemini']['status'] == 'ok'
    assert groq_first._cooldown_until['groq'] - lo.time.time() > 1700    # a daily limit: left alone a while


def test_a_model_off_the_free_plan_is_replaced_from_groqs_own_list(groq_first, monkeypatch):
    # The account's list can still name models the free plan cannot use, so
    # a substitute that is refused too gives way to the next.
    listing = {'data': [
        {'id': 'whisper-large-v3', 'active': True, 'context_window': 448},
        {'id': 'meta-llama/llama-guard-4-12b', 'active': True, 'context_window': 131072},
        {'id': 'llama-3.3-70b-specdec', 'active': False, 'context_window': 8192},
        {'id': 'qwen/qwen3.6-27b', 'active': True, 'context_window': 131072},
        {'id': 'qwen/qwen3.8-27b', 'active': True, 'context_window': 131072},
        {'id': 'openai/gpt-oss-20b', 'active': True, 'context_window': 131072},
    ]}
    sent, listed = [], []

    def post(url, headers=None, json=None, timeout=None):
        sent.append(json['model'])
        if json['model'] in ('openai/gpt-oss-120b', 'qwen/qwen3.8-27b'):
            return _reply(404, GONE)
        return _groq_reply('{"action": "hold"}')

    def get(url, headers=None, timeout=None):
        listed.append(url)
        return _reply(200, listing)
    monkeypatch.setattr(lo.requests, 'post', post)
    monkeypatch.setattr(lo.requests, 'get', get)
    assert groq_first.propose_json('sys', 'user') == {'action': 'hold'}
    assert sent == ['openai/gpt-oss-120b', 'qwen/qwen3.8-27b', 'qwen/qwen3.6-27b']
    assert groq_first.propose_json('sys', 'user') == {'action': 'hold'}
    assert sent[-1] == 'qwen/qwen3.6-27b' and len(listed) == 1          # remembered; listed once
    assert {p['provider']: p for p in groq_first.provider_health()}['groq']['model'] == 'qwen/qwen3.6-27b'


def test_when_no_listed_model_answers_the_refusal_is_reported(groq_first, monkeypatch):
    monkeypatch.setattr(lo.requests, 'post', lambda *a, **k: _reply(404, GONE))
    monkeypatch.setattr(lo.requests, 'get', lambda *a, **k: _reply(200, {'data': [
        {'id': 'whisper-large-v3'}, {'id': 'openai/gpt-oss-20b'}]}))
    with pytest.raises(requests.HTTPError):
        groq_first._groq_post([{'role': 'user', 'content': 'x'}], vision=True)
    assert groq_first.groq_vision_model == 'qwen/qwen3.8-27b'              # nothing better was found


def test_groq_model_choice_by_family():
    models = [{'id': 'qwen/qwen3-32b', 'context_window': 131072},                  # text only
              {'id': 'qwen/qwen3.6-27b', 'context_window': 131072},
              {'id': 'qwen/qwen3.8-27b', 'context_window': 131072},
              {'id': 'meta-llama/llama-prompt-guard-2-86m', 'context_window': 512},
              {'id': 'playai-tts', 'context_window': 8192},
              {'id': 'openai/gpt-oss-20b', 'context_window': 131072},
              {'id': 'openai/gpt-oss-120b', 'context_window': 131072}]
    assert lo.pick_groq_model(models, vision=True) == 'qwen/qwen3.8-27b'          # the newer version
    assert lo.pick_groq_model(models, vision=True, exclude={'qwen/qwen3.8-27b'}) == 'qwen/qwen3.6-27b'
    assert lo.pick_groq_model(models, vision=False) == 'openai/gpt-oss-120b'
    assert lo.pick_groq_model(models[5:], vision=True) is None                  # no model that reads pictures
    assert lo.pick_groq_model(models[5:6], vision=False) == 'openai/gpt-oss-20b'


def test_a_reply_with_a_code_fence_reasoning_or_words_around_the_json_is_still_read():
    assert lo.json_from_text('```json\n{"a": 1}\n```') == {'a': 1}
    assert lo.json_from_text('Here is the table:\n{"a": {"b": 2}}\nDone.') == {'a': {'b': 2}}
    assert lo.json_from_text('<think>rows are {x}</think>\n{"a": 3}') == {'a': 3}
    assert lo.json_from_text('[1, 2]') is None and lo.json_from_text('no json here') is None


def test_groq_reads_the_picture_before_gemini(groq_first, monkeypatch, tmp_path):
    from src.agent import daily_whispers as dw
    from tests.test_daily_whispers import SHEET
    sent, urls = {}, []

    def post(url, headers=None, json=None, timeout=None):
        urls.append(url)
        sent.update(json)
        return _groq_reply('```json\n' + __import__('json').dumps(SHEET) + '\n```')
    monkeypatch.setattr(lo.requests, 'post', post)
    img = tmp_path / 'whispers.jpg'
    img.write_bytes(b'\xff\xd8')
    sheet = dw.read(str(img), groq_first, last_close=lambda s, d: None)
    assert len(sheet['accepted']) == 5 and sheet['as_of'] == '2026-09-24'
    assert urls == [lo.GROQ_CHAT_URL]                                      # Gemini's allowance untouched
    assert sent['model'] == 'qwen/qwen3.8-27b' and sent['max_completion_tokens'] == 4096
    assert 'reasoning_effort' not in sent
    text, image = sent['messages'][0]['content']
    assert 'Reply with JSON only' in text['text']
    assert image['image_url']['url'].startswith('data:image/jpeg;base64,')


def test_gemini_reads_the_picture_when_groq_cannot(groq_first, monkeypatch, tmp_path):
    from src.agent import daily_whispers as dw
    from tests.test_daily_whispers import SHEET

    def post(url, headers=None, json=None, timeout=None):
        if url == lo.GROQ_CHAT_URL:
            return _reply(429, {'error': {'message': 'Rate limit reached for model `qwen/qwen3.8-27b`.'}})
        return _reply(200, {'candidates': [{'content': {'parts': [{'text': __import__('json').dumps(SHEET)}]}}]})
    monkeypatch.setattr(lo.requests, 'post', post)
    img = tmp_path / 'whispers.jpg'
    img.write_bytes(b'\xff\xd8')
    assert len(dw.read(str(img), groq_first, last_close=lambda s, d: None)['accepted']) == 5


def test_geminis_quota_message_keeps_the_limit_and_the_model():
    body = {'error': {'message': (
        'You exceeded your current quota, please check your plan and billing details. For more '
        'information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. '
        '\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, '
        'limit: 20, model: gemini-3.5-flash\nPlease retry in 22.5s.')}}
    err = requests.HTTPError('429', response=SimpleNamespace(status_code=429, json=lambda: body))
    assert lo.describe_error(err) == ('HTTP 429: You exceeded your current quota, please check your plan and '
                                      'billing details. (limit 20, model gemini-3.5-flash)')
