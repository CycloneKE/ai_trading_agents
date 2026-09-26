"""The paid AI tier and its monthly cap (src/agent/ai_budget.py,
src/agent/claude_provider.py, LLMOrchestrator). A stand-in client replaces
the Anthropic SDK: nothing here calls the API or spends money."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.agent.ai_budget import AiBudget, cost_usd
from src.agent.claude_provider import ClaudeProvider, ClaudeUnavailable


class _Client:
    """Records each request and answers with a canned reply."""

    def __init__(self, text='{"action": "buy", "confidence": 0.7, "position_size": 0.03, '
                            '"reasoning": "ok"}',
                 stop_reason='end_turn', usage=(1000, 200), error=None):
        self.requests = []
        self.text, self.stop_reason, self.usage, self.error = text, stop_reason, usage, error
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type='text', text=self.text)],
            usage=SimpleNamespace(input_tokens=self.usage[0], output_tokens=self.usage[1],
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0))


OCT = datetime(2026, 10, 5, tzinfo=timezone.utc)


# ------------------------------------------------------------------ budget

def test_calls_are_priced_from_their_token_counts():
    # Sonnet 5: $2 in, $10 out per million tokens.
    assert cost_usd('claude-sonnet-5', 1_000_000, 100_000) == pytest.approx(3.0)
    assert cost_usd('claude-haiku-4-5', 1_000_000, 0) == pytest.approx(1.0)
    # An unknown model is priced at the dearest known rate.
    assert cost_usd('claude-new', 0, 1_000_000) == pytest.approx(25.0)


def test_spending_is_kept_per_month_and_survives_a_restart(tmp_path):
    budget = AiBudget(tmp_path / 'spend.json', 20)
    budget.record('claude-sonnet-5', {'input_tokens': 1_000_000, 'output_tokens': 0}, 'review', now=OCT)
    assert AiBudget(tmp_path / 'spend.json', 20).spent(OCT) == pytest.approx(2.0)
    assert budget.spent(datetime(2026, 11, 1, tzinfo=timezone.utc)) == 0.0
    s = budget.summary(OCT)
    assert (s['cap_usd'], s['remaining_usd'], s['calls']) == (20, 18.0, 1)


def test_a_call_that_could_pass_the_cap_is_refused(tmp_path):
    budget = AiBudget(tmp_path / 'spend.json', 1.0)
    budget.record('claude-sonnet-5', {'input_tokens': 0, 'output_tokens': 95_000}, now=OCT)
    assert budget.allows(0.04, now=OCT) is True
    assert budget.allows(0.06, now=OCT) is False


# ---------------------------------------------------------------- provider

def _provider(tmp_path, client, cap=20.0):
    return ClaudeProvider('key', AiBudget(tmp_path / 'spend.json', cap), {}, client=client)


def test_a_trade_review_uses_sonnet_5_with_the_verdict_schema_and_is_charged(tmp_path):
    client = _Client()
    p = _provider(tmp_path, client)
    verdict = p.complete_json('review this', 'AAPL buy', purpose='review',
                              model_override='meta-llama/llama-3.1-8b-instruct')
    assert verdict['action'] == 'buy'
    (req,) = client.requests
    assert req['model'] == 'claude-sonnet-5'
    assert req['output_config']['format']['type'] == 'json_schema'
    assert req['output_config']['effort'] == 'medium'
    assert p.budget.spent() == pytest.approx(cost_usd('claude-sonnet-5', 1000, 200))


def test_volume_work_uses_haiku_without_an_effort_setting(tmp_path):
    client = _Client(text='```json\n{"outlook_score": 0.4}\n```')
    p = _provider(tmp_path, client)
    assert p.complete_json('sector', 'banks') == {'outlook_score': 0.4}
    (req,) = client.requests
    assert req['model'] == 'claude-haiku-4-5' and 'output_config' not in req


def test_a_refusal_or_a_cut_off_reply_is_no_answer_but_is_still_charged(tmp_path):
    for stop in ('refusal', 'max_tokens'):
        p = _provider(tmp_path / stop, _Client(stop_reason=stop))
        with pytest.raises(ClaudeUnavailable):
            p.complete_json('s', 'u', purpose='review')
        assert p.budget.spent() > 0


def test_with_the_cap_reached_claude_is_not_called(tmp_path):
    client = _Client()
    p = _provider(tmp_path, client, cap=0.01)
    with pytest.raises(ClaudeUnavailable):
        p.complete_json('s', 'u', purpose='review')
    assert client.requests == []


# ------------------------------------------------------------ orchestrator

def _orchestrator(monkeypatch, tmp_path, client, free=True, cap=20.0):
    from src.agent import llm_orchestrator as lo
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.setenv('GEMINI_API_KEY', 'g') if free else monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    orch = lo.LLMOrchestrator({'primary_llm_provider': 'gemini'})
    orch.claude = ClaudeProvider('key', AiBudget(tmp_path / 'spend.json', cap), {}, client=client)
    orch.enabled = True
    orch._call_gemini = lambda s, u, fb, model_override=None: {
        'action': 'hold', 'confidence': 0.2, 'position_size': 0.0, 'reasoning': 'free model',
        'strategy': 'llm_orchestrated'}
    return orch


SIGNAL = {'action': 'buy', 'confidence': 0.8, 'position_size': 0.04}


def test_claude_reviews_first_while_it_has_budget(monkeypatch, tmp_path):
    orch = _orchestrator(monkeypatch, tmp_path, _Client())
    assert orch._provider_order() == ['anthropic', 'gemini']
    verdict = orch.validate_trade('AAPL', SIGNAL, {'close': 100.0})
    assert verdict['reasoning'] == 'ok' and verdict['strategy'] == 'llm_orchestrated'


def test_the_free_model_takes_over_when_claude_fails_or_the_cap_is_reached(monkeypatch, tmp_path):
    broken = _orchestrator(monkeypatch, tmp_path / 'a', _Client(stop_reason='refusal'))
    assert broken.validate_trade('AAPL', SIGNAL, {'close': 100.0})['reasoning'] == 'free model'
    spent = _orchestrator(monkeypatch, tmp_path / 'b', _Client(), cap=0.0)
    assert spent._provider_order() == ['gemini']
    assert spent.validate_trade('MSFT', SIGNAL, {'close': 100.0})['reasoning'] == 'free model'


def test_without_a_key_claude_is_off_and_the_dashboard_says_so(monkeypatch):
    from src.agent import llm_orchestrator as lo
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.setenv('GEMINI_API_KEY', 'g')
    orch = lo.LLMOrchestrator({})
    assert orch.claude is None and 'anthropic' not in orch._provider_order()
    assert orch.ai_budget()['paid_tier'] is False
