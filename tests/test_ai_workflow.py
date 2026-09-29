"""The AI workflow's economy: it is asked only when there is something new
to ask, in few tokens, and what it answers is kept while it is valid. The
free plans limit tokens a day (Groq: about 200,000), not just requests."""
import json
from types import SimpleNamespace

import pytest

from src.agent import llm_orchestrator as lo
from src.agent.bias_detector import BiasDetector
from src.agent.main import TradingAgent
from src.agent.sector_specialist import SectorSpecialistManager
from src.utils.config_validator import load_config


class _LLM:
    enabled = True

    def __init__(self):
        self.config, self.calls = {'swarm': {'agents': {}}}, 0

    def propose_json(self, system, user, model_override=None):
        self.calls += 1
        self.last_user = user
        return {'outlook_score': 0.4, 'updated_profile_text': 'firm', 'risk_factors': [], 'catalysts': []}


@pytest.fixture
def manager(tmp_path):
    llm = _LLM()
    m = SectorSpecialistManager(llm, db_path=str(tmp_path / 'esc.db'))
    yield m, llm
    m.close()


NEWS = [{'title': 'Apple beats estimates', 'sentiment': 0.6}, {'title': 'Chip supply eases', 'sentiment': 0.3}]


def test_no_headlines_means_no_ai_call(manager):
    m, llm = manager
    m.run_sector_analysis('technology', [])
    m.run_sector_analysis('technology', [{'title': ''}, {'no': 'title'}])
    assert llm.calls == 0


def test_the_same_headlines_are_not_analysed_twice(manager):
    m, llm = manager
    first = m.run_sector_analysis('technology', NEWS)
    assert llm.calls == 1 and first['outlook_score'] == 0.4
    m.run_sector_analysis('technology', list(reversed(NEWS)))              # same headlines, other order
    assert llm.calls == 1
    m.run_sector_analysis('technology', NEWS + [{'title': 'New export rules'}])
    assert llm.calls == 2
    assert '_ai' not in llm.last_user and '_news_digest' not in llm.last_user


def test_a_redeploy_does_not_ask_again_within_the_interval(manager):
    m, llm = manager
    m.run_sector_analysis('technology', NEWS)
    m.run_sector_analysis('technology', NEWS + [{'title': 'Fresh story'}], min_interval_seconds=6 * 3600)
    assert llm.calls == 1                                                  # answered a moment ago
    m.run_sector_analysis('technology', NEWS + [{'title': 'Fresh story'}], min_interval_seconds=0)
    assert llm.calls == 2


def test_reviews_get_the_saved_outlook_after_a_restart(manager):
    m, llm = manager
    assert m.load_recent_profile('technology', 3600) is None               # only a seeded starting profile
    m.run_sector_analysis('technology', NEWS)
    got = m.load_recent_profile('technology', 3600)
    assert got['outlook_score'] == 0.4 and not any(k.startswith('_') for k in got)
    assert m.load_recent_profile('technology', -1) is None                 # too old to use


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'k')
    for key in ('ANTHROPIC_API_KEY', 'OPENROUTER_API_KEY', 'GEMINI_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    return lo.LLMOrchestrator({'primary_llm_provider': 'groq'})


def test_a_review_is_short_capped_and_counted(groq, monkeypatch):
    sent = []

    def post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        body = {'choices': [{'message': {'content': '{"action": "buy", "confidence": 0.5, "reasoning": "ok"}'}}],
                'usage': {'total_tokens': 640}}
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: body)
    monkeypatch.setattr(lo.requests, 'post', post)
    signal = {'action': 'buy', 'confidence': 0.7, 'symbol': 'AAPL', 'position_size': 0.05,
              'per_strategy': {'momentum': {'action': 'buy', 'confidence': 0.7, 'metadata': {'x': [0] * 200}},
                               'rsi': {'action': 'hold', 'confidence': 0.1}}}
    news = [{'title': 'Apple beats estimates', 'sentiment': 0.6, 'body': 'x' * 5000, 'url': 'https://x'}]
    verdict = groq.validate_trade('AAPL', signal, {'close': 227.5}, news)
    assert verdict['action'] == 'buy'
    [body] = sent
    assert body['max_completion_tokens'] == lo.GROQ_TEXT_MAX_TOKENS
    user = body['messages'][1]['content']
    assert 'momentum buy 0.70, rsi hold 0.10' in user and 'Apple beats estimates' in user
    assert 'metadata' not in user and 'xxxx' not in user and len(user) < 500
    row = next(p for p in groq.provider_health() if p['provider'] == 'groq')
    assert row['calls_today'] == 1 and row['tokens_today'] == 640
    groq.validate_trade('AAPL', signal, {'close': 227.6}, news)            # the same question: cached
    assert len(sent) == 1


def test_a_verdict_is_reused_for_an_hour(groq):
    assert groq.cache_ttl == 3600


def test_a_buy_the_pace_limit_would_hold_is_not_sent_to_the_ai(tmp_path):
    cfg = load_config('config/config.json')
    bias = BiasDetector(cfg)
    from datetime import datetime
    now = datetime.utcnow()
    for sym in ('NVDA', 'TSLA', 'XLV'):
        bias._approved_buys.setdefault('us_equity', {})[sym] = now
    broker = SimpleNamespace(is_connected=True, fetch_positions=lambda: [], fetch_open_orders=lambda s=None: [])
    agent = SimpleNamespace(
        config=cfg, order_journal=None,
        components={'broker_manager': SimpleNamespace(get_broker=lambda *a: broker), 'bias_detector': bias})
    import types
    agent._position_gate = types.MethodType(TradingAgent._position_gate, agent)
    open_session = datetime(2026, 9, 30, 15, 0, tzinfo=__import__('datetime').timezone.utc)
    assert TradingAgent._held_back_before_review(agent, 'QQQ', 'buy', 500.0, now=open_session) == 'bias_downgrade'
    assert 'QQQ' not in bias._approved_buys['us_equity']                   # asking used no slot
    assert TradingAgent._held_back_before_review(agent, 'NVDA', 'buy', 500.0, now=open_session) is None
