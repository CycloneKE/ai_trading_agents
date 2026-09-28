"""Every decision and order records the version of the agent that made it,
and every AI review the model that gave it, so a 90-day run that spans many
deploys and a change of AI provider can be judged version by version."""
import sqlite3

from src.agent.decision_journal import DecisionJournal
from src.agent.order_journal import OrderJournal
from src.utils.build_info import code_version


def test_decisions_and_orders_carry_the_code_version(tmp_path):
    dj = DecisionJournal(db_path=str(tmp_path / 'd.db'), heartbeat_cycles=1)
    oj = OrderJournal(db_path=str(tmp_path / 'o.db'))
    dj.record({'symbol': 'AAPL', 'cycle': 0, 'action': 'hold', 'skip_reason': 'hold'})
    oj.record_intent('x1', 'AAPL', 'buy', 1, 'market', strategy='momentum')
    version = code_version()
    assert version.startswith('src-') or len(version) == 12
    d = sqlite3.connect(str(tmp_path / 'd.db')).execute('SELECT code_version FROM decisions').fetchone()
    o = sqlite3.connect(str(tmp_path / 'o.db')).execute('SELECT code_version FROM orders').fetchone()
    assert d == (version,) and o == (version,)
    dj.close()
    oj.close()


def test_a_journal_from_before_version_tags_gains_the_column(tmp_path):
    path = tmp_path / 'old.db'
    conn = sqlite3.connect(str(path))
    conn.execute('CREATE TABLE decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, cycle INTEGER,'
                 ' symbol TEXT NOT NULL, action TEXT, executed INTEGER DEFAULT 0, skip_reason TEXT,'
                 ' ensemble_confidence REAL, per_strategy_json TEXT, llm_verdict_json TEXT, price REAL,'
                 ' target_value REAL, client_order_id TEXT)')
    conn.execute("INSERT INTO decisions (ts, symbol, action) VALUES ('2026-09-23T10:00:00', 'SPY', 'hold')")
    conn.commit()
    conn.close()
    dj = DecisionJournal(db_path=str(path), heartbeat_cycles=1)
    dj.record({'symbol': 'SPY', 'cycle': 5, 'action': 'buy', 'skip_reason': 'market_closed'})
    rows = sqlite3.connect(str(path)).execute('SELECT code_version FROM decisions ORDER BY id').fetchall()
    assert rows == [(None,), (code_version(),)]
    dj.close()


def test_an_ai_review_names_the_model_that_gave_it(monkeypatch):
    from src.agent import llm_orchestrator as lo
    monkeypatch.setenv('GROQ_API_KEY', 'k')
    for key in ('ANTHROPIC_API_KEY', 'OPENROUTER_API_KEY', 'GEMINI_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    llm = lo.LLMOrchestrator({'primary_llm_provider': 'groq'})
    monkeypatch.setattr(llm, '_call_groq', lambda s, u, fb, model_override=None: {
        'action': 'buy', 'confidence': 0.5, 'reasoning': 'ok'})
    verdict = llm.validate_trade('AAPL', {'action': 'buy', 'confidence': 0.6}, {'close': 1.0})
    assert verdict['_model'] == 'groq:openai/gpt-oss-120b'
