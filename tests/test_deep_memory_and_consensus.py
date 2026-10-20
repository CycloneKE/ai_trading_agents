"""Unit tests for Phase 2: Multi-Model LLM Consensus and Episodic Memory Ingestion."""

import pytest
from unittest.mock import MagicMock

from src.agent.llm_orchestrator import LLMOrchestrator
from src.agent.decision_journal import SKIP_REASONS


def make_dual_orch(primary_response, secondary_response, **cfg_overrides):
    """Create an orchestrator with dual providers configured and mocked responses."""
    config = {
        'llm_enabled': True,
        'primary_llm_provider': 'openrouter',
        'llm_consensus_enabled': True,
        'llm_consensus_threshold': 0.65,
        'llm_consensus_strict': True,
        'llm_cache_ttl': 0,  # disable caching for deterministic testing
    }
    config.update(cfg_overrides)
    orch = LLMOrchestrator(config)
    orch.enabled = True
    orch.openrouter_api_key = 'fake-openrouter-key'
    orch.gemini_api_key = 'fake-gemini-key'

    orch.primary_calls = []
    orch.secondary_calls = []

    def fake_primary(system_prompt, user_prompt, fallback, model_override=None, **kwargs):
        orch.primary_calls.append((system_prompt, user_prompt))
        return primary_response if primary_response is not None else fallback

    def fake_secondary(system_prompt, user_prompt):
        orch.secondary_calls.append((system_prompt, user_prompt))
        return secondary_response

    orch._complete = fake_primary
    orch.propose_json_secondary = fake_secondary
    return orch


BUY_SIGNAL = {'action': 'buy', 'confidence': 0.75, 'position_size': 0.05, 'strategy': 'momentum'}
SELL_SIGNAL = {'action': 'sell', 'confidence': 0.80, 'position_size': 0.04, 'strategy': 'mean_reversion'}
HOLD_SIGNAL = {'action': 'hold', 'confidence': 0.40, 'position_size': 0.0, 'strategy': 'momentum'}
MARKET_DATA = {'close': 150.0, 'open': 148.0, 'price': 150.0}


def test_dual_model_consensus_agreement_boosts_confidence():
    """When primary and secondary models both approve BUY, consensus is verified and confidence boosted."""
    primary = {'action': 'buy', 'confidence': 0.80, 'position_size': 0.05, 'reasoning': 'Clear breakout'}
    secondary = {'agree': True, 'action': 'buy', 'confidence': 0.85, 'reasoning': 'Volume profile confirms'}

    orch = make_dual_orch(primary, secondary)
    out = orch.validate_trade('AAPL', dict(BUY_SIGNAL), MARKET_DATA)

    assert out['action'] == 'buy'
    assert out['consensus_verified'] is True
    assert out['confidence'] >= 0.80
    assert 'secondary_verdict' in out
    assert out['secondary_verdict']['action'] == 'buy'
    assert len(orch.secondary_calls) == 1


def test_dual_model_consensus_disagreement_vetoes_to_hold():
    """When secondary model disagrees with primary BUY, trade is vetoed down to hold."""
    primary = {'action': 'buy', 'confidence': 0.80, 'position_size': 0.05, 'reasoning': 'Breakout'}
    secondary = {'agree': False, 'action': 'hold', 'reasoning': 'Overbought divergence'}

    orch = make_dual_orch(primary, secondary)
    out = orch.validate_trade('AAPL', dict(BUY_SIGNAL), MARKET_DATA)

    assert out['action'] == 'hold'
    assert out['consensus_verified'] is False
    assert 'secondary_verdict' in out
    assert 'Consensus Veto' in out.get('reasoning', '')


def test_dual_model_consensus_opposite_direction_vetoes_to_hold():
    """When secondary model votes SELL against primary BUY, trade is vetoed to hold (never flipped to sell)."""
    primary = {'action': 'buy', 'confidence': 0.80, 'position_size': 0.05, 'reasoning': 'Breakout'}
    secondary = {'agree': False, 'action': 'sell', 'reasoning': 'Reversal imminent'}

    orch = make_dual_orch(primary, secondary)
    out = orch.validate_trade('AAPL', dict(BUY_SIGNAL), MARKET_DATA)

    assert out['action'] == 'hold'
    assert out['consensus_verified'] is False


def test_consensus_skipped_when_secondary_unavailable():
    """When secondary provider is unavailable (returns None), single-model verdict stands safely."""
    primary = {'action': 'buy', 'confidence': 0.80, 'position_size': 0.05, 'reasoning': 'Breakout'}
    secondary = None

    orch = make_dual_orch(primary, secondary)
    out = orch.validate_trade('AAPL', dict(BUY_SIGNAL), MARKET_DATA)

    assert out['action'] == 'buy'
    assert out['consensus_verified'] is None
    assert out['confidence'] == 0.75  # bounded to strategy signal confidence by guardrails


def test_hold_signals_skip_consensus_voting():
    """When base signal or primary model outputs HOLD, secondary consensus is not queried."""
    primary = {'action': 'hold', 'confidence': 0.30, 'reasoning': 'No edge'}
    secondary = {'agree': True, 'action': 'hold'}

    orch = make_dual_orch(primary, secondary)
    out = orch.validate_trade('AAPL', dict(HOLD_SIGNAL), MARKET_DATA)

    assert out['action'] == 'hold'
    assert len(orch.secondary_calls) == 0


def test_low_confidence_signals_skip_consensus():
    """Signals below consensus threshold do not trigger secondary consensus queries."""
    primary = {'action': 'buy', 'confidence': 0.55, 'position_size': 0.02, 'reasoning': 'Weak signal'}
    secondary = {'agree': True, 'action': 'buy'}

    orch = make_dual_orch(primary, secondary, llm_consensus_threshold=0.70)
    out = orch.validate_trade('AAPL', dict(BUY_SIGNAL), MARKET_DATA)

    assert out['action'] == 'buy'
    assert len(orch.secondary_calls) == 0
    assert out.get('consensus_verified') is None


def test_episodic_memory_ingestion_in_prompt():
    """Episodic memory dict is formatted and injected into the user prompt."""
    primary = {'action': 'buy', 'confidence': 0.80, 'position_size': 0.05, 'reasoning': 'Breakout'}
    secondary = {'agree': True, 'action': 'buy'}

    orch = make_dual_orch(primary, secondary)
    memory = {
        'recent_decisions': ['BUY @ 148.0 (executed=1)', 'HOLD (skip_reason=below_confidence)'],
        'recent_fills': ['BUY 10 @ 148.50 (filled)', 'SELL 10 @ 155.00 (filled)'],
        'veto_accuracy': '75.0%',
        'summary': 'Profitable recent momentum trades.'
    }

    orch.validate_trade('AAPL', dict(BUY_SIGNAL), MARKET_DATA, episodic_memory=memory)

    assert len(orch.primary_calls) == 1
    sys_prompt, user_prompt = orch.primary_calls[0]
    assert 'Episodic Memory' in user_prompt
    assert 'Recent Decisions' in user_prompt
    assert 'BUY @ 148.0' in user_prompt
    assert 'Recent Fills' in user_prompt
    assert '75.0%' in user_prompt


def test_consensus_disagreement_is_valid_skip_reason():
    """consensus_disagreement should be present in canonical SKIP_REASONS."""
    assert 'consensus_disagreement' in SKIP_REASONS


def test_build_symbol_episodic_memory_in_trading_agent(tmp_path):
    """TradingAgent should construct episodic memory from decision and order journals."""
    from src.agent.main import TradingAgent
    from src.agent.decision_journal import DecisionJournal
    from src.agent.order_journal import OrderJournal

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    dj = DecisionJournal(db_path=str(data_dir / "decision_journal.db"))
    oj = OrderJournal(db_path=str(data_dir / "order_journal.db"))

    # Seed decisions
    dj.record({'symbol': 'AAPL', 'cycle': 10, 'action': 'buy', 'price': 150.0, 'executed': True, 'skip_reason': None})
    dj.record({'symbol': 'AAPL', 'cycle': 11, 'action': 'hold', 'price': 152.0, 'executed': False, 'skip_reason': 'llm_veto'})

    # Seed orders
    oj.record_intent("ord-1", "AAPL", "buy", 10, 150.0, strategy="momentum")
    oj.mark_final("ord-1", "filled", 10, 150.0)

    agent = TradingAgent.__new__(TradingAgent)
    agent.decision_journal = dj
    agent.order_journal = oj
    agent.components = {'order_journal': oj}
    agent._last_veto_accuracy = 80.0

    memory = agent._build_symbol_episodic_memory('AAPL')

    assert 'recent_decisions' in memory
    assert len(memory['recent_decisions']) >= 2
    assert 'Cycle 10' in memory['recent_decisions'][0] or 'Cycle 11' in memory['recent_decisions'][0]
    assert 'recent_fills' in memory
    assert len(memory['recent_fills']) == 1
    assert 'BUY 10' in memory['recent_fills'][0]
    assert memory.get('veto_accuracy') == '80.0%'

