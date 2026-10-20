"""Tests for Multi-Model LLM Consensus Voting in broker research signal evaluation."""
import pytest
from unittest.mock import MagicMock, patch
from src.agent.broker_research_ingest import BrokerResearchIngest


@pytest.fixture
def sample_config():
    return {
        'research_ingest': {
            'enabled': True,
            'upload_dir': 'data/test_uploads',
            'auto_follow_rules': {
                'allowed_recommendations': ['BUY', 'ACCUMULATE', 'HOLD'],
                'min_upside_pct': 5.0,
                'require_existing_symbol': True,
            }
        },
        'data_manager': {
            'symbols': ['AAPL', 'MSFT'],
            'nse_symbols': ['SCOM', 'KCB'],
        }
    }


def test_consensus_agreement_boosts_confidence(sample_config):
    """When both LLMs agree on BUY, confidence is boosted and auto_follow returned."""
    llm = MagicMock()
    llm.propose_json_secondary.return_value = {
        'agree': True,
        'recommendation': 'BUY',
        'reasoning': 'Strong fundamentals support the buy recommendation.'
    }
    
    ingest = BrokerResearchIngest(llm, MagicMock(), sample_config)
    signal = {
        'symbol': 'AAPL',
        'recommendation': 'BUY',
        'current_price': 150.0,
        'target_price': 180.0,
        'confidence': 0.75,
        'reason': 'Strong growth outlook.'
    }
    
    action, reason, risk = ingest._evaluate_signal(signal)
    assert action == 'auto_follow'
    assert 'consensus' in reason.lower() or 'dual' in reason.lower()
    assert signal.get('consensus_verified') is True
    assert signal.get('confidence', 0) >= 0.90


def test_consensus_disagreement_escalates(sample_config):
    """When secondary LLM disagrees, signal is escalated for operator review."""
    llm = MagicMock()
    llm.propose_json_secondary.return_value = {
        'agree': False,
        'recommendation': 'HOLD',
        'reasoning': 'Valuation is stretched, recommend waiting.'
    }
    
    ingest = BrokerResearchIngest(llm, MagicMock(), sample_config)
    signal = {
        'symbol': 'MSFT',
        'recommendation': 'BUY',
        'current_price': 400.0,
        'target_price': 450.0,
        'confidence': 0.80,
        'reason': 'Cloud growth acceleration.'
    }
    
    action, reason, risk = ingest._evaluate_signal(signal)
    assert action == 'escalate'
    assert 'disagreement' in reason.lower()
    assert signal.get('consensus_verified') is False


def test_consensus_unavailable_falls_through(sample_config):
    """When secondary LLM is unavailable, signal proceeds with single-model auto_follow."""
    llm = MagicMock()
    llm.propose_json_secondary.return_value = None
    
    ingest = BrokerResearchIngest(llm, MagicMock(), sample_config)
    signal = {
        'symbol': 'SCOM',
        'recommendation': 'BUY',
        'current_price': 15.0,
        'target_price': 20.0,
        'confidence': 0.80,
        'reason': 'Dividend growth play.',
        'market': 'kenyan'
    }
    
    action, reason, risk = ingest._evaluate_signal(signal)
    assert action == 'auto_follow'
    assert signal.get('consensus_verified') is None


def test_hold_signals_skip_consensus(sample_config):
    """HOLD signals should not trigger consensus voting (only BUY/ACCUMULATE do)."""
    llm = MagicMock()
    
    ingest = BrokerResearchIngest(llm, MagicMock(), sample_config)
    signal = {
        'symbol': 'AAPL',
        'recommendation': 'HOLD',
        'current_price': 150.0,
        'target_price': 160.0,
        'confidence': 0.80,
    }
    
    action, reason, risk = ingest._evaluate_signal(signal)
    assert action == 'auto_follow'
    # propose_json_secondary should NOT have been called for HOLD signals
    llm.propose_json_secondary.assert_not_called()


def test_consensus_exception_is_safe(sample_config):
    """If consensus voting throws an exception, it should not crash the evaluation."""
    llm = MagicMock()
    llm.propose_json_secondary.side_effect = Exception("Network timeout")
    
    ingest = BrokerResearchIngest(llm, MagicMock(), sample_config)
    signal = {
        'symbol': 'AAPL',
        'recommendation': 'BUY',
        'current_price': 150.0,
        'target_price': 180.0,
        'confidence': 0.75,
        'reason': 'Strong outlook.'
    }
    
    action, reason, risk = ingest._evaluate_signal(signal)
    assert action == 'auto_follow'
    assert signal.get('consensus_verified') is None
