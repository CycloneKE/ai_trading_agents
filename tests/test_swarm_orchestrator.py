"""
Unit tests for swarm validation in llm_orchestrator.py.
"""
import pytest
import json
from unittest.mock import MagicMock
from src.agent.llm_orchestrator import LLMOrchestrator


@pytest.fixture
def mock_config():
    return {
        "primary_llm_provider": "openrouter",
        "swarm": {
            "agents": {
                "synthesizer": "deepseek/deepseek-r1"
            }
        }
    }


def test_swarm_orchestration_passes_sector_outlook(mock_config):
    """Verify validate_trade fetches synthesizer model and injects sector context."""
    orchestrator = LLMOrchestrator(mock_config)
    orchestrator.enabled = True
    orchestrator.openrouter_api_key = "test_key"
    
    # Mock OpenRouter API call method
    orchestrator._call_openrouter = MagicMock()
    orchestrator._call_openrouter.return_value = {
        "action": "buy",
        "confidence": 0.9,
        "position_size": 0.1,
        "reasoning": "Strong sector support"
    }
    
    strategy_signal = {"action": "buy", "confidence": 0.8}
    market_data = {"close": 150.0}
    sector_outlook = {
        "outlook_score": 0.6,
        "updated_profile_text": "Tech sector demand is high",
        "risk_factors": ["supply constraints"]
    }
    
    res = orchestrator.validate_trade(
        symbol="AAPL",
        strategy_signal=strategy_signal,
        market_data=market_data,
        news_data=[],
        research_context=None,
        sector_outlook=sector_outlook
    )
    
    # Verify return value
    assert res["action"] == "buy"
    assert res["confidence"] == 0.9
    
    # Verify OpenRouter was called with the correct model and prompt content
    assert orchestrator._call_openrouter.call_count == 1
    args, kwargs = orchestrator._call_openrouter.call_args
    
    system_prompt, user_prompt, fallback_signal = args
    model_override = kwargs.get("model_override")
    
    assert model_override == "deepseek/deepseek-r1"
    assert "Sector Outlook Context" in user_prompt
    assert "Outlook Score=0.6" in user_prompt
    assert "Tech sector demand is high" in user_prompt
