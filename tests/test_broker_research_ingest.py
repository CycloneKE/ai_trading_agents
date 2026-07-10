"""
Unit tests for broker_research_ingest.py.
"""
import pytest
from unittest.mock import MagicMock
from src.agent.broker_research_ingest import BrokerResearchIngest


@pytest.fixture
def mock_escalation_manager():
    em = MagicMock()
    em.record_upload.return_value = 1
    em.record_signal.return_value = 10
    return em


@pytest.fixture
def mock_llm_orchestrator():
    llm = MagicMock()
    llm.enabled = True
    return llm


@pytest.fixture
def sample_config():
    return {
        "research_ingest": {
            "enabled": True,
            "upload_dir": "data/test_uploads",
            "auto_follow_rules": {
                "require_existing_symbol": True,
                "allowed_recommendations": ["BUY", "HOLD"],
                "min_upside_pct": 5.0
            }
        },
        "data_manager": {
            "symbols": ["AAPL", "MSFT"],
            "nse_symbols": ["SCOM", "KCB"]
        }
    }


def test_evaluate_signal_auto_follow(mock_llm_orchestrator, mock_escalation_manager, sample_config):
    """Verify known symbol with BUY rating and high upside triggers auto_follow."""
    ingest = BrokerResearchIngest(mock_llm_orchestrator, mock_escalation_manager, sample_config)
    
    signal = {
        "symbol": "SCOM",
        "market": "kenyan",
        "current_price": 30.0,
        "target_price": 35.0,
        "recommendation": "BUY",
        "confidence": 0.8,
        "rationale": "Bullish structure"
    }
    
    action, reason, risk = ingest._evaluate_signal(signal)
    assert action == "auto_follow"
    assert risk == "low"


def test_evaluate_signal_escalate_unknown_symbol(mock_llm_orchestrator, mock_escalation_manager, sample_config):
    """Verify unknown symbol triggers escalation."""
    ingest = BrokerResearchIngest(mock_llm_orchestrator, mock_escalation_manager, sample_config)
    
    signal = {
        "symbol": "UNKNOWN_TICKER",
        "market": "kenyan",
        "current_price": 10.0,
        "target_price": 15.0,
        "recommendation": "BUY",
        "confidence": 0.8,
        "rationale": "High growth"
    }
    
    action, reason, risk = ingest._evaluate_signal(signal)
    assert action == "escalate"
    assert "not in active trading configuration" in reason
    assert risk == "high"


def test_evaluate_signal_escalate_low_upside(mock_llm_orchestrator, mock_escalation_manager, sample_config):
    """Verify low upside BUY triggers escalation."""
    ingest = BrokerResearchIngest(mock_llm_orchestrator, mock_escalation_manager, sample_config)
    
    signal = {
        "symbol": "SCOM",
        "market": "kenyan",
        "current_price": 30.0,
        "target_price": 30.5, # 1.6% upside
        "recommendation": "BUY",
        "confidence": 0.8,
        "rationale": "Minor recovery"
    }
    
    action, reason, risk = ingest._evaluate_signal(signal)
    assert action == "escalate"
    assert "below minimum target threshold" in reason
    assert risk == "medium"


def test_evaluate_signal_escalate_sell_rating(mock_llm_orchestrator, mock_escalation_manager, sample_config):
    """Verify SELL rating triggers escalation."""
    ingest = BrokerResearchIngest(mock_llm_orchestrator, mock_escalation_manager, sample_config)
    
    signal = {
        "symbol": "SCOM",
        "market": "kenyan",
        "current_price": 30.0,
        "target_price": 25.0,
        "recommendation": "SELL",
        "confidence": 0.8,
        "rationale": "Downside risks"
    }
    
    action, reason, risk = ingest._evaluate_signal(signal)
    assert action == "escalate"
    assert "not in pre-approved list" in reason
    assert risk == "medium"


def test_heuristic_fallback_extractor(mock_llm_orchestrator, mock_escalation_manager, sample_config):
    """Verify that regex fallback correctly parses symbols and buy/sell recommendations from raw text."""
    ingest = BrokerResearchIngest(mock_llm_orchestrator, mock_escalation_manager, sample_config)
    
    # Test text simulating a PDF report
    report_text = """
    AIB AXYS Daily Whispers.
    Safaricom (SCOM) is currently trading at 15.20. We recommend a BUY with a target price of 18.50.
    Equity Group (EQTY) is at 38.0. We maintain a HOLD recommendation.
    KCB Group (KCB) faces pressure, recommend SELL due to non-performing loans.
    """
    
    signals = ingest._extract_signals_fallback(report_text)
    
    # Check that SCOM was extracted as BUY
    scom_signal = next((s for s in signals if s["symbol"] == "SCOM"), None)
    assert scom_signal is not None
    assert scom_signal["recommendation"] == "buy"
    assert scom_signal["current_price"] == 15.20
    assert scom_signal["target_price"] == 18.50
    assert scom_signal["market"] == "kenyan"
    
    # Check that EQTY was extracted as HOLD
    eqty_signal = next((s for s in signals if s["symbol"] == "EQTY"), None)
    assert eqty_signal is not None
    assert eqty_signal["recommendation"] == "hold"
    
    # Check that KCB was extracted as SELL
    kcb_signal = next((s for s in signals if s["symbol"] == "KCB"), None)
    assert kcb_signal is not None
    assert kcb_signal["recommendation"] == "sell"


def test_llm_fails_triggering_fallback(mock_llm_orchestrator, mock_escalation_manager, sample_config):
    """Verify that if LLM fails (returns None), the extraction falls back to the heuristic extractor."""
    mock_llm_orchestrator.propose_json.return_value = None
    ingest = BrokerResearchIngest(mock_llm_orchestrator, mock_escalation_manager, sample_config)
    
    report_text = "We recommend a BUY on Safaricom (SCOM) at 15.20 with a target of 18.50."
    signals = ingest._extract_signals_via_llm(report_text)
    
    assert len(signals) > 0
    scom = next((s for s in signals if s["symbol"] == "SCOM"), None)
    assert scom is not None
    assert scom["recommendation"] == "buy"

