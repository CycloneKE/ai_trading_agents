import os
import pytest
from unittest.mock import MagicMock, patch

from src.agent.broker_research_ingest import detect_research_source, BrokerResearchIngest
from src.agent.escalation_manager import EscalationManager
from src.agent.llm_orchestrator import LLMOrchestrator


def test_detect_research_source_kenyan_publishers():
    assert detect_research_source("Daily report from AIB-AXYS Africa Research Desk", "report.pdf") == "AIB AXYS Africa"
    assert detect_research_source("Genghis Capital weekly equity strategy note", "report.pdf") == "Genghis Capital"
    assert detect_research_source("Faida Investment Bank valuation update", "report.pdf") == "Faida Investment Bank"
    assert detect_research_source("Standard Investment Bank SIB monthly outlook", "report.pdf") == "Standard Investment Bank"
    assert detect_research_source("Dyer and Blair daily market report", "report.pdf") == "Dyer & Blair"
    assert detect_research_source("KCB Capital equities update", "report.pdf") == "KCB Capital"
    assert detect_research_source("NCBA Investment Bank research briefing", "report.pdf") == "NCBA Investment Bank"


def test_detect_research_source_international_publishers():
    assert detect_research_source("Goldman Sachs Equity Research: Semiconductor Sector", "gs_tech.pdf") == "Goldman Sachs"
    assert detect_research_source("Morgan Stanley Research: Megacap tech update", "ms_report.pdf") == "Morgan Stanley"
    assert detect_research_source("JPMorgan Global Equities Research", "jpm_macro.pdf") == "J.P. Morgan"
    assert detect_research_source("Bank of America BofA Global Research", "bofa.pdf") == "Bank of America"
    assert detect_research_source("Citigroup Citi Research: Cloud infrastructure", "citi.pdf") == "Citigroup"
    assert detect_research_source("Morningstar quantitative equity assessment", "morningstar.pdf") == "Morningstar"


def test_detect_research_source_fallback_and_overrides():
    # Unknown header falls back to Institutional Research
    assert detect_research_source("Generic commentary on equities without publisher tag", "memo.pdf") == "Institutional Research"
    
    # Explicit alias override
    assert detect_research_source("Any text", "report.pdf", preferred_source="goldman") == "Goldman Sachs"
    assert detect_research_source("Any text", "report.pdf", preferred_source="genghis") == "Genghis Capital"
    assert detect_research_source("Any text", "report.pdf", preferred_source="Custom Broker House") == "Custom Broker House"


def test_fallback_extractor_supports_both_nse_and_us(tmp_path):
    em_mock = MagicMock(spec=EscalationManager)
    cfg = {
        "research_ingest": {"auto_follow_rules": {"min_upside_pct": 5.0}},
        "data_manager": {
            "symbols": ["NVDA", "AAPL", "MSFT"],
            "nse_symbols": ["SCOM", "EQTY", "KCB"]
        }
    }
    ingest = BrokerResearchIngest(llm_orchestrator=None, escalation_manager=em_mock, config=cfg)
    
    # Text with both a US company (NVIDIA) and a Kenyan company (Safaricom)
    sample_text = """
    Market Analysis:
    1. Safaricom (SCOM): Recommendation is BUY with current price at KES 18.0 and target price at KES 25.0. Strong M-Pesa growth.
    2. Nvidia (NVDA): Recommending BUY at current price $120.0 with target price $160.0 driven by Blackwell data center demand.
    """
    signals = ingest._extract_signals_fallback(sample_text)
    
    symbols_found = {s["symbol"]: s for s in signals}
    assert "SCOM" in symbols_found
    assert symbols_found["SCOM"]["market"] == "kenyan"
    assert symbols_found["SCOM"]["recommendation"] == "BUY"
    assert symbols_found["SCOM"]["target_price"] == 25.0
    
    assert "NVDA" in symbols_found
    assert symbols_found["NVDA"]["market"] == "international"
    assert symbols_found["NVDA"]["recommendation"] == "BUY"
    assert symbols_found["NVDA"]["target_price"] == 160.0


def test_llm_orchestrator_prompt_reflects_dynamic_source():
    cfg = {"swarm": {"agents": {"supervisor": "openai/gpt-4o"}}}
    orch = LLMOrchestrator(config=cfg)
    
    captured_prompts = []
    def fake_complete(sys_prompt, user_prompt, fallback, **kwargs):
        captured_prompts.append(user_prompt)
        return {"action": "buy", "confidence": 0.8, "reasoning": "Strong report"}
    
    orch._complete = fake_complete
    orch.enabled = True
    
    research_ctx = {
        "source": "Goldman Sachs",
        "recommendation": "BUY",
        "target_price": 175.0,
        "rationale": "Massive generative AI pipeline"
    }
    
    orch.validate_trade(
        symbol="NVDA",
        strategy_signal={"action": "buy", "confidence": 0.8},
        market_data={"price": 130.0},
        research_context=research_ctx
    )
    
    assert len(captured_prompts) == 1
    assert "Broker Research Context (Goldman Sachs):" in captured_prompts[0]
    assert "Target Price=175.0" in captured_prompts[0]


def test_process_pdf_end_to_end_with_detected_source(tmp_path):
    em_mock = MagicMock(spec=EscalationManager)
    em_mock.record_upload.return_value = 42
    em_mock.record_signal.return_value = 101
    
    cfg = {
        "research_ingest": {"auto_follow_rules": {"min_upside_pct": 5.0, "require_existing_symbol": False}},
        "data_manager": {
            "symbols": ["NVDA"],
            "nse_symbols": []
        }
    }
    
    fake_llm = MagicMock()
    fake_llm.enabled = True
    fake_llm.propose_json.return_value = [{
        "symbol": "NVDA",
        "market": "international",
        "current_price": 120.0,
        "target_price": 150.0,
        "upside_pct": 25.0,
        "recommendation": "BUY",
        "rationale": "High data center demand",
        "confidence": 0.9
    }]
    fake_llm.propose_json_secondary.return_value = {
        "agree": True,
        "recommendation": "BUY",
        "reasoning": "Strong fundamentals"
    }
    
    ingest = BrokerResearchIngest(llm_orchestrator=fake_llm, escalation_manager=em_mock, config=cfg)
    
    pdf_file = tmp_path / "morgan_stanley_ai.pdf"
    pdf_file.write_text("dummy")
    
    with patch("src.utils.pdf_parser.extract_all") as mock_extract:
        mock_extract.return_value = {
            "text": "Morgan Stanley Research: Semiconductor Outlook\nNvidia NVDA Buy target $150",
            "pages": [{"page_number": 1, "text": "Morgan Stanley Research", "tables": [], "combined": "Morgan Stanley Research: NVDA Buy target $150"}]
        }
        
        result = ingest.process_pdf(str(pdf_file), source="auto")
        
        assert result["status"] == "completed"
        assert result["source"] == "Morgan Stanley"
        assert result["signals_processed"] == 1
        assert "NVDA" in result["auto_followed"]
        
        # Verify upload recorded with detected source
        em_mock.record_upload.assert_called_once_with("morgan_stanley_ai.pdf", "Morgan Stanley")
        
        # Verify added to watchlist with detected source
        em_mock.add_to_watchlist.assert_called_once_with(
            symbol="NVDA",
            market="international",
            source="Morgan Stanley",
            recommendation="BUY",
            target_price=150.0,
            rationale="High data center demand"
        )
