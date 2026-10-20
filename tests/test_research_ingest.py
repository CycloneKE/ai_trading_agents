"""
Test suite for Broker Research PDF Ingestion, Multi-Page Extraction, and Embedded Link Capture.
"""
import pytest
from src.agent.broker_research_ingest import BrokerResearchIngest

def test_broker_research_fallback_extraction_all_stocks():
    """Test heuristic signal extraction across all 20+ covered Kenyan & African equities."""
    ingest = BrokerResearchIngest(None, None, {})
    
    sample_report_text = """
    AIB-AXYS June 2026 Stock Recommendations & Market Pulse.
    Top Traded: Equity Group 79.75, Co-operative Bank 34.50, KenGen 9.90, Diamond Trust Bank 141.00, KCB Group 76.00.
    Top Gainers: Williamson Tea Kenya 150.25 (up 13.2%), Family Bank 24.50 (up 9.4%), Limuru Tea 538.00 (up 9.0%), I&M Group 65.75 (up 8.7%), Kapchorua Tea 321.25 (up 7.5%).
    Top Losers: Longhorn 2.72, WPP Scangroup 2.06, Eveready 1.04, Eaagads 29.20, Uchumi 1.70.
    Equities Highlights: NASI advanced 1.5% to 222.42 supported by Safaricom (up 2.0%), KCB Group (up 2.0%), East African Breweries (up 1.3%), Equity Group (up 0.9%).
    Weighed down by TotalEnergies Marketing Kenya (down 2.3%), Co-operative Bank (down 0.1%), BK Group (down 0.9%), Jubilee Holdings (down 0.5%).
    
    Capital News Update & Corporate Filings:
    Kapchorua Tea Audited Results (here: https://aib-axysafrica.com/reports/kapchorua-2026.pdf)
    Williamson Tea Audited Results (here: https://aib-axysafrica.com/reports/williamson-2026.pdf)
    Diamond Trust Bank AGM Resolutions (here: https://aib-axysafrica.com/reports/dtb-agm.pdf)
    Equity Group AGM Results (here: https://aib-axysafrica.com/reports/equity-agm.pdf)
    ABSA Bank Kenya Director Appointment (here: https://aib-axysafrica.com/reports/absa-dir.pdf)
    East African Portland Cement Audited Results (here: https://aib-axysafrica.com/reports/eapc-2025.pdf)
    """
    
    signals = ingest._extract_signals_fallback(sample_report_text)
    
    # Assert signals were extracted across all major companies
    extracted_symbols = [s["symbol"] for s in signals]
    
    assert "ABSA" in extracted_symbols
    assert "COOP" in extracted_symbols
    assert "DTB" in extracted_symbols
    assert "EQTY" in extracted_symbols
    assert "KCB" in extracted_symbols
    assert "SCOM" in extracted_symbols
    assert "WILLIAMSON" in extracted_symbols
    assert "KAPCHORUA" in extracted_symbols
    assert "PORTLAND" in extracted_symbols
    
    # Assert rationale retention and embedded link capture
    portland_signal = next(s for s in signals if s["symbol"] == "PORTLAND")
    assert portland_signal["rationale"] is not None
    assert len(portland_signal["document_links"]) > 0
    assert "https://aib-axysafrica.com/reports/eapc-2025.pdf" in portland_signal["document_links"]
