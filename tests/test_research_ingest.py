"""
Test suite for Broker Research PDF Ingestion, Multi-Page Extraction, and Embedded Link Capture.
"""
import pytest
from src.agent.broker_research_ingest import BrokerResearchIngest

def test_a_market_report_without_ratings_yields_no_recommendations():
    """A market report names companies without rating them. This used to
    return a "HOLD" for every company named, with "prices" read from the
    percentages beside the names; now a company needs a rating near its
    name. (The daily Market Pulse itself is read by market_pulse.py.)"""
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
    
    assert ingest._extract_signals_fallback(sample_report_text) == []

    # A rated mention in the same kind of text is still found, with its link.
    rated = sample_report_text + (
        "\n    East African Portland Cement: we upgrade to BUY, target 140.00 "
        "(here: https://aib-axysafrica.com/reports/eapc-2025.pdf)\n")
    (portland,) = ingest._extract_signals_fallback(rated)
    assert (portland["symbol"], portland["recommendation"]) == ("PORTLAND", "BUY")
    assert "https://aib-axysafrica.com/reports/eapc-2025.pdf" in portland["document_links"]
