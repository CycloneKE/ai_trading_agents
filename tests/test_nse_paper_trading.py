"""
Test suite for African & Kenyan NSE Equities Automated Paper Trading and Dividend Sleeve.
"""
import pytest
from src.agent.nse_order_queue import NseOrderQueue
from src.agent.sleeve.sleeve_manager import SleeveManager

def test_nse_ticket_creation_and_auto_fill(tmp_path):
    """Test ticket creation and automated paper fill simulation in NseOrderQueue."""
    db_file = str(tmp_path / "test_escalations.db")
    queue = NseOrderQueue(db_path=db_file)
    
    # Create NSE Ticket
    ticket_id = queue.create_ticket(
        symbol="SCOM",
        side="buy",
        quantity=500,
        suggested_limit_price=15.50,
        rationale="Strong M-Pesa growth and technical BUY signal",
        ensemble_confidence=0.85,
        llm_reasoning="LLM validated trade"
    )
    assert ticket_id is not None
    assert ticket_id > 0
    
    # Verify ticket enters pending status
    pending = queue.get_pending()
    assert len(pending) == 1
    assert pending[0]["symbol"] == "SCOM"
    assert pending[0]["status"] == "pending"
    
    # Mark filled via Auto Paper Trader
    ok_fill, fill_data = queue.mark_filled(
        ticket_id,
        fill_price=15.50,
        fill_quantity=500,
        resolved_by="auto_paper_trader",
        notes="Automated Paper Trade Simulation Fill"
    )
    assert ok_fill is True
    assert fill_data["status"] == "filled"
    assert fill_data["fill_price"] == 15.50
    assert fill_data["fill_quantity"] == 500
    
    # Verify recent fills list
    fills = queue.recent_fills()
    assert len(fills) == 1
    assert fills[0]["symbol"] == "SCOM"
    assert fills[0]["resolved_by"] == "auto_paper_trader"

def test_dividend_sleeve_universe_and_monthly_pass(tmp_path):
    """Test dividend accumulation sleeve universe and monthly allocation pass."""
    db_file = str(tmp_path / "test_sleeve.db")
    sleeve = SleeveManager({}, None, None, None, db_path=db_file)
    
    assert sleeve._db_path == db_file
    assert sleeve.enabled is False
    
    # Run monthly cycle pass with mock quotes
    mock_quotes = {
        "BAT": 520.0,
        "EABL": 240.0,
        "SCBK": 330.0,
        "COOP": 32.0,
        "NCBA": 88.0,
        "SCOM": 31.0
    }
    
    results = sleeve.run_monthly_cycle(mock_quotes)
    # Sleeve manager returns accumulation targets
    assert results is not None
    assert isinstance(results, list)
