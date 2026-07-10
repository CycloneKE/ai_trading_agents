"""
Unit tests for sector_specialist.py.
"""
import pytest
import os
import tempfile
from unittest.mock import MagicMock
from src.agent.sector_specialist import SectorSpecialistManager


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp()
    yield path
    os.close(fd)
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def mock_llm_orchestrator():
    llm = MagicMock()
    llm.enabled = True
    llm.config = {
        "swarm": {
            "agents": {
                "sector_specialist": "anthropic/claude-3.5-sonnet"
            }
        }
    }
    return llm


def test_sector_mapping(mock_llm_orchestrator, temp_db):
    """Verify sector lookup mappings work correctly."""
    manager = SectorSpecialistManager(mock_llm_orchestrator, temp_db)
    assert manager.get_sector_for_symbol("AAPL") == "technology"
    assert manager.get_sector_for_symbol("SCOM") == "telecommunications"
    assert manager.get_sector_for_symbol("XLE") == "energy"
    assert manager.get_sector_for_symbol("UNKNOWN_TICKER") == "general"
    manager.close()


def test_profile_load_save(mock_llm_orchestrator, temp_db):
    """Verify sector profile loads, saves, and updates SQLite correctly."""
    manager = SectorSpecialistManager(mock_llm_orchestrator, temp_db)
    
    # Verify default profile is returned for absent sector
    empty_profile = manager.load_sector_profile("absent_sector")
    assert empty_profile["outlook_score"] == 0.0
    assert "Initial profiling" in empty_profile["updated_profile_text"]
    
    # Save a profile
    custom_profile = {
        "outlook_score": 0.75,
        "updated_profile_text": "Tech sector is highly bullish.",
        "risk_factors": ["semiconductor shortage"],
        "catalysts": ["earnings reports"]
    }
    manager.save_sector_profile("healthcare", custom_profile)
    
    # Load and check
    loaded = manager.load_sector_profile("healthcare")
    assert loaded["outlook_score"] == 0.75
    assert loaded["updated_profile_text"] == "Tech sector is highly bullish."
    assert loaded["risk_factors"] == ["semiconductor shortage"]
    assert loaded["catalysts"] == ["earnings reports"]
    
    manager.close()


def test_run_sector_analysis(mock_llm_orchestrator, temp_db):
    """Verify specialist requests LLM proposal and updates SQLite state."""
    manager = SectorSpecialistManager(mock_llm_orchestrator, temp_db)
    
    mock_llm_orchestrator.propose_json.return_value = {
        "outlook_score": 0.45,
        "updated_profile_text": "Fitted telecom analysis.",
        "risk_factors": ["regulatory oversight"],
        "catalysts": ["5G expansion"]
    }
    
    res = manager.run_sector_analysis("telecommunications", [{"title": "Safaricom expands 5G"}])
    
    assert res["outlook_score"] == 0.45
    assert res["updated_profile_text"] == "Fitted telecom analysis."
    
    # Verify saved to database automatically
    db_loaded = manager.load_sector_profile("telecommunications")
    assert db_loaded["outlook_score"] == 0.45
    
    manager.close()
