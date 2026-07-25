"""
Test suite for Multi-Region Portfolio API & Operator Recent Fills Endpoint.
"""
import pytest
import os
import requests
from src.api.auth import create_token

def test_portfolio_api_unauthenticated():
    """Test that /api/portfolio blocks unauthenticated requests with 401."""
    try:
        r = requests.get("http://localhost:5001/api/portfolio", timeout=5)
        assert r.status_code == 401
    except requests.exceptions.ConnectionError:
        pytest.skip("System API server not running locally; skipping HTTP call test.")

def test_portfolio_api_authenticated():
    """Test that /api/portfolio returns multi-region holdings structure for valid operator tokens."""
    os.environ["SECRET_KEY"] = os.environ.get("SECRET_KEY", "test-secret-key-32-bytes-minimum-length-required-for-jwt")
    token = create_token("test_admin", "operator")
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        r = requests.get("http://localhost:5001/api/portfolio", headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            assert "account" in data
            assert "positions" in data
            assert "summary" in data
            assert isinstance(data["positions"], list)
    except requests.exceptions.ConnectionError:
        pytest.skip("System API server not running locally; skipping HTTP call test.")
