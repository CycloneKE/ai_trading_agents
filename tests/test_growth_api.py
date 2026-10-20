"""Unit tests for growth report and shadow strategies API endpoints."""

import json
import os

# Must set SECRET_KEY before importing auth
os.environ["SECRET_KEY"] = "test-secret-key-32-bytes-minimum-length-required-for-jwt"

from unittest.mock import MagicMock
import pytest

from src.api.api_server import TradingAPI
from src.api import auth


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    # Set up temporary users file
    users_path = str(tmp_path / "users.json")
    with open(users_path, "w", encoding="utf-8") as f:
        json.dump({"test_operator": {"role": "operator"}}, f)
    monkeypatch.setattr(auth, "USERS_FILE", users_path)

    mock_agent = MagicMock()
    mock_agent.config = {
        "ensemble_method": "adaptive_confidence",
        "strategies": {"momentum": {"weight": 1.0}, "mean_reversion": {"weight": 1.0}}
    }
    mock_agent.components = {}
    api = TradingAPI(mock_agent, {"api": {"port": 5001}})
    return api.app.test_client()


def test_growth_api_requires_auth(api_client):
    res = api_client.get("/api/growth-report")
    assert res.status_code == 401

    res = api_client.get("/api/shadow-strategies")
    assert res.status_code == 401


def test_growth_api_returns_growth_report(api_client):
    token = auth.create_token("test_operator", "operator")
    headers = {"Authorization": f"Bearer {token}"}

    res = api_client.get("/api/growth-report", headers=headers)
    assert res.status_code == 200
    data = res.get_json()
    assert "timestamp" in data
    assert "strategies" in data


def test_growth_api_returns_shadow_strategies(api_client):
    token = auth.create_token("test_operator", "operator")
    headers = {"Authorization": f"Bearer {token}"}

    res = api_client.get("/api/shadow-strategies", headers=headers)
    assert res.status_code == 200
    data = res.get_json()
    assert "champion" in data
    assert data["champion"]["ensemble_method"] == "adaptive_confidence"
    assert "challengers" in data
    assert "learning_cycle" in data
