#!/usr/bin/env python3
"""Mocked API tests (unit-style) — use responses to fake external API replies.

These are safe to run in CI. If you want to run real network checks locally, run a separate integration test
that uses real API keys and does not use the mocked endpoints.
"""
import responses


@responses.activate
def test_alpha_vantage_mocked():
    url = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=testkey"
    mock_body = {"Global Quote": {"01. symbol": "AAPL", "05. price": "150.00"}}
    responses.add(responses.GET, url, json=mock_body, status=200)

    import requests

    resp = requests.get(url, timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert "Global Quote" in data
    assert data["Global Quote"]["01. symbol"] == "AAPL"


@responses.activate
def test_fmp_mocked():
    url = "https://financialmodelingprep.com/api/v3/quote/AAPL?apikey=testkey"
    mock_body = [{"symbol": "AAPL", "price": 150.0}]
    responses.add(responses.GET, url, json=mock_body, status=200)

    import requests

    resp = requests.get(url, timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) and data[0]["symbol"] == "AAPL"


@responses.activate
def test_finnhub_mocked():
    url = "https://finnhub.io/api/v1/quote?symbol=AAPL&token=testkey"
    mock_body = {"c": 150.0}
    responses.add(responses.GET, url, json=mock_body, status=200)

    import requests

    resp = requests.get(url, timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("c") == 150.0


@responses.activate
def test_news_api_mocked():
    url = "https://newsapi.org/v2/everything?q=AAPL&apiKey=testkey&pageSize=1"
    mock_body = {"totalResults": 1, "articles": [{"title": "AAPL news"}]}
    responses.add(responses.GET, url, json=mock_body, status=200)

    import requests

    resp = requests.get(url, timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("totalResults", 0) >= 0
