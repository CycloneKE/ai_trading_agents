#!/usr/bin/env python3
"""
Test API keys to verify they're working
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_alpha_vantage():
    """Test Alpha Vantage API"""
    api_key = os.getenv('TRADING_ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("[FAIL] Alpha Vantage API key not found")
        return
    
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'Global Quote' in data:
                print(f"[PASS] Alpha Vantage: {data['Global Quote']['01. symbol']} - ${data['Global Quote']['05. price']}")
            else:
                print(f"[WARN] Alpha Vantage: {data}")
        else:
            print(f"[FAIL] Alpha Vantage: HTTP {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Alpha Vantage: {e}")

def test_fmp():
    """Test Financial Modeling Prep API"""
    api_key = os.getenv('TRADING_FMP_API_KEY')
    if not api_key:
        print("[FAIL] FMP API key not found")
        return
    
    url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                print(f"[PASS] FMP: {data[0]['symbol']} - ${data[0]['price']}")
            else:
                print(f"[WARN] FMP: Empty response")
        else:
            print(f"[FAIL] FMP: HTTP {response.status_code}")
    except Exception as e:
        print(f"[ERROR] FMP: {e}")

def test_finnhub():
    """Test Finnhub API"""
    api_key = os.getenv('TRADING_FINNHUB_API_KEY')
    if not api_key:
        print("[FAIL] Finnhub API key not found")
        return
    
    url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'c' in data:
                print(f"[PASS] Finnhub: AAPL - ${data['c']}")
            else:
                print(f"[WARN] Finnhub: {data}")
        else:
            print(f"[FAIL] Finnhub: HTTP {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Finnhub: {e}")

def test_news_api():
    """Test News API"""
    api_key = os.getenv('TRADING_NEWS_API_KEY')
    if not api_key:
        print("[FAIL] News API key not found")
        return
    
    url = f"https://newsapi.org/v2/everything?q=AAPL&apiKey={api_key}&pageSize=1"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('totalResults', 0) > 0:
                print(f"[PASS] News API: {data['totalResults']} articles found")
            else:
                print(f"[WARN] News API: No articles found")
        else:
            print(f"[FAIL] News API: HTTP {response.status_code}")
    except Exception as e:
        print(f"[ERROR] News API: {e}")

if __name__ == "__main__":
    print("Testing API Keys...")
    print("=" * 30)
    test_alpha_vantage()
    test_fmp()
    test_finnhub()
    test_news_api()
    print("=" * 30)
    print("Test complete")