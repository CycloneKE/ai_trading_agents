#!/usr/bin/env python3
"""
Quick test of real data APIs
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_fmp():
    api_key = os.getenv('TRADING_FMP_API_KEY')
    if not api_key:
        return False, "No API key"
    
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return True, f"AAPL: ${data[0]['price']}"
        return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_alpha_vantage():
    api_key = os.getenv('TRADING_ALPHA_VANTAGE_API_KEY')
    if not api_key:
        return False, "No API key"
    
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'Global Quote' in data:
                price = data['Global Quote']['05. price']
                return True, f"AAPL: ${price}"
        return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, str(e)

print("🔍 Testing Real Data APIs...")
print("=" * 40)

fmp_ok, fmp_msg = test_fmp()
print(f"FMP API: {'✅' if fmp_ok else '❌'} {fmp_msg}")

av_ok, av_msg = test_alpha_vantage()
print(f"Alpha Vantage: {'✅' if av_ok else '❌'} {av_msg}")

print("=" * 40)
if fmp_ok or av_ok:
    print("✅ Ready for real data!")
else:
    print("❌ No working APIs")