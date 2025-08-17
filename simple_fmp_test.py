#!/usr/bin/env python3
"""Simple test of FMP provider functionality"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fmp_provider import FMPProvider

def test_fmp_provider():
    """Test FMP provider basic functionality"""
    
    print("Testing FMP Provider...")
    
    # Initialize provider
    fmp = FMPProvider()
    
    # Test with AAPL
    symbol = "AAPL"
    
    print(f"\nTesting with symbol: {symbol}")
    
    # Test quote
    print("Getting quote...")
    quote = fmp.get_quote(symbol)
    print(f"Quote data: {quote}")
    
    # Test financial ratios
    print("\nGetting financial ratios...")
    ratios = fmp.get_financial_ratios(symbol)
    print(f"Ratios data: {ratios}")
    
    # Test market cap extraction
    print("\nGetting market cap...")
    market_cap = fmp.get_market_cap(symbol)
    print(f"Market cap: {market_cap}")
    
    # Show what features would be available for ML
    print("\n=== ML Features Available ===")
    features = {}
    
    if quote:
        features['price'] = quote.get('price', 0)
        features['market_cap'] = quote.get('marketCap', 0)
    
    if ratios:
        features['pe_ratio'] = ratios.get('priceEarningsRatio', 0)
        features['debt_ratio'] = ratios.get('debtRatio', 0)
        features['current_ratio'] = ratios.get('currentRatio', 0)
    
    print(f"Extracted features: {features}")
    
    if features:
        print("[SUCCESS] FMP integration working - fundamental data available for ML")
    else:
        print("[WARNING] No fundamental data available (check API key)")
        print("Note: Replace 'your_fmp_api_key_here' in .env with your actual FMP API key")

if __name__ == "__main__":
    test_fmp_provider()