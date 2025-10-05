#!/usr/bin/env python3
"""
Quick improvements that can be done immediately
"""

import json
import os

def add_api_keys():
    """Add optional API keys for better data"""
    print("=== Optional API Keys Setup ===")
    print("1. Alpha Vantage (free): https://www.alphavantage.co/support/#api-key")
    print("2. News API (free): https://newsapi.org/register")
    print("\nAdd to .env file:")
    print("TRADING_ALPHA_VANTAGE_API_KEY=your_key_here")
    print("TRADING_FMP_API_KEY=your_key_here") 
    print("TRADING_FINNHUB_API_KEY=your_key_here")

def lower_thresholds():
    """Lower trading thresholds for more activity"""
    config_file = "config/config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Lower all strategy thresholds
        if 'strategies' in config:
            for strategy in config['strategies'].values():
                if 'threshold' in strategy:
                    strategy['threshold'] = 0.001  # Very low threshold
                if 'sentiment_threshold' in strategy:
                    strategy['sentiment_threshold'] = 0.3  # Lower sentiment threshold
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("[DONE] Lowered all trading thresholds for more activity")
    else:
        print("[ERROR] Config file not found")

def increase_update_frequency():
    """Increase data update frequency"""
    config_file = "config/config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Faster updates
        config['data_manager']['update_interval'] = 30  # 30 seconds
        config['trading_loop_interval'] = 15  # 15 seconds
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("[DONE] Increased update frequency for faster trading")

if __name__ == "__main__":
    print("Quick Improvements Available:")
    print("1. Lower trading thresholds (more trades)")
    print("2. Increase update frequency (faster response)")
    print("3. Add API keys (better data)")
    
    choice = input("\nRun improvements? (y/n): ")
    if choice.lower() == 'y':
        lower_thresholds()
        increase_update_frequency()
        add_api_keys()
        print("\n[COMPLETE] Restart the agent to apply changes")