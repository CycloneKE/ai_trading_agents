#!/usr/bin/env python3
"""
System Verification Script
Quick verification of AI Trading Agent functionality
"""

import requests
import json
import subprocess
import os
from datetime import datetime

def check_system_status():
    """Check if the system is working properly"""
    print("AI Trading Agent System Verification")
    print("=" * 50)
    
    # 1. Check if monitoring service is running
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            print("[OK] Monitoring service: RUNNING")
            health_data = response.json()
            print(f"   Status: {health_data.get('status', 'unknown')}")
        else:
            print("[FAIL] Monitoring service: NOT RESPONDING")
    except:
        print("[FAIL] Monitoring service: NOT RUNNING")
    
    # 2. Check logs for recent activity
    log_file = "logs/trading_agent.log"
    if os.path.exists(log_file):
        print("[OK] Log file: EXISTS")
        # Get last few lines
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    print(f"   Last log: {last_line[:100]}...")
        except:
            print("   Could not read log file")
    else:
        print("[FAIL] Log file: NOT FOUND")
    
    # 3. Check paper trading state
    state_file = "data/paper_trading_state.json"
    if os.path.exists(state_file):
        print("[OK] Paper trading state: EXISTS")
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                print(f"   Cash: ${state.get('cash', 0):,.2f}")
                print(f"   Positions: {len(state.get('positions', {}))}")
        except:
            print("   Could not read state file")
    else:
        print("[FAIL] Paper trading state: NOT FOUND")
    
    # 4. Check optimization results
    today = datetime.now().strftime('%Y%m%d')
    opt_file = f"data/optimization_results_{today}.json"
    if os.path.exists(opt_file):
        print("[OK] Today's optimization: EXISTS")
    else:
        print("[WARN] Today's optimization: NOT FOUND (may not have run yet)")
    
    # 5. Check configuration
    config_file = "config/config.json"
    if os.path.exists(config_file):
        print("[OK] Configuration: EXISTS")
    else:
        print("[FAIL] Configuration: NOT FOUND")
    
    print("\nSystem Metrics:")
    try:
        response = requests.get("http://localhost:8080/metrics", timeout=5)
        if response.status_code == 200:
            print("[OK] Metrics endpoint: ACCESSIBLE")
        else:
            print("[FAIL] Metrics endpoint: NOT ACCESSIBLE")
    except:
        print("[FAIL] Metrics endpoint: NOT RESPONDING")

def check_data_flow():
    """Check if data is flowing through the system"""
    print("\nData Flow Verification")
    print("=" * 30)
    
    # Check if data manager is collecting data
    log_file = "logs/trading_agent.log"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Look for data collection indicators
            if "Connected yahoo_finance" in content:
                print("[OK] Yahoo Finance: CONNECTED")
            else:
                print("[FAIL] Yahoo Finance: NOT CONNECTED")
                
            if "Alpha Vantage API key not provided" in content:
                print("[WARN] Alpha Vantage: API KEY MISSING")
            elif "Connected alpha_vantage" in content:
                print("[OK] Alpha Vantage: CONNECTED")
            else:
                print("[FAIL] Alpha Vantage: NOT CONNECTED")
                
            if "NewsAPI API key not provided" in content:
                print("[WARN] News API: API KEY MISSING")
            elif "Connected news_api" in content:
                print("[OK] News API: CONNECTED")
            else:
                print("[FAIL] News API: NOT CONNECTED")
                
            # Check for strategy execution
            if "Strategy manager initialized with" in content:
                print("[OK] Strategies: INITIALIZED")
            else:
                print("[FAIL] Strategies: NOT INITIALIZED")
                
        except Exception as e:
            print(f"[FAIL] Could not analyze logs: {e}")

def check_trading_activity():
    """Check for trading activity"""
    print("\nTrading Activity")
    print("=" * 20)
    
    log_file = "logs/trading_agent.log"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Look for trading indicators
            if "Placing" in content and "order" in content:
                print("[OK] Orders: BEING PLACED")
            else:
                print("[WARN] Orders: NO RECENT ACTIVITY")
                
            if "Portfolio optimization completed" in content:
                print("[OK] Portfolio optimization: RUNNING")
            else:
                print("[WARN] Portfolio optimization: NOT RUNNING")
                
            # Count errors
            error_count = content.count("ERROR")
            warning_count = content.count("WARNING")
            
            print(f"[WARN] Warnings: {warning_count}")
            print(f"[ERROR] Errors: {error_count}")
            
        except Exception as e:
            print(f"[FAIL] Could not analyze trading activity: {e}")

def main():
    """Main verification function"""
    check_system_status()
    check_data_flow()
    check_trading_activity()
    
    print("\nQuick Actions:")
    print("1. View logs: tail -f logs/trading_agent.log")
    print("2. Check health: curl http://localhost:8080/health")
    print("3. View metrics: curl http://localhost:8080/metrics")
    print("4. Monitor dashboard: http://localhost:8080")

if __name__ == "__main__":
    main()