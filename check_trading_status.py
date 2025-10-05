#!/usr/bin/env python3
"""
Check if paper trading is active and can start trading
"""

import json
import os
import requests
from datetime import datetime

def check_paper_trading_status():
    """Check current paper trading status"""
    print("Paper Trading Status Check")
    print("=" * 30)
    
    # Check paper trading state file
    state_file = "data/paper_trading_state.json"
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        print(f"Cash Available: ${state.get('cash', 0):,.2f}")
        print(f"Active Positions: {len(state.get('positions', {}))}")
        print(f"Trade History: {len(state.get('trade_history', []))} trades")
        print(f"Last Updated: {state.get('last_updated', 'Never')}")
        
        # Check if ready to trade
        if state.get('cash', 0) > 0:
            print("\n[READY] Paper trading account has funds available")
        else:
            print("\n[NOT READY] No cash available for trading")
    else:
        print("[NOT FOUND] Paper trading state file missing")
    
    # Check system health for trading readiness
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            broker_status = health.get('checks', {}).get('broker_manager', {})
            
            print(f"\nBroker Status:")
            print(f"Connected Brokers: {broker_status.get('connected_brokers', 0)}")
            print(f"Primary Broker: {broker_status.get('primary_broker', 'None')}")
            
            # Check if paper broker is connected
            brokers = broker_status.get('brokers', [])
            paper_broker = next((b for b in brokers if b.get('type') == 'paper'), None)
            
            if paper_broker and paper_broker.get('is_connected'):
                print("[READY] Paper trading broker is connected")
                return True
            else:
                print("[NOT READY] Paper trading broker not connected")
                return False
        else:
            print("[ERROR] Cannot check system health")
            return False
    except:
        print("[ERROR] System not responding")
        return False

def check_strategy_signals():
    """Check if strategies are generating signals"""
    print("\n" + "=" * 30)
    print("Strategy Signal Check")
    print("=" * 30)
    
    # Check recent logs for strategy activity
    log_file = "logs/trading_agent.log"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Look for recent strategy activity
        recent_lines = lines[-100:]  # Last 100 lines
        
        signal_count = 0
        error_count = 0
        
        for line in recent_lines:
            if "generate_signals" in line or "signal" in line.lower():
                signal_count += 1
            if "ERROR" in line and "executing trades" in line:
                error_count += 1
        
        print(f"Recent Signal Activity: {signal_count} entries")
        print(f"Trade Execution Errors: {error_count}")
        
        if signal_count > 0:
            print("[ACTIVE] Strategies are generating signals")
        else:
            print("[INACTIVE] No recent signal generation")
        
        if error_count > 0:
            print("[ISSUE] Trade execution has errors")
        else:
            print("[OK] No trade execution errors")

if __name__ == "__main__":
    ready = check_paper_trading_status()
    check_strategy_signals()
    
    print("\n" + "=" * 30)
    if ready:
        print("RESULT: System is READY for paper trading")
        print("The agent can start making trades when signals are strong enough")
    else:
        print("RESULT: System is NOT READY for paper trading")
        print("Check broker connection and system health")