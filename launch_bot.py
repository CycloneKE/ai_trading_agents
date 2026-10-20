#!/usr/bin/env python3
"""
Simple launcher for just the trading bot
"""

import os
import sys
import subprocess

def main():
    """Launch just the trading bot"""
    project_root = r"C:\Users\joesy\ai_trading_agents"
    
    print("🚀 Starting AI Trading Bot...")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(project_root)
    
    # Start the trading bot
    try:
        subprocess.run([sys.executable, "main.py", "--config", "config/config.json"])
    except KeyboardInterrupt:
        print("\n🛑 Trading bot stopped.")
    except Exception as e:
        print(f"❌ Error starting trading bot: {e}")

if __name__ == "__main__":
    main()
