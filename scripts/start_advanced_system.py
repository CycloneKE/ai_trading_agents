#!/usr/bin/env python3
"""
Start the advanced trading dashboard system
"""

import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def start_advanced_api():
    """Start advanced API server with all features"""
    print("🚀 Starting Advanced API Server...")
    subprocess.run([sys.executable, "advanced_api_server.py"], cwd=PROJECT_ROOT)

def start_frontend():
    """Start frontend dashboard"""
    print("🎨 Starting Advanced Dashboard...")
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    if os.path.exists(frontend_dir):
        subprocess.run(["npm", "run", "dev"], shell=True, cwd=frontend_dir)

def main():
    print("=" * 70)
    print("🤖 ADVANCED AI TRADING DASHBOARD")
    print("📊 Market Intelligence • 🛡️ Risk Management • 🧠 AI Analytics")
    print("=" * 70)
    
    # Start advanced API server
    api_thread = Thread(target=start_advanced_api, daemon=True)
    api_thread.start()
    
    time.sleep(3)
    
    # Start frontend
    frontend_thread = Thread(target=start_frontend, daemon=True)
    frontend_thread.start()
    
    time.sleep(5)
    
    print("🌐 Opening Advanced Dashboard...")
    webbrowser.open("http://localhost:3001")
    
    print("\n✅ Advanced System Running:")
    print("   📊 Dashboard: http://localhost:3001")
    print("   🔌 Advanced API: http://localhost:5001")
    print("   📈 Live Market Data + AI Analytics")
    print("   🛡️ Risk Management + Trading Controls")
    print("   🧠 ML Models + Backtesting")
    print("\n⚠️  Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Advanced system stopped")

if __name__ == "__main__":
    main()