#!/usr/bin/env python3
"""
Start dashboard with real market data
"""

import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def start_real_api():
    """Start real data API server"""
    print("🚀 Starting Real Data API Server...")
    subprocess.run([sys.executable, "real_api_server.py"], cwd=PROJECT_ROOT)

def start_frontend():
    """Start frontend dashboard"""
    print("🎨 Starting Dashboard...")
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    if os.path.exists(frontend_dir):
        subprocess.run(["npm", "run", "dev"], shell=True, cwd=frontend_dir)
    else:
        print("⚠️  Frontend not found")

def main():
    print("=" * 60)
    print("🚀 AI TRADING DASHBOARD - REAL DATA")
    print("=" * 60)
    
    # Start API server
    api_thread = Thread(target=start_real_api, daemon=True)
    api_thread.start()
    
    time.sleep(3)
    
    # Start frontend
    frontend_thread = Thread(target=start_frontend, daemon=True)
    frontend_thread.start()
    
    time.sleep(5)
    
    print("🌐 Opening dashboard...")
    webbrowser.open("http://localhost:3001")
    
    print("\n✅ System Running:")
    print("   📊 Dashboard: http://localhost:3001")
    print("   🔌 Real API: http://localhost:5001")
    print("   📈 Live Market Data")
    print("\n⚠️  Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped")

if __name__ == "__main__":
    main()