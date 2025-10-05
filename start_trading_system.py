"""
Single command startup script for the complete AI Trading System.
Launches trading bot, API server, and frontend dashboard.
"""

import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def start_api_server():
    """Start the API server."""
    print("🚀 Starting API server...")
    subprocess.run([sys.executable, "simple_api.py"], cwd=os.getcwd())

def start_frontend():
    """Start the frontend dashboard."""
    print("🎨 Starting frontend dashboard...")
    os.chdir("frontend")
    subprocess.run(["npm", "run", "dev"], shell=True)

def start_trading_bot():
    """Start the main trading bot."""
    print("🤖 Starting AI trading bot...")
    subprocess.run([sys.executable, "main.py"])

def main():
    """Main startup function."""
    print("=" * 60)
    print("🚀 AI TRADING SYSTEM STARTUP")
    print("=" * 60)
    
    # Start API server in background thread
    api_thread = Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # Wait for API to start
    time.sleep(3)
    
    # Kill any existing Node.js processes
    try:
        subprocess.run(["taskkill", "/F", "/IM", "node.exe"], capture_output=True, shell=True)
        time.sleep(2)
    except:
        pass
    
    # Start frontend in background thread
    frontend_thread = Thread(target=start_frontend, daemon=True)
    frontend_thread.start()
    
    # Wait for frontend to start
    time.sleep(5)
    
    # Open browser to dashboard
    print("🌐 Opening dashboard in browser...")
    webbrowser.open("http://localhost:3001")
    
    print("\n✅ System Status:")
    print("   📊 Dashboard: http://localhost:3001")
    print("   🔌 API Server: http://localhost:5001")
    print("   🤖 Trading Bot: Starting...")
    print("\n⚠️  Press Ctrl+C to stop all services")
    
    # Start trading bot (main process)
    try:
        start_trading_bot()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down AI Trading System...")
        print("   All services stopped.")

if __name__ == "__main__":
    main()