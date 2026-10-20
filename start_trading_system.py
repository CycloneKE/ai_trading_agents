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

# Set project root directory
PROJECT_ROOT = r"C:\Users\joesy\ai_trading_agents"

def start_api_server():
    """Start the API server."""
    print("[API] Starting API server...")
    print("Starting API server on http://localhost:5001")
    subprocess.run([sys.executable, os.path.join("scripts", "simple_api.py")], cwd=PROJECT_ROOT)

def start_frontend():
    """Start the frontend dashboard."""
    print("[UI] Starting frontend dashboard...")
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    if os.path.exists(frontend_dir):
        subprocess.run(["npm", "run", "dev", "--", "-p", "3001"], shell=True, cwd=frontend_dir)
    else:
        print("[UI] Warning: Frontend directory not found, skipping...")

def start_trading_bot():
    """Start the main trading bot."""
    print("[BOT] Starting AI trading bot...")
    subprocess.run([sys.executable, "main.py"], cwd=PROJECT_ROOT)

def main():
    """Main startup function."""
    print("=" * 60)
    print("AI TRADING SYSTEM STARTUP")
    print("=" * 60)
    
    # API server is started automatically inside the trading bot on port 5001.
    time.sleep(1)
    
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
    print("[UI] Opening dashboard in browser...")
    webbrowser.open("http://localhost:3001")
    
    print("\n[OK] System Status:")
    print("   Dashboard: http://localhost:3001")
    print("   API Server: http://localhost:5001")
    print("   Trading Bot: Starting...")
    print("\nPress Ctrl+C to stop all services")
    
    # Start trading bot (main process)
    try:
        start_trading_bot()
    except KeyboardInterrupt:
        print("\nShutting down AI Trading System...")
        print("   All services stopped.")

if __name__ == "__main__":
    main()
