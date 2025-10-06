#!/usr/bin/env python3
"""
Quick fix for startup issues
"""

import os
import sys
import subprocess
import time
import webbrowser
from threading import Thread

def fix_startup_script():
    """Fix the startup script to handle directory paths correctly"""
    
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    startup_script_content = f'''"""
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
PROJECT_ROOT = r"{project_root}"

def start_api_server():
    """Start the API server."""
    print("🚀 Starting API server...")
    print("Starting API server on http://localhost:5001")
    subprocess.run([sys.executable, "simple_api.py"], cwd=PROJECT_ROOT)

def start_frontend():
    """Start the frontend dashboard."""
    print("🎨 Starting frontend dashboard...")
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    if os.path.exists(frontend_dir):
        subprocess.run(["npm", "run", "dev"], shell=True, cwd=frontend_dir)
    else:
        print("⚠️  Frontend directory not found, skipping...")

def start_trading_bot():
    """Start the main trading bot."""
    print("🤖 Starting AI trading bot...")
    subprocess.run([sys.executable, "main.py"], cwd=PROJECT_ROOT)

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
    
    print("\\n✅ System Status:")
    print("   📊 Dashboard: http://localhost:3001")
    print("   🔌 API Server: http://localhost:5001")
    print("   🤖 Trading Bot: Starting...")
    print("\\n⚠️  Press Ctrl+C to stop all services")
    
    # Start trading bot (main process)
    try:
        start_trading_bot()
    except KeyboardInterrupt:
        print("\\n🛑 Shutting down AI Trading System...")
        print("   All services stopped.")

if __name__ == "__main__":
    main()
'''
    
    # Write the fixed startup script
    with open(os.path.join(project_root, "start_trading_system.py"), "w") as f:
        f.write(startup_script_content)
    
    print("✅ Fixed startup script")

def create_simple_launcher():
    """Create a simple launcher that just starts the trading bot"""
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    launcher_content = f'''#!/usr/bin/env python3
"""
Simple launcher for just the trading bot
"""

import os
import sys
import subprocess

def main():
    """Launch just the trading bot"""
    project_root = r"{project_root}"
    
    print("🚀 Starting AI Trading Bot...")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(project_root)
    
    # Start the trading bot
    try:
        subprocess.run([sys.executable, "main.py", "--config", "config/config.json"])
    except KeyboardInterrupt:
        print("\\n🛑 Trading bot stopped.")
    except Exception as e:
        print(f"❌ Error starting trading bot: {{e}}")

if __name__ == "__main__":
    main()
'''
    
    with open(os.path.join(project_root, "launch_bot.py"), "w") as f:
        f.write(launcher_content)
    
    print("✅ Created simple launcher: launch_bot.py")

def main():
    """Main function"""
    print("🔧 Fixing startup issues...")
    
    # Fix the startup script
    fix_startup_script()
    
    # Create simple launcher
    create_simple_launcher()
    
    print("\n✅ Startup fixes completed!")
    print("\nYou can now use:")
    print("  python start_trading_system.py  # Full system with dashboard")
    print("  python launch_bot.py           # Just the trading bot")

if __name__ == "__main__":
    main()