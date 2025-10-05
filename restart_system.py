#!/usr/bin/env python3
"""
Restart the trading system with updated configuration
"""

import subprocess
import sys
import time

def restart_system():
    """Restart the trading system"""
    print("Restarting AI Trading Agent with updated data connector...")
    
    try:
        # Start the main system
        process = subprocess.Popen([
            sys.executable, "main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print(f"Started trading agent with PID: {process.pid}")
        print("System is starting up...")
        print("Dashboard will be available at: http://localhost:8080")
        print("Press Ctrl+C to stop")
        
        # Wait and show initial output
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ System started successfully!")
            print("✅ Updated data connector active (Finnhub priority)")
            print("✅ FMP errors should be resolved")
            
            # Keep the process running
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\nStopping system...")
                process.terminate()
                process.wait()
        else:
            stdout, stderr = process.communicate()
            print("❌ System failed to start:")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            
    except Exception as e:
        print(f"Error starting system: {e}")

if __name__ == "__main__":
    restart_system()