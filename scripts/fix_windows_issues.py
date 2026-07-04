#!/usr/bin/env python3
"""
Fix Windows-specific Issues
"""

import os
import sys
import subprocess
import time

def fix_port_conflicts():
    """Fix port conflicts on Windows"""
    try:
        print("Checking for port conflicts...")
        
        # Kill any processes on port 5001
        result = subprocess.run(
            ['netstat', '-ano'], 
            capture_output=True, text=True, shell=True
        )
        
        for line in result.stdout.split('\n'):
            if ':5001' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    print(f"Killing process {pid} on port 5001")
                    subprocess.run(['taskkill', '/F', '/PID', pid], 
                                 capture_output=True, shell=True)
        
        print("✓ Port conflicts resolved")
        return True
        
    except Exception as e:
        print(f"✗ Error fixing port conflicts: {e}")
        return False

def start_simple_api():
    """Start simple API server"""
    try:
        print("Starting simple API server...")
        
        # Start in background
        subprocess.Popen([
            sys.executable, 'simple_api_server.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(3)  # Wait for startup
        
        # Test connection
        import requests
        response = requests.get('http://localhost:5001/api/status', timeout=5)
        
        if response.status_code == 200:
            print("✓ API server started successfully")
            return True
        else:
            print("✗ API server not responding")
            return False
            
    except Exception as e:
        print(f"✗ Error starting API server: {e}")
        return False

def main():
    """Main fix function"""
    print("Fixing Windows-specific Issues")
    print("=" * 35)
    
    success_count = 0
    
    if fix_port_conflicts():
        success_count += 1
    
    if start_simple_api():
        success_count += 1
    
    print(f"\nCompleted {success_count}/2 fixes")
    
    if success_count == 2:
        print("✓ All Windows issues fixed!")
        print("\nTest with: python check_system_status.py")
    else:
        print("⚠ Some fixes failed")

if __name__ == '__main__':
    main()