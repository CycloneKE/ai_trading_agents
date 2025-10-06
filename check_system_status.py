#!/usr/bin/env python3
"""
System Status Checker - Quick health check for the trading system
"""

import os
import sys
import json
import requests
import psutil
from datetime import datetime

def check_files():
    """Check if required files exist"""
    required_files = [
        'main.py',
        'config/config.json',
        '.env'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    return len(missing) == 0, missing

def check_api_server():
    """Check if API server is running"""
    try:
        response = requests.get('http://localhost:5001/api/status', timeout=5)
        return response.status_code == 200, "Running"
    except requests.exceptions.ConnectionError:
        return False, "Not running - connection refused"
    except Exception as e:
        return False, str(e)

def check_system_resources():
    """Check system resource usage"""
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('C:' if os.name == 'nt' else '/').percent
        
        healthy = cpu < 95 and memory < 98 and disk < 95
        
        return {
            'cpu': cpu,
            'memory': memory,
            'disk': disk,
            'healthy': healthy,
            'critical': cpu > 95 or memory > 95
        }
    except:
        return {'cpu': 0, 'memory': 0, 'disk': 0, 'healthy': True, 'critical': False}

def main():
    """Main status check"""
    print("AI Trading System Status Check")
    print("=" * 40)
    
    # Check files
    files_ok, missing_files = check_files()
    status_icon = "✓" if files_ok else "✗"
    print(f"{status_icon} Required Files: {'OK' if files_ok else 'Missing: ' + ', '.join(missing_files)}")
    
    # Check API server
    api_ok, api_msg = check_api_server()
    status_icon = "✓" if api_ok else "✗"
    print(f"{status_icon} API Server: {api_msg}")
    
    # Check resources
    resources = check_system_resources()
    if resources.get('critical', False):
        status_icon = "🔥"
        status_msg = f"CRITICAL - CPU {resources['cpu']:.1f}%, Memory {resources['memory']:.1f}%"
    elif resources['healthy']:
        status_icon = "✓"
        status_msg = f"CPU {resources['cpu']:.1f}%, Memory {resources['memory']:.1f}%, Disk {resources['disk']:.1f}%"
    else:
        status_icon = "⚠"
        status_msg = f"HIGH - CPU {resources['cpu']:.1f}%, Memory {resources['memory']:.1f}%"
    
    print(f"{status_icon} System Resources: {status_msg}")
    
    if resources.get('critical', False):
        print("🚨 CRITICAL: System resources critically high!")
        print("💡 Run: python emergency_fix.py")
    
    # Overall status
    print()
    if files_ok and api_ok and resources['healthy']:
        print("✓ System is healthy and running")
    elif not api_ok:
        print("✗ System not responding - check if it's running")
        print("💡 Run: python quick_start_system.py")
    else:
        print("⚠ System has issues - run diagnostics")
        print("💡 Run: python enhanced_error_handler.py")

if __name__ == '__main__':
    main()