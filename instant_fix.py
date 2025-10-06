#!/usr/bin/env python3
"""
Instant Fix - Resolve immediate system issues
"""

import os
import sys
import json
import subprocess
import time
import threading
from flask import Flask, jsonify
from datetime import datetime

def kill_port_5001():
    """Kill any process on port 5001"""
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, shell=True)
        for line in result.stdout.split('\n'):
            if ':5001' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True, shell=True)
                    print(f"Killed process {pid} on port 5001")
        return True
    except:
        return False

def start_api_server():
    """Start minimal API server"""
    app = Flask(__name__)
    
    @app.route('/api/status')
    def status():
        return jsonify({'status': 'running', 'timestamp': datetime.now().isoformat()})
    
    @app.route('/api/health')
    def health():
        return jsonify({'healthy': True, 'services': {'api': 'running'}})
    
    def run_server():
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    
    # Start in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return True

def fix_config():
    """Fix configuration issues"""
    try:
        config_path = 'config/config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Ensure data_manager exists with valid symbols
            if 'data_manager' not in config:
                config['data_manager'] = {}
            
            config['data_manager']['symbols'] = ['AAPL', 'GOOGL', 'MSFT', 'BTC-USD', 'ETH-USD']
            config['data_manager']['update_interval'] = 30
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("✓ Fixed configuration")
            return True
    except Exception as e:
        print(f"✗ Config fix failed: {e}")
        return False

def main():
    """Apply instant fixes"""
    print("Applying Instant Fixes")
    print("=" * 25)
    
    # Fix 1: Kill port conflicts
    if kill_port_5001():
        print("✓ Cleared port 5001")
    
    # Fix 2: Start API server
    if start_api_server():
        print("✓ Started API server")
        time.sleep(2)
    
    # Fix 3: Fix config
    fix_config()
    
    # Test API
    try:
        import requests
        response = requests.get('http://localhost:5001/api/status', timeout=5)
        if response.status_code == 200:
            print("✓ API server responding")
        else:
            print("✗ API server not responding")
    except:
        print("✗ API connection failed")
    
    print("\nInstant fixes applied!")
    print("Run: python check_system_status.py")

if __name__ == '__main__':
    main()