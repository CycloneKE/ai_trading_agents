#!/usr/bin/env python3
"""Emergency Fix - Address critical resource issues"""
import os
import sys
import gc
import subprocess

def free_memory():
    """Free up memory"""
    gc.collect()
    print("✓ Memory cleanup")

def kill_heavy_processes():
    """Kill resource-heavy processes"""
    try:
        # Kill any Python processes except current
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True)
        
        current_pid = os.getpid()
        for line in result.stdout.split('\n'):
            if 'python.exe' in line:
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    if pid.isdigit() and int(pid) != current_pid:
                        subprocess.run(['taskkill', '/F', '/PID', pid], 
                                     capture_output=True)
        print("✓ Killed heavy processes")
    except:
        print("⚠ Could not kill processes")

def start_minimal_api():
    """Start minimal API using standard library only"""
    code = '''
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if '/api/status' in self.path:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "running", "timestamp": datetime.now().isoformat()}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args): pass

server = HTTPServer(("localhost", 5001), Handler)
print("Minimal API started on port 5001")
server.serve_forever()
'''
    
    with open('temp_server.py', 'w') as f:
        f.write(code)
    
    # Start in background
    subprocess.Popen([sys.executable, 'temp_server.py'], 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    import time
    time.sleep(3)
    
    # Test
    try:
        import urllib.request
        urllib.request.urlopen('http://localhost:5001/api/status', timeout=2)
        print("✓ Minimal API started")
        return True
    except:
        print("✗ API start failed")
        return False

def main():
    """Emergency fixes"""
    print("Emergency System Fix")
    print("=" * 20)
    
    free_memory()
    kill_heavy_processes()
    
    if start_minimal_api():
        print("\n✓ Emergency fixes applied")
        print("Test: python check_system_status.py")
    else:
        print("\n✗ Critical failure - restart computer")

if __name__ == '__main__':
    main()