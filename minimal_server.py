#!/usr/bin/env python3
"""Minimal API Server - Ultra lightweight"""
import json
from datetime import datetime
try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading
    
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/status':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'status': 'running', 'timestamp': datetime.now().isoformat()}
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            pass  # Suppress logs
    
    def start_server():
        server = HTTPServer(('localhost', 5001), Handler)
        server.serve_forever()
    
    if __name__ == '__main__':
        print("Starting minimal server on port 5001...")
        thread = threading.Thread(target=start_server, daemon=True)
        thread.start()
        
        import time
        time.sleep(2)
        
        # Test
        import urllib.request
        try:
            response = urllib.request.urlopen('http://localhost:5001/api/status')
            print("✓ Server running")
        except:
            print("✗ Server failed")
        
        input("Press Enter to stop...")
        
except ImportError:
    print("✗ Missing dependencies")