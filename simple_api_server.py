#!/usr/bin/env python3
"""
Simple API Server - Lightweight API server for system status
"""

from flask import Flask, jsonify
import threading
import time
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'message': 'Trading system API is operational'
    })

@app.route('/api/health')
def health():
    return jsonify({
        'healthy': True,
        'uptime': time.time(),
        'services': {
            'api': 'running',
            'trading': 'unknown'
        }
    })

def run_server():
    """Run the Flask server"""
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

if __name__ == '__main__':
    print("Starting simple API server on port 5001...")
    run_server()