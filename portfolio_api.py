#!/usr/bin/env python3
"""
Simple portfolio API server to show paper trading data
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading

class PortfolioHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logs
    
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        if path == '/portfolio':
            self.handle_portfolio()
        elif path == '/':
            self.handle_dashboard()
        else:
            self.send_error(404)
    
    def handle_portfolio(self):
        try:
            state_file = "data/paper_trading_state.json"
            portfolio_data = {
                'cash': 100000,
                'positions': {},
                'total_value': 100000,
                'trade_history': [],
                'last_updated': 'Never'
            }
            
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                portfolio_data.update(state)
                
                # Calculate total value
                total_value = state.get('cash', 0)
                for symbol, position in state.get('positions', {}).items():
                    if isinstance(position, dict):
                        quantity = position.get('quantity', 0)
                        price = state.get('market_prices', {}).get(symbol, position.get('avg_price', 0))
                        total_value += quantity * price
                
                portfolio_data['total_value'] = total_value
            
            self.send_json_response(portfolio_data)
            
        except Exception as e:
            self.send_json_response({'error': str(e)})
    
    def handle_dashboard(self):
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Paper Trading Portfolio</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .card {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ text-align: center; color: #333; }}
                .metric {{ display: flex; justify-content: space-between; margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }}
                .metric-value {{ font-weight: bold; color: #28a745; }}
                .refresh-btn {{ background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }}
                .refresh-btn:hover {{ background: #0056b3; }}
            </style>
            <script>
                async function loadPortfolio() {{
                    try {{
                        const response = await fetch('/portfolio');
                        const data = await response.json();
                        
                        document.getElementById('cash').textContent = '$' + (data.cash || 0).toLocaleString();
                        document.getElementById('total-value').textContent = '$' + (data.total_value || 0).toLocaleString();
                        document.getElementById('positions').textContent = Object.keys(data.positions || {{}}).length;
                        document.getElementById('trades').textContent = (data.trade_history || []).length;
                        document.getElementById('last-updated').textContent = data.last_updated || 'Never';
                        
                        // Show positions
                        const positionsDiv = document.getElementById('positions-list');
                        const positions = data.positions || {{}};
                        if (Object.keys(positions).length > 0) {{
                            let html = '<h3>Current Positions:</h3>';
                            for (const [symbol, pos] of Object.entries(positions)) {{
                                if (typeof pos === 'object') {{
                                    html += `<div class="metric">
                                        <span>${{symbol}}</span>
                                        <span>${{pos.quantity}} @ ${{pos.avg_price}}</span>
                                    </div>`;
                                }}
                            }}
                            positionsDiv.innerHTML = html;
                        }} else {{
                            positionsDiv.innerHTML = '<p>No positions currently held</p>';
                        }}
                        
                    }} catch (e) {{
                        console.error('Error loading portfolio:', e);
                    }}
                }}
                
                window.onload = loadPortfolio;
                setInterval(loadPortfolio, 10000); // Refresh every 10 seconds
            </script>
        </head>
        <body>
            <div class="container">
                <h1>Paper Trading Portfolio</h1>
                
                <div class="card">
                    <h2>Account Summary</h2>
                    <div class="metric">
                        <span>Available Cash:</span>
                        <span class="metric-value" id="cash">Loading...</span>
                    </div>
                    <div class="metric">
                        <span>Total Portfolio Value:</span>
                        <span class="metric-value" id="total-value">Loading...</span>
                    </div>
                    <div class="metric">
                        <span>Active Positions:</span>
                        <span class="metric-value" id="positions">Loading...</span>
                    </div>
                    <div class="metric">
                        <span>Total Trades:</span>
                        <span class="metric-value" id="trades">Loading...</span>
                    </div>
                    <div class="metric">
                        <span>Last Updated:</span>
                        <span class="metric-value" id="last-updated">Loading...</span>
                    </div>
                </div>
                
                <div class="card">
                    <div id="positions-list">Loading positions...</div>
                </div>
                
                <div class="card">
                    <button class="refresh-btn" onclick="loadPortfolio()">Refresh Data</button>
                    <p><a href="http://localhost:8080">← Back to Main Dashboard</a></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

def start_portfolio_server(port=8081):
    """Start the portfolio server"""
    server = HTTPServer(('localhost', port), PortfolioHandler)
    print(f"Portfolio dashboard running at http://localhost:{port}")
    server.serve_forever()

if __name__ == "__main__":
    # Run in background thread
    thread = threading.Thread(target=start_portfolio_server, daemon=True)
    thread.start()
    
    print("Portfolio API started at http://localhost:8081")
    print("Press Ctrl+C to stop")
    
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nStopping portfolio server...")