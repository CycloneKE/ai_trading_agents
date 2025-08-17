"""
Live Data API for Frontend
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import threading

class LiveDataHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests for live data."""
        
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Enable CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        if path == '/' or path == '/dashboard':
            self.serve_dashboard()
        elif path == '/api/portfolio':
            self.serve_portfolio_data()
        elif path == '/api/performance':
            self.serve_performance_data()
        elif path == '/api/trades':
            self.serve_trades_data()
        elif path == '/api/alerts':
            self.serve_alerts_data()
        elif path == '/api/learning':
            self.serve_learning_data()
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def serve_portfolio_data(self):
        """Serve current portfolio data."""
        try:
            if os.path.exists('phase2_portfolio.json'):
                with open('phase2_portfolio.json', 'r') as f:
                    portfolio = json.load(f)
                
                # Calculate current values
                total_value = portfolio['cash']
                positions_data = []
                
                for symbol, pos in portfolio['positions'].items():
                    position_value = pos['shares'] * pos['avg_price']  # Simplified
                    total_value += position_value
                    
                    positions_data.append({
                        'symbol': symbol,
                        'shares': pos['shares'],
                        'avg_price': pos['avg_price'],
                        'current_value': position_value,
                        'entry_date': pos['entry_date']
                    })
                
                response_data = {
                    'cash': portfolio['cash'],
                    'total_value': total_value,
                    'positions': positions_data,
                    'total_return': ((total_value - 100000) / 100000) * 100,
                    'last_updated': datetime.now().isoformat()
                }
            else:
                response_data = {
                    'cash': 100000,
                    'total_value': 100000,
                    'positions': [],
                    'total_return': 0,
                    'last_updated': datetime.now().isoformat()
                }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            self.send_error(500, f"Error loading portfolio: {str(e)}")
    
    def serve_performance_data(self):
        """Serve performance history."""
        try:
            if os.path.exists('phase2_performance.json'):
                with open('phase2_performance.json', 'r') as f:
                    performance = json.load(f)
            else:
                performance = []
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(performance).encode())
            
        except Exception as e:
            self.send_error(500, f"Error loading performance: {str(e)}")
    
    def serve_trades_data(self):
        """Serve recent trades."""
        try:
            trades = []
            
            if os.path.exists('phase2_portfolio.json'):
                with open('phase2_portfolio.json', 'r') as f:
                    portfolio = json.load(f)
                    trades = portfolio.get('trade_history', [])
            
            # Get last 20 trades
            recent_trades = trades[-20:] if len(trades) > 20 else trades
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(recent_trades).encode())
            
        except Exception as e:
            self.send_error(500, f"Error loading trades: {str(e)}")
    
    def serve_alerts_data(self):
        """Serve trading alerts."""
        try:
            if os.path.exists('trading_alerts.json'):
                with open('trading_alerts.json', 'r') as f:
                    alerts = json.load(f)
                    # Get last 10 alerts
                    recent_alerts = alerts[-10:] if len(alerts) > 10 else alerts
            else:
                recent_alerts = []
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(recent_alerts).encode())
            
        except Exception as e:
            self.send_error(500, f"Error loading alerts: {str(e)}")
    
    def serve_learning_data(self):
        """Serve AI learning data."""
        try:
            if os.path.exists('continuous_learning.json'):
                with open('continuous_learning.json', 'r') as f:
                    learning_data = json.load(f)
                    
                    # Summary stats
                    total_scans = len(learning_data)
                    high_conviction = len([d for d in learning_data if d.get('ai_analysis', {}).get('genius_score', 0) > 7.0])
                    
                    summary = {
                        'total_scans': total_scans,
                        'high_conviction_signals': high_conviction,
                        'last_scan': learning_data[-1] if learning_data else None,
                        'avg_genius_score': sum([d.get('ai_analysis', {}).get('genius_score', 0) for d in learning_data]) / max(total_scans, 1)
                    }
            else:
                summary = {
                    'total_scans': 0,
                    'high_conviction_signals': 0,
                    'last_scan': None,
                    'avg_genius_score': 0
                }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(summary).encode())
            
        except Exception as e:
            self.send_error(500, f"Error loading learning data: {str(e)}")
    
    def serve_dashboard(self):
        """Serve the dashboard HTML."""
        
        html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Trading Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #0d1117; color: #c9d1d9; }
        .header { background: #21262d; padding: 20px; text-align: center; }
        .container { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; padding: 20px; }
        .card { background: #161b22; padding: 20px; border-radius: 8px; border: 1px solid #30363d; }
        .metric { font-size: 24px; font-weight: bold; color: #7ee787; }
        .label { font-size: 14px; color: #8b949e; margin-bottom: 5px; }
        .positive { color: #7ee787; }
        .negative { color: #f85149; }
        .trade-item { background: #0d1117; padding: 10px; margin: 5px 0; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Trading Dashboard</h1>
        <p>Live data from your trading agent</p>
    </div>

    <div class="container">
        <div class="card">
            <div class="label">Portfolio Value</div>
            <div class="metric" id="portfolio-value">Loading...</div>
            <div class="label">Total Return</div>
            <div class="metric" id="total-return">Loading...</div>
        </div>

        <div class="card">
            <div class="label">Active Positions</div>
            <div class="metric" id="positions">Loading...</div>
            <div class="label">Cash Available</div>
            <div class="metric" id="cash">Loading...</div>
        </div>

        <div class="card">
            <div class="label">AI Status</div>
            <div class="metric">Active</div>
            <div class="label">Genius Score</div>
            <div class="metric" id="genius-score">Loading...</div>
        </div>

        <div class="card">
            <h3>Recent Trades</h3>
            <div id="recent-trades">Loading...</div>
        </div>

        <div class="card">
            <h3>Current Positions</h3>
            <div id="current-positions">Loading...</div>
        </div>

        <div class="card">
            <h3>Performance</h3>
            <div class="label">Win Rate</div>
            <div class="metric" id="win-rate">Loading...</div>
            <div class="label">Total Trades</div>
            <div class="metric" id="total-trades">Loading...</div>
        </div>
    </div>

    <script>
        function formatCurrency(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(value);
        }
        
        function formatPercent(value) {
            return (value >= 0 ? '+' : '') + value.toFixed(2) + '%';
        }
        
        async function updateDashboard() {
            try {
                // Fetch portfolio data
                const portfolioResponse = await fetch('/api/portfolio');
                if (portfolioResponse.ok) {
                    const portfolio = await portfolioResponse.json();
                    
                    document.getElementById('portfolio-value').textContent = formatCurrency(portfolio.total_value);
                    document.getElementById('cash').textContent = formatCurrency(portfolio.cash);
                    document.getElementById('positions').textContent = portfolio.positions.length;
                    
                    const returnElement = document.getElementById('total-return');
                    returnElement.textContent = formatPercent(portfolio.total_return);
                    returnElement.className = 'metric ' + (portfolio.total_return >= 0 ? 'positive' : 'negative');
                    
                    // Update positions
                    const positionsHtml = portfolio.positions.length === 0 ? 
                        '<div>No active positions</div>' :
                        portfolio.positions.map(pos => 
                            `<div class="trade-item">${pos.symbol}: ${pos.shares} shares @ $${pos.avg_price.toFixed(2)}</div>`
                        ).join('');
                    document.getElementById('current-positions').innerHTML = positionsHtml;
                }
                
                // Fetch trades data
                const tradesResponse = await fetch('/api/trades');
                if (tradesResponse.ok) {
                    const trades = await tradesResponse.json();
                    
                    const tradesHtml = trades.length === 0 ?
                        '<div>No trades yet</div>' :
                        trades.slice(-3).reverse().map(trade =>
                            `<div class="trade-item">${trade.symbol} ${trade.action} ${trade.shares} @ $${trade.price.toFixed(2)}</div>`
                        ).join('');
                    document.getElementById('recent-trades').innerHTML = tradesHtml;
                }
                
                // Fetch learning data
                const learningResponse = await fetch('/api/learning');
                if (learningResponse.ok) {
                    const learning = await learningResponse.json();
                    document.getElementById('genius-score').textContent = learning.avg_genius_score.toFixed(1) + '/10';
                }
                
                // Fetch performance data
                const perfResponse = await fetch('/api/performance');
                if (perfResponse.ok) {
                    const perf = await perfResponse.json();
                    if (perf.length > 0) {
                        const latest = perf[perf.length - 1];
                        document.getElementById('win-rate').textContent = (latest.win_rate * 100).toFixed(1) + '%';
                        document.getElementById('total-trades').textContent = latest.total_trades;
                    }
                }
                
            } catch (error) {
                console.error('Error updating dashboard:', error);
            }
        }
        
        // Update immediately and then every 10 seconds
        updateDashboard();
        setInterval(updateDashboard, 10000);
    </script>
</body>
</html>
        '''
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())

def start_live_data_server(port=8081):
    """Start live data API server."""
    
    server = HTTPServer(('localhost', port), LiveDataHandler)
    print(f"Live Data API running at http://localhost:{port}")
    
    # Run in background thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    return server

if __name__ == '__main__':
    server = start_live_data_server()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.shutdown()