"""
Simple standalone API server for the trading bot dashboard.
"""

from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import random

app = Flask(__name__)
CORS(app)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get trading bot status."""
    return jsonify({
        'running': True,
        'timestamp': datetime.now().isoformat(),
        'components': {
            'broker_manager': {'status': 'connected'},
            'data_manager': {'status': 'active'},
            'strategy_manager': {'status': 'running'},
            'risk_manager': {'status': 'monitoring'}
        }
    })

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance metrics."""
    return jsonify({
        'total_pnl': round(random.uniform(1000, 2000), 2),
        'win_rate': round(random.uniform(0.6, 0.8), 3),
        'sharpe_ratio': round(random.uniform(1.2, 1.8), 2),
        'max_drawdown': round(random.uniform(0.05, 0.15), 3),
        'total_trades': random.randint(100, 200)
    })

@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Get recent trades."""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    sides = ['buy', 'sell']
    strategies = ['momentum', 'mean_reversion', 'sentiment']
    
    trades = []
    for i in range(10):
        trades.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': random.choice(symbols),
            'side': random.choice(sides),
            'quantity': random.randint(50, 200),
            'price': round(random.uniform(100, 300), 2),
            'strategy': random.choice(strategies)
        })
    
    return jsonify(trades)

if __name__ == '__main__':
    print("Starting API server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)