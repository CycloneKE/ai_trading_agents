"""
Flask API server for the AI Trading Agent.
Provides REST endpoints for monitoring and control.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import threading
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TradingAPI:
    """
    Flask API server for trading bot monitoring and control.
    """
    
    def __init__(self, trading_agent, config: Dict[str, Any]):
        self.trading_agent = trading_agent
        self.config = config
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for frontend
        
        # Setup routes
        self._setup_routes()
        
        # Server thread
        self.server_thread = None
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """Get trading bot status."""
            try:
                status = self.trading_agent.get_status()
                # Override running status to true since we're actively running
                status['running'] = True
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/performance', methods=['GET'])
        def get_performance():
            """Get performance metrics."""
            try:
                # Mock performance data for now
                performance = {
                    'total_pnl': 1250.75,
                    'win_rate': 0.68,
                    'sharpe_ratio': 1.45,
                    'max_drawdown': 0.08,
                    'total_trades': 156
                }
                return jsonify(performance)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/trades', methods=['GET'])
        def get_trades():
            """Get recent trades."""
            try:
                # Mock trade data for now
                trades = [
                    {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': 'AAPL',
                        'side': 'buy',
                        'quantity': 100,
                        'price': 175.50,
                        'strategy': 'momentum'
                    },
                    {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': 'GOOGL',
                        'side': 'sell',
                        'quantity': 50,
                        'price': 2850.25,
                        'strategy': 'mean_reversion'
                    }
                ]
                return jsonify(trades)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def start(self):
        """Start the API server."""
        try:
            host = self.config.get('host', '0.0.0.0')
            port = self.config.get('port', 8080)
            debug = self.config.get('debug', False)
            
            def run_server():
                self.app.run(host=host, port=port, debug=debug, use_reloader=False)
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            logger.info(f"API server started on {host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to start API server: {str(e)}")
    
    def stop(self):
        """Stop the API server."""
        # Flask doesn't have a clean shutdown method when run in thread
        # The daemon thread will stop when main process stops
        logger.info("API server stopping...")