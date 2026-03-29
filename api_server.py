"""
Flask API server for the AI Trading Agent.
Provides REST endpoints for monitoring and control.
Secured with JWT authentication, rate limiting, and restricted CORS.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import threading
import time
import os
from collections import defaultdict
from datetime import datetime
from functools import wraps
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------
_request_counts = defaultdict(list)
RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', '100'))  # requests per minute


def _rate_limit_check() -> bool:
    """Return True if the request is within rate limits."""
    client_ip = request.remote_addr or '127.0.0.1'
    now = time.time()
    # Remove entries older than 60 seconds
    _request_counts[client_ip] = [
        t for t in _request_counts[client_ip] if now - t < 60
    ]
    if len(_request_counts[client_ip]) >= RATE_LIMIT:
        return False
    _request_counts[client_ip].append(now)
    return True


def require_rate_limit(f):
    """Rate limiting decorator."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not _rate_limit_check():
            return jsonify({'error': 'Rate limit exceeded', 'retry_after': 60}), 429
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Optional JWT authentication (enabled when auth module is available)
# ---------------------------------------------------------------------------
try:
    from auth import token_required
    AUTH_AVAILABLE = True
except Exception:
    AUTH_AVAILABLE = False
    # Provide a pass-through decorator so the server still starts
    def token_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            return f(*args, **kwargs)
        return decorated
    logger.warning("Auth module not available — API routes are UNPROTECTED. "
                   "Set SECRET_KEY env var and create users to enable JWT auth.")


class TradingAPI:
    """
    Flask API server for trading bot monitoring and control.
    """

    def __init__(self, trading_agent, config: Dict[str, Any]):
        self.trading_agent = trading_agent
        self.config = config
        self.app = Flask(__name__)

        # Restricted CORS — only allow configured origins
        allowed_origins = os.getenv(
            'CORS_ALLOWED_ORIGINS',
            'http://localhost:3000,http://localhost:3001'
        ).split(',')
        CORS(self.app, origins=allowed_origins)

        # Setup routes
        self._setup_routes()

        # Server thread
        self.server_thread = None

    def _setup_routes(self):
        """Setup API routes with authentication and rate limiting."""

        # --- Public health endpoint (no auth required) ---
        @self.app.route('/api/health', methods=['GET'])
        @require_rate_limit
        def health_check():
            """Public health check (no sensitive data)."""
            return jsonify({
                'status': 'ok',
                'timestamp': datetime.utcnow().isoformat()
            })

        # --- Protected endpoints ---
        @self.app.route('/api/status', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_status():
            """Get trading bot status (protected)."""
            try:
                status = self.trading_agent.get_status()
                status['running'] = True
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/performance', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_performance():
            """Get performance metrics (protected)."""
            try:
                period = request.args.get('period', '1M')
                report = self.trading_agent.performance_analytics.generate_performance_report(period)
                return jsonify(report)
            except Exception as e:
                logger.error(f"Error getting performance: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/trades', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_trades():
            """Get recent trades (protected)."""
            try:
                offset = int(request.args.get('offset', 0))
                limit = min(int(request.args.get('limit', 100)), 500)  # Cap at 500
                trades = self.trading_agent.performance_analytics.trades_history
                return jsonify(trades[offset:offset + limit])
            except Exception as e:
                logger.error(f"Error getting trades: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/portfolio', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_portfolio():
            """Get portfolio data (protected)."""
            try:
                broker = self.trading_agent.components.get('broker_manager')
                if broker:
                    primary = broker.get_broker()
                    if primary:
                        account = primary.get_account_info()
                        positions = primary.get_positions()
                        return jsonify({
                            'account': {
                                'cash': account.cash if account else 0,
                                'equity': account.equity if account else 0,
                            },
                            'positions': [
                                {
                                    'symbol': p.symbol,
                                    'quantity': p.quantity,
                                    'avg_entry_price': p.avg_entry_price,
                                    'current_price': p.current_price,
                                    'unrealized_pl': p.unrealized_pl,
                                }
                                for p in (positions or [])
                            ]
                        })
                return jsonify({'error': 'No broker available'}), 503
            except Exception as e:
                logger.error(f"Error getting portfolio: {e}")
                return jsonify({'error': str(e)}), 500

        # --- Error handlers ---
        @self.app.errorhandler(429)
        def rate_limit_exceeded(error):
            return jsonify({'error': 'Rate limit exceeded', 'retry_after': 60}), 429

        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500

    def start(self):
        """Start the API server."""
        try:
            host = self.config.get('host', '0.0.0.0')
            port = self.config.get('port', 5001)

            def run_server():
                # Use threaded=True for better concurrent handling
                self.app.run(
                    host=host,
                    port=port,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                )

            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()

            logger.info(f"API server started on {host}:{port}")
            if AUTH_AVAILABLE:
                logger.info("API endpoints are JWT-protected")
            else:
                logger.warning("API endpoints are UNPROTECTED — configure SECRET_KEY to enable auth")

        except Exception as e:
            logger.error(f"Failed to start API server: {str(e)}")

    def stop(self):
        """Stop the API server."""
        logger.info("API server stopping...")