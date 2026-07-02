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
from typing import Dict, Any, List
from src.agent.sentiment_analyzer import FinancialSentimentAnalyzer
from src.api.auth import token_required

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
    from .auth import token_required, verify_password, create_token, USERS_FILE, json
    AUTH_AVAILABLE = True
except Exception as e:
    AUTH_AVAILABLE = False
    logger.critical(
        "Auth module failed to load (%s). API is locked down: all protected "
        "routes will return 503 until SECRET_KEY is set and auth imports cleanly.",
        e,
    )

    # Fail CLOSED: never serve protected routes without authentication. Reject
    # every protected request rather than silently making the API public.
    def token_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            return jsonify({
                'error': 'Authentication unavailable; service is locked down'
            }), 503
        return decorated


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

        # Initializing sentiment analyzer for news feed enrichment
        self.sentiment_analyzer = FinancialSentimentAnalyzer(config)

        # Response cache: with several traders polling the dashboard every
        # 20-30s, per-request vendor fetches (news, sector quotes) would
        # multiply upstream API calls by the number of clients. Each cached
        # entry is shared by all clients until its TTL expires.
        self._response_cache: Dict[str, Any] = {}
        self._response_cache_lock = threading.Lock()

        # Define API routes
        self._setup_routes()

        # Server thread
        self.server_thread = None

    def _cached(self, key: str, ttl_seconds: int, producer):
        """Return the cached value for ``key`` if younger than ``ttl_seconds``,
        else call ``producer()`` and cache its result. Errors from the
        producer are not cached."""
        now = time.time()
        with self._response_cache_lock:
            entry = self._response_cache.get(key)
            if entry and now - entry['at'] < ttl_seconds:
                return entry['value']
        value = producer()
        with self._response_cache_lock:
            self._response_cache[key] = {'value': value, 'at': now}
        return value

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

        @self.app.route('/api/login', methods=['POST'])
        @require_rate_limit
        def login():
            """Authenticate user and return JWT token."""
            if not AUTH_AVAILABLE:
                return jsonify({'error': 'Authentication service unavailable'}), 503
            
            data = request.get_json()
            if not data or not data.get('username') or not data.get('password'):
                return jsonify({'error': 'Missing credentials'}), 400
            
            username = data.get('username')
            password = data.get('password')
            
            try:
                if not os.path.exists(USERS_FILE):
                    return jsonify({'error': 'User database not initialized'}), 503
                
                with open(USERS_FILE, 'r') as f:
                    users = json.load(f)
                
                if username in users and verify_password(users[username], password):
                    token = create_token(username)
                    return jsonify({
                        'token': token,
                        'message': 'Login successful',
                        'username': username
                    })
                
                return jsonify({'error': 'Invalid username or password'}), 401
            except Exception as e:
                logger.error(f"Login error: {e}")
                return jsonify({'error': 'Internal server error during login'}), 500

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
                                # AccountInfo has no currency field; Alpaca accounts are USD
                                'currency': getattr(account, 'currency', 'USD') or 'USD'
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

        @self.app.route('/api/positions', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_positions():
            """Alias for portfolio positions."""
            try:
                broker = self.trading_agent.components.get('broker_manager')
                if broker:
                    primary = broker.get_broker()
                    if primary:
                        positions = primary.get_positions()
                        return jsonify([
                            {
                                'symbol': p.symbol,
                                'quantity': p.quantity,
                                'avg_entry_price': p.avg_entry_price,
                                'current_price': p.current_price,
                                'unrealized_pl': p.unrealized_pl,
                                'unrealized_pl_pct': (p.unrealized_pl / (p.avg_entry_price * p.quantity)) * 100 if p.avg_entry_price and p.quantity else 0
                            }
                            for p in (positions or [])
                        ])
                return jsonify([])
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/risk-metrics', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_risk_metrics():
            """Get current risk metrics."""
            try:
                risk_manager = getattr(self.trading_agent, 'risk_manager', None)
                if risk_manager:
                    return jsonify(risk_manager.get_risk_report())
                return jsonify({
                    'status': 'nominal',
                    'portfolio_var': 0,
                    'drawdown': 0,
                    'is_risk_exceeded': False
                })
            except Exception as e:
                logger.error(f"Error getting risk metrics: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/alerts', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_alerts():
            """Get recent system alerts."""
            # Placeholder for alerts system
            return jsonify([
                {
                    'id': 1,
                    'type': 'info',
                    'message': 'System initialized successfully',
                    'timestamp': datetime.utcnow().isoformat()
                }
            ])

        @self.app.route('/api/news-feed', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_news_feed():
            """Get latest news and sentiment (cached, shared across clients)."""
            def produce():
                data_manager = self.trading_agent.components.get('data_manager')
                news = []
                if data_manager and 'real_data' in data_manager.connectors:
                    real_connector = data_manager.connectors['real_data']
                    news = real_connector.get_market_news()

                # Enrich news with live sentiment analysis if available
                # Using faster VADER/Keyword logic for low-latency dashboard updates
                for item in news:
                    if isinstance(item, dict) and 'title' in item:
                        text_to_analyze = item.get('title', '') + " " + item.get('summary', '')
                        sentiment_result = self.sentiment_analyzer.analyze_sentiment(text_to_analyze)
                        item['sentiment'] = sentiment_result.get('combined', {}).get('overall', 0.0)
                        item['sentiment_label'] = sentiment_result.get('combined', {}).get('label', 'neutral')
                        item['sentiment_confidence'] = sentiment_result.get('combined', {}).get('confidence', 0.0)
                return news

            try:
                return jsonify(self._cached('news_feed', 60, produce))
            except Exception as e:
                logger.error(f"Error getting news feed: {e}")
                return jsonify([])

        @self.app.route('/api/model-performance', methods=['GET'])
        @self.app.route('/api/strategy-performance', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_strategy_performance():
            """Get performance broken down by strategy."""
            try:
                # Mock performance per strategy until database integration is complete
                # In a real app, we'd query performance per strategy over time
                return jsonify({
                    'momentum': {'win_rate': 0.65, 'sharpe': 1.8},
                    'mean_reversion': {'win_rate': 0.58, 'sharpe': 1.4},
                    'rsi_strategy': {'win_rate': 0.52, 'sharpe': 1.1}
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/system-health', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_system_health():
            """Get detailed system health."""
            try:
                health = {
                    'status': 'healthy',
                    'components': {},
                    'uptime': time.time() - (getattr(self.trading_agent, 'start_time', time.time())),
                    'memory_usage': 'nominal'
                }
                for name, component in self.trading_agent.components.items():
                    if hasattr(component, 'get_status'):
                        health['components'][name] = component.get_status()
                return jsonify(health)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/agent-activity', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_agent_activity():
            """Get recent agent actions."""
            return jsonify([])

        @self.app.route('/api/correlation-matrix', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_correlation_matrix():
            return jsonify({})

        @self.app.route('/api/market-heatmap', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_market_heatmap():
            """Get sector performance heatmap (cached: 8 ETF quotes per refresh)."""
            def produce():
                data_manager = self.trading_agent.components.get('data_manager')
                if data_manager and 'real_data' in data_manager.connectors:
                    real_connector = data_manager.connectors['real_data']
                    return real_connector.get_sector_performance()
                return []

            try:
                return jsonify(self._cached('market_heatmap', 120, produce))
            except Exception as e:
                logger.error(f"Error getting market heatmap: {e}")
                return jsonify([])

        @self.app.route('/api/economic-calendar', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_economic_calendar():
            return jsonify([])

        @self.app.route('/api/nse-market', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_nse_market():
            """Get Kenya NSE market data: quotes, movers, sectors (cached)."""
            try:
                data_manager = self.trading_agent.components.get('data_manager')
                nse = data_manager.connectors.get('nse') if data_manager else None

                if not nse:
                    return jsonify({'error': 'NSE connector not available', 'quotes': [], 'movers': {}, 'sectors': [], 'status': {}}), 200

                def produce():
                    # Include periodic scraper status if available
                    scraper_status = {}
                    if hasattr(self.trading_agent, 'nse_scraper') and self.trading_agent.nse_scraper:
                        scraper_status = self.trading_agent.nse_scraper.get_status()
                    return {
                        'quotes': nse.get_all_quotes(),
                        'movers': nse.get_top_movers(),
                        'sectors': nse.get_sector_performance(),
                        'status': nse.get_status(),
                        'scraper': scraper_status,
                        'kes_usd_rate': nse.get_kes_usd_rate(),
                        'market_open': nse.is_market_open(),
                    }

                return jsonify(self._cached('nse_market', 60, produce))
            except Exception as e:
                logger.error(f"Error getting NSE market data: {e}")
                return jsonify({'error': str(e), 'quotes': [], 'movers': {}, 'sectors': []}), 500

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