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
# NOTE: token_required is imported via the guarded try/except below, NOT here.
# A top-level import defeats the fail-closed guard: auth.py raises RuntimeError
# (not ImportError) when SECRET_KEY is unset, which would crash the whole agent
# at import instead of starting with protected routes locked down.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------
_request_counts = defaultdict(list)
_rate_lock = threading.Lock()
RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', '100'))  # requests per minute
# When behind a trusted reverse proxy (Caddy), remote_addr is always the
# proxy, so all clients would share one bucket. Set TRUST_PROXY=1 to key
# the limiter on the real client via X-Forwarded-For instead.
TRUST_PROXY = os.getenv('TRUST_PROXY', '').lower() in ('1', 'true', 'yes')


def _client_key() -> str:
    if TRUST_PROXY:
        xff = request.headers.get('X-Forwarded-For', '')
        if xff:
            return xff.split(',')[0].strip()
    return request.remote_addr or '127.0.0.1'


def _rate_limit_check() -> bool:
    """Return True if the request is within rate limits. Thread-safe (waitress
    runs multiple worker threads) and self-pruning to bound memory."""
    client_ip = _client_key()
    now = time.time()
    with _rate_lock:
        recent = [t for t in _request_counts[client_ip] if now - t < 60]
        if len(recent) >= RATE_LIMIT:
            _request_counts[client_ip] = recent
            return False
        recent.append(now)
        _request_counts[client_ip] = recent
        # Prune idle client keys so a scanner can't grow the map unbounded.
        if len(_request_counts) > 1024:
            for k in [k for k, v in _request_counts.items()
                      if not v or now - v[-1] > 120]:
                del _request_counts[k]
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
# Login brute-force protection (per username, independent of the IP limiter:
# the IP limiter alone allows ~144k guesses/day against one account)
# ---------------------------------------------------------------------------
LOGIN_MAX_FAILURES = int(os.getenv('LOGIN_MAX_FAILURES', '5'))
LOGIN_LOCKOUT_SECONDS = int(os.getenv('LOGIN_LOCKOUT_SECONDS', '900'))  # 15 min
_login_failures = defaultdict(list)   # username -> [failure timestamps]
_login_lock = threading.Lock()


def _login_locked(username: str):
    """Return seconds remaining in lockout, or 0 if the account may try."""
    now = time.time()
    with _login_lock:
        recent = [t for t in _login_failures[username]
                  if now - t < LOGIN_LOCKOUT_SECONDS]
        _login_failures[username] = recent
        if len(recent) >= LOGIN_MAX_FAILURES:
            return int(LOGIN_LOCKOUT_SECONDS - (now - recent[0])) + 1
    return 0


def _login_failed(username: str, remote_addr: str):
    with _login_lock:
        _login_failures[username].append(time.time())
        count = len(_login_failures[username])
    logger.warning(f"AUTH FAILURE for '{username}' from {remote_addr} "
                   f"({count}/{LOGIN_MAX_FAILURES} before lockout)")


def _login_succeeded(username: str):
    with _login_lock:
        _login_failures.pop(username, None)


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
            
            username = str(data.get('username'))[:64]
            password = data.get('password')

            lockout = _login_locked(username)
            if lockout:
                logger.warning(f"AUTH LOCKOUT active for '{username}' from {request.remote_addr}")
                return jsonify({'error': 'Too many failed attempts; account temporarily locked',
                                'retry_after': lockout}), 429

            try:
                if not os.path.exists(USERS_FILE):
                    return jsonify({'error': 'User database not initialized'}), 503

                with open(USERS_FILE, 'r') as f:
                    users = json.load(f)

                if username in users and verify_password(users[username], password):
                    _login_succeeded(username)
                    token = create_token(username)
                    return jsonify({
                        'token': token,
                        'message': 'Login successful',
                        'username': username
                    })

                _login_failed(username, request.remote_addr or '?')
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
                # Report the REAL running flag from the agent, not a constant —
                # the dashboard must be able to see when the loop has stopped.
                status = self.trading_agent.get_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting status: {e}")
                return jsonify({'error': 'Failed to get status'}), 500

        @self.app.route('/api/performance', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_performance():
            """Get performance metrics (protected)."""
            try:
                period = request.args.get('period', '1M')
                report = self.trading_agent.performance_analytics.generate_performance_report(period)
                # Flatten the headline numbers the dashboard cards read at the
                # top level (they live under metrics/summary in the report).
                metrics = report.get('metrics', {})
                summary = report.get('summary', {})
                current_value = summary.get('current_value', 0) or 0
                total_return = metrics.get('total_return', 0) or 0
                # Derive dollar P&L from period return and current equity.
                initial = current_value / (1 + total_return) if total_return > -1 else current_value
                report['portfolio_value'] = current_value
                report['total_pnl'] = round(current_value - initial, 2)
                report['win_rate'] = metrics.get('win_rate', 0)
                report['total_trades'] = metrics.get('total_trades', 0)
                return jsonify(report)
            except Exception as e:
                logger.error(f"Error getting performance: {e}")
                return jsonify({'error': 'Failed to generate performance report'}), 500

        @self.app.route('/api/trades', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_trades():
            """Get recent trades (protected)."""
            try:
                # Clamp to non-negative: a negative offset/limit produces
                # confusing wrap-around slices of the trade history.
                offset = max(0, int(request.args.get('offset', 0)))
                limit = min(max(0, int(request.args.get('limit', 100))), 500)  # Cap at 500
                trades = self.trading_agent.performance_analytics.trades_history
                return jsonify(trades[offset:offset + limit])
            except (ValueError, TypeError):
                return jsonify({'error': 'offset and limit must be integers'}), 400
            except Exception as e:
                logger.error(f"Error getting trades: {e}")
                return jsonify({'error': 'Failed to get trades'}), 500

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

        @self.app.route('/api/trading/halt', methods=['POST'])
        @require_rate_limit
        @token_required
        def halt_trading():
            """Kill switch: stop all new orders; optionally flatten the book.

            Body: {"confirm": true, "flatten": false}
            """
            try:
                data = request.get_json(silent=True) or {}
                if data.get('confirm') is not True:
                    return jsonify({'error': 'confirmation required: pass {"confirm": true}'}), 400
                flatten = bool(data.get('flatten', False))
                result = self.trading_agent.halt_trading(
                    flatten=flatten,
                    reason=f"API request (flatten={flatten})")
                logger.warning(f"Kill switch engaged via API: {result}")
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error engaging kill switch: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/trading/resume', methods=['POST'])
        @require_rate_limit
        @token_required
        def resume_trading():
            """Release the kill switch."""
            try:
                data = request.get_json(silent=True) or {}
                if data.get('confirm') is not True:
                    return jsonify({'error': 'confirmation required: pass {"confirm": true}'}), 400
                result = self.trading_agent.resume_trading()
                logger.warning("Kill switch released via API")
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error resuming trading: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/alerts', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_alerts():
            """Get recent system alerts from the risk manager (real, not mock)."""
            try:
                risk_manager = getattr(self.trading_agent, 'risk_manager', None)
                if risk_manager and hasattr(risk_manager, 'get_risk_report'):
                    report = risk_manager.get_risk_report() or {}
                    alerts = report.get('alerts', {})
                    recent = alerts.get('recent', []) if isinstance(alerts, dict) else alerts
                    return jsonify(recent or [])
                return jsonify([])
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                return jsonify([])

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
            """Real per-strategy P&L attribution from the order journal."""
            def produce():
                journal = getattr(self.trading_agent, 'order_journal', None)
                if not journal:
                    return {}
                from src.agent.strategy_attribution import compute_attribution

                def price_lookup(symbol):
                    data_manager = self.trading_agent.components.get('data_manager')
                    real = data_manager.connectors.get('real_data') if data_manager else None
                    if real:
                        quote = real.get_real_time_data(symbol)
                        return (quote or {}).get('price')
                    return None

                return compute_attribution(journal.filled_orders(), price_lookup)

            try:
                return jsonify(self._cached('strategy_performance', 30, produce))
            except Exception as e:
                logger.error(f"Error computing strategy attribution: {e}")
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
                # Prefer waitress: a production WSGI server (Flask's builtin
                # dev server is single-purpose and warns against real use).
                try:
                    from waitress import serve
                    logger.info("Serving API with waitress")
                    serve(self.app, host=host, port=port, threads=8,
                          ident=None)  # ident=None: no Server version header
                except ImportError:
                    logger.warning("waitress not installed; falling back to Flask dev server")
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