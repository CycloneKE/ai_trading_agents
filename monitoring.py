"""
Monitoring service for health checks and metrics.
"""

import threading
import time
import json
import logging
import socket
import os
import platform
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    from prometheus_metrics import get_prometheus_metrics
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    get_prometheus_metrics = None

logger = logging.getLogger(__name__)

class MonitoringService:
    """
    Service for monitoring system health and metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the monitoring service.
        
        Args:
            config: Configuration dict
        """
        self.config = config
        self.enabled = config.get('enabled', True) and PSUTIL_AVAILABLE
        self.port = config.get('port', 8080)
        self.metrics_interval = config.get('metrics_interval', 15)  # seconds
        self.system_metrics_enabled = config.get('system_metrics_enabled', True) and PSUTIL_AVAILABLE
        
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available. System metrics disabled. Install psutil to enable monitoring.")
            self.enabled = False
        
        # Metrics storage
        self.metrics = {
            'system': {},
            'trading': {
                'orders': [],
                'positions': [],
                'portfolio_value': [],
                'cash': [],
                'equity': []
            },
            'risk': {
                'portfolio_var': [],
                'max_drawdown': [],
                'sharpe_ratio': []
            },
            'performance': {
                'strategy_returns': {},
                'ensemble_returns': []
            }
        }
        
        # Health checks
        self.health_checks = {}
        
        # HTTP server
        self.server = None
        self.server_thread = None
        self.is_running = False
        
        # Metrics collection thread
        self.metrics_thread = None
        
        logger.info("Monitoring service initialized")
    
    def start(self, port: Optional[int] = None):
        """
        Start the monitoring service.
        
        Args:
            port: Optional port override
        """
        if not self.enabled:
            logger.info("Monitoring service is disabled")
            return
        
        if port is not None:
            self.port = port
        
        try:
            # Start HTTP server
            self.server = HTTPServer(('0.0.0.0', self.port), self._create_request_handler())
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            
            # Start metrics collection
            self.metrics_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
            self.metrics_thread.start()
            
            self.is_running = True
            logger.info(f"Monitoring service started on port {self.port}")
            
        except Exception as e:
            logger.error(f"Error starting monitoring service: {str(e)}")
    
    def stop(self):
        """Stop the monitoring service."""
        if not self.is_running:
            return
        
        try:
            # Stop HTTP server
            if self.server:
                self.server.shutdown()
                self.server.server_close()
            
            self.is_running = False
            logger.info("Monitoring service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring service: {str(e)}")
    
    def register_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """
        Register a health check function.
        
        Args:
            name: Health check name
            check_func: Function that returns health status
        """
        self.health_checks[name] = check_func
        logger.debug(f"Registered health check: {name}")
    
    def record_trade(self, action: str, symbol: str, strategy: str, quantity: float, price: float):
        """
        Record a trade for monitoring.
        
        Args:
            action: Trade action ('buy' or 'sell')
            symbol: Trading symbol
            strategy: Strategy name
            quantity: Trade quantity
            price: Trade price
        """
        try:
            trade = {
                'action': action,
                'symbol': symbol,
                'strategy': strategy,
                'quantity': quantity,
                'price': price,
                'value': quantity * price,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.metrics['trading']['orders'].append(trade)
            
            # Keep only last 100 trades
            if len(self.metrics['trading']['orders']) > 100:
                self.metrics['trading']['orders'] = self.metrics['trading']['orders'][-100:]
                
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
    
    def update_portfolio(self, portfolio_value: float, cash: float, equity: float, positions: List[Dict[str, Any]]):
        """
        Update portfolio metrics.
        
        Args:
            portfolio_value: Total portfolio value
            cash: Available cash
            equity: Total equity
            positions: List of positions
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            self.metrics['trading']['portfolio_value'].append({
                'value': portfolio_value,
                'timestamp': timestamp
            })
            
            self.metrics['trading']['cash'].append({
                'value': cash,
                'timestamp': timestamp
            })
            
            self.metrics['trading']['equity'].append({
                'value': equity,
                'timestamp': timestamp
            })
            
            self.metrics['trading']['positions'] = positions
            
            # Keep only last 1000 data points
            for key in ['portfolio_value', 'cash', 'equity']:
                if len(self.metrics['trading'][key]) > 1000:
                    self.metrics['trading'][key] = self.metrics['trading'][key][-1000:]
                    
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {str(e)}")
    
    def update_risk_metrics(self, risk_metrics: Dict[str, float]):
        """
        Update risk metrics.
        
        Args:
            risk_metrics: Dict of risk metrics
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            if 'portfolio_var' in risk_metrics:
                self.metrics['risk']['portfolio_var'].append({
                    'value': risk_metrics['portfolio_var'],
                    'timestamp': timestamp
                })
            
            if 'max_drawdown' in risk_metrics:
                self.metrics['risk']['max_drawdown'].append({
                    'value': risk_metrics['max_drawdown'],
                    'timestamp': timestamp
                })
            
            if 'sharpe_ratio' in risk_metrics:
                self.metrics['risk']['sharpe_ratio'].append({
                    'value': risk_metrics['sharpe_ratio'],
                    'timestamp': timestamp
                })
            
            # Keep only last 1000 data points
            for key in self.metrics['risk']:
                if len(self.metrics['risk'][key]) > 1000:
                    self.metrics['risk'][key] = self.metrics['risk'][key][-1000:]
                    
        except Exception as e:
            logger.error(f"Error updating risk metrics: {str(e)}")
    
    def update_strategy_performance(self, strategy_name: str, returns: float):
        """
        Update strategy performance metrics.
        
        Args:
            strategy_name: Strategy name
            returns: Strategy returns
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            if strategy_name not in self.metrics['performance']['strategy_returns']:
                self.metrics['performance']['strategy_returns'][strategy_name] = []
            
            self.metrics['performance']['strategy_returns'][strategy_name].append({
                'value': returns,
                'timestamp': timestamp
            })
            
            # Keep only last 1000 data points
            if len(self.metrics['performance']['strategy_returns'][strategy_name]) > 1000:
                self.metrics['performance']['strategy_returns'][strategy_name] = \
                    self.metrics['performance']['strategy_returns'][strategy_name][-1000:]
                    
        except Exception as e:
            logger.error(f"Error updating strategy performance: {str(e)}")
    
    def update_ensemble_performance(self, returns: float):
        """
        Update ensemble performance metrics.
        
        Args:
            returns: Ensemble returns
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            self.metrics['performance']['ensemble_returns'].append({
                'value': returns,
                'timestamp': timestamp
            })
            
            # Keep only last 1000 data points
            if len(self.metrics['performance']['ensemble_returns']) > 1000:
                self.metrics['performance']['ensemble_returns'] = \
                    self.metrics['performance']['ensemble_returns'][-1000:]
                    
        except Exception as e:
            logger.error(f"Error updating ensemble performance: {str(e)}")
    
    def _collect_metrics_loop(self):
        """Background thread for collecting system metrics."""
        while self.is_running:
            try:
                if self.system_metrics_enabled:
                    self._collect_system_metrics()
                
                time.sleep(self.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
                time.sleep(self.metrics_interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            timestamp = datetime.utcnow().isoformat()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_percent = memory.percent
            memory_used = memory.used
            memory_total = memory.total
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_used_percent = disk.percent
            disk_used = disk.used
            disk_total = disk.total
            
            # Network stats
            net_io = psutil.net_io_counters()
            bytes_sent = net_io.bytes_sent
            bytes_recv = net_io.bytes_recv
            
            # Process info
            process = psutil.Process(os.getpid())
            process_cpu = process.cpu_percent(interval=1)
            process_memory = process.memory_info().rss
            
            # Store metrics
            self.metrics['system'] = {
                'timestamp': timestamp,
                'cpu': {
                    'percent': cpu_percent
                },
                'memory': {
                    'percent': memory_used_percent,
                    'used': memory_used,
                    'total': memory_total
                },
                'disk': {
                    'percent': disk_used_percent,
                    'used': disk_used,
                    'total': disk_total
                },
                'network': {
                    'bytes_sent': bytes_sent,
                    'bytes_recv': bytes_recv
                },
                'process': {
                    'cpu_percent': process_cpu,
                    'memory_rss': process_memory
                },
                'host': {
                    'hostname': socket.gethostname(),
                    'platform': platform.platform()
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    def _run_server(self):
        """Run the HTTP server."""
        try:
            self.server.serve_forever()
        except Exception as e:
            if self.is_running:
                logger.error(f"HTTP server error: {str(e)}")
    
    def _create_request_handler(self):
        """Create HTTP request handler class with access to monitoring service."""
        monitoring_service = self
        
        class MonitoringRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Redirect logs to our logger
                logger.debug(format % args)
            
            def do_GET(self):
                try:
                    # Parse URL
                    parsed_url = urlparse(self.path)
                    path = parsed_url.path
                    query = parse_qs(parsed_url.query)
                    
                    # Handle different endpoints
                    if path == '/health':
                        self._handle_health()
                    elif path == '/metrics':
                        if 'prometheus' in query.get('format', []):
                            self._handle_prometheus_metrics()
                        else:
                            self._handle_metrics(query)
                    elif path == '/prometheus':
                        self._handle_prometheus_metrics()
                    elif path == '/status':
                        self._handle_status()
                    elif path == '/portfolio':
                        self._handle_portfolio()
                    elif path == '/trades':
                        self._handle_trades()
                    elif path == '/':
                        self._handle_index()
                    else:
                        self.send_error(404, "Not Found")
                        
                except Exception as e:
                    logger.error(f"Error handling request: {str(e)}")
                    self.send_error(500, "Internal Server Error")
            
            def _handle_health(self):
                """Handle health check endpoint."""
                health_status = {
                    'status': 'ok',
                    'timestamp': datetime.utcnow().isoformat(),
                    'checks': {}
                }
                
                # Run all registered health checks
                all_healthy = True
                for name, check_func in monitoring_service.health_checks.items():
                    try:
                        result = check_func()
                        health_status['checks'][name] = result
                        
                        if isinstance(result, dict) and result.get('status') == 'error':
                            all_healthy = False
                    except Exception as e:
                        health_status['checks'][name] = {
                            'status': 'error',
                            'error': str(e)
                        }
                        all_healthy = False
                
                if not all_healthy:
                    health_status['status'] = 'error'
                
                self._send_json_response(health_status)
            
            def _handle_metrics(self, query):
                """Handle metrics endpoint."""
                # Filter metrics based on query
                metrics_type = query.get('type', ['all'])[0]
                
                if metrics_type == 'all':
                    metrics = monitoring_service.metrics
                elif metrics_type in monitoring_service.metrics:
                    metrics = {metrics_type: monitoring_service.metrics[metrics_type]}
                else:
                    metrics = {'error': f"Unknown metrics type: {metrics_type}"}
                
                self._send_json_response(metrics)
                
            def _handle_prometheus_metrics(self):
                """Handle Prometheus metrics endpoint."""
                if not PROMETHEUS_AVAILABLE:
                    self.send_error(503, "Prometheus metrics not available")
                    return
                    
                # Get Prometheus metrics
                prometheus_metrics = get_prometheus_metrics()
                
                # Update Prometheus metrics from monitoring service
                self._update_prometheus_metrics(prometheus_metrics)
                
                # Get metrics in Prometheus format
                metrics_text = prometheus_metrics.get_prometheus_metrics()
                
                # Send response
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(metrics_text.encode())
                
            def _update_prometheus_metrics(self, prometheus_metrics):
                """Update Prometheus metrics from monitoring service."""
                try:
                    # Portfolio metrics
                    portfolio_value = 0.0
                    cash = 0.0
                    equity = 0.0
                    total_pnl = 0.0
                    daily_pnl = 0.0
                    positions = []
                    
                    if monitoring_service.metrics['trading']['portfolio_value']:
                        portfolio_value = monitoring_service.metrics['trading']['portfolio_value'][-1]['value']
                    
                    if monitoring_service.metrics['trading']['cash']:
                        cash = monitoring_service.metrics['trading']['cash'][-1]['value']
                    
                    if monitoring_service.metrics['trading']['equity']:
                        equity = monitoring_service.metrics['trading']['equity'][-1]['value']
                    
                    positions = monitoring_service.metrics['trading']['positions']
                    
                    # Calculate PnL
                    if positions:
                        total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)
                        daily_pnl = sum(pos.get('daily_pnl', 0) for pos in positions)
                    
                    prometheus_metrics.update_portfolio_metrics(
                        portfolio_value=portfolio_value,
                        cash=cash,
                        equity=equity,
                        total_pnl=total_pnl,
                        daily_pnl=daily_pnl,
                        positions=positions
                    )
                    
                    # Risk metrics
                    max_drawdown = 0.0
                    sharpe_ratio = 0.0
                    portfolio_var = 0.0
                    volatility = 0.0
                    
                    if monitoring_service.metrics['risk']['max_drawdown']:
                        max_drawdown = monitoring_service.metrics['risk']['max_drawdown'][-1]['value']
                    
                    if monitoring_service.metrics['risk']['sharpe_ratio']:
                        sharpe_ratio = monitoring_service.metrics['risk']['sharpe_ratio'][-1]['value']
                    
                    if monitoring_service.metrics['risk']['portfolio_var']:
                        portfolio_var = monitoring_service.metrics['risk']['portfolio_var'][-1]['value']
                    
                    prometheus_metrics.update_risk_metrics(
                        max_drawdown=max_drawdown,
                        sharpe_ratio=sharpe_ratio,
                        portfolio_var=portfolio_var,
                        volatility=volatility
                    )
                    
                    # Trade metrics
                    trade_count = len(monitoring_service.metrics['trading']['orders'])
                    win_count = sum(1 for trade in monitoring_service.metrics['trading']['orders'] 
                                  if (trade['action'] == 'sell' and trade['value'] > 0) or 
                                     (trade['action'] == 'buy' and trade['value'] < 0))
                    
                    prometheus_metrics.update_trade_metrics(
                        trade_count=trade_count,
                        win_count=win_count
                    )
                    
                    # Strategy returns
                    for strategy_name, returns in monitoring_service.metrics['performance']['strategy_returns'].items():
                        if returns:
                            prometheus_metrics.update_strategy_returns(
                                strategy_name=strategy_name,
                                returns=returns[-1]['value']
                            )
                    
                    # System metrics
                    system = monitoring_service.metrics['system']
                    cpu_usage = system.get('cpu', {}).get('percent', 0.0)
                    memory_usage = system.get('memory', {}).get('percent', 0.0)
                    disk_usage = system.get('disk', {}).get('percent', 0.0)
                    
                    prometheus_metrics.update_system_metrics(
                        cpu_usage=cpu_usage,
                        memory_usage=memory_usage,
                        disk_usage=disk_usage
                    )
                    
                except Exception as e:
                    logger.error(f"Error updating Prometheus metrics: {str(e)}")
            
            def _handle_status(self):
                """Handle status endpoint."""
                status = {
                    'status': 'running' if monitoring_service.is_running else 'stopped',
                    'timestamp': datetime.utcnow().isoformat(),
                    'uptime': time.time() - psutil.boot_time(),
                    'process_uptime': time.time() - psutil.Process(os.getpid()).create_time(),
                    'system': monitoring_service.metrics['system']
                }
                
                self._send_json_response(status)
            
            def _handle_portfolio(self):
                """Handle portfolio endpoint."""
                try:
                    # Get paper trading state
                    state_file = "data/paper_trading_state.json"
                    portfolio_data = {
                        'cash': 0,
                        'positions': {},
                        'market_prices': {},
                        'trade_history': [],
                        'total_value': 0,
                        'last_updated': None
                    }
                    
                    if os.path.exists(state_file):
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                        
                        portfolio_data.update(state)
                        
                        # Calculate total portfolio value
                        total_value = state.get('cash', 0)
                        for symbol, position in state.get('positions', {}).items():
                            if isinstance(position, dict):
                                quantity = position.get('quantity', 0)
                                current_price = state.get('market_prices', {}).get(symbol, position.get('avg_price', 0))
                                total_value += quantity * current_price
                        
                        portfolio_data['total_value'] = total_value
                    
                    self._send_json_response(portfolio_data)
                    
                except Exception as e:
                    self._send_json_response({'error': str(e)})
            
            def _handle_trades(self):
                """Handle trades endpoint."""
                try:
                    # Get recent trades from monitoring service
                    trades = monitoring_service.metrics['trading']['orders']
                    
                    # Also get trades from paper trading state
                    state_file = "data/paper_trading_state.json"
                    if os.path.exists(state_file):
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                        
                        trade_history = state.get('trade_history', [])
                        
                        # Combine trades
                        all_trades = {
                            'recent_orders': trades,
                            'trade_history': trade_history,
                            'total_trades': len(trade_history)
                        }
                    else:
                        all_trades = {
                            'recent_orders': trades,
                            'trade_history': [],
                            'total_trades': 0
                        }
                    
                    self._send_json_response(all_trades)
                    
                except Exception as e:
                    self._send_json_response({'error': str(e)})
            
            def _handle_index(self):
                """Handle index page."""
                html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>AI Trading Agent Monitoring</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                        .container { max-width: 1200px; margin: 0 auto; }
                        h1 { color: #333; text-align: center; }
                        .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
                        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                        .card h3 { margin-top: 0; color: #555; }
                        .endpoint { margin-bottom: 10px; }
                        .endpoint a { color: #0066cc; text-decoration: none; }
                        .endpoint a:hover { text-decoration: underline; }
                        .status { padding: 5px 10px; border-radius: 4px; color: white; font-weight: bold; }
                        .status.ok { background: #28a745; }
                        .status.warning { background: #ffc107; color: #333; }
                        .status.error { background: #dc3545; }
                        #portfolio-data, #trades-data { margin-top: 10px; }
                        .metric { display: flex; justify-content: space-between; margin: 5px 0; }
                        .metric-value { font-weight: bold; }
                    </style>
                    <script>
                        async function loadPortfolio() {
                            try {
                                const response = await fetch('/portfolio');
                                const data = await response.json();
                                document.getElementById('portfolio-data').innerHTML = `
                                    <div class="metric"><span>Cash:</span><span class="metric-value">$${data.cash?.toLocaleString() || 0}</span></div>
                                    <div class="metric"><span>Total Value:</span><span class="metric-value">$${data.total_value?.toLocaleString() || 0}</span></div>
                                    <div class="metric"><span>Positions:</span><span class="metric-value">${Object.keys(data.positions || {}).length}</span></div>
                                    <div class="metric"><span>Last Updated:</span><span class="metric-value">${data.last_updated || 'Never'}</span></div>
                                `;
                            } catch (e) {
                                document.getElementById('portfolio-data').innerHTML = '<div style="color: red;">Error loading portfolio</div>';
                            }
                        }
                        
                        async function loadTrades() {
                            try {
                                const response = await fetch('/trades');
                                const data = await response.json();
                                document.getElementById('trades-data').innerHTML = `
                                    <div class="metric"><span>Total Trades:</span><span class="metric-value">${data.total_trades || 0}</span></div>
                                    <div class="metric"><span>Recent Orders:</span><span class="metric-value">${data.recent_orders?.length || 0}</span></div>
                                `;
                            } catch (e) {
                                document.getElementById('trades-data').innerHTML = '<div style="color: red;">Error loading trades</div>';
                            }
                        }
                        
                        async function loadHealth() {
                            try {
                                const response = await fetch('/health');
                                const data = await response.json();
                                const statusEl = document.getElementById('health-status');
                                statusEl.className = `status ${data.status}`;
                                statusEl.textContent = data.status.toUpperCase();
                                
                                const componentsEl = document.getElementById('components-count');
                                componentsEl.textContent = Object.keys(data.checks || {}).length;
                            } catch (e) {
                                document.getElementById('health-status').innerHTML = '<span class="status error">ERROR</span>';
                            }
                        }
                        
                        // Load data on page load and refresh every 30 seconds
                        window.onload = function() {
                            loadPortfolio();
                            loadTrades();
                            loadHealth();
                            setInterval(() => {
                                loadPortfolio();
                                loadTrades();
                                loadHealth();
                            }, 30000);
                        };
                    </script>
                </head>
                <body>
                    <div class="container">
                        <h1>AI Trading Agent Dashboard</h1>
                        
                        <div class="dashboard">
                            <div class="card">
                                <h3>System Health</h3>
                                <div class="metric">
                                    <span>Status:</span>
                                    <span id="health-status" class="status ok">Loading...</span>
                                </div>
                                <div class="metric">
                                    <span>Components:</span>
                                    <span class="metric-value" id="components-count">-</span>
                                </div>
                            </div>
                            
                            <div class="card">
                                <h3>Paper Trading Portfolio</h3>
                                <div id="portfolio-data">Loading...</div>
                            </div>
                            
                            <div class="card">
                                <h3>Trading Activity</h3>
                                <div id="trades-data">Loading...</div>
                            </div>
                            
                            <div class="card">
                                <h3>API Endpoints</h3>
                                <div class="endpoint"><a href="/health">/health</a> - Health check</div>
                                <div class="endpoint"><a href="/portfolio">/portfolio</a> - Portfolio data</div>
                                <div class="endpoint"><a href="/trades">/trades</a> - Trading history</div>
                                <div class="endpoint"><a href="/metrics">/metrics</a> - All metrics</div>
                                <div class="endpoint"><a href="/status">/status</a> - System status</div>
                            </div>
                        </div>
                    </div>
                </body>
                </html>
                """
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
            
            def _send_json_response(self, data):
                """Send JSON response."""
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data, default=str).encode())
        
        return MonitoringRequestHandler

# Singleton instance
_monitoring_service = None

def get_monitoring_service(config: Optional[Dict[str, Any]] = None) -> MonitoringService:
    """
    Get the monitoring service instance.
    
    Args:
        config: Optional configuration dict
        
    Returns:
        MonitoringService instance
    """
    global _monitoring_service
    
    if _monitoring_service is None:
        if config is None:
            config = {
                'enabled': True,
                'port': 8080,
                'metrics_interval': 15,
                'system_metrics_enabled': True
            }
        
        _monitoring_service = MonitoringService(config)
    
    return _monitoring_service