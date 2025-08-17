"""
REST API Server for Trading Agent
Provides external access to trading system functionality
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import json

logger = logging.getLogger(__name__)

class TradingAPI:
    """REST API server for trading agent"""
    
    def __init__(self, trading_agent, config: Dict[str, Any]):
        self.trading_agent = trading_agent
        self.config = config
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web interfaces
        
        # API configuration
        self.port = config.get('port', 5000)
        self.host = config.get('host', '0.0.0.0')
        self.debug = config.get('debug', False)
        
        # Setup routes
        self._setup_routes()
        
        # Server thread
        self.server_thread = None
        self.running = False
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            try:
                status = self.trading_agent.get_status()
                return jsonify({
                    'status': 'healthy',
                    'timestamp': datetime.utcnow().isoformat(),
                    'agent_running': status.get('running', False)
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }), 500
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """Get comprehensive system status"""
            try:
                status = self.trading_agent.get_status()
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/portfolio', methods=['GET'])
        def get_portfolio():
            """Get portfolio information"""
            try:
                if hasattr(self.trading_agent, 'components') and 'broker_manager' in self.trading_agent.components:
                    broker_manager = self.trading_agent.components['broker_manager']
                    
                    # Get account info
                    account_info = broker_manager.get_account_info()
                    positions = broker_manager.get_positions()
                    
                    portfolio = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'account': {
                            'cash': account_info.cash if account_info else 0,
                            'equity': account_info.equity if account_info else 0,
                            'buying_power': account_info.buying_power if account_info else 0
                        },
                        'positions': [
                            {
                                'symbol': pos.symbol,
                                'quantity': pos.quantity,
                                'market_value': pos.market_value,
                                'unrealized_pnl': pos.unrealized_pnl,
                                'avg_entry_price': pos.avg_entry_price
                            }
                            for pos in positions
                        ]
                    }
                    
                    return jsonify(portfolio)
                else:
                    return jsonify({'error': 'Broker manager not available'}), 503
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/orders', methods=['GET'])
        def get_orders():
            """Get order history"""
            try:
                symbol = request.args.get('symbol')
                
                if hasattr(self.trading_agent, 'components') and 'broker_manager' in self.trading_agent.components:
                    broker_manager = self.trading_agent.components['broker_manager']
                    orders = broker_manager.get_orders(symbol)
                    
                    order_list = [
                        {
                            'order_id': order.order_id,
                            'symbol': order.symbol,
                            'side': order.side,
                            'quantity': order.quantity,
                            'price': order.price,
                            'status': order.status.value if hasattr(order.status, 'value') else str(order.status),
                            'timestamp': order.timestamp.isoformat() if order.timestamp else None
                        }
                        for order in orders
                    ]
                    
                    return jsonify({
                        'orders': order_list,
                        'count': len(order_list),
                        'timestamp': datetime.utcnow().isoformat()
                    })
                else:
                    return jsonify({'error': 'Broker manager not available'}), 503
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/orders', methods=['POST'])
        def place_order():
            """Place a new order"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                # Validate required fields
                required_fields = ['symbol', 'side', 'quantity']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400
                
                # Create order request
                from order_execution_engine import OrderRequest, OrderType
                
                order_request = OrderRequest(
                    symbol=data['symbol'],
                    side=data['side'],
                    quantity=float(data['quantity']),
                    order_type=OrderType(data.get('order_type', 'market')),
                    price=float(data['price']) if data.get('price') else None,
                    time_in_force=data.get('time_in_force', 'day'),
                    strategy=data.get('strategy', 'api'),
                    metadata={'source': 'api', 'timestamp': datetime.utcnow().isoformat()}
                )
                
                # Execute order through execution engine
                if hasattr(self.trading_agent, 'execution_engine'):
                    order_id = self.trading_agent.execution_engine.execute_order(order_request)
                    
                    if order_id:
                        return jsonify({
                            'order_id': order_id,
                            'status': 'submitted',
                            'timestamp': datetime.utcnow().isoformat()
                        })
                    else:
                        return jsonify({'error': 'Order execution failed'}), 500
                else:
                    return jsonify({'error': 'Order execution engine not available'}), 503
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/orders/<order_id>', methods=['DELETE'])
        def cancel_order(order_id):
            """Cancel an order"""
            try:
                if hasattr(self.trading_agent, 'execution_engine'):
                    success = self.trading_agent.execution_engine.cancel_order(order_id)
                    
                    return jsonify({
                        'order_id': order_id,
                        'cancelled': success,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                else:
                    return jsonify({'error': 'Order execution engine not available'}), 503
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/signals', methods=['GET'])
        def get_signals():
            """Get latest trading signals"""
            try:
                symbol = request.args.get('symbol')
                
                if hasattr(self.trading_agent, 'components') and 'strategy_manager' in self.trading_agent.components:
                    strategy_manager = self.trading_agent.components['strategy_manager']
                    
                    # Get latest market data
                    market_data = {'symbol': symbol} if symbol else {}
                    
                    # Generate signals
                    signals = strategy_manager.generate_signals(market_data)
                    
                    return jsonify({
                        'signals': signals,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                else:
                    return jsonify({'error': 'Strategy manager not available'}), 503
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/risk', methods=['GET'])
        def get_risk_metrics():
            """Get risk metrics"""
            try:
                if hasattr(self.trading_agent, 'risk_manager'):
                    risk_report = self.trading_agent.risk_manager.get_risk_report()
                    return jsonify(risk_report)
                else:
                    return jsonify({'error': 'Risk manager not available'}), 503
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/risk/emergency-stop', methods=['POST'])
        def emergency_stop():
            """Trigger emergency stop"""
            try:
                if hasattr(self.trading_agent, 'risk_manager'):
                    self.trading_agent.risk_manager.emergency_stop = True
                    
                    return jsonify({
                        'emergency_stop': True,
                        'timestamp': datetime.utcnow().isoformat(),
                        'message': 'Emergency stop activated'
                    })
                else:
                    return jsonify({'error': 'Risk manager not available'}), 503
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/risk/emergency-stop', methods=['DELETE'])
        def reset_emergency_stop():
            """Reset emergency stop"""
            try:
                if hasattr(self.trading_agent, 'risk_manager'):
                    success = self.trading_agent.risk_manager.reset_emergency_stop()
                    
                    return jsonify({
                        'emergency_stop': not success,
                        'reset_successful': success,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                else:
                    return jsonify({'error': 'Risk manager not available'}), 503
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/performance', methods=['GET'])
        def get_performance():
            """Get performance metrics"""
            try:
                # Get performance from various components
                performance = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'strategy_performance': {},
                    'execution_metrics': {},
                    'risk_metrics': {}
                }
                
                # Strategy performance
                if hasattr(self.trading_agent, 'components') and 'strategy_manager' in self.trading_agent.components:
                    strategy_status = self.trading_agent.components['strategy_manager'].get_status()
                    performance['strategy_performance'] = strategy_status.get('strategies', {})
                
                # Execution metrics
                if hasattr(self.trading_agent, 'execution_engine'):
                    performance['execution_metrics'] = self.trading_agent.execution_engine.get_execution_metrics()
                
                # Risk metrics
                if hasattr(self.trading_agent, 'risk_manager'):
                    performance['risk_metrics'] = self.trading_agent.risk_manager.get_status()
                
                return jsonify(performance)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/adaptive', methods=['GET'])
        def get_adaptive_status():
            """Get adaptive agent status"""
            try:
                if hasattr(self.trading_agent, 'components') and 'strategy_manager' in self.trading_agent.components:
                    # Try to get adaptive status from supervised learning strategy
                    strategy_manager = self.trading_agent.components['strategy_manager']
                    strategies = strategy_manager.strategies
                    
                    adaptive_status = {}
                    for name, strategy in strategies.items():
                        if hasattr(strategy, 'get_adaptive_status'):
                            adaptive_status[name] = strategy.get_adaptive_status()
                        elif hasattr(strategy, 'adaptive_agent'):
                            adaptive_status[name] = strategy.adaptive_agent.get_agent_state()
                    
                    return jsonify({
                        'adaptive_strategies': adaptive_status,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                else:
                    return jsonify({'error': 'Strategy manager not available'}), 503
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/market-data/<symbol>', methods=['GET'])
        def get_market_data(symbol):
            """Get market data for symbol"""
            try:
                if hasattr(self.trading_agent, 'components') and 'data_manager' in self.trading_agent.components:
                    data_manager = self.trading_agent.components['data_manager']
                    market_data = data_manager.get_latest_data('market_data')
                    
                    return jsonify({
                        'symbol': symbol,
                        'data': market_data,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                else:
                    return jsonify({'error': 'Data manager not available'}), 503
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def start(self):
        """Start the API server"""
        try:
            if self.running:
                logger.warning("API server already running")
                return
            
            self.running = True
            
            # Start server in separate thread
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            
            logger.info(f"API server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            self.running = False
    
    def stop(self):
        """Stop the API server"""
        try:
            self.running = False
            logger.info("API server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping API server: {e}")
    
    def _run_server(self):
        """Run the Flask server"""
        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False,
                threaded=True
            )
        except Exception as e:
            logger.error(f"API server error: {e}")
            self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get API server status"""
        return {
            'running': self.running,
            'host': self.host,
            'port': self.port,
            'endpoints': [
                'GET /api/health',
                'GET /api/status',
                'GET /api/portfolio',
                'GET /api/orders',
                'POST /api/orders',
                'DELETE /api/orders/<order_id>',
                'GET /api/signals',
                'GET /api/risk',
                'POST /api/risk/emergency-stop',
                'DELETE /api/risk/emergency-stop',
                'GET /api/performance',
                'GET /api/adaptive',
                'GET /api/market-data/<symbol>'
            ]
        }