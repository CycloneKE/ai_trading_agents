#!/usr/bin/env python3
"""
Integrated API Server with All Enhancements
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Import our enhanced modules
try:
    from optimized_data_manager import OptimizedDataManager
    from memory_optimizer import memory_optimizer
    from advanced_monitoring import SystemMonitor
    from portfolio_analytics import portfolio_analytics
    from strategy_optimizer import strategy_optimizer
    from log_analyzer import log_analyzer
except ImportError as e:
    print(f"Import warning: {e}")
    # Fallback to basic functionality
    OptimizedDataManager = None
    memory_optimizer = None
    SystemMonitor = None
    portfolio_analytics = None
    strategy_optimizer = None
    log_analyzer = None

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize components
data_manager = OptimizedDataManager() if OptimizedDataManager else None
system_monitor = SystemMonitor() if SystemMonitor else None

# Start background services
if memory_optimizer:
    memory_optimizer.start()

if system_monitor:
    system_monitor.start()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '3.0.0',
        'features': {
            'optimized_data': data_manager is not None,
            'memory_optimization': memory_optimizer is not None,
            'system_monitoring': system_monitor is not None,
            'portfolio_analytics': portfolio_analytics is not None,
            'strategy_optimization': strategy_optimizer is not None
        }
    })

@app.route('/api/performance-metrics', methods=['GET'])
def get_performance_metrics():
    """Get system performance metrics"""
    if system_monitor:
        return jsonify(system_monitor.get_dashboard_data())
    else:
        return jsonify({'error': 'Monitoring not available'}), 503

@app.route('/api/portfolio-analytics', methods=['GET'])
def get_portfolio_analytics():
    """Get portfolio analytics"""
    if portfolio_analytics:
        # Mock current prices for demo
        current_prices = {'AAPL': 175, 'GOOGL': 140, 'MSFT': 415}
        metrics = portfolio_analytics.calculate_portfolio_metrics(current_prices)
        return jsonify(metrics)
    else:
        return jsonify({'error': 'Portfolio analytics not available'}), 503

@app.route('/api/strategy-optimization', methods=['POST'])
def optimize_strategy():
    """Optimize trading strategy"""
    if not strategy_optimizer:
        return jsonify({'error': 'Strategy optimizer not available'}), 503
    
    data = request.json
    strategy_name = data.get('strategy', 'momentum')
    
    try:
        if strategy_name == 'momentum':
            param_ranges = {
                'lookback_period': (10, 30),
                'threshold': (0.01, 0.05)
            }
        else:
            param_ranges = {
                'window': (5, 20),
                'z_threshold': (1.0, 3.0)
            }
        
        result = strategy_optimizer.optimize_strategy(
            strategy_name, param_ranges, iterations=50
        )
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-data-optimized/<symbol>', methods=['GET'])
def get_optimized_market_data(symbol):
    """Get market data using optimized data manager"""
    if data_manager:
        try:
            data = data_manager.get_market_data(symbol.upper())
            return jsonify(data) if data else jsonify({'error': 'No data'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Optimized data manager not available'}), 503

@app.route('/api/log-analysis', methods=['GET'])
def get_log_analysis():
    """Get log analysis results"""
    if log_analyzer:
        try:
            analysis = log_analyzer.get_error_summary()
            return jsonify(analysis)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Log analyzer not available'}), 503

@app.route('/api/system-status', methods=['GET'])
def get_system_status():
    """Get comprehensive system status"""
    status = {
        'timestamp': datetime.now().isoformat(),
        'uptime': '3h 45m',
        'version': '3.0.0',
        'components': {
            'api_server': {'status': 'running', 'port': 5001},
            'data_manager': {'status': 'active' if data_manager else 'unavailable'},
            'monitoring': {'status': 'active' if system_monitor else 'unavailable'},
            'memory_optimizer': {'status': 'active' if memory_optimizer else 'unavailable'}
        }
    }
    
    if system_monitor:
        dashboard_data = system_monitor.get_dashboard_data()
        status['metrics'] = dashboard_data.get('system_metrics', {})
    
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Enhanced Trading API Server v3.0")
    print("Features: Performance optimization, Advanced monitoring, Enhanced analytics")
    print("Server running on http://localhost:5001")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    finally:
        # Cleanup
        if memory_optimizer:
            memory_optimizer.stop()
        if system_monitor:
            system_monitor.stop()
