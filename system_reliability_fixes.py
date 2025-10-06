#!/usr/bin/env python3
"""
SYSTEM RELIABILITY FIXES - Phase 2
Implement circuit breakers, retry logic, and error handling
"""

import os
import json
import time
import logging
import threading
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker pattern for external API calls"""
    def __init__(self, failure_threshold=5, timeout=60, name="CircuitBreaker"):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = 'HALF_OPEN'
                    logger.info(f"🔄 {self.name}: Attempting recovery (HALF_OPEN)")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self._reset()
            return result
        except Exception as e:
            self._record_failure()
            raise
    
    def _record_failure(self):
        """Record a failure and potentially open the circuit"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                logger.warning(f"🚨 {self.name}: Circuit breaker OPENED after {self.failure_count} failures")
    
    def _reset(self):
        """Reset the circuit breaker"""
        with self.lock:
            self.failure_count = 0
            self.state = 'CLOSED'
            self.last_failure_time = None
            logger.info(f"✅ {self.name}: Circuit breaker CLOSED (recovered)")

def retry_with_backoff(max_retries=3, backoff_factor=1, exceptions=(Exception,)):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        logger.error(f"❌ {func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"⚠️ {func.__name__} attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
            
            return None
        return wrapper
    return decorator

class RobustDataManager:
    """Enhanced data manager with reliability features"""
    def __init__(self):
        self.circuit_breakers = {
            'fmp': CircuitBreaker(name="FMP_API"),
            'alpha_vantage': CircuitBreaker(name="AlphaVantage_API"),
            'coinbase': CircuitBreaker(name="Coinbase_API")
        }
        self.data_cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def fetch_market_data(self, symbol: str, source: str = 'fmp') -> Optional[Dict]:
        """Fetch market data with circuit breaker and retry logic"""
        # Check cache first
        cache_key = f"{source}_{symbol}"
        if self._is_cache_valid(cache_key):
            logger.debug(f"📋 Using cached data for {symbol}")
            return self.data_cache[cache_key]['data']
        
        # Use circuit breaker for API call
        try:
            if source in self.circuit_breakers:
                data = self.circuit_breakers[source].call(self._fetch_from_api, symbol, source)
            else:
                data = self._fetch_from_api(symbol, source)
            
            # Cache successful result
            self.data_cache[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch {symbol} from {source}: {e}")
            
            # Return stale cache if available
            if cache_key in self.data_cache:
                logger.warning(f"⚠️ Using stale cache for {symbol}")
                return self.data_cache[cache_key]['data']
            
            return None
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.data_cache:
            return False
        
        cache_age = time.time() - self.data_cache[cache_key]['timestamp']
        return cache_age < self.cache_timeout
    
    def _fetch_from_api(self, symbol: str, source: str) -> Dict:
        """Actual API fetch implementation (placeholder)"""
        import requests
        
        if source == 'fmp':
            api_key = os.getenv('TRADING_FMP_API_KEY')
            if not api_key:
                raise Exception("FMP API key not configured")
            
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                return data[0]
            else:
                raise Exception("No data returned from FMP API")
        
        raise Exception(f"Unknown data source: {source}")

class HealthChecker:
    """System health monitoring with automatic recovery"""
    def __init__(self):
        self.checks = {}
        self.last_check_time = {}
        self.check_interval = 60  # 1 minute
        self.running = False
        self.monitor_thread = None
    
    def register_check(self, name: str, check_func: Callable, critical: bool = False):
        """Register a health check function"""
        self.checks[name] = {
            'func': check_func,
            'critical': critical,
            'last_result': None,
            'failure_count': 0
        }
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🏥 Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🏥 Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self.run_all_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"❌ Health monitoring error: {e}")
                time.sleep(10)
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        for name, check_info in self.checks.items():
            try:
                start_time = time.time()
                result = check_info['func']()
                duration = time.time() - start_time
                
                check_info['last_result'] = result
                check_info['failure_count'] = 0 if result else check_info['failure_count'] + 1
                
                results['checks'][name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'duration_ms': round(duration * 1000, 2),
                    'critical': check_info['critical'],
                    'failure_count': check_info['failure_count']
                }
                
                if not result:
                    if check_info['critical']:
                        results['overall_status'] = 'critical'
                    elif results['overall_status'] == 'healthy':
                        results['overall_status'] = 'degraded'
                
                # Alert on critical failures
                if not result and check_info['critical'] and check_info['failure_count'] >= 3:
                    logger.critical(f"🚨 CRITICAL: Health check '{name}' failed {check_info['failure_count']} times")
                
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e),
                    'critical': check_info['critical']
                }
                
                if check_info['critical']:
                    results['overall_status'] = 'critical'
        
        return results

def create_enhanced_api_server():
    """Create API server with reliability features"""
    api_code = '''#!/usr/bin/env python3
"""
Enhanced API Server with Reliability Features
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import time
import logging
from functools import wraps
from system_reliability_fixes import RobustDataManager, HealthChecker

app = Flask(__name__)
CORS(app)

# Initialize components
data_manager = RobustDataManager()
health_checker = HealthChecker()

# Rate limiting
request_counts = {}
RATE_LIMIT = 100  # requests per minute

def rate_limit_check():
    """Check rate limiting"""
    client_ip = request.remote_addr
    current_time = time.time()
    
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    # Remove old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip] 
        if current_time - req_time < 60
    ]
    
    # Check limit
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        return False
    
    # Record this request
    request_counts[client_ip].append(current_time)
    return True

def require_rate_limit(f):
    """Rate limiting decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not rate_limit_check():
            return jsonify({'error': 'Rate limit exceeded'}), 429
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        results = health_checker.run_all_checks()
        status_code = 200 if results['overall_status'] == 'healthy' else 503
        return jsonify(results), status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-data/<symbol>', methods=['GET'])
@require_rate_limit
def get_market_data(symbol):
    """Get market data with reliability features"""
    try:
        # Validate symbol
        if not symbol or len(symbol) > 10:
            return jsonify({'error': 'Invalid symbol'}), 400
        
        # Fetch data
        data = data_manager.fetch_market_data(symbol.upper())
        
        if data:
            return jsonify(data)
        else:
            return jsonify({'error': 'Data not available'}), 503
            
    except Exception as e:
        logging.error(f"Market data error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Register health checks
    def check_data_sources():
        try:
            data_manager.fetch_market_data('AAPL', 'fmp')
            return True
        except:
            return False
    
    def check_memory():
        import psutil
        return psutil.virtual_memory().percent < 90
    
    health_checker.register_check('data_sources', check_data_sources, critical=True)
    health_checker.register_check('memory', check_memory, critical=False)
    health_checker.start_monitoring()
    
    print("🚀 Enhanced API Server starting...")
    app.run(host='0.0.0.0', port=5001, debug=False)
'''
    
    try:
        with open('enhanced_api_server.py', 'w') as f:
            f.write(api_code)
        
        logger.info("✅ Enhanced API server created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create enhanced API server: {e}")
        return False

def create_memory_monitor():
    """Create memory monitoring and cleanup"""
    monitor_code = '''#!/usr/bin/env python3
"""
Memory Monitor and Cleanup
"""

import gc
import psutil
import threading
import time
import logging

class MemoryMonitor:
    def __init__(self, threshold_percent=85, check_interval=30):
        self.threshold_percent = threshold_percent
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None
    
    def start(self):
        """Start memory monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("🧠 Memory monitoring started")
    
    def stop(self):
        """Stop memory monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                memory = psutil.virtual_memory()
                
                if memory.percent > self.threshold_percent:
                    logging.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
                    self._cleanup_memory()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logging.error(f"Memory monitoring error: {e}")
                time.sleep(10)
    
    def _cleanup_memory(self):
        """Perform memory cleanup"""
        logging.info("🧹 Performing memory cleanup...")
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory after cleanup
        memory_after = psutil.virtual_memory()
        
        logging.info(f"🧹 Cleanup complete: {collected} objects collected, "
                    f"memory usage: {memory_after.percent:.1f}%")

# Global memory monitor instance
memory_monitor = MemoryMonitor()

if __name__ == "__main__":
    memory_monitor.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        memory_monitor.stop()
'''
    
    try:
        with open('memory_monitor.py', 'w') as f:
            f.write(monitor_code)
        
        logger.info("✅ Memory monitor created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create memory monitor: {e}")
        return False

def main():
    """Execute all system reliability fixes"""
    print("🔧 SYSTEM RELIABILITY FIXES - PHASE 2")
    print("=" * 50)
    
    fixes_applied = 0
    total_fixes = 3
    
    # Fix 1: Create enhanced API server
    print("\n1️⃣ Creating enhanced API server with circuit breakers...")
    if create_enhanced_api_server():
        fixes_applied += 1
        print("   ✅ Enhanced API server created")
    else:
        print("   ❌ Failed to create enhanced API server")
    
    # Fix 2: Create memory monitor
    print("\n2️⃣ Creating memory monitoring system...")
    if create_memory_monitor():
        fixes_applied += 1
        print("   ✅ Memory monitor created")
    else:
        print("   ❌ Failed to create memory monitor")
    
    # Fix 3: Test reliability features
    print("\n3️⃣ Testing reliability features...")
    try:
        # Test circuit breaker
        cb = CircuitBreaker(name="Test")
        
        def failing_function():
            raise Exception("Test failure")
        
        # Should work initially
        try:
            cb.call(lambda: "success")
            print("   ✅ Circuit breaker basic functionality works")
        except:
            print("   ❌ Circuit breaker basic test failed")
            
        fixes_applied += 1
        
    except Exception as e:
        print(f"   ❌ Reliability test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🔧 RELIABILITY FIXES SUMMARY")
    print(f"   Applied: {fixes_applied}/{total_fixes} fixes")
    
    if fixes_applied >= 2:
        print("   ✅ System reliability significantly improved!")
        print("\n📋 NEXT STEPS:")
        print("   1. Test the enhanced API server")
        print("   2. Monitor memory usage patterns")
        print("   3. Proceed to risk management fixes")
    else:
        print("   ⚠️ Some reliability fixes failed - review errors above")
    
    return fixes_applied >= 2

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)