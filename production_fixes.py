#!/usr/bin/env python3
"""
Critical Production Fixes Implementation
"""

import os
import json
import logging
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps
import requests

class CircuitBreaker:
    """Circuit breaker pattern for external API calls"""
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
    
    def reset(self):
        self.failure_count = 0
        self.state = 'CLOSED'
        self.last_failure_time = None

class SecureConfigManager:
    """Secure configuration management"""
    def __init__(self):
        self.config_cache = {}
        self.last_reload = 0
        self.reload_interval = 300  # 5 minutes
    
    def get_config(self, reload=False):
        """Get configuration with caching and validation"""
        current_time = time.time()
        
        if reload or (current_time - self.last_reload) > self.reload_interval:
            self._load_config()
            self.last_reload = current_time
        
        return self.config_cache.copy()
    
    def _load_config(self):
        """Load and validate configuration"""
        try:
            with open('config/config.json', 'r') as f:
                config = json.load(f)
            
            # Remove hardcoded secrets
            self._sanitize_config(config)
            
            # Validate configuration
            self._validate_config(config)
            
            self.config_cache = config
            
        except Exception as e:
            logging.error(f"Config loading failed: {e}")
            raise
    
    def _sanitize_config(self, config):
        """Remove hardcoded secrets from config"""
        # Remove hardcoded OANDA keys
        if 'data_manager' in config and 'connectors' in config['data_manager']:
            oanda_config = config['data_manager']['connectors'].get('oanda', {})
            if 'api_key' in oanda_config:
                del oanda_config['api_key']
            if 'account_id' in oanda_config:
                del oanda_config['account_id']
        
        # Ensure broker configs use environment variables
        if 'brokers' in config:
            for broker_name, broker_config in config['brokers'].items():
                for key in ['api_key', 'api_secret', 'passphrase', 'account_id']:
                    if key in broker_config and not broker_config[key].startswith('${'):
                        broker_config[key] = f"${{{key.upper()}}}"
    
    def _validate_config(self, config):
        """Validate configuration structure"""
        required_sections = ['data_manager', 'strategies', 'risk_management', 'brokers']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate risk limits
        risk_mgmt = config.get('risk_management', {})
        if risk_mgmt.get('max_position_size', 0) > 0.2:
            logging.warning("Position size limit > 20% - high risk!")
        
        if risk_mgmt.get('max_drawdown', 0) > 0.3:
            logging.warning("Drawdown limit > 30% - very high risk!")

class InputValidator:
    """Input validation for API endpoints"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate trading symbol format"""
        if not symbol or len(symbol) > 20:
            return False
        
        # Allow alphanumeric, dash, underscore
        allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
        return all(c in allowed_chars for c in symbol.upper())
    
    @staticmethod
    def validate_quantity(quantity: float) -> bool:
        """Validate order quantity"""
        return isinstance(quantity, (int, float)) and 0 < quantity <= 1000000
    
    @staticmethod
    def validate_price(price: float) -> bool:
        """Validate price value"""
        return isinstance(price, (int, float)) and 0 < price <= 1000000
    
    @staticmethod
    def validate_order_request(data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate complete order request"""
        required_fields = ['symbol', 'side', 'quantity']
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        if not InputValidator.validate_symbol(data['symbol']):
            return False, "Invalid symbol format"
        
        if data['side'] not in ['buy', 'sell']:
            return False, "Invalid side - must be 'buy' or 'sell'"
        
        if not InputValidator.validate_quantity(data['quantity']):
            return False, "Invalid quantity"
        
        if 'price' in data and not InputValidator.validate_price(data['price']):
            return False, "Invalid price"
        
        return True, "Valid"

class RobustAPIClient:
    """Robust API client with retry logic and circuit breaker"""
    
    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url
        self.timeout = timeout
        self.circuit_breaker = CircuitBreaker()
        self.session = requests.Session()
    
    def make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make API request with retry logic"""
        max_retries = 3
        backoff_factor = 1
        
        for attempt in range(max_retries):
            try:
                return self.circuit_breaker.call(self._do_request, method, endpoint, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"API request failed after {max_retries} attempts: {e}")
                    return None
                
                wait_time = backoff_factor * (2 ** attempt)
                logging.warning(f"Request failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
        
        return None
    
    def _do_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Execute the actual HTTP request"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault('timeout', self.timeout)
        
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        
        return response.json()

class DataValidator:
    """Validate market data integrity"""
    
    @staticmethod
    def validate_price_data(data: Dict[str, Any]) -> bool:
        """Validate price data structure and values"""
        required_fields = ['symbol', 'price', 'timestamp']
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                logging.error(f"Missing field in price data: {field}")
                return False
        
        # Validate price
        price = data.get('price')
        if not isinstance(price, (int, float)) or price <= 0:
            logging.error(f"Invalid price: {price}")
            return False
        
        # Validate timestamp
        try:
            datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            logging.error(f"Invalid timestamp: {data.get('timestamp')}")
            return False
        
        # Check for reasonable price ranges
        if price > 1000000 or price < 0.0001:
            logging.warning(f"Unusual price detected: {price}")
        
        return True
    
    @staticmethod
    def calculate_checksum(data: Dict[str, Any]) -> str:
        """Calculate data checksum for integrity verification"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

def rate_limit(max_calls: int, window: int):
    """Rate limiting decorator"""
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls outside the window
            calls[:] = [call_time for call_time in calls if now - call_time < window]
            
            # Check if we're at the limit
            if len(calls) >= max_calls:
                raise Exception(f"Rate limit exceeded: {max_calls} calls per {window} seconds")
            
            # Record this call
            calls.append(now)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.checks = {}
    
    def register_check(self, name: str, check_func):
        """Register a health check function"""
        self.checks[name] = check_func
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                status = check_func()
                duration = time.time() - start_time
                
                results['checks'][name] = {
                    'status': 'healthy' if status else 'unhealthy',
                    'duration_ms': round(duration * 1000, 2),
                    'details': status if isinstance(status, dict) else {}
                }
                
                if not status:
                    results['overall_status'] = 'unhealthy'
                    
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e),
                    'duration_ms': 0
                }
                results['overall_status'] = 'unhealthy'
        
        return results

def fix_config_security():
    """Fix security issues in configuration"""
    print("🔧 Fixing configuration security issues...")
    
    config_manager = SecureConfigManager()
    
    try:
        # Load and sanitize config
        config = config_manager.get_config(reload=True)
        
        # Save sanitized config
        with open('config/config_secure.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuration security fixed")
        print("   • Removed hardcoded API keys")
        print("   • Added environment variable references")
        print("   • Validated configuration structure")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix config security: {e}")
        return False

def create_health_checks():
    """Create system health check endpoints"""
    print("🔧 Creating health check system...")
    
    health_checker = HealthChecker()
    
    # Register basic health checks
    def check_config():
        try:
            config_manager = SecureConfigManager()
            config_manager.get_config()
            return True
        except:
            return False
    
    def check_disk_space():
        import shutil
        total, used, free = shutil.disk_usage('.')
        free_percent = (free / total) * 100
        return free_percent > 10  # At least 10% free space
    
    def check_memory():
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent < 90  # Less than 90% memory usage
    
    health_checker.register_check('config', check_config)
    health_checker.register_check('disk_space', check_disk_space)
    health_checker.register_check('memory', check_memory)
    
    # Test health checks
    results = health_checker.run_checks()
    
    print("✅ Health check system created")
    print(f"   • Overall status: {results['overall_status']}")
    print(f"   • Checks configured: {len(results['checks'])}")
    
    return health_checker

def main():
    """Apply all production fixes"""
    print("🚀 APPLYING PRODUCTION FIXES")
    print("=" * 50)
    
    fixes_applied = 0
    total_fixes = 2
    
    # Fix 1: Configuration security
    if fix_config_security():
        fixes_applied += 1
    
    # Fix 2: Health checks
    if create_health_checks():
        fixes_applied += 1
    
    print("\n" + "=" * 50)
    print(f"✅ Applied {fixes_applied}/{total_fixes} critical fixes")
    
    if fixes_applied == total_fixes:
        print("🎉 All critical fixes applied successfully!")
        print("\n📋 Next steps:")
        print("   1. Test the system thoroughly")
        print("   2. Deploy to staging environment")
        print("   3. Implement remaining improvements")
        print("   4. Monitor system performance")
    else:
        print("⚠️  Some fixes failed - review errors above")
    
    return fixes_applied == total_fixes

if __name__ == "__main__":
    main()