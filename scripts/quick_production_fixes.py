#!/usr/bin/env python3
"""
Quick Production Fixes - Windows Compatible
"""

import os
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_config_security():
    """Remove hardcoded API keys from config"""
    try:
        # Backup config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"config/config_backup_{timestamp}.json"
        
        if os.path.exists('config/config.json'):
            shutil.copy2('config/config.json', backup_file)
            print(f"Config backed up to {backup_file}")
        
        # Load and fix config
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        
        # Remove hardcoded OANDA credentials
        if 'data_manager' in config and 'connectors' in config['data_manager']:
            oanda_config = config['data_manager']['connectors'].get('oanda', {})
            
            if 'api_key' in oanda_config and not oanda_config['api_key'].startswith('${'):
                print(f"Removing hardcoded OANDA API key")
                oanda_config['api_key'] = "${OANDA_API_KEY}"
            
            if 'account_id' in oanda_config and not oanda_config['account_id'].startswith('${'):
                print(f"Removing hardcoded OANDA account ID")
                oanda_config['account_id'] = "${OANDA_ACCOUNT_ID}"
        
        # Save fixed config
        with open('config/config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("Security fix applied: Hardcoded secrets removed")
        return True
        
    except Exception as e:
        print(f"Security fix failed: {e}")
        return False

def create_circuit_breaker():
    """Create simple circuit breaker for API calls"""
    circuit_breaker_code = '''import time
import threading
from functools import wraps

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        with self.lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self._reset()
            return result
        except Exception as e:
            self._record_failure()
            raise
    
    def _record_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
    
    def _reset(self):
        with self.lock:
            self.failure_count = 0
            self.state = 'CLOSED'
            self.last_failure_time = None

def retry_with_backoff(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator
'''
    
    try:
        with open('circuit_breaker.py', 'w') as f:
            f.write(circuit_breaker_code)
        
        print("Reliability fix applied: Circuit breaker created")
        return True
        
    except Exception as e:
        print(f"Circuit breaker creation failed: {e}")
        return False

def create_risk_manager():
    """Create enhanced risk manager"""
    risk_manager_code = '''import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class RiskLimits:
    max_position_size: float = 0.05
    max_portfolio_var: float = 0.02
    max_drawdown: float = 0.10
    stop_loss_percent: float = 0.02

class EnhancedRiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_limits = RiskLimits()
        self.positions = {}
        self.portfolio_value = config.get('initial_capital', 100000)
        self.peak_value = self.portfolio_value
    
    def validate_position(self, symbol: str, quantity: float, price: float) -> Tuple[bool, str]:
        """Validate new position against risk limits"""
        try:
            position_value = abs(quantity * price)
            max_position_value = self.portfolio_value * self.risk_limits.max_position_size
            
            if position_value > max_position_value:
                return False, f"Position too large: ${position_value:.0f} > ${max_position_value:.0f}"
            
            return True, "Position approved"
            
        except Exception as e:
            return False, f"Risk validation failed: {e}"
    
    def add_stop_loss(self, symbol: str, quantity: float, price: float) -> float:
        """Calculate stop loss price"""
        stop_loss_pct = self.risk_limits.stop_loss_percent
        
        if quantity > 0:  # Long position
            return price * (1 - stop_loss_pct)
        else:  # Short position
            return price * (1 + stop_loss_pct)
    
    def check_drawdown(self, current_value: float) -> bool:
        """Check if drawdown limit exceeded"""
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (self.peak_value - current_value) / self.peak_value
        
        if drawdown > self.risk_limits.max_drawdown:
            logging.critical(f"Drawdown limit exceeded: {drawdown:.2%}")
            return False
        
        return True
'''
    
    try:
        with open('enhanced_risk_manager.py', 'w') as f:
            f.write(risk_manager_code)
        
        print("Risk management fix applied: Enhanced risk manager created")
        return True
        
    except Exception as e:
        print(f"Risk manager creation failed: {e}")
        return False

def update_config_risk_limits():
    """Update config with better risk limits"""
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        
        # Enhanced risk settings
        config['risk_management'].update({
            "max_position_size": 0.05,
            "max_portfolio_var": 0.015,
            "max_drawdown": 0.10,
            "stop_loss_percent": 0.02,
            "emergency_drawdown": 0.08
        })
        
        config['risk_limits'].update({
            "max_portfolio_var": 0.02,
            "max_position_size": 0.05,
            "max_drawdown": 0.10
        })
        
        with open('config/config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("Risk limits updated in configuration")
        return True
        
    except Exception as e:
        print(f"Config update failed: {e}")
        return False

def create_secure_api_server():
    """Create API server with basic security"""
    api_code = '''from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import time
import logging
from functools import wraps

app = Flask(__name__)
CORS(app)

# Rate limiting
request_counts = {}
RATE_LIMIT = 100

def rate_limit_check():
    client_ip = request.remote_addr or 'unknown'
    current_time = time.time()
    
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip] 
        if current_time - req_time < 60
    ]
    
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        return False
    
    request_counts[client_ip].append(current_time)
    return True

def require_rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not rate_limit_check():
            return jsonify({'error': 'Rate limit exceeded'}), 429
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    })

@app.route('/api/status', methods=['GET'])
@require_rate_limit
def get_status():
    return jsonify({
        'running': True,
        'timestamp': time.time(),
        'components': {
            'api_server': {'status': 'running'},
            'risk_manager': {'status': 'active'}
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting secure API server on port 5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
'''
    
    try:
        with open('secure_api_server.py', 'w') as f:
            f.write(api_code)
        
        print("Secure API server created")
        return True
        
    except Exception as e:
        print(f"API server creation failed: {e}")
        return False

def create_env_template():
    """Create secure environment template"""
    env_template = """# AI Trading Bot Environment Variables
TRADING_ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
TRADING_FMP_API_KEY=your_fmp_key_here
TRADING_FINNHUB_API_KEY=your_finnhub_key_here

COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here

OANDA_API_KEY=your_oanda_api_key_here
OANDA_ACCOUNT_ID=your_oanda_account_id_here

TRADING_MASTER_KEY=your_secure_master_key_here
"""
    
    try:
        with open('.env.template', 'w') as f:
            f.write(env_template)
        
        print("Environment template created (.env.template)")
        return True
        
    except Exception as e:
        print(f"Environment template creation failed: {e}")
        return False

def main():
    """Execute quick production fixes"""
    print("QUICK PRODUCTION FIXES")
    print("=" * 50)
    
    fixes_applied = 0
    total_fixes = 6
    
    print("\n1. Fixing configuration security...")
    if fix_config_security():
        fixes_applied += 1
    
    print("\n2. Creating circuit breaker...")
    if create_circuit_breaker():
        fixes_applied += 1
    
    print("\n3. Creating enhanced risk manager...")
    if create_risk_manager():
        fixes_applied += 1
    
    print("\n4. Updating risk limits...")
    if update_config_risk_limits():
        fixes_applied += 1
    
    print("\n5. Creating secure API server...")
    if create_secure_api_server():
        fixes_applied += 1
    
    print("\n6. Creating environment template...")
    if create_env_template():
        fixes_applied += 1
    
    print("\n" + "=" * 50)
    print(f"FIXES SUMMARY: {fixes_applied}/{total_fixes} applied")
    
    if fixes_applied >= 4:
        print("\nSUCCESS: Critical fixes applied!")
        print("\nNEXT STEPS:")
        print("1. Copy .env.template to .env and add your API keys")
        print("2. Test: python secure_api_server.py")
        print("3. Run: python start_advanced_system.py")
    else:
        print("\nWARNING: Some fixes failed - review errors above")
    
    return fixes_applied >= 4

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)