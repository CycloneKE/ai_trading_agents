#!/usr/bin/env python3
"""
CRITICAL SECURITY FIXES - Phase 1
Fix hardcoded API keys and implement secure configuration
"""

import os
import json
import shutil
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_current_config():
    """Backup current configuration before making changes"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"config/config_backup_{timestamp}.json"
        
        if os.path.exists('config/config.json'):
            shutil.copy2('config/config.json', backup_file)
            logger.info(f"✅ Config backed up to {backup_file}")
            return True
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        return False

def remove_hardcoded_secrets():
    """Remove hardcoded API keys from config.json"""
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        
        # Remove hardcoded OANDA credentials
        if 'data_manager' in config and 'connectors' in config['data_manager']:
            oanda_config = config['data_manager']['connectors'].get('oanda', {})
            
            # Check for hardcoded keys
            if 'api_key' in oanda_config and not oanda_config['api_key'].startswith('${'):
                logger.warning(f"🔒 Removing hardcoded OANDA API key: {oanda_config['api_key'][:8]}...")
                oanda_config['api_key'] = "${OANDA_API_KEY}"
            
            if 'account_id' in oanda_config and not oanda_config['account_id'].startswith('${'):
                logger.warning(f"🔒 Removing hardcoded OANDA account ID: {oanda_config['account_id'][:8]}...")
                oanda_config['account_id'] = "${OANDA_ACCOUNT_ID}"
        
        # Ensure all broker configs use environment variables
        if 'brokers' in config:
            for broker_name, broker_config in config['brokers'].items():
                for key in ['api_key', 'api_secret', 'passphrase', 'account_id']:
                    if key in broker_config:
                        value = broker_config[key]
                        if isinstance(value, str) and not value.startswith('${'):
                            # Only replace if it looks like a real key (not empty or placeholder)
                            if value and value != 'your_key_here' and len(value) > 10:
                                logger.warning(f"🔒 Securing {broker_name}.{key}")
                                broker_config[key] = f"${{{key.upper()}}}"
        
        # Save sanitized config
        with open('config/config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("✅ Hardcoded secrets removed from config")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to remove secrets: {e}")
        return False

def create_secure_env_template():
    """Create secure .env template with all required variables"""
    env_template = """# AI Trading Bot - Secure Environment Variables
# Copy this to .env and fill in your actual API keys

# Market Data APIs
TRADING_ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
TRADING_FMP_API_KEY=your_fmp_key_here
TRADING_FINNHUB_API_KEY=your_finnhub_key_here

# Crypto Trading (Coinbase Pro)
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here

# Forex Trading (OANDA)
OANDA_API_KEY=your_oanda_api_key_here
OANDA_ACCOUNT_ID=your_oanda_account_id_here

# Security
TRADING_MASTER_KEY=your_secure_master_key_32_chars_min
JWT_SECRET_KEY=your_jwt_secret_key_here

# Database (if using external DB)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_agent
DB_USER=trading_user
DB_PASSWORD=your_secure_db_password_here

# Redis (if using external Redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password_here
"""
    
    try:
        with open('.env.secure', 'w') as f:
            f.write(env_template)
        
        logger.info("✅ Secure .env template created (.env.secure)")
        logger.info("📋 Copy .env.secure to .env and fill in your API keys")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create .env template: {e}")
        return False

def validate_current_env():
    """Validate current environment variables"""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        'TRADING_FMP_API_KEY',
        'COINBASE_API_KEY', 
        'COINBASE_API_SECRET',
        'OANDA_API_KEY',
        'OANDA_ACCOUNT_ID'
    ]
    
    missing_vars = []
    configured_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value or value == 'your_key_here':
            missing_vars.append(var)
        else:
            configured_vars.append(var)
    
    logger.info(f"✅ Configured variables: {len(configured_vars)}")
    for var in configured_vars:
        logger.info(f"   • {var}: {os.getenv(var)[:8]}...")
    
    if missing_vars:
        logger.warning(f"⚠️  Missing variables: {len(missing_vars)}")
        for var in missing_vars:
            logger.warning(f"   • {var}")
    
    return len(missing_vars) == 0

def create_config_validator():
    """Create configuration validator to prevent future security issues"""
    validator_code = '''#!/usr/bin/env python3
"""
Configuration Security Validator
Prevents hardcoded secrets in config files
"""

import json
import re
import sys

def validate_config_security(config_path):
    """Validate configuration file for security issues"""
    issues = []
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            config = json.loads(content)
        
        # Check for potential API keys (long alphanumeric strings)
        api_key_pattern = r'[A-Za-z0-9]{20,}'
        matches = re.findall(api_key_pattern, content)
        
        for match in matches:
            # Skip environment variable references
            if not match.startswith('${') and match != 'your_key_here':
                # Check if it looks like a real API key
                if len(match) > 15 and any(c.isdigit() for c in match) and any(c.isalpha() for c in match):
                    issues.append(f"Potential hardcoded API key: {match[:8]}...")
        
        # Check for specific patterns
        dangerous_patterns = [
            (r'sk-[A-Za-z0-9]{40,}', 'OpenAI API key'),
            (r'[A-Z0-9]{32,}', 'Generic API key'),
            (r'[a-f0-9]{64}', 'SHA256 hash/secret')
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, content):
                issues.append(f"Detected {description}")
        
        return issues
        
    except Exception as e:
        return [f"Validation error: {e}"]

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/config.json"
    issues = validate_config_security(config_file)
    
    if issues:
        print("🚨 SECURITY ISSUES FOUND:")
        for issue in issues:
            print(f"   ❌ {issue}")
        sys.exit(1)
    else:
        print("✅ Configuration security validation passed")
        sys.exit(0)
'''
    
    try:
        with open('validate_config_security.py', 'w') as f:
            f.write(validator_code)
        
        # Make it executable
        os.chmod('validate_config_security.py', 0o755)
        
        logger.info("✅ Configuration validator created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create validator: {e}")
        return False

def main():
    """Execute all critical security fixes"""
    print("🔒 CRITICAL SECURITY FIXES - PHASE 1")
    print("=" * 50)
    
    fixes_applied = 0
    total_fixes = 5
    
    # Fix 1: Backup current config
    print("\n1️⃣ Backing up current configuration...")
    if backup_current_config():
        fixes_applied += 1
        print("   ✅ Configuration backed up")
    else:
        print("   ❌ Backup failed")
    
    # Fix 2: Remove hardcoded secrets
    print("\n2️⃣ Removing hardcoded API keys...")
    if remove_hardcoded_secrets():
        fixes_applied += 1
        print("   ✅ Hardcoded secrets removed")
    else:
        print("   ❌ Failed to remove secrets")
    
    # Fix 3: Create secure .env template
    print("\n3️⃣ Creating secure environment template...")
    if create_secure_env_template():
        fixes_applied += 1
        print("   ✅ Secure .env template created")
    else:
        print("   ❌ Failed to create template")
    
    # Fix 4: Validate current environment
    print("\n4️⃣ Validating current environment...")
    if validate_current_env():
        fixes_applied += 1
        print("   ✅ Environment validation passed")
    else:
        print("   ⚠️  Some environment variables missing")
        fixes_applied += 0.5  # Partial credit
    
    # Fix 5: Create config validator
    print("\n5️⃣ Creating configuration validator...")
    if create_config_validator():
        fixes_applied += 1
        print("   ✅ Security validator created")
    else:
        print("   ❌ Failed to create validator")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🔒 SECURITY FIXES SUMMARY")
    print(f"   Applied: {fixes_applied}/{total_fixes} fixes")
    
    if fixes_applied >= 4:
        print("   ✅ Critical security issues resolved!")
        print("\n📋 NEXT STEPS:")
        print("   1. Copy .env.secure to .env")
        print("   2. Fill in your actual API keys in .env")
        print("   3. Run: python validate_config_security.py")
        print("   4. Proceed to system reliability fixes")
    else:
        print("   ⚠️  Some critical fixes failed - review errors above")
    
    return fixes_applied >= 4

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)