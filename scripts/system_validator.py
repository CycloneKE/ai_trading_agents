#!/usr/bin/env python3
"""
System Validator - Check production readiness
"""

import os
import json
import psutil
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def check_configuration():
    """Check configuration security"""
    issues = []
    
    if not os.path.exists('config/config.json'):
        issues.append("config.json not found")
        return issues
    
    try:
        with open('config/config.json', 'r') as f:
            content = f.read()
            config = json.loads(content)
        
        # Check for hardcoded OANDA key
        if 'cf87c3c69c323de28aa987c2d29a601c' in content:
            issues.append("Hardcoded OANDA key still present")
        
        # Check risk limits
        risk_mgmt = config.get('risk_management', {})
        if risk_mgmt.get('max_position_size', 1) > 0.1:
            issues.append("Position size limit too high (>10%)")
        
        if risk_mgmt.get('max_drawdown', 1) > 0.2:
            issues.append("Drawdown limit too high (>20%)")
            
    except json.JSONDecodeError:
        issues.append("config.json is not valid JSON")
    except Exception as e:
        issues.append(f"Config check error: {e}")
    
    return issues

def check_environment():
    """Check environment variables"""
    required_vars = [
        'TRADING_FMP_API_KEY',
        'COINBASE_API_KEY',
        'COINBASE_API_SECRET',
        'OANDA_API_KEY',
        'OANDA_ACCOUNT_ID'
    ]
    
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value == 'your_key_here':
            missing.append(var)
    
    return missing

def check_system_resources():
    """Check system resources"""
    issues = []
    
    # Memory check
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        issues.append(f"High memory usage: {memory.percent:.1f}%")
    
    # Disk space check
    disk = psutil.disk_usage('.')
    free_percent = (disk.free / disk.total) * 100
    if free_percent < 10:
        issues.append(f"Low disk space: {free_percent:.1f}% free")
    
    return issues

def check_critical_files():
    """Check critical files exist"""
    critical_files = [
        'main.py',
        'config/config.json',
        'requirements.txt',
        'circuit_breaker.py',
        'enhanced_risk_manager.py',
        'secure_api_server.py'
    ]
    
    missing = []
    for file in critical_files:
        if not os.path.exists(file):
            missing.append(file)
    
    return missing

def check_api_connectivity():
    """Test API connectivity"""
    issues = []
    
    # Test FMP API if key is available
    fmp_key = os.getenv('TRADING_FMP_API_KEY')
    if fmp_key and fmp_key != 'your_fmp_key_here':
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={fmp_key}"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                issues.append(f"FMP API error: {response.status_code}")
        except Exception as e:
            issues.append(f"FMP API connection failed: {str(e)[:50]}")
    
    return issues

def calculate_readiness_score():
    """Calculate overall production readiness score"""
    config_issues = check_configuration()
    env_issues = check_environment()
    resource_issues = check_system_resources()
    file_issues = check_critical_files()
    api_issues = check_api_connectivity()
    
    total_issues = len(config_issues) + len(env_issues) + len(resource_issues) + len(file_issues) + len(api_issues)
    
    # Maximum possible issues (rough estimate)
    max_issues = 20
    
    score = max(0, (max_issues - total_issues) / max_issues * 100)
    
    return score, {
        'config': config_issues,
        'environment': env_issues,
        'resources': resource_issues,
        'files': file_issues,
        'api': api_issues
    }

def main():
    """Run comprehensive system validation"""
    print("SYSTEM PRODUCTION READINESS VALIDATOR")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    score, issues = calculate_readiness_score()
    
    print(f"\nOVERALL READINESS SCORE: {score:.1f}%")
    
    if score >= 80:
        print("STATUS: PRODUCTION READY")
    elif score >= 60:
        print("STATUS: NEEDS MINOR FIXES")
    else:
        print("STATUS: NEEDS MAJOR FIXES")
    
    # Detailed results
    print("\nDETAILED RESULTS:")
    print("-" * 30)
    
    for category, category_issues in issues.items():
        if category_issues:
            print(f"\n{category.upper()} ISSUES:")
            for issue in category_issues:
                print(f"  - {issue}")
        else:
            print(f"\n{category.upper()}: OK")
    
    # Recommendations
    print("\nRECOMMENDations:")
    print("-" * 30)
    
    if issues['config']:
        print("1. Run: python quick_production_fixes.py")
    
    if issues['environment']:
        print("2. Copy .env.template to .env and add API keys")
    
    if issues['files']:
        print("3. Run production fixes to create missing files")
    
    if score < 80:
        print("4. Address issues above before production deployment")
    else:
        print("System ready for production deployment!")
    
    return score >= 80

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)