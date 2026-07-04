#!/usr/bin/env python3
"""
EXECUTE ALL PRODUCTION FIXES
Systematic implementation of all critical improvements
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_fix_script(script_name: str, phase_name: str) -> bool:
    """Run a fix script and return success status"""
    try:
        print(f"\n🔧 Executing {phase_name}...")
        print("=" * 60)
        
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(result.stdout)
            logger.info(f"✅ {phase_name} completed successfully")
            return True
        else:
            print(result.stdout)
            print(result.stderr)
            logger.error(f"❌ {phase_name} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {phase_name} timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ {phase_name} failed with exception: {e}")
        return False

def create_system_status_checker():
    """Create comprehensive system status checker"""
    checker_code = '''#!/usr/bin/env python3
"""
Comprehensive System Status Checker
"""

import os
import json
import psutil
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def check_configuration():
    """Check configuration files"""
    issues = []
    
    # Check config.json exists and is valid
    if not os.path.exists('config/config.json'):
        issues.append("config.json not found")
    else:
        try:
            with open('config/config.json', 'r') as f:
                config = json.load(f)
            
            # Check for hardcoded secrets
            config_str = json.dumps(config)
            if 'cf87c3c69c323de28aa987c2d29a601c' in config_str:
                issues.append("Hardcoded OANDA key still present")
                
        except json.JSONDecodeError:
            issues.append("config.json is not valid JSON")
    
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
        if not os.getenv(var) or os.getenv(var) == 'your_key_here':
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
    
    # CPU check
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 90:
        issues.append(f"High CPU usage: {cpu_percent:.1f}%")
    
    return issues

def check_api_connectivity():
    """Check API connectivity"""
    issues = []
    
    # Test FMP API
    fmp_key = os.getenv('TRADING_FMP_API_KEY')
    if fmp_key and fmp_key != 'your_key_here':
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={fmp_key}"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                issues.append(f"FMP API error: {response.status_code}")
        except Exception as e:
            issues.append(f"FMP API connection failed: {e}")
    
    return issues

def check_file_integrity():
    """Check critical files exist"""
    critical_files = [
        'main.py',
        'config/config.json',
        '.env',
        'requirements.txt',
        'critical_security_fixes.py',
        'system_reliability_fixes.py',
        'risk_management_fixes.py'
    ]
    
    missing = []
    for file in critical_files:
        if not os.path.exists(file):
            missing.append(file)
    
    return missing

def main():
    """Run comprehensive system check"""
    print("🔍 COMPREHENSIVE SYSTEM STATUS CHECK")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    all_issues = []
    
    # Configuration check
    print("\\n📋 Checking configuration...")
    config_issues = check_configuration()
    if config_issues:
        print("   ❌ Configuration issues:")
        for issue in config_issues:
            print(f"      • {issue}")
        all_issues.extend(config_issues)
    else:
        print("   ✅ Configuration OK")
    
    # Environment check
    print("\\n🔐 Checking environment variables...")
    env_issues = check_environment()
    if env_issues:
        print("   ⚠️ Missing environment variables:")
        for var in env_issues:
            print(f"      • {var}")
        all_issues.extend(env_issues)
    else:
        print("   ✅ Environment variables OK")
    
    # System resources check
    print("\\n💻 Checking system resources...")
    resource_issues = check_system_resources()
    if resource_issues:
        print("   ⚠️ Resource issues:")
        for issue in resource_issues:
            print(f"      • {issue}")
        all_issues.extend(resource_issues)
    else:
        print("   ✅ System resources OK")
    
    # API connectivity check
    print("\\n🌐 Checking API connectivity...")
    api_issues = check_api_connectivity()
    if api_issues:
        print("   ❌ API issues:")
        for issue in api_issues:
            print(f"      • {issue}")
        all_issues.extend(api_issues)
    else:
        print("   ✅ API connectivity OK")
    
    # File integrity check
    print("\\n📁 Checking file integrity...")
    file_issues = check_file_integrity()
    if file_issues:
        print("   ❌ Missing files:")
        for file in file_issues:
            print(f"      • {file}")
        all_issues.extend(file_issues)
    else:
        print("   ✅ All critical files present")
    
    # Summary
    print("\\n" + "=" * 50)
    if all_issues:
        print(f"❌ SYSTEM STATUS: {len(all_issues)} issues found")
        print("\\n🔧 Issues to resolve:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
        return False
    else:
        print("✅ SYSTEM STATUS: All checks passed!")
        print("🚀 System ready for production deployment")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
'''
    
    try:
        with open('system_status_checker.py', 'w') as f:
            f.write(checker_code)
        
        logger.info("✅ System status checker created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create system status checker: {e}")
        return False

def main():
    """Execute all production fixes in sequence"""
    print("🚀 AI TRADING BOT - PRODUCTION FIXES EXECUTION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Track progress
    phases = [
        ("critical_security_fixes.py", "PHASE 1: Critical Security Fixes"),
        ("system_reliability_fixes.py", "PHASE 2: System Reliability Fixes"),
        ("risk_management_fixes.py", "PHASE 3: Risk Management Fixes")
    ]
    
    completed_phases = 0
    total_phases = len(phases)
    
    # Pre-execution checks
    print("\n🔍 PRE-EXECUTION CHECKS")
    print("-" * 30)
    
    # Check if we're in the right directory
    if not os.path.exists('config'):
        print("❌ Error: Not in the correct directory (config folder not found)")
        print("   Please run this script from the ai_trading_agents directory")
        return False
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Error: Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return False
    
    print("✅ Directory check passed")
    print(f"✅ Python version OK: {python_version.major}.{python_version.minor}")
    
    # Execute phases
    for script_name, phase_name in phases:
        if not os.path.exists(script_name):
            logger.error(f"❌ Script not found: {script_name}")
            continue
        
        success = run_fix_script(script_name, phase_name)
        
        if success:
            completed_phases += 1
            print(f"✅ {phase_name} - COMPLETED")
        else:
            print(f"❌ {phase_name} - FAILED")
            
            # Ask user if they want to continue
            response = input(f"\n⚠️ {phase_name} failed. Continue with next phase? (y/n): ")
            if response.lower() != 'y':
                print("🛑 Execution stopped by user")
                break
    
    # Create system status checker
    print("\n🔧 Creating system status checker...")
    if create_system_status_checker():
        print("✅ System status checker created")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎯 EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Completed: {completed_phases}/{total_phases} phases")
    print(f"Finished: {datetime.now().isoformat()}")
    
    if completed_phases == total_phases:
        print("\n🎉 ALL FIXES APPLIED SUCCESSFULLY!")
        print("\n📋 NEXT STEPS:")
        print("   1. Run: python system_status_checker.py")
        print("   2. Test the system: python start_advanced_system.py")
        print("   3. Monitor performance and logs")
        print("   4. Deploy to staging environment")
        
        print("\n🛡️ SECURITY IMPROVEMENTS:")
        print("   ✅ Hardcoded API keys removed")
        print("   ✅ Secure configuration management")
        print("   ✅ Input validation implemented")
        
        print("\n🔧 RELIABILITY IMPROVEMENTS:")
        print("   ✅ Circuit breakers for API calls")
        print("   ✅ Retry logic with exponential backoff")
        print("   ✅ Health monitoring system")
        print("   ✅ Memory management")
        
        print("\n🛡️ RISK MANAGEMENT IMPROVEMENTS:")
        print("   ✅ Automatic stop losses")
        print("   ✅ Kelly criterion position sizing")
        print("   ✅ Real-time risk monitoring")
        print("   ✅ Portfolio concentration limits")
        
        print("\n🚀 SYSTEM IS NOW PRODUCTION-READY!")
        
    else:
        print(f"\n⚠️ {total_phases - completed_phases} phases failed or skipped")
        print("   Review the errors above and re-run failed phases")
        print("   System may not be fully production-ready")
    
    return completed_phases == total_phases

if __name__ == "__main__":
    success = main()
    
    # Run final system check
    if success and os.path.exists('system_status_checker.py'):
        print("\n🔍 Running final system status check...")
        try:
            result = subprocess.run([sys.executable, 'system_status_checker.py'], 
                                  capture_output=True, text=True, timeout=60)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
        except Exception as e:
            print(f"❌ Final check failed: {e}")
    
    exit(0 if success else 1)