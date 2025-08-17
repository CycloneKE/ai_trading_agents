#!/usr/bin/env python3
"""
Quick Paper Trading Deployment - One Command Setup
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def check_dependencies():
    """Check if required packages are installed"""
    required = ['yfinance', 'pandas', 'numpy', 'scikit-learn', 'xgboost']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} - MISSING")
    
    return missing

def install_dependencies(packages):
    """Install missing packages"""
    if not packages:
        return True
    
    print(f"\nInstalling missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        print("✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install packages")
        return False

def verify_system():
    """Verify system components"""
    try:
        from supervised_learning import SupervisedLearningStrategy
        print("✓ AI Strategy module loaded")
        
        from paper_trading_deployment import PaperTradingEngine
        print("✓ Paper trading engine loaded")
        
        return True
    except ImportError as e:
        print(f"✗ System verification failed: {e}")
        return False

def create_quick_config():
    """Create optimized configuration for quick deployment"""
    config = {
        'model_type': 'xgboost',
        'threshold': 0.01,
        'features': ['close', 'volume', 'rsi', 'macd', 'ema_12', 'ema_26', 'atr'],
        'stop_loss_atr_multiplier': 2.0,
        'trailing_stop_enabled': True,
        'trend_following_enabled': True,
        'symbols': ['AAPL', 'MSFT'],  # Start with 2 symbols for quick testing
        'initial_capital': 100000,
        'max_positions': 2,
        'check_interval': 60,  # 1 minute for quick testing
        'bias_threshold': 0.15
    }
    
    with open('quick_deploy_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Quick deployment configuration created")
    return config

def run_quick_test():
    """Run a quick 30-minute test session"""
    try:
        from paper_trading_deployment import PaperTradingEngine
        
        config = create_quick_config()
        engine = PaperTradingEngine(config)
        
        print(f"""
{'='*50}
QUICK PAPER TRADING TEST - 30 MINUTES
{'='*50}

Starting quick test with:
• Symbols: {', '.join(config['symbols'])}
• Capital: ${config['initial_capital']:,}
• Check Interval: {config['check_interval']} seconds

Press Ctrl+C to stop early...
{'='*50}
""")
        
        # Run for 30 minutes (0.5 hours)
        engine.run_trading_session(duration_hours=0.5)
        
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        return False

def main():
    """Main quick deployment function"""
    print("""
🚀 AI TRADING AGENT - QUICK PAPER TRADING DEPLOYMENT
====================================================

This script will:
1. Check and install required dependencies
2. Verify system components
3. Run a 30-minute paper trading test
4. Generate performance report

Let's get started!
""")
    
    # Step 1: Check dependencies
    print("\n1. Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        install_ok = install_dependencies(missing)
        if not install_ok:
            print("\n❌ Deployment failed - could not install dependencies")
            return False
    
    # Step 2: Verify system
    print("\n2. Verifying system components...")
    if not verify_system():
        print("\n❌ Deployment failed - system verification error")
        return False
    
    # Step 3: Run quick test
    print("\n3. Running quick paper trading test...")
    if not run_quick_test():
        print("\n❌ Deployment failed - paper trading test error")
        return False
    
    print(f"""
🎉 QUICK DEPLOYMENT SUCCESSFUL!
===============================

Your AI trading agent has been successfully deployed and tested!

Next Steps:
1. Review the generated session report
2. If satisfied, run full paper trading: python paper_trading_deployment.py
3. Monitor performance for 1-2 weeks before considering live trading

Files Created:
• quick_deploy_config.json - Configuration used
• paper_trading_session_*.json - Test results

Ready for full deployment! 🚀
""")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nDeployment interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)