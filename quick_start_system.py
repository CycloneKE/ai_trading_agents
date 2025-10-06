#!/usr/bin/env python3
"""
Quick Start System - One-click system initialization and startup
Handles all system setup, validation, and startup in correct order
"""

import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickStartSystem:
    def __init__(self):
        self.steps_completed = []
        self.errors = []
        
    def log_step(self, step: str, success: bool, message: str = ""):
        """Log completion of a step"""
        status = "✓" if success else "✗"
        print(f"{status} {step}")
        if message:
            print(f"  {message}")
        
        self.steps_completed.append({
            'step': step,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        if not success:
            self.errors.append(f"{step}: {message}")
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            self.log_step("Python Version Check", True, f"Python {version.major}.{version.minor}")
            return True
        else:
            self.log_step("Python Version Check", False, f"Python {version.major}.{version.minor} - Need 3.8+")
            return False
    
    def check_dependencies(self) -> bool:
        """Check and install dependencies"""
        try:
            # Check if requirements.txt exists
            if not os.path.exists('requirements.txt'):
                self.log_step("Dependencies Check", False, "requirements.txt not found")
                return False
            
            # Try importing key packages
            required_packages = ['requests', 'pandas', 'numpy', 'flask']
            missing = []
            
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing.append(package)
            
            if missing:
                self.log_step("Dependencies Check", False, f"Missing: {', '.join(missing)}")
                
                # Attempt to install
                print("  Installing missing dependencies...")
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log_step("Dependencies Install", True, "All dependencies installed")
                    return True
                else:
                    self.log_step("Dependencies Install", False, result.stderr[:100])
                    return False
            else:
                self.log_step("Dependencies Check", True, "All dependencies available")
                return True
                
        except Exception as e:
            self.log_step("Dependencies Check", False, str(e))
            return False
    
    def setup_environment(self) -> bool:
        """Setup environment variables"""
        try:
            # Check if .env exists
            if not os.path.exists('.env'):
                if os.path.exists('.env.example'):
                    # Copy example file
                    with open('.env.example', 'r') as src, open('.env', 'w') as dst:
                        dst.write(src.read())
                    self.log_step("Environment Setup", True, "Created .env from example")
                else:
                    # Create basic .env
                    env_content = """# Trading API Keys
TRADING_ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
TRADING_FMP_API_KEY=your_fmp_key
TRADING_FINNHUB_API_KEY=your_finnhub_key

# Coinbase API
COINBASE_API_KEY=your_coinbase_key
COINBASE_API_SECRET=your_coinbase_secret
COINBASE_PASSPHRASE=your_coinbase_passphrase

# OANDA API
OANDA_API_KEY=your_oanda_key
OANDA_ACCOUNT_ID=your_oanda_account_id
"""
                    with open('.env', 'w') as f:
                        f.write(env_content)
                    self.log_step("Environment Setup", True, "Created basic .env file")
            else:
                self.log_step("Environment Setup", True, ".env file exists")
            
            # Load environment variables
            if os.path.exists('.env'):
                from dotenv import load_dotenv
                load_dotenv()
            
            return True
            
        except Exception as e:
            self.log_step("Environment Setup", False, str(e))
            return False
    
    def setup_configuration(self) -> bool:
        """Setup configuration files"""
        try:
            # Create config directory
            os.makedirs('config', exist_ok=True)
            
            # Check if config exists
            config_path = 'config/config.json'
            if not os.path.exists(config_path):
                # Create basic config
                basic_config = {
                    "data_manager": {
                        "symbols": ["AAPL", "GOOGL", "MSFT", "BTC-USD", "ETH-USD"],
                        "update_interval": 30,
                        "market_data_interval": 60,
                        "connectors": ["alpha_vantage", "fmp"],
                        "redis": {"enabled": False}
                    },
                    "strategies": {
                        "momentum": {"enabled": True, "lookback_period": 20},
                        "mean_reversion": {"enabled": True, "lookback_period": 50},
                        "sentiment": {"enabled": False}
                    },
                    "risk_management": {
                        "max_position_size": 0.05,
                        "max_portfolio_risk": 0.02,
                        "stop_loss_pct": 0.02,
                        "take_profit_pct": 0.04
                    },
                    "brokers": {
                        "paper": {"enabled": True, "initial_capital": 100000},
                        "coinbase": {"enabled": False},
                        "oanda": {"enabled": False}
                    },
                    "trading": {
                        "initial_capital": 100000,
                        "commission": 0.001,
                        "slippage": 0.0005
                    },
                    "monitoring": {
                        "enabled": True,
                        "port": 8080,
                        "metrics_interval": 60
                    },
                    "logging": {
                        "level": "INFO",
                        "file": "logs/trading_agent.log",
                        "max_size": "10MB",
                        "backup_count": 5
                    }
                }
                
                with open(config_path, 'w') as f:
                    json.dump(basic_config, f, indent=2)
                
                self.log_step("Configuration Setup", True, "Created basic config.json")
            else:
                self.log_step("Configuration Setup", True, "config.json exists")
            
            return True
            
        except Exception as e:
            self.log_step("Configuration Setup", False, str(e))
            return False
    
    def setup_directories(self) -> bool:
        """Setup required directories"""
        try:
            directories = ['logs', 'data', 'backups', 'src']
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            
            self.log_step("Directory Setup", True, f"Created: {', '.join(directories)}")
            return True
            
        except Exception as e:
            self.log_step("Directory Setup", False, str(e))
            return False
    
    def validate_system(self) -> bool:
        """Validate system is ready"""
        try:
            # Run system validator if available
            if os.path.exists('system_validator.py'):
                result = subprocess.run([sys.executable, 'system_validator.py'], 
                                      capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.log_step("System Validation", True, "All checks passed")
                    return True
                else:
                    self.log_step("System Validation", False, "Some checks failed")
                    return False
            else:
                # Basic validation
                required_files = ['main.py', 'config/config.json']
                missing = [f for f in required_files if not os.path.exists(f)]
                
                if missing:
                    self.log_step("System Validation", False, f"Missing: {', '.join(missing)}")
                    return False
                else:
                    self.log_step("System Validation", True, "Basic validation passed")
                    return True
                    
        except Exception as e:
            self.log_step("System Validation", False, str(e))
            return False
    
    def start_services(self) -> bool:
        """Start system services"""
        try:
            # Use system recovery if available
            if os.path.exists('system_recovery.py'):
                print("  Starting services using system recovery...")
                result = subprocess.run([sys.executable, 'system_recovery.py'], 
                                      capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    self.log_step("Service Startup", True, "All services started")
                    return True
                else:
                    self.log_step("Service Startup", False, "Some services failed to start")
                    print(f"  Error: {result.stderr[:200]}")
                    return False
            else:
                # Basic startup
                print("  Starting main trading system...")
                # Don't wait for completion, just start it
                subprocess.Popen([sys.executable, 'main.py'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                time.sleep(3)  # Give it time to start
                self.log_step("Service Startup", True, "Trading system started")
                return True
                
        except Exception as e:
            self.log_step("Service Startup", False, str(e))
            return False
    
    def run_quick_start(self) -> Dict:
        """Run complete quick start process"""
        print("AI Trading System - Quick Start")
        print("=" * 40)
        print(f"Starting system initialization at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        start_time = time.time()
        
        # Run all setup steps
        steps = [
            ("Checking Python Version", self.check_python_version),
            ("Setting up Directories", self.setup_directories),
            ("Setting up Environment", self.setup_environment),
            ("Setting up Configuration", self.setup_configuration),
            ("Checking Dependencies", self.check_dependencies),
            ("Validating System", self.validate_system),
            ("Starting Services", self.start_services)
        ]
        
        success_count = 0
        for step_name, step_func in steps:
            try:
                if step_func():
                    success_count += 1
            except Exception as e:
                self.log_step(step_name, False, str(e))
        
        # Calculate results
        total_steps = len(steps)
        success_rate = (success_count / total_steps) * 100
        duration = time.time() - start_time
        
        # Print summary
        print()
        print("Quick Start Summary")
        print("-" * 20)
        print(f"Steps Completed: {success_count}/{total_steps} ({success_rate:.1f}%)")
        print(f"Duration: {duration:.1f} seconds")
        
        if success_count == total_steps:
            print("✓ System is ready!")
            print("\nNext steps:")
            print("1. Edit .env file with your API keys")
            print("2. Customize config/config.json as needed")
            print("3. Check system status: python check_system_status.py")
        else:
            print("⚠ Some issues detected:")
            for error in self.errors:
                print(f"  - {error}")
        
        return {
            'success': success_count == total_steps,
            'success_rate': success_rate,
            'duration': duration,
            'steps_completed': self.steps_completed,
            'errors': self.errors
        }

def main():
    """Main function"""
    quick_start = QuickStartSystem()
    
    try:
        results = quick_start.run_quick_start()
        
        # Save results
        os.makedirs('logs', exist_ok=True)
        with open('logs/quick_start_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return 0 if results['success'] else 1
        
    except KeyboardInterrupt:
        print("\nQuick start interrupted by user")
        return 1
    except Exception as e:
        print(f"\nQuick start failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())