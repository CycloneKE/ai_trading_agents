#!/usr/bin/env python3
"""
Run All Fixes - Master script to apply all system improvements and fixes
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MasterFixer:
    def __init__(self):
        self.fixes_applied = []
        self.errors = []
    
    def run_script(self, script_name: str, description: str) -> bool:
        """Run a fix script and return success status"""
        try:
            print(f"Running: {description}")
            
            if not os.path.exists(script_name):
                print(f"  ⚠ Script not found: {script_name}")
                return False
            
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                print(f"  ✓ {description} completed")
                self.fixes_applied.append(description)
                return True
            else:
                print(f"  ✗ {description} failed")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")
                self.errors.append(f"{description}: {result.stderr[:100]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ {description} timed out")
            self.errors.append(f"{description}: Timeout")
            return False
        except Exception as e:
            print(f"  ✗ {description} error: {e}")
            self.errors.append(f"{description}: {str(e)}")
            return False
    
    def apply_all_fixes(self):
        """Apply all available fixes in order"""
        print("AI Trading System - Master Fix Application")
        print("=" * 50)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Define fix scripts in order of importance
        fix_scripts = [
            # Core system fixes
            ("enhanced_error_handler.py", "Enhanced Error Handling Setup"),
            ("fix_symbol_issues.py", "Symbol Issues Resolution"),
            ("fix_trading_errors.py", "Trading Error Fixes"),
            
            # System improvements
            ("system_validator.py", "System Validation"),
            ("performance_optimization.py", "Performance Optimization"),
            ("advanced_monitoring.py", "Advanced Monitoring Setup"),
            
            # Enhanced features
            ("enhanced_features.py", "Enhanced Trading Features"),
            ("apply_next_improvements.py", "Next Level Improvements"),
            
            # System recovery
            ("system_recovery.py", "System Recovery Setup"),
            ("quick_start_system.py", "Quick Start System")
        ]
        
        print("Applying fixes in order:")
        print("-" * 30)
        
        success_count = 0
        total_fixes = len(fix_scripts)
        
        for script, description in fix_scripts:
            if self.run_script(script, description):
                success_count += 1
            time.sleep(1)  # Brief pause between fixes
        
        # Summary
        print()
        print("Fix Application Summary")
        print("=" * 30)
        print(f"Fixes Applied: {success_count}/{total_fixes}")
        print(f"Success Rate: {(success_count/total_fixes)*100:.1f}%")
        
        if self.fixes_applied:
            print("\\nSuccessful Fixes:")
            for fix in self.fixes_applied:
                print(f"  ✓ {fix}")
        
        if self.errors:
            print("\\nErrors Encountered:")
            for error in self.errors:
                print(f"  ✗ {error}")
        
        # Final recommendations
        print("\\nNext Steps:")
        if success_count >= total_fixes * 0.8:  # 80% success rate
            print("✓ System is significantly improved!")
            print("1. Run: python quick_start_system.py")
            print("2. Check: python check_system_status.py")
            print("3. Monitor: python system_recovery.py --monitor")
        else:
            print("⚠ Some critical fixes failed")
            print("1. Check error logs in logs/ directory")
            print("2. Manually run failed scripts")
            print("3. Verify dependencies: pip install -r requirements.txt")
        
        return success_count >= total_fixes * 0.7  # 70% minimum success

def main():
    """Main function"""
    fixer = MasterFixer()
    
    try:
        success = fixer.apply_all_fixes()
        
        # Save results
        os.makedirs('logs', exist_ok=True)
        with open('logs/master_fix_results.txt', 'w') as f:
            f.write(f"Master Fix Results - {datetime.now()}\\n")
            f.write("=" * 50 + "\\n\\n")
            
            f.write("Successful Fixes:\\n")
            for fix in fixer.fixes_applied:
                f.write(f"  ✓ {fix}\\n")
            
            f.write("\\nErrors:\\n")
            for error in fixer.errors:
                f.write(f"  ✗ {error}\\n")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\\nFix application interrupted by user")
        return 1
    except Exception as e:
        print(f"\\nMaster fix failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())