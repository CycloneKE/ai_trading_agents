#!/usr/bin/env python3
"""
Enhanced Error Handler and System Diagnostics
Provides comprehensive error detection, logging, and recovery
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading
import time

@dataclass
class ErrorReport:
    timestamp: str
    error_type: str
    message: str
    traceback: str
    context: Dict[str, Any]
    severity: str
    recovery_action: Optional[str] = None

class EnhancedErrorHandler:
    def __init__(self):
        self.error_log = []
        self.error_counts = {}
        self.recovery_actions = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Setup enhanced logging"""
        os.makedirs('logs', exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # File handler for errors
        error_handler = logging.FileHandler('logs/errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # File handler for all logs
        all_handler = logging.FileHandler('logs/system.log')
        all_handler.setLevel(logging.INFO)
        all_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(all_handler)
        root_logger.addHandler(console_handler)
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorReport:
        """Handle and log errors with context"""
        error_type = type(error).__name__
        message = str(error)
        tb = traceback.format_exc()
        
        # Determine severity
        severity = self.determine_severity(error_type, message)
        
        # Create error report
        report = ErrorReport(
            timestamp=datetime.now().isoformat(),
            error_type=error_type,
            message=message,
            traceback=tb,
            context=context or {},
            severity=severity
        )
        
        # Log error
        logger = logging.getLogger(__name__)
        if severity == 'CRITICAL':
            logger.critical(f"{error_type}: {message}")
        elif severity == 'HIGH':
            logger.error(f"{error_type}: {message}")
        elif severity == 'MEDIUM':
            logger.warning(f"{error_type}: {message}")
        else:
            logger.info(f"{error_type}: {message}")
        
        # Track error frequency
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to error log
        self.error_log.append(report)
        
        # Attempt recovery
        recovery_action = self.suggest_recovery(error_type, message, context)
        if recovery_action:
            report.recovery_action = recovery_action
            logger.info(f"Recovery suggestion: {recovery_action}")
        
        return report
    
    def determine_severity(self, error_type: str, message: str) -> str:
        """Determine error severity"""
        critical_errors = ['SystemExit', 'KeyboardInterrupt', 'MemoryError']
        high_errors = ['ConnectionError', 'TimeoutError', 'AuthenticationError', 'APIError']
        medium_errors = ['ValueError', 'KeyError', 'FileNotFoundError']
        
        if error_type in critical_errors:
            return 'CRITICAL'
        elif error_type in high_errors or 'connection' in message.lower():
            return 'HIGH'
        elif error_type in medium_errors:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def suggest_recovery(self, error_type: str, message: str, context: Dict) -> Optional[str]:
        """Suggest recovery actions based on error type"""
        recovery_map = {
            'ConnectionError': 'Check network connectivity and API endpoints',
            'TimeoutError': 'Increase timeout values or check server response time',
            'AuthenticationError': 'Verify API keys and credentials in .env file',
            'FileNotFoundError': 'Check file paths and ensure required files exist',
            'KeyError': 'Verify configuration keys and data structure',
            'ValueError': 'Check input data format and validation',
            'ImportError': 'Install missing dependencies with pip install',
            'PermissionError': 'Check file permissions and user access rights'
        }
        
        # Check for specific patterns
        if 'api key' in message.lower():
            return 'Check API key configuration in .env file'
        elif 'port' in message.lower() and 'refused' in message.lower():
            return 'Start the required service or check port availability'
        elif 'symbol' in message.lower() and 'unknown' in message.lower():
            return 'Verify symbol format and data source configuration'
        
        return recovery_map.get(error_type)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary and statistics"""
        total_errors = len(self.error_log)
        recent_errors = [e for e in self.error_log if 
                        (datetime.now() - datetime.fromisoformat(e.timestamp)).seconds < 3600]
        
        severity_counts = {}
        for error in self.error_log:
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1
        
        return {
            'total_errors': total_errors,
            'recent_errors': len(recent_errors),
            'error_types': dict(self.error_counts),
            'severity_distribution': severity_counts,
            'most_common_errors': sorted(self.error_counts.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]
        }
    
    def diagnose_system_issues(self) -> Dict[str, Any]:
        """Diagnose common system issues"""
        issues = []
        
        # Check configuration
        if not os.path.exists('config/config.json'):
            issues.append({
                'type': 'configuration',
                'message': 'Missing config/config.json file',
                'severity': 'HIGH',
                'fix': 'Create configuration file or run setup'
            })
        
        # Check environment variables
        required_env_vars = [
            'TRADING_ALPHA_VANTAGE_API_KEY',
            'TRADING_FMP_API_KEY',
            'OANDA_API_KEY'
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            issues.append({
                'type': 'environment',
                'message': f'Missing environment variables: {", ".join(missing_vars)}',
                'severity': 'HIGH',
                'fix': 'Set missing variables in .env file'
            })
        
        # Check dependencies
        try:
            import requests
            import pandas
            import numpy
        except ImportError as e:
            issues.append({
                'type': 'dependencies',
                'message': f'Missing dependency: {e}',
                'severity': 'HIGH',
                'fix': 'Run: pip install -r requirements.txt'
            })
        
        # Check disk space
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
        if free_space < 1:
            issues.append({
                'type': 'system',
                'message': f'Low disk space: {free_space:.1f}GB remaining',
                'severity': 'MEDIUM',
                'fix': 'Free up disk space'
            })
        
        return {
            'issues_found': len(issues),
            'issues': issues,
            'system_health': 'GOOD' if len(issues) == 0 else 'ISSUES_DETECTED'
        }

class SystemDiagnostics:
    def __init__(self):
        self.error_handler = EnhancedErrorHandler()
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostic"""
        print("Running System Diagnostics...")
        print("=" * 40)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'diagnostics': {},
            'recommendations': []
        }
        
        # Check system issues
        system_issues = self.error_handler.diagnose_system_issues()
        results['diagnostics']['system_issues'] = system_issues
        
        # Check error history
        error_summary = self.error_handler.get_error_summary()
        results['diagnostics']['error_summary'] = error_summary
        
        # Generate recommendations
        if system_issues['issues_found'] > 0:
            for issue in system_issues['issues']:
                results['recommendations'].append({
                    'priority': issue['severity'],
                    'action': issue['fix'],
                    'reason': issue['message']
                })
        
        # Check recent error patterns
        if error_summary['recent_errors'] > 5:
            results['recommendations'].append({
                'priority': 'HIGH',
                'action': 'Investigate recent error spike',
                'reason': f'{error_summary["recent_errors"]} errors in last hour'
            })
        
        return results
    
    def print_diagnostic_report(self, results: Dict[str, Any]):
        """Print formatted diagnostic report"""
        print(f"\nDiagnostic Report - {results['timestamp']}")
        print("=" * 50)
        
        # System Issues
        issues = results['diagnostics']['system_issues']
        print(f"\nSystem Health: {issues['system_health']}")
        print(f"Issues Found: {issues['issues_found']}")
        
        if issues['issues']:
            print("\nDetected Issues:")
            for i, issue in enumerate(issues['issues'], 1):
                print(f"{i}. [{issue['severity']}] {issue['message']}")
                print(f"   Fix: {issue['fix']}")
        
        # Error Summary
        errors = results['diagnostics']['error_summary']
        print(f"\nError Statistics:")
        print(f"Total Errors: {errors['total_errors']}")
        print(f"Recent Errors: {errors['recent_errors']}")
        
        if errors['most_common_errors']:
            print("\nMost Common Errors:")
            for error_type, count in errors['most_common_errors']:
                print(f"  {error_type}: {count}")
        
        # Recommendations
        if results['recommendations']:
            print("\nRecommendations:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"{i}. [{rec['priority']}] {rec['action']}")
                print(f"   Reason: {rec['reason']}")

def main():
    """Main diagnostic function"""
    diagnostics = SystemDiagnostics()
    
    try:
        # Run diagnostics
        results = diagnostics.run_full_diagnostic()
        
        # Print report
        diagnostics.print_diagnostic_report(results)
        
        # Save report
        os.makedirs('logs', exist_ok=True)
        with open('logs/diagnostic_report.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nFull report saved to: logs/diagnostic_report.json")
        
    except Exception as e:
        print(f"Error running diagnostics: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()