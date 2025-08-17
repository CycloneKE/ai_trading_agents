"""
Security Audit Report for AI Trading Agent
"""

import os
import json
from typing import Dict, List

class SecurityAudit:
    def __init__(self):
        self.vulnerabilities = []
        self.missing_safeguards = []
        self.recommendations = []
    
    def audit_api_security(self):
        """Audit API key security."""
        issues = []
        
        # Check .env file exposure
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                content = f.read()
                if 'AIzaSy' in content:  # Exposed Gemini key
                    issues.append("CRITICAL: API keys exposed in .env file")
        
        # Check hardcoded keys
        py_files = [f for f in os.listdir('.') if f.endswith('.py')]
        for file in py_files:
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    if 'AIzaSy' in content or 'sk-' in content:
                        issues.append(f"CRITICAL: Hardcoded API key in {file}")
            except:
                pass
        
        return issues
    
    def audit_trading_safeguards(self):
        """Audit trading safety mechanisms."""
        missing = []
        
        # Check for position limits
        if not self._check_position_limits():
            missing.append("CRITICAL: No position size limits")
        
        # Check for daily loss limits
        if not self._check_loss_limits():
            missing.append("CRITICAL: No daily loss limits")
        
        # Check for emergency stop
        if not self._check_emergency_stop():
            missing.append("HIGH: No emergency stop mechanism")
        
        # Check for human approval for large trades
        if not self._check_human_approval():
            missing.append("HIGH: No human approval for large trades")
        
        return missing
    
    def audit_ai_guardrails(self):
        """Audit AI decision guardrails."""
        missing = []
        
        # Check for confidence thresholds
        if not self._check_confidence_thresholds():
            missing.append("MEDIUM: No AI confidence thresholds")
        
        # Check for decision logging
        if not self._check_decision_logging():
            missing.append("HIGH: No AI decision audit trail")
        
        # Check for bias detection
        if not self._check_bias_detection():
            missing.append("MEDIUM: No AI bias detection")
        
        return missing
    
    def audit_data_security(self):
        """Audit data protection."""
        issues = []
        
        # Check database encryption
        if os.path.exists('trading_memory.db'):
            issues.append("MEDIUM: Database not encrypted")
        
        # Check log file security
        if os.path.exists('logs/'):
            issues.append("LOW: Log files may contain sensitive data")
        
        return issues
    
    def _check_position_limits(self) -> bool:
        """Check if position limits are implemented."""
        # Look for position limit code
        try:
            with open('main.py', 'r') as f:
                content = f.read()
                return 'max_position_size' in content
        except:
            return False
    
    def _check_loss_limits(self) -> bool:
        """Check if daily loss limits exist."""
        return False  # Not implemented
    
    def _check_emergency_stop(self) -> bool:
        """Check if emergency stop exists."""
        try:
            with open('unified_control_center.py', 'r') as f:
                content = f.read()
                return 'emergency' in content.lower()
        except:
            return False
    
    def _check_human_approval(self) -> bool:
        """Check if human approval system exists."""
        return False  # Not implemented
    
    def _check_confidence_thresholds(self) -> bool:
        """Check if AI confidence thresholds exist."""
        try:
            with open('base_strategy.py', 'r') as f:
                content = f.read()
                return 'confidence' in content and 'threshold' in content
        except:
            return False
    
    def _check_decision_logging(self) -> bool:
        """Check if AI decisions are logged."""
        return os.path.exists('learning_log.json')
    
    def _check_bias_detection(self) -> bool:
        """Check if AI bias detection exists."""
        return False  # Not implemented
    
    def generate_report(self) -> Dict:
        """Generate comprehensive security report."""
        
        api_issues = self.audit_api_security()
        trading_issues = self.audit_trading_safeguards()
        ai_issues = self.audit_ai_guardrails()
        data_issues = self.audit_data_security()
        
        all_issues = api_issues + trading_issues + ai_issues + data_issues
        
        # Calculate risk score
        critical_count = len([i for i in all_issues if 'CRITICAL' in i])
        high_count = len([i for i in all_issues if 'HIGH' in i])
        medium_count = len([i for i in all_issues if 'MEDIUM' in i])
        
        risk_score = critical_count * 10 + high_count * 5 + medium_count * 2
        
        if risk_score >= 30:
            risk_level = "EXTREME RISK"
        elif risk_score >= 20:
            risk_level = "HIGH RISK"
        elif risk_score >= 10:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'total_issues': len(all_issues),
            'critical_issues': critical_count,
            'high_issues': high_count,
            'medium_issues': medium_count,
            'api_security': api_issues,
            'trading_safeguards': trading_issues,
            'ai_guardrails': ai_issues,
            'data_security': data_issues,
            'recommendations': self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get security recommendations."""
        return [
            "IMMEDIATE: Remove API keys from .env file",
            "IMMEDIATE: Implement position size limits (max 5% per trade)",
            "IMMEDIATE: Add daily loss limits (max 10% portfolio)",
            "HIGH: Add human approval for trades >$10,000",
            "HIGH: Implement emergency stop mechanism",
            "HIGH: Add AI decision audit logging",
            "MEDIUM: Encrypt trading database",
            "MEDIUM: Add AI confidence thresholds",
            "MEDIUM: Implement bias detection",
            "LOW: Secure log files"
        ]

def run_security_audit():
    """Run complete security audit."""
    
    print("AI Trading Agent - Security Audit")
    print("=" * 40)
    
    auditor = SecurityAudit()
    report = auditor.generate_report()
    
    print(f"\nRISK LEVEL: {report['risk_level']}")
    print(f"RISK SCORE: {report['risk_score']}/100")
    print(f"TOTAL ISSUES: {report['total_issues']}")
    
    print(f"\nISSUE BREAKDOWN:")
    print(f"  Critical: {report['critical_issues']}")
    print(f"  High: {report['high_issues']}")
    print(f"  Medium: {report['medium_issues']}")
    
    print(f"\nCRITICAL ISSUES:")
    for issue in report['api_security'] + report['trading_safeguards']:
        if 'CRITICAL' in issue:
            print(f"  - {issue}")
    
    print(f"\nHIGH PRIORITY ISSUES:")
    for issue in report['trading_safeguards'] + report['ai_guardrails']:
        if 'HIGH' in issue:
            print(f"  - {issue}")
    
    print(f"\nRECOMMENDATIONS:")
    for rec in report['recommendations'][:5]:  # Top 5
        print(f"  - {rec}")
    
    return report

if __name__ == '__main__':
    run_security_audit()