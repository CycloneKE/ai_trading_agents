#!/usr/bin/env python3
"""
Bias Monitoring Dashboard for AI Trading Agent
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from src.bias_detector import BiasDetector

logger = logging.getLogger(__name__)

class BiasMonitor:
    """Monitor and report on algorithmic bias in trading decisions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bias_detector = BiasDetector(config)
        self.reports_history = []
    
    def generate_daily_report(self, decisions: List[Dict]) -> Dict[str, Any]:
        """Generate daily bias monitoring report"""
        if not decisions:
            return {'error': 'No decisions to analyze'}
        
        # Generate comprehensive bias report
        bias_report = self.bias_detector.generate_bias_report(decisions)
        
        # Add timestamp and summary
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': self._create_summary(bias_report),
            'detailed_analysis': bias_report,
            'recommendations': self.bias_detector.suggest_mitigation(bias_report)
        }
        
        # Store report
        self.reports_history.append(report)
        if len(self.reports_history) > 30:  # Keep 30 days
            self.reports_history = self.reports_history[-30:]
        
        return report
    
    def _create_summary(self, bias_report: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of bias analysis"""
        summary = {
            'overall_status': 'HEALTHY' if not bias_report.get('bias_detected', False) else 'BIAS_DETECTED',
            'bias_score': bias_report.get('overall_bias_score', 0.0),
            'total_decisions': bias_report.get('total_decisions', 0),
            'key_findings': []
        }
        
        # Sector bias findings
        if bias_report.get('sector_bias_detected', False):
            summary['key_findings'].append('Sector bias detected - uneven distribution across sectors')
        
        # Market cap bias findings
        if bias_report.get('cap_bias_detected', False):
            summary['key_findings'].append('Market cap bias detected - preference for certain company sizes')
        
        # Feature bias findings
        if bias_report.get('feature_bias_detected', False):
            summary['key_findings'].append('Feature bias detected - over-reliance on single indicators')
        
        # Performance impact
        if summary['bias_score'] > 0.5:
            summary['key_findings'].append('High bias score may impact performance and fairness')
        
        return summary
    
    def get_bias_trends(self, days: int = 7) -> Dict[str, Any]:
        """Analyze bias trends over time"""
        if len(self.reports_history) < 2:
            return {'error': 'Insufficient historical data'}
        
        recent_reports = self.reports_history[-days:] if len(self.reports_history) >= days else self.reports_history
        
        # Extract bias scores over time
        bias_scores = [r['detailed_analysis'].get('overall_bias_score', 0.0) for r in recent_reports]
        dates = [r['date'] for r in recent_reports]
        
        # Calculate trend
        if len(bias_scores) >= 2:
            trend = 'IMPROVING' if bias_scores[-1] < bias_scores[0] else 'WORSENING'
            change = bias_scores[-1] - bias_scores[0]
        else:
            trend = 'STABLE'
            change = 0.0
        
        return {
            'trend': trend,
            'change': change,
            'current_score': bias_scores[-1] if bias_scores else 0.0,
            'average_score': sum(bias_scores) / len(bias_scores) if bias_scores else 0.0,
            'dates': dates,
            'scores': bias_scores
        }
    
    def export_report(self, filepath: str, report: Dict[str, Any] = None) -> bool:
        """Export bias report to file"""
        try:
            if report is None:
                report = self.reports_history[-1] if self.reports_history else {}
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Bias report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export bias report: {e}")
            return False
    
    def print_summary(self, report: Dict[str, Any] = None):
        """Print bias report summary to console"""
        if report is None:
            report = self.reports_history[-1] if self.reports_history else {}
        
        if not report:
            print("No bias report available")
            return
        
        summary = report.get('summary', {})
        
        print("\n" + "="*50)
        print("AI TRADING AGENT - BIAS MONITORING REPORT")
        print("="*50)
        print(f"Date: {report.get('date', 'Unknown')}")
        print(f"Status: {summary.get('overall_status', 'Unknown')}")
        print(f"Bias Score: {summary.get('bias_score', 0.0):.3f}")
        print(f"Decisions Analyzed: {summary.get('total_decisions', 0)}")
        
        print("\nKey Findings:")
        findings = summary.get('key_findings', [])
        if findings:
            for finding in findings:
                print(f"  • {finding}")
        else:
            print("  • No significant bias detected")
        
        print("\nRecommendations:")
        recommendations = report.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                print(f"  • {rec}")
        else:
            print("  • Continue current practices")
        
        print("="*50 + "\n")

def main():
    """Test bias monitoring system"""
    # Sample configuration
    config = {
        'bias_threshold': 0.15,
        'bias_lookback_days': 30
    }
    
    # Create monitor
    monitor = BiasMonitor(config)
    
    # Sample decisions for testing
    sample_decisions = [
        {
            'action': 'buy',
            'symbol': 'AAPL',
            'sector': 'technology',
            'market_cap': 3000000000000,
            'returns': 0.02,
            'timestamp': datetime.now()
        },
        {
            'action': 'buy',
            'symbol': 'MSFT',
            'sector': 'technology',
            'market_cap': 2800000000000,
            'returns': 0.015,
            'timestamp': datetime.now()
        },
        {
            'action': 'sell',
            'symbol': 'XOM',
            'sector': 'energy',
            'market_cap': 400000000000,
            'returns': -0.01,
            'timestamp': datetime.now()
        }
    ]
    
    # Generate report
    report = monitor.generate_daily_report(sample_decisions)
    monitor.print_summary(report)

if __name__ == "__main__":
    main()