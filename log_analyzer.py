import re
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter

class LogAnalyzer:
    def __init__(self, log_file='logs/trading_agent.log'):
        self.log_file = log_file
        self.error_patterns = [
            r'ERROR.*',
            r'CRITICAL.*',
            r'Exception.*',
            r'Traceback.*'
        ]
    
    def analyze_recent_logs(self, hours=24):
        """Analyze logs from recent hours"""
        if not os.path.exists(self.log_file):
            return {'error': 'Log file not found'}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        stats = {
            'total_lines': 0,
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0,
            'error_types': Counter(),
            'hourly_errors': defaultdict(int),
            'recent_errors': []
        }
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    stats['total_lines'] += 1
                    
                    # Parse timestamp
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        try:
                            log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                            if log_time < cutoff_time:
                                continue
                        except:
                            continue
                    
                    # Count log levels
                    if 'ERROR' in line:
                        stats['error_count'] += 1
                        stats['recent_errors'].append(line.strip())
                        hour = timestamp_match.group(1)[:13] if timestamp_match else 'unknown'
                        stats['hourly_errors'][hour] += 1
                    elif 'WARNING' in line:
                        stats['warning_count'] += 1
                    elif 'INFO' in line:
                        stats['info_count'] += 1
                    
                    # Categorize errors
                    for pattern in self.error_patterns:
                        if re.search(pattern, line):
                            error_type = pattern.split('.*')[0]
                            stats['error_types'][error_type] += 1
        
        except Exception as e:
            stats['analysis_error'] = str(e)
        
        # Keep only recent errors
        stats['recent_errors'] = stats['recent_errors'][-10:]
        
        return stats
    
    def get_error_summary(self):
        """Get error summary for dashboard"""
        analysis = self.analyze_recent_logs(24)
        
        return {
            'error_rate': analysis['error_count'] / max(analysis['total_lines'], 1),
            'total_errors': analysis['error_count'],
            'error_trend': 'increasing' if analysis['error_count'] > 10 else 'stable',
            'top_errors': dict(analysis['error_types'].most_common(5)),
            'recent_errors': analysis['recent_errors']
        }

import os
log_analyzer = LogAnalyzer()
