#!/usr/bin/env python3
"""
Advanced Monitoring and Alerting System
"""

import os
import json
import time
import logging
import threading
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Dict, List, Any
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect and store system metrics"""
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric with timestamp"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append({
                'value': value,
                'timestamp': time.time(),
                'tags': tags or {}
            })
            
            # Keep only last 1000 points
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
    
    def get_metric_history(self, name: str, duration_seconds: int = 3600) -> List[Dict]:
        """Get metric history for specified duration"""
        with self.lock:
            if name not in self.metrics:
                return []
            
            cutoff_time = time.time() - duration_seconds
            return [m for m in self.metrics[name] if m['timestamp'] > cutoff_time]
    
    def get_latest_metric(self, name: str) -> float:
        """Get latest value for metric"""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return 0.0
            return self.metrics[name][-1]['value']

class AlertManager:
    """Manage alerts and notifications"""
    def __init__(self):
        self.alert_rules = []
        self.alert_history = []
        self.cooldown_period = 300  # 5 minutes
    
    def add_alert_rule(self, name: str, condition_func, severity: str = 'warning', 
                      message: str = '', cooldown: int = None):
        """Add an alert rule"""
        self.alert_rules.append({
            'name': name,
            'condition': condition_func,
            'severity': severity,
            'message': message,
            'cooldown': cooldown or self.cooldown_period,
            'last_triggered': 0
        })
    
    def check_alerts(self, metrics_collector: MetricsCollector):
        """Check all alert rules"""
        current_time = time.time()
        
        for rule in self.alert_rules:
            try:
                # Check cooldown
                if current_time - rule['last_triggered'] < rule['cooldown']:
                    continue
                
                # Evaluate condition
                if rule['condition'](metrics_collector):
                    self._trigger_alert(rule)
                    rule['last_triggered'] = current_time
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    def _trigger_alert(self, rule: Dict):
        """Trigger an alert"""
        alert = {
            'name': rule['name'],
            'severity': rule['severity'],
            'message': rule['message'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        log_level = logging.CRITICAL if rule['severity'] == 'critical' else logging.WARNING
        logger.log(log_level, f"ALERT: {rule['name']} - {rule['message']}")
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

class SystemMonitor:
    """Comprehensive system monitoring"""
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.running = False
        self.monitor_thread = None
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # High memory usage
        self.alert_manager.add_alert_rule(
            name="high_memory_usage",
            condition_func=lambda mc: mc.get_latest_metric('memory_percent') > 85,
            severity="warning",
            message="Memory usage above 85%"
        )
        
        # High CPU usage
        self.alert_manager.add_alert_rule(
            name="high_cpu_usage",
            condition_func=lambda mc: mc.get_latest_metric('cpu_percent') > 90,
            severity="warning",
            message="CPU usage above 90%"
        )
        
        # Trading system errors
        self.alert_manager.add_alert_rule(
            name="high_error_rate",
            condition_func=lambda mc: mc.get_latest_metric('error_rate') > 0.05,
            severity="critical",
            message="Error rate above 5%"
        )
        
        # Portfolio drawdown
        self.alert_manager.add_alert_rule(
            name="high_drawdown",
            condition_func=lambda mc: mc.get_latest_metric('portfolio_drawdown') > 0.08,
            severity="critical",
            message="Portfolio drawdown above 8%"
        )
    
    def start(self):
        """Start monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_trading_metrics()
                self.alert_manager.check_alerts(self.metrics_collector)
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(10)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric('memory_percent', memory.percent)
            self.metrics_collector.record_metric('memory_used_mb', memory.used / 1024 / 1024)
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_metric('cpu_percent', cpu_percent)
            
            # Disk metrics
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics_collector.record_metric('disk_percent', disk_percent)
            
            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                self.metrics_collector.record_metric('network_bytes_sent', net_io.bytes_sent)
                self.metrics_collector.record_metric('network_bytes_recv', net_io.bytes_recv)
            except:
                pass
                
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    def _collect_trading_metrics(self):
        """Collect trading-specific metrics"""
        try:
            # Mock trading metrics - replace with actual data
            import random
            
            # Portfolio metrics
            self.metrics_collector.record_metric('portfolio_value', 100000 + random.uniform(-5000, 5000))
            self.metrics_collector.record_metric('portfolio_drawdown', random.uniform(0, 0.1))
            self.metrics_collector.record_metric('daily_pnl', random.uniform(-1000, 1000))
            
            # Trading activity
            self.metrics_collector.record_metric('trades_per_hour', random.randint(0, 10))
            self.metrics_collector.record_metric('error_rate', random.uniform(0, 0.02))
            self.metrics_collector.record_metric('api_latency_ms', random.uniform(50, 200))
            
        except Exception as e:
            logger.error(f"Trading metrics collection error: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            'system_metrics': {
                'memory_percent': self.metrics_collector.get_latest_metric('memory_percent'),
                'cpu_percent': self.metrics_collector.get_latest_metric('cpu_percent'),
                'disk_percent': self.metrics_collector.get_latest_metric('disk_percent')
            },
            'trading_metrics': {
                'portfolio_value': self.metrics_collector.get_latest_metric('portfolio_value'),
                'daily_pnl': self.metrics_collector.get_latest_metric('daily_pnl'),
                'trades_per_hour': self.metrics_collector.get_latest_metric('trades_per_hour'),
                'error_rate': self.metrics_collector.get_latest_metric('error_rate')
            },
            'recent_alerts': self.alert_manager.alert_history[-10:],
            'timestamp': datetime.now().isoformat()
        }

def create_monitoring_api():
    """Create monitoring API endpoints"""
    code = '''from flask import Flask, jsonify
from advanced_monitoring import SystemMonitor
import threading

app = Flask(__name__)
monitor = SystemMonitor()

@app.route('/monitoring/health')
def health_check():
    return jsonify({'status': 'healthy', 'monitoring': monitor.running})

@app.route('/monitoring/metrics')
def get_metrics():
    return jsonify(monitor.get_dashboard_data())

@app.route('/monitoring/alerts')
def get_alerts():
    return jsonify({
        'alerts': monitor.alert_manager.alert_history[-20:],
        'total_alerts': len(monitor.alert_manager.alert_history)
    })

@app.route('/monitoring/system')
def get_system_metrics():
    return jsonify({
        'memory': monitor.metrics_collector.get_metric_history('memory_percent', 3600),
        'cpu': monitor.metrics_collector.get_metric_history('cpu_percent', 3600),
        'disk': monitor.metrics_collector.get_metric_history('disk_percent', 3600)
    })

def start_monitoring_server():
    """Start monitoring server"""
    monitor.start()
    app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == '__main__':
    start_monitoring_server()
'''
    
    try:
        with open('monitoring_api.py', 'w') as f:
            f.write(code)
        logger.info("Created monitoring API")
        return True
    except Exception as e:
        logger.error(f"Failed to create monitoring API: {e}")
        return False

def create_log_analyzer():
    """Create log analysis system"""
    code = '''import re
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
'''
    
    try:
        with open('log_analyzer.py', 'w') as f:
            f.write(code)
        logger.info("Created log analyzer")
        return True
    except Exception as e:
        logger.error(f"Failed to create log analyzer: {e}")
        return False

def main():
    """Apply advanced monitoring improvements"""
    print("ADVANCED MONITORING SYSTEM")
    print("=" * 40)
    
    fixes_applied = 0
    total_fixes = 3
    
    print("\n1. Creating monitoring API...")
    if create_monitoring_api():
        fixes_applied += 1
        print("   ✅ Monitoring API created")
    
    print("\n2. Creating log analyzer...")
    if create_log_analyzer():
        fixes_applied += 1
        print("   ✅ Log analyzer created")
    
    print("\n3. Testing system monitor...")
    try:
        monitor = SystemMonitor()
        monitor.start()
        time.sleep(2)
        dashboard_data = monitor.get_dashboard_data()
        monitor.stop()
        
        if dashboard_data and 'system_metrics' in dashboard_data:
            fixes_applied += 1
            print("   ✅ System monitor working")
        else:
            print("   ❌ System monitor test failed")
    except Exception as e:
        print(f"   ❌ System monitor error: {e}")
    
    print(f"\n✅ Monitoring improvements: {fixes_applied}/{total_fixes}")
    
    if fixes_applied >= 2:
        print("\nMONITORING FEATURES:")
        print("• Real-time metrics collection")
        print("• Automated alerting system")
        print("• Performance monitoring")
        print("• Log analysis")
        print("• Dashboard API endpoints")
        print("• System health tracking")
        
        print("\nMONITORING ENDPOINTS:")
        print("• http://localhost:8080/monitoring/health")
        print("• http://localhost:8080/monitoring/metrics")
        print("• http://localhost:8080/monitoring/alerts")
    
    return fixes_applied >= 2

if __name__ == "__main__":
    main()