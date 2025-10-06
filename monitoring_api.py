from flask import Flask, jsonify
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
