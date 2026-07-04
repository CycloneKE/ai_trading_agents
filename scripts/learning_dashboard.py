"""
Learning Dashboard - Web interface to see AI learning progress.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
from learning_tracker import get_learning_tracker

class LearningDashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/learning' or self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = self._generate_dashboard()
            self.wfile.write(html.encode())
        elif self.path == '/api/learning':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            tracker = get_learning_tracker()
            summary = tracker.get_learning_summary()
            self.wfile.write(json.dumps(summary, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def _generate_dashboard(self):
        tracker = get_learning_tracker()
        summary = tracker.get_learning_summary()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI Learning Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #0d1117; color: #c9d1d9; }}
        .header {{ background: #21262d; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metric {{ background: #161b22; padding: 15px; margin: 10px; border-radius: 8px; display: inline-block; min-width: 200px; }}
        .metric h3 {{ margin: 0 0 10px 0; color: #58a6ff; }}
        .metric .value {{ font-size: 24px; font-weight: bold; color: #7ee787; }}
        .event {{ background: #21262d; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #58a6ff; }}
        .model {{ background: #161b22; padding: 15px; margin: 10px 0; border-radius: 8px; }}
        .improvement {{ color: #7ee787; }}
        .degradation {{ color: #f85149; }}
        .learning-score {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; }}
        .score-high {{ background: #238636; color: white; }}
        .score-medium {{ background: #bf8700; color: white; }}
        .score-low {{ background: #da3633; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 AI Learning Dashboard</h1>
        <p>Monitoring how the AI learns and improves from market data</p>
        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    {self._format_metrics(summary)}
    {self._format_models(summary.get('model_versions', {}))}
    {self._format_recent_events(summary.get('recent_events', []))}
    
    <script>
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
        """
        return html
    
    def _format_metrics(self, summary):
        if summary.get('status'):
            return f'<div class="metric"><h3>Status</h3><div class="value">{summary["status"]}</div></div>'
            
        total = summary.get('total_learning_events', 0)
        avg_score = summary.get('average_learning_score', 0)
        event_types = summary.get('event_types', {})
        
        html = '<div style="display: flex; flex-wrap: wrap;">'
        html += f'<div class="metric"><h3>Total Learning Events</h3><div class="value">{total}</div></div>'
        html += f'<div class="metric"><h3>Average Learning Score</h3><div class="value">{avg_score:.3f}</div></div>'
        
        for event_type, count in event_types.items():
            html += f'<div class="metric"><h3>{event_type.replace("_", " ").title()}</h3><div class="value">{count}</div></div>'
            
        html += '</div>'
        return html
    
    def _format_models(self, models):
        if not models:
            return '<h2>📊 Model Versions</h2><p>No model updates recorded yet</p>'
            
        html = '<h2>📊 Model Versions</h2>'
        
        for strategy, info in models.items():
            improvement_class = 'improvement' if info.get('improvement', 0) > 0 else 'degradation'
            improvement_sign = '+' if info.get('improvement', 0) > 0 else ''
            
            html += f'''
            <div class="model">
                <h3>{strategy.title()} Strategy</h3>
                <p><strong>Version:</strong> {info.get('version', 'Unknown')}</p>
                <p><strong>Accuracy:</strong> {info.get('accuracy', 0):.3f}</p>
                <p><strong>Improvement:</strong> <span class="{improvement_class}">{improvement_sign}{info.get('improvement', 0):.3f}</span></p>
                <p><strong>Training Samples:</strong> {info.get('training_samples', 0)}</p>
                <p><strong>Last Updated:</strong> {info.get('last_updated', 'Unknown')}</p>
            </div>
            '''
            
        return html
    
    def _format_recent_events(self, events):
        if not events:
            return '<h2>📝 Recent Learning Events</h2><p>No recent events</p>'
            
        html = '<h2>📝 Recent Learning Events</h2>'
        
        for event in reversed(events):  # Most recent first
            score = event.get('learning_score', 0)
            score_class = 'score-high' if score > 0.7 else 'score-medium' if score > 0.3 else 'score-low'
            
            html += f'''
            <div class="event">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{event.get('strategy', 'Unknown')} - {event.get('event_type', 'Unknown').replace('_', ' ').title()}</strong>
                        <span class="learning-score {score_class}">{score:.3f}</span>
                    </div>
                    <div style="font-size: 12px; color: #8b949e;">{event.get('timestamp', 'Unknown time')}</div>
                </div>
                <div style="margin-top: 5px; font-size: 14px;">
                    {self._format_event_data(event.get('data', {}))}
                </div>
            </div>
            '''
            
        return html
    
    def _format_event_data(self, data):
        if not data:
            return 'No additional data'
            
        formatted = []
        for key, value in data.items():
            if isinstance(value, float):
                formatted.append(f"{key}: {value:.3f}")
            else:
                formatted.append(f"{key}: {value}")
                
        return ' | '.join(formatted)

def start_learning_dashboard(port=8084):
    """Start the learning dashboard server."""
    server = HTTPServer(('0.0.0.0', port), LearningDashboardHandler)
    print(f"Learning dashboard running on http://localhost:{port}/learning")
    server.serve_forever()

if __name__ == '__main__':
    start_learning_dashboard()