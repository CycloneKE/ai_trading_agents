"""
Always-Online Trading Agent - Continuous Market Scanning
"""

import os
import time
import schedule
import threading
from datetime import datetime, timedelta
import yfinance as yf
from phase2_daily_trading import Phase2Trader
from genius_trading_ai import GeniusTradingAI
import json

class AlwaysOnlineAgent:
    def __init__(self):
        self.trader = Phase2Trader()
        self.genius_ai = GeniusTradingAI()
        self.is_running = True
        self.market_hours = {'start': 9.5, 'end': 16}  # 9:30 AM to 4:00 PM EST
        self.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
        self.learning_data = []
        
        # Initialize webhook system
        from webhook_system import TradingWebhookSystem, start_webhook_receiver
        self.webhook_system = TradingWebhookSystem()
        self.webhook_system.setup_default_webhooks()
        # Add enhanced unified platform webhook
        self.webhook_system.add_webhook("http://localhost:8087/webhook", "Enhanced Platform")
        start_webhook_receiver()
        print("Webhook system initialized with real-time dashboard")
        
    def is_market_open(self):
        """Check if market is currently open."""
        now = datetime.now()
        current_hour = now.hour + now.minute/60
        
        # Check if weekday and within market hours
        if now.weekday() < 5 and self.market_hours['start'] <= current_hour <= self.market_hours['end']:
            return True
        return False
    
    def continuous_market_scan(self):
        """Continuously scan market every 5 minutes during market hours."""
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market scan started...")
        
        for symbol in self.watchlist:
            try:
                # Get real-time data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="5m")
                
                if len(data) > 0:
                    latest = data.iloc[-1]
                    prev = data.iloc[-2] if len(data) > 1 else latest
                    
                    market_data = {
                        'symbol': symbol,
                        'price': latest['Close'],
                        'volume': latest['Volume'],
                        'change': ((latest['Close'] - prev['Close']) / prev['Close']) * 100,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # AI analysis
                    analysis = self.genius_ai.generate_genius_analysis(symbol, {
                        'close': latest['Close'],
                        'volume': latest['Volume'],
                        'high': latest['High'],
                        'low': latest['Low']
                    })
                    
                    # Store learning data
                    learning_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'market_data': market_data,
                        'ai_analysis': analysis,
                        'market_open': self.is_market_open()
                    }
                    
                    self.learning_data.append(learning_entry)
                    
                    # Keep only last 1000 entries
                    if len(self.learning_data) > 1000:
                        self.learning_data = self.learning_data[-1000:]
                    
                    # Alert on high-conviction opportunities
                    if analysis.get('genius_score', 0) > 8.0:
                        self.send_alert(symbol, market_data, analysis)
                        # Send webhook alert
                        self.webhook_system.send_high_conviction_alert(analysis, symbol, latest['Close'])
                        
            except Exception as e:
                print(f"Error scanning {symbol}: {str(e)}")
        
        # Save learning data
        self.save_learning_data()
    
    def send_alert(self, symbol, market_data, analysis):
        """Send high-conviction trading alert."""
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'price': market_data['price'],
            'genius_score': analysis.get('genius_score', 0),
            'profit_probability': analysis.get('profit_probability', 0),
            'message': f"HIGH CONVICTION: {symbol} at ${market_data['price']:.2f}"
        }
        
        print(f"ALERT: {alert['message']}")
        
        # Save alert
        alerts_file = 'trading_alerts.json'
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
        else:
            alerts = []
        
        alerts.append(alert)
        
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
    
    def save_learning_data(self):
        """Save continuous learning data."""
        
        with open('continuous_learning.json', 'w') as f:
            json.dump(self.learning_data, f, indent=2)
    
    def daily_trading_session(self):
        """Run daily trading session at market open."""
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running daily trading session...")
        
        # Get portfolio before trading
        old_portfolio = self.trader.get_portfolio_data() if hasattr(self.trader, 'get_portfolio_data') else None
        
        # Run trading session
        self.trader.daily_trading_session()
        
        # Send portfolio update webhook
        try:
            if hasattr(self.trader, 'calculate_performance'):
                performance = self.trader.calculate_performance()
                self.webhook_system.send_portfolio_update({
                    'total_value': performance.get('portfolio_value', 100000),
                    'total_return': performance.get('total_return_pct', 0)
                })
        except:
            pass
    
    def end_of_day_analysis(self):
        """Analyze day's performance and learn."""
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] End of day analysis...")
        
        # Analyze today's learning data
        today_data = [d for d in self.learning_data 
                     if d['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))]
        
        if today_data:
            # Calculate accuracy of AI predictions
            high_score_predictions = [d for d in today_data if d['ai_analysis'].get('genius_score', 0) > 7.0]
            
            analysis_summary = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'total_scans': len(today_data),
                'high_conviction_signals': len(high_score_predictions),
                'market_hours_scans': len([d for d in today_data if d['market_open']]),
                'top_opportunities': sorted(high_score_predictions, 
                                          key=lambda x: x['ai_analysis'].get('genius_score', 0), 
                                          reverse=True)[:3]
            }
            
            # Save daily analysis
            daily_file = f"daily_analysis_{datetime.now().strftime('%Y%m%d')}.json"
            with open(daily_file, 'w') as f:
                json.dump(analysis_summary, f, indent=2)
            
            print(f"Daily analysis saved: {len(high_score_predictions)} high-conviction signals")
    
    def start_continuous_operation(self):
        """Start always-online operation."""
        
        print("STARTING ALWAYS-ONLINE TRADING AGENT")
        print("=" * 50)
        print("Features:")
        print("• Continuous market scanning every 5 minutes")
        print("• Daily trading sessions at market open")
        print("• End-of-day analysis and learning")
        print("• High-conviction alerts")
        print("• 24/7 operation with market hour awareness")
        print("=" * 50)
        
        # Schedule daily tasks
        schedule.every().day.at("09:35").do(self.daily_trading_session)  # 5 min after market open
        schedule.every().day.at("16:05").do(self.end_of_day_analysis)   # 5 min after market close
        
        # Start continuous scanning in separate thread
        scan_thread = threading.Thread(target=self.continuous_scan_loop)
        scan_thread.daemon = True
        scan_thread.start()
        
        # Main scheduler loop
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def continuous_scan_loop(self):
        """Continuous scanning loop."""
        
        while self.is_running:
            if self.is_market_open():
                self.continuous_market_scan()
                time.sleep(300)  # 5 minutes
            else:
                # Scan less frequently outside market hours
                time.sleep(1800)  # 30 minutes
    
    def stop(self):
        """Stop the agent."""
        self.is_running = False
        print("Agent stopped.")

def deploy_always_online():
    """Deploy always-online agent."""
    
    agent = AlwaysOnlineAgent()
    
    try:
        agent.start_continuous_operation()
    except KeyboardInterrupt:
        print("\nShutting down agent...")
        agent.stop()

if __name__ == '__main__':
    deploy_always_online()