"""
Phase 2: Daily Trading with Performance Tracking
"""

import os
import json
import yfinance as yf
from datetime import datetime, timedelta
from genius_trading_ai import GeniusTradingAI
import pandas as pd

class Phase2Trader:
    def __init__(self):
        self.genius_ai = GeniusTradingAI()
        self.portfolio_file = 'phase2_portfolio.json'
        self.performance_file = 'phase2_performance.json'
        self.load_portfolio()
        
    def load_portfolio(self):
        """Load existing portfolio or create new one."""
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, 'r') as f:
                data = json.load(f)
                self.cash = data.get('cash', 100000)
                self.positions = data.get('positions', {})
                self.trade_history = data.get('trade_history', [])
        else:
            self.cash = 100000
            self.positions = {}
            self.trade_history = []
    
    def save_portfolio(self):
        """Save portfolio state."""
        data = {
            'cash': self.cash,
            'positions': self.positions,
            'trade_history': self.trade_history,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.portfolio_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_market_data(self, symbol):
        """Get enhanced market data."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            info = ticker.info
            
            if len(hist) > 0:
                latest = hist.iloc[-1]
                prev_close = hist.iloc[-2]['Close'] if len(hist) > 1 else latest['Close']
                
                # Calculate technical indicators
                rsi = self.calculate_rsi(hist['Close'])
                sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else latest['Close']
                
                return {
                    'close': latest['Close'],
                    'prev_close': prev_close,
                    'change_pct': ((latest['Close'] - prev_close) / prev_close) * 100,
                    'volume': latest['Volume'],
                    'high_52w': info.get('fiftyTwoWeekHigh', latest['High']),
                    'low_52w': info.get('fiftyTwoWeekLow', latest['Low']),
                    'rsi': rsi,
                    'sma_20': sma_20,
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0)
                }
        except Exception as e:
            print(f"Error getting data for {symbol}: {str(e)}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        if len(prices) < period:
            return 50
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def enhanced_analysis(self, symbol, market_data):
        """Enhanced analysis with improved scoring."""
        
        # Get genius AI analysis
        analysis = self.genius_ai.generate_genius_analysis(symbol, market_data)
        
        # Enhanced scoring based on multiple factors
        price = market_data['close']
        rsi = market_data['rsi']
        change_pct = market_data['change_pct']
        volume = market_data['volume']
        
        # Technical score
        tech_score = 0
        if 30 < rsi < 70:  # Not overbought/oversold
            tech_score += 2
        if price > market_data['sma_20']:  # Above 20-day average
            tech_score += 2
        if volume > 1000000:  # High volume
            tech_score += 1
        
        # Momentum score
        momentum_score = 0
        if -2 < change_pct < 5:  # Healthy daily move
            momentum_score += 2
        if change_pct > 0:  # Positive momentum
            momentum_score += 1
        
        # Value score
        value_score = 0
        pe_ratio = market_data.get('pe_ratio', 25)
        if 10 < pe_ratio < 30:  # Reasonable valuation
            value_score += 2
        
        # Calculate enhanced genius score
        base_score = analysis.get('genius_score', 0)
        enhanced_score = min(10, base_score + (tech_score + momentum_score + value_score) * 0.5)
        
        analysis['enhanced_genius_score'] = enhanced_score
        analysis['technical_score'] = tech_score
        analysis['momentum_score'] = momentum_score
        analysis['value_score'] = value_score
        
        return analysis
    
    def make_trading_decision(self, symbol, analysis, market_data):
        """Make enhanced trading decision."""
        
        enhanced_score = analysis.get('enhanced_genius_score', 0)
        profit_prob = analysis.get('profit_probability', 0)
        current_price = market_data['close']
        
        # Position sizing based on conviction
        if enhanced_score >= 8.5 and profit_prob >= 0.80:
            action = "BUY"
            conviction = "VERY HIGH"
            position_size = min(5000, self.cash * 0.10)  # 10% max
        elif enhanced_score >= 7.0 and profit_prob >= 0.70:
            action = "BUY"
            conviction = "HIGH"
            position_size = min(3000, self.cash * 0.06)  # 6% max
        elif enhanced_score >= 5.5 and profit_prob >= 0.60:
            action = "BUY"
            conviction = "MEDIUM"
            position_size = min(1500, self.cash * 0.03)  # 3% max
        else:
            action = "HOLD"
            conviction = "LOW"
            position_size = 0
        
        # Check if we already have position
        if symbol in self.positions:
            current_value = self.positions[symbol]['shares'] * current_price
            if enhanced_score < 4.0:  # Sell if conviction drops
                action = "SELL"
                conviction = "EXIT"
        
        return {
            'action': action,
            'conviction': conviction,
            'position_size': position_size,
            'enhanced_score': enhanced_score
        }
    
    def execute_trade(self, symbol, decision, market_data, analysis):
        """Execute virtual trade."""
        
        action = decision['action']
        price = market_data['close']
        
        if action == "BUY" and decision['position_size'] > 0:
            shares = int(decision['position_size'] / price)
            cost = shares * price
            
            if cost <= self.cash and shares > 0:
                self.cash -= cost
                
                if symbol in self.positions:
                    # Add to existing position
                    old_shares = self.positions[symbol]['shares']
                    old_cost = self.positions[symbol]['total_cost']
                    self.positions[symbol] = {
                        'shares': old_shares + shares,
                        'avg_price': (old_cost + cost) / (old_shares + shares),
                        'total_cost': old_cost + cost,
                        'entry_date': self.positions[symbol]['entry_date']
                    }
                else:
                    # New position
                    self.positions[symbol] = {
                        'shares': shares,
                        'avg_price': price,
                        'total_cost': cost,
                        'entry_date': datetime.now().isoformat()
                    }
                
                # Log trade
                self.log_trade(symbol, action, shares, price, decision, analysis)
                
                # Send webhook notification
                try:
                    from webhook_system import TradingWebhookSystem
                    webhook_system = TradingWebhookSystem()
                    trade_data = {
                        'symbol': symbol,
                        'action': action,
                        'shares': shares,
                        'price': price,
                        'value': cost,
                        'conviction': decision['conviction'],
                        'genius_score': decision['enhanced_score']
                    }
                    webhook_system.send_trade_alert(trade_data)
                except:
                    pass
                
                print(f"BOUGHT {shares} shares of {symbol} at ${price:.2f}")
                print(f"Total cost: ${cost:.2f} | Remaining cash: ${self.cash:.2f}")
                
                return True
        
        elif action == "SELL" and symbol in self.positions:
            shares = self.positions[symbol]['shares']
            proceeds = shares * price
            
            self.cash += proceeds
            cost_basis = self.positions[symbol]['total_cost']
            pnl = proceeds - cost_basis
            pnl_pct = (pnl / cost_basis) * 100
            
            # Log trade
            self.log_trade(symbol, action, shares, price, decision, analysis, pnl, pnl_pct)
            
            print(f"SOLD {shares} shares of {symbol} at ${price:.2f}")
            print(f"Proceeds: ${proceeds:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
            
            del self.positions[symbol]
            return True
        
        return False
    
    def log_trade(self, symbol, action, shares, price, decision, analysis, pnl=0, pnl_pct=0):
        """Log trade details."""
        
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'value': shares * price,
            'conviction': decision['conviction'],
            'enhanced_score': decision['enhanced_score'],
            'genius_score': analysis.get('genius_score', 0),
            'profit_probability': analysis.get('profit_probability', 0),
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }
        
        self.trade_history.append(trade_record)
    
    def calculate_performance(self):
        """Calculate portfolio performance metrics."""
        
        # Current portfolio value
        portfolio_value = self.cash
        unrealized_pnl = 0
        
        for symbol, position in self.positions.items():
            current_data = self.get_market_data(symbol)
            if current_data:
                current_value = position['shares'] * current_data['close']
                portfolio_value += current_value
                unrealized_pnl += current_value - position['total_cost']
        
        # Realized P&L from completed trades
        realized_pnl = sum([trade['pnl'] for trade in self.trade_history if trade['action'] == 'SELL'])
        
        # Win rate
        completed_trades = [t for t in self.trade_history if t['action'] == 'SELL']
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        
        # Total return
        total_return = ((portfolio_value - 100000) / 100000) * 100
        
        return {
            'portfolio_value': portfolio_value,
            'total_return_pct': total_return,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'win_rate': win_rate,
            'total_trades': len(completed_trades),
            'active_positions': len(self.positions)
        }
    
    def daily_trading_session(self):
        """Run daily trading session."""
        
        print(f"PHASE 2 DAILY TRADING - {datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        # Expanded watchlist
        watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
        
        for symbol in watchlist:
            print(f"\nAnalyzing {symbol}...")
            
            market_data = self.get_market_data(symbol)
            if not market_data:
                continue
            
            analysis = self.enhanced_analysis(symbol, market_data)
            decision = self.make_trading_decision(symbol, analysis, market_data)
            
            print(f"Price: ${market_data['close']:.2f} ({market_data['change_pct']:+.1f}%)")
            print(f"Enhanced Score: {analysis['enhanced_genius_score']:.1f}/10")
            print(f"Decision: {decision['action']} ({decision['conviction']})")
            
            if decision['action'] in ['BUY', 'SELL']:
                self.execute_trade(symbol, decision, market_data, analysis)
            
            print("-" * 40)
        
        # Show performance
        performance = self.calculate_performance()
        print(f"\nPORTFOLIO PERFORMANCE:")
        print(f"Total Value: ${performance['portfolio_value']:,.2f}")
        print(f"Total Return: {performance['total_return_pct']:+.2f}%")
        print(f"Win Rate: {performance['win_rate']:.1%}")
        print(f"Active Positions: {performance['active_positions']}")
        
        # Save portfolio
        self.save_portfolio()
        
        # Save performance history
        perf_record = {
            'date': datetime.now().isoformat(),
            **performance
        }
        
        if os.path.exists(self.performance_file):
            with open(self.performance_file, 'r') as f:
                perf_history = json.load(f)
        else:
            perf_history = []
        
        perf_history.append(perf_record)
        
        with open(self.performance_file, 'w') as f:
            json.dump(perf_history, f, indent=2)
        
        print(f"\nSession complete! Data saved to {self.portfolio_file}")

def start_phase2():
    """Start Phase 2 daily trading."""
    
    trader = Phase2Trader()
    trader.daily_trading_session()
    
    return trader

if __name__ == '__main__':
    start_phase2()