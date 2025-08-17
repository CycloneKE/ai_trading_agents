#!/usr/bin/env python3
"""
Paper Trading Deployment System
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import yfinance as yf
import pandas as pd
from supervised_learning import SupervisedLearningStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperTradingEngine:
    """Paper trading engine for AI agent deployment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initial_capital = config.get('initial_capital', 100000)
        self.cash = self.initial_capital
        self.positions = {}  # {symbol: {'shares': int, 'avg_price': float, 'entry_time': datetime}}
        self.trade_log = []
        self.performance_log = []
        
        # Initialize AI strategy
        self.strategy = SupervisedLearningStrategy("paper_trader", config)
        
        # Trading parameters
        self.symbols = config.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
        self.max_positions = config.get('max_positions', 4)
        self.check_interval = config.get('check_interval', 300)  # 5 minutes
        
        logger.info(f"Paper trading engine initialized with ${self.initial_capital:,.2f}")
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time market data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get recent data
            hist = ticker.history(period="5d", interval="1h")
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            
            # Calculate indicators
            close_prices = hist['Close']
            
            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = close_prices.ewm(span=12).mean()
            ema26 = close_prices.ewm(span=26).mean()
            macd = ema12 - ema26
            
            # ATR
            high_low = hist['High'] - hist['Low']
            high_close = abs(hist['High'] - hist['Close'].shift())
            low_close = abs(hist['Low'] - hist['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            return {
                'symbol': symbol,
                'close': float(latest['Close']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': int(latest['Volume']),
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
                'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0,
                'ema_12': float(ema12.iloc[-1]) if not pd.isna(ema12.iloc[-1]) else float(latest['Close']),
                'ema_26': float(ema26.iloc[-1]) if not pd.isna(ema26.iloc[-1]) else float(latest['Close']),
                'atr': float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else float(latest['Close']) * 0.02,
                'adx': 25.0,  # Mock ADX
                'pe_ratio': 15.0,  # Mock fundamental data
                'debt_ratio': 0.3,
                'current_ratio': 1.5,
                'market_cap': 1000000000,
                'sector': 'technology'
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def execute_trade(self, symbol: str, action: str, signal: Dict[str, Any]) -> bool:
        """Execute paper trade"""
        try:
            market_data = self.get_market_data(symbol)
            if not market_data:
                return False
            
            current_price = market_data['close']
            confidence = signal.get('confidence', 0)
            
            if action == 'buy':
                # Calculate position size
                max_investment = self.cash * 0.25  # Max 25% per position
                shares = int(max_investment / current_price)
                cost = shares * current_price
                
                if shares > 0 and cost <= self.cash and len(self.positions) < self.max_positions:
                    self.cash -= cost
                    
                    if symbol in self.positions:
                        # Add to existing position
                        old_shares = self.positions[symbol]['shares']
                        old_cost = old_shares * self.positions[symbol]['avg_price']
                        new_avg_price = (old_cost + cost) / (old_shares + shares)
                        
                        self.positions[symbol]['shares'] += shares
                        self.positions[symbol]['avg_price'] = new_avg_price
                    else:
                        # New position
                        self.positions[symbol] = {
                            'shares': shares,
                            'avg_price': current_price,
                            'entry_time': datetime.now()
                        }
                    
                    # Log trade
                    trade = {
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': current_price,
                        'value': cost,
                        'confidence': confidence,
                        'cash_remaining': self.cash
                    }
                    self.trade_log.append(trade)
                    
                    logger.info(f"BUY {shares} {symbol} @ ${current_price:.2f} (Confidence: {confidence:.2f})")
                    return True
            
            elif action == 'sell' and symbol in self.positions:
                # Sell entire position
                shares = self.positions[symbol]['shares']
                proceeds = shares * current_price
                entry_price = self.positions[symbol]['avg_price']
                
                self.cash += proceeds
                pnl = proceeds - (shares * entry_price)
                pnl_pct = (pnl / (shares * entry_price)) * 100
                
                # Log trade
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares,
                    'price': current_price,
                    'value': proceeds,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'confidence': confidence,
                    'cash_remaining': self.cash,
                    'entry_price': entry_price,
                    'stop_loss_triggered': signal.get('stop_loss_triggered', False)
                }
                self.trade_log.append(trade)
                
                # Remove position
                del self.positions[symbol]
                
                logger.info(f"SELL {shares} {symbol} @ ${current_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.1f}%)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing {action} for {symbol}: {e}")
            return False
    
    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            market_data = self.get_market_data(symbol)
            if market_data:
                current_price = market_data['close']
                position_value = position['shares'] * current_price
                total_value += position_value
        
        return total_value
    
    def log_performance(self):
        """Log current performance metrics"""
        total_value = self.calculate_portfolio_value()
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        # Calculate position details
        position_details = {}
        for symbol, position in self.positions.items():
            market_data = self.get_market_data(symbol)
            if market_data:
                current_price = market_data['close']
                position_value = position['shares'] * current_price
                unrealized_pnl = position_value - (position['shares'] * position['avg_price'])
                
                position_details[symbol] = {
                    'shares': position['shares'],
                    'avg_price': position['avg_price'],
                    'current_price': current_price,
                    'value': position_value,
                    'unrealized_pnl': unrealized_pnl
                }
        
        performance = {
            'timestamp': datetime.now(),
            'total_value': total_value,
            'cash': self.cash,
            'total_return': total_return,
            'positions': position_details,
            'num_positions': len(self.positions)
        }
        
        self.performance_log.append(performance)
        
        logger.info(f"Portfolio Value: ${total_value:,.2f} | Return: {total_return:.2%} | Positions: {len(self.positions)}")
    
    def run_trading_session(self, duration_hours: int = 8):
        """Run paper trading session"""
        logger.info(f"Starting paper trading session for {duration_hours} hours")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            try:
                # Check each symbol
                for symbol in self.symbols:
                    market_data = self.get_market_data(symbol)
                    if not market_data:
                        continue
                    
                    # Get AI signal
                    signal = self.strategy.generate_signals(market_data)
                    action = signal.get('action', 'hold')
                    
                    if action != 'hold':
                        self.execute_trade(symbol, action, signal)
                
                # Log performance
                self.log_performance()
                
                # Wait before next check
                logger.info(f"Waiting {self.check_interval} seconds until next check...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Trading session interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
        
        logger.info("Paper trading session completed")
        self.generate_session_report()
    
    def generate_session_report(self):
        """Generate trading session report"""
        if not self.trade_log:
            logger.info("No trades executed during session")
            return
        
        # Calculate metrics
        completed_trades = [t for t in self.trade_log if 'pnl' in t]
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        
        total_value = self.calculate_portfolio_value()
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_pnl = sum(t['pnl'] for t in completed_trades) / len(completed_trades) if completed_trades else 0
        
        report = f"""
{'='*60}
PAPER TRADING SESSION REPORT
{'='*60}

PERFORMANCE SUMMARY:
  Initial Capital:     ${self.initial_capital:,.2f}
  Final Value:         ${total_value:,.2f}
  Total Return:        {total_return:.2%}
  Cash Remaining:      ${self.cash:,.2f}

TRADING STATISTICS:
  Total Trades:        {len(self.trade_log)}
  Completed Trades:    {len(completed_trades)}
  Win Rate:            {win_rate:.1%}
  Average P&L:         ${avg_pnl:.2f}

CURRENT POSITIONS:
"""
        
        for symbol, position in self.positions.items():
            market_data = self.get_market_data(symbol)
            if market_data:
                current_price = market_data['close']
                unrealized_pnl = (current_price - position['avg_price']) * position['shares']
                report += f"  {symbol}: {position['shares']} shares @ ${position['avg_price']:.2f} | Current: ${current_price:.2f} | P&L: ${unrealized_pnl:.2f}\n"
        
        report += f"\n{'='*60}"
        
        print(report)
        
        # Export detailed logs
        session_data = {
            'session_summary': {
                'start_capital': self.initial_capital,
                'final_value': total_value,
                'total_return': total_return,
                'win_rate': win_rate,
                'total_trades': len(self.trade_log)
            },
            'trade_log': self.trade_log,
            'performance_log': self.performance_log
        }
        
        filename = f"paper_trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        logger.info(f"Session data exported to {filename}")

def main():
    """Main deployment function"""
    
    # Paper trading configuration
    config = {
        'model_type': 'xgboost',
        'threshold': 0.01,
        'features': ['close', 'volume', 'rsi', 'macd', 'ema_12', 'ema_26', 'adx', 'atr'],
        'stop_loss_atr_multiplier': 2.0,
        'trailing_stop_enabled': True,
        'trend_following_enabled': True,
        'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        'initial_capital': 100000,
        'max_positions': 4,
        'check_interval': 300,  # 5 minutes
        'bias_threshold': 0.15
    }
    
    # Create paper trading engine
    engine = PaperTradingEngine(config)
    
    print(f"""
{'='*60}
AI TRADING AGENT - PAPER TRADING DEPLOYMENT
{'='*60}

Configuration:
  Initial Capital: ${config['initial_capital']:,}
  Symbols: {', '.join(config['symbols'])}
  Max Positions: {config['max_positions']}
  Check Interval: {config['check_interval']} seconds
  
Starting paper trading session...
Press Ctrl+C to stop at any time.
{'='*60}
""")
    
    try:
        # Run trading session (8 hours by default)
        engine.run_trading_session(duration_hours=8)
    except KeyboardInterrupt:
        print("\nTrading session stopped by user")
        engine.generate_session_report()

if __name__ == "__main__":
    main()