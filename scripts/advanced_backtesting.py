import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class AdvancedBacktester:
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.portfolio_history = []
        self.positions = {}
        self.cash = initial_capital
        self.portfolio_value = initial_capital
    
    def add_trade(self, timestamp, symbol, action, quantity, price, strategy='manual'):
        """Add a trade to the backtest"""
        trade_value = quantity * price
        commission_cost = abs(trade_value) * self.commission
        
        if action == 'buy':
            if self.cash >= trade_value + commission_cost:
                self.cash -= (trade_value + commission_cost)
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            else:
                return False  # Insufficient funds
        elif action == 'sell':
            if self.positions.get(symbol, 0) >= quantity:
                self.cash += (trade_value - commission_cost)
                self.positions[symbol] = self.positions.get(symbol, 0) - quantity
                if self.positions[symbol] == 0:
                    del self.positions[symbol]
            else:
                return False  # Insufficient shares
        
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'commission': commission_cost,
            'strategy': strategy
        }
        
        self.trades.append(trade)
        return True
    
    def update_portfolio_value(self, timestamp, market_prices):
        """Update portfolio value based on current market prices"""
        positions_value = 0
        
        for symbol, quantity in self.positions.items():
            if symbol in market_prices:
                positions_value += quantity * market_prices[symbol]
        
        self.portfolio_value = self.cash + positions_value
        
        self.portfolio_history.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': self.portfolio_value,
            'return': (self.portfolio_value - self.initial_capital) / self.initial_capital
        })
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if len(self.portfolio_history) < 2:
            return {}
        
        returns = [h['return'] for h in self.portfolio_history]
        daily_returns = np.diff(returns)
        
        total_return = returns[-1]
        
        # Sharpe ratio (assuming 252 trading days)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate([h['total_value'] for h in self.portfolio_history])
        drawdown = [(h['total_value'] - peak[i]) / peak[i] for i, h in enumerate(self.portfolio_history)]
        max_drawdown = min(drawdown) if drawdown else 0
        
        # Win rate
        profitable_trades = sum(1 for t in self.trades if self._is_profitable_trade(t))
        win_rate = profitable_trades / len(self.trades) if self.trades else 0
        
        return {
            'total_return': total_return,
            'total_return_percent': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'max_drawdown_percent': abs(max_drawdown) * 100,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'final_portfolio_value': self.portfolio_value,
            'profit_loss': self.portfolio_value - self.initial_capital
        }
    
    def _is_profitable_trade(self, trade):
        """Check if a trade was profitable (simplified)"""
        # This is a simplified check - in reality, you'd need to track
        # the complete buy/sell cycle for each position
        return trade['action'] == 'sell' and trade['value'] > 0
    
    def export_results(self, filename='backtest_results.json'):
        """Export backtest results"""
        results = {
            'metrics': self.calculate_metrics(),
            'trades': self.trades,
            'portfolio_history': self.portfolio_history,
            'final_positions': self.positions,
            'parameters': {
                'initial_capital': self.initial_capital,
                'commission': self.commission
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results

def run_sample_backtest():
    """Run a sample backtest"""
    backtester = AdvancedBacktester(initial_capital=100000)
    
    # Sample market data
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    base_prices = {'AAPL': 150, 'GOOGL': 2500, 'MSFT': 300}
    
    # Simulate 30 days of trading
    for day in range(30):
        timestamp = datetime.now() - timedelta(days=30-day)
        
        # Generate random price movements
        market_prices = {}
        for symbol in symbols:
            price_change = np.random.normal(0, 0.02)  # 2% daily volatility
            market_prices[symbol] = base_prices[symbol] * (1 + price_change)
            base_prices[symbol] = market_prices[symbol]
        
        # Simple momentum strategy
        for symbol in symbols:
            if day > 5:  # Need some history
                if np.random.random() > 0.7:  # 30% chance of trade
                    action = 'buy' if np.random.random() > 0.5 else 'sell'
                    quantity = 10
                    backtester.add_trade(timestamp, symbol, action, quantity, 
                                       market_prices[symbol], 'momentum')
        
        backtester.update_portfolio_value(timestamp, market_prices)
    
    return backtester.export_results()
