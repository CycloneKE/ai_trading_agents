#!/usr/bin/env python3
"""
Enhanced Trading Features
"""

import os
import json
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_advanced_backtesting():
    """Create comprehensive backtesting system"""
    code = '''import pandas as pd
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
'''
    
    try:
        with open('advanced_backtesting.py', 'w') as f:
            f.write(code)
        logger.info("Created advanced backtesting system")
        return True
    except Exception as e:
        logger.error(f"Failed to create backtesting system: {e}")
        return False

def create_portfolio_analytics():
    """Create portfolio analytics system"""
    code = '''import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

class PortfolioAnalytics:
    def __init__(self):
        self.positions = {}
        self.price_history = {}
        self.trade_history = []
    
    def add_position(self, symbol: str, quantity: float, price: float):
        """Add or update position"""
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0, 'total_cost': 0}
        
        pos = self.positions[symbol]
        new_total_cost = pos['total_cost'] + (quantity * price)
        new_quantity = pos['quantity'] + quantity
        
        if new_quantity != 0:
            pos['avg_price'] = new_total_cost / new_quantity
        else:
            pos['avg_price'] = 0
        
        pos['quantity'] = new_quantity
        pos['total_cost'] = new_total_cost
        
        if pos['quantity'] == 0:
            del self.positions[symbol]
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current market prices"""
        timestamp = datetime.now()
        
        for symbol, price in prices.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'timestamp': timestamp,
                'price': price
            })
            
            # Keep only last 100 price points
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
    
    def calculate_portfolio_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        total_value = 0
        total_cost = 0
        total_pnl = 0
        position_details = []
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position['avg_price'])
            market_value = position['quantity'] * current_price
            cost_basis = position['quantity'] * position['avg_price']
            pnl = market_value - cost_basis
            pnl_percent = (pnl / cost_basis * 100) if cost_basis != 0 else 0
            
            total_value += market_value
            total_cost += cost_basis
            total_pnl += pnl
            
            position_details.append({
                'symbol': symbol,
                'quantity': position['quantity'],
                'avg_price': position['avg_price'],
                'current_price': current_price,
                'market_value': market_value,
                'cost_basis': cost_basis,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'weight': 0  # Will be calculated below
            })
        
        # Calculate position weights
        for pos in position_details:
            pos['weight'] = (pos['market_value'] / total_value * 100) if total_value > 0 else 0
        
        # Portfolio-level metrics
        total_return_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        # Risk metrics
        portfolio_volatility = self._calculate_portfolio_volatility(current_prices)
        beta = self._calculate_portfolio_beta()
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_pnl': total_pnl,
            'total_return_percent': total_return_percent,
            'position_count': len(self.positions),
            'positions': position_details,
            'risk_metrics': {
                'volatility': portfolio_volatility,
                'beta': beta,
                'sharpe_ratio': sharpe_ratio
            }
        }
    
    def _calculate_portfolio_volatility(self, current_prices: Dict[str, float]) -> float:
        """Calculate portfolio volatility"""
        if not self.price_history:
            return 0.0
        
        # Simplified volatility calculation
        all_returns = []
        
        for symbol in self.positions.keys():
            if symbol in self.price_history and len(self.price_history[symbol]) > 1:
                prices = [p['price'] for p in self.price_history[symbol]]
                returns = np.diff(np.log(prices))
                all_returns.extend(returns)
        
        if len(all_returns) > 1:
            return float(np.std(all_returns) * np.sqrt(252))  # Annualized
        
        return 0.0
    
    def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta (simplified)"""
        # This would require market index data in a real implementation
        return 1.0  # Placeholder
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (simplified)"""
        # This would require risk-free rate and return history
        return 1.5  # Placeholder
    
    def get_sector_allocation(self) -> Dict[str, float]:
        """Get sector allocation (simplified)"""
        # This would require sector mapping in a real implementation
        tech_symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
        
        tech_value = 0
        other_value = 0
        
        for symbol, position in self.positions.items():
            # Use avg_price as approximation for current value
            value = position['quantity'] * position['avg_price']
            
            if symbol in tech_symbols:
                tech_value += value
            else:
                other_value += value
        
        total_value = tech_value + other_value
        
        if total_value == 0:
            return {}
        
        return {
            'Technology': tech_value / total_value * 100,
            'Other': other_value / total_value * 100
        }

# Global portfolio analytics instance
portfolio_analytics = PortfolioAnalytics()
'''
    
    try:
        with open('portfolio_analytics.py', 'w') as f:
            f.write(code)
        logger.info("Created portfolio analytics system")
        return True
    except Exception as e:
        logger.error(f"Failed to create portfolio analytics: {e}")
        return False

def create_strategy_optimizer():
    """Create strategy optimization system"""
    code = '''import numpy as np
from typing import Dict, List, Tuple, Any
import json

class StrategyOptimizer:
    def __init__(self):
        self.strategies = {}
        self.optimization_history = []
    
    def register_strategy(self, name: str, parameters: Dict[str, Any], 
                         performance_func: callable):
        """Register a strategy for optimization"""
        self.strategies[name] = {
            'parameters': parameters,
            'performance_func': performance_func,
            'best_params': parameters.copy(),
            'best_performance': 0
        }
    
    def optimize_strategy(self, strategy_name: str, param_ranges: Dict[str, Tuple], 
                         iterations: int = 100) -> Dict[str, Any]:
        """Optimize strategy parameters using random search"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not registered")
        
        strategy = self.strategies[strategy_name]
        best_params = strategy['parameters'].copy()
        best_performance = float('-inf')
        
        results = []
        
        for i in range(iterations):
            # Generate random parameters within ranges
            test_params = {}
            for param, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    test_params[param] = np.random.randint(min_val, max_val + 1)
                else:
                    test_params[param] = np.random.uniform(min_val, max_val)
            
            # Test performance
            try:
                performance = strategy['performance_func'](test_params)
                
                results.append({
                    'iteration': i,
                    'parameters': test_params.copy(),
                    'performance': performance
                })
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = test_params.copy()
                    
            except Exception as e:
                print(f"Error in iteration {i}: {e}")
                continue
        
        # Update strategy with best parameters
        strategy['best_params'] = best_params
        strategy['best_performance'] = best_performance
        
        optimization_result = {
            'strategy_name': strategy_name,
            'best_parameters': best_params,
            'best_performance': best_performance,
            'iterations': iterations,
            'improvement': best_performance - strategy.get('baseline_performance', 0),
            'all_results': results
        }
        
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimizations"""
        return {
            'total_optimizations': len(self.optimization_history),
            'strategies_optimized': list(self.strategies.keys()),
            'recent_optimizations': self.optimization_history[-5:],
            'best_performers': self._get_best_performers()
        }
    
    def _get_best_performers(self) -> List[Dict]:
        """Get best performing strategies"""
        performers = []
        
        for name, strategy in self.strategies.items():
            performers.append({
                'name': name,
                'performance': strategy['best_performance'],
                'parameters': strategy['best_params']
            })
        
        return sorted(performers, key=lambda x: x['performance'], reverse=True)

# Example strategy performance functions
def momentum_strategy_performance(params):
    """Example momentum strategy performance function"""
    lookback = params.get('lookback_period', 20)
    threshold = params.get('threshold', 0.02)
    
    # Simulate performance based on parameters
    # In reality, this would run backtests
    base_performance = 0.1
    lookback_factor = max(0, 1 - abs(lookback - 15) / 50)
    threshold_factor = max(0, 1 - abs(threshold - 0.015) / 0.1)
    
    return base_performance * lookback_factor * threshold_factor + np.random.normal(0, 0.02)

def mean_reversion_strategy_performance(params):
    """Example mean reversion strategy performance function"""
    window = params.get('window', 10)
    z_threshold = params.get('z_threshold', 2.0)
    
    base_performance = 0.08
    window_factor = max(0, 1 - abs(window - 12) / 30)
    z_factor = max(0, 1 - abs(z_threshold - 1.8) / 5)
    
    return base_performance * window_factor * z_factor + np.random.normal(0, 0.015)

# Global optimizer instance
strategy_optimizer = StrategyOptimizer()

# Register example strategies
strategy_optimizer.register_strategy(
    'momentum', 
    {'lookback_period': 20, 'threshold': 0.02},
    momentum_strategy_performance
)

strategy_optimizer.register_strategy(
    'mean_reversion',
    {'window': 10, 'z_threshold': 2.0},
    mean_reversion_strategy_performance
)
'''
    
    try:
        with open('strategy_optimizer.py', 'w') as f:
            f.write(code)
        logger.info("Created strategy optimizer")
        return True
    except Exception as e:
        logger.error(f"Failed to create strategy optimizer: {e}")
        return False

def main():
    """Apply enhanced trading features"""
    print("ENHANCED TRADING FEATURES")
    print("=" * 40)
    
    fixes_applied = 0
    total_fixes = 3
    
    print("\n1. Creating advanced backtesting system...")
    if create_advanced_backtesting():
        fixes_applied += 1
        print("   ✅ Advanced backtesting created")
    
    print("\n2. Creating portfolio analytics...")
    if create_portfolio_analytics():
        fixes_applied += 1
        print("   ✅ Portfolio analytics created")
    
    print("\n3. Creating strategy optimizer...")
    if create_strategy_optimizer():
        fixes_applied += 1
        print("   ✅ Strategy optimizer created")
    
    print(f"\n✅ Enhanced features: {fixes_applied}/{total_fixes}")
    
    if fixes_applied >= 2:
        print("\nNEW FEATURES ADDED:")
        print("• Advanced backtesting with metrics")
        print("• Portfolio analytics and risk metrics")
        print("• Strategy parameter optimization")
        print("• Performance tracking and analysis")
        print("• Sector allocation analysis")
        print("• Automated strategy tuning")
    
    return fixes_applied >= 2

if __name__ == "__main__":
    main()