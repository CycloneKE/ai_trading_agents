import numpy as np
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
