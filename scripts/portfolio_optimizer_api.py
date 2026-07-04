"""
Portfolio Optimizer API integration for advanced portfolio optimization.
"""

import os
import logging
import requests
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

class PortfolioOptimizerAPI:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.getenv('TRADING_PORTFOLIO_OPTIMIZER_API_KEY', config.get('api_key', ''))
        self.base_url = config.get('base_url', 'https://api.portfoliooptimizer.io')
        self.is_connected = False
        
    def connect(self) -> bool:
        """Connect to Portfolio Optimizer API."""
        if not self.api_key:
            logger.warning("Portfolio Optimizer API key not provided, using local optimization")
            return True  # Can still work without API
        
        try:
            # Test connection
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.get(f"{self.base_url}/v1/status", headers=headers)
            
            if response.status_code == 200:
                self.is_connected = True
                logger.info("Connected to Portfolio Optimizer API")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to Portfolio Optimizer API: {str(e)}")
        
        return False
    
    def optimize_portfolio(self, returns_data: Dict[str, List[float]], 
                          risk_tolerance: float = 0.5) -> Dict[str, Any]:
        """Optimize portfolio allocation."""
        
        if self.is_connected and self.api_key:
            return self._api_optimize(returns_data, risk_tolerance)
        else:
            return self._local_optimize(returns_data, risk_tolerance)
    
    def _api_optimize(self, returns_data: Dict[str, List[float]], 
                     risk_tolerance: float) -> Dict[str, Any]:
        """Use API for optimization."""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'returns': returns_data,
                'risk_tolerance': risk_tolerance,
                'optimization_method': 'mean_variance',
                'constraints': {
                    'max_weight': 0.4,  # Max 40% in any single asset
                    'min_weight': 0.05  # Min 5% in any asset
                }
            }
            
            response = requests.post(
                f"{self.base_url}/v1/optimize",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'weights': result.get('weights', {}),
                    'expected_return': result.get('expected_return', 0),
                    'expected_risk': result.get('expected_risk', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'method': 'api'
                }
        except Exception as e:
            logger.error(f"API optimization failed: {str(e)}")
        
        # Fallback to local optimization
        return self._local_optimize(returns_data, risk_tolerance)
    
    def _local_optimize(self, returns_data: Dict[str, List[float]], 
                       risk_tolerance: float) -> Dict[str, Any]:
        """Local portfolio optimization using simple methods."""
        try:
            symbols = list(returns_data.keys())
            n_assets = len(symbols)
            
            if n_assets == 0:
                return {'weights': {}, 'expected_return': 0, 'expected_risk': 0}
            
            # Calculate mean returns and volatilities
            mean_returns = {}
            volatilities = {}
            
            for symbol, returns in returns_data.items():
                if len(returns) > 0:
                    mean_returns[symbol] = np.mean(returns)
                    volatilities[symbol] = np.std(returns)
                else:
                    mean_returns[symbol] = 0
                    volatilities[symbol] = 0.2  # Default volatility
            
            # Simple risk-adjusted weighting
            weights = {}
            total_score = 0
            
            for symbol in symbols:
                # Score based on return/risk ratio adjusted by risk tolerance
                if volatilities[symbol] > 0:
                    score = mean_returns[symbol] / volatilities[symbol]
                    # Adjust for risk tolerance (0 = risk averse, 1 = risk seeking)
                    score = score * (1 + risk_tolerance)
                else:
                    score = mean_returns[symbol]
                
                # Ensure positive weights
                score = max(score, 0.1)
                weights[symbol] = score
                total_score += score
            
            # Normalize weights
            if total_score > 0:
                for symbol in weights:
                    weights[symbol] = weights[symbol] / total_score
                    # Apply constraints
                    weights[symbol] = max(0.05, min(0.4, weights[symbol]))
            else:
                # Equal weighting fallback
                equal_weight = 1.0 / n_assets
                weights = {symbol: equal_weight for symbol in symbols}
            
            # Renormalize after constraints
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {symbol: w / total_weight for symbol, w in weights.items()}
            
            # Calculate expected portfolio metrics
            expected_return = sum(mean_returns[symbol] * weights[symbol] for symbol in symbols)
            expected_risk = np.sqrt(sum((volatilities[symbol] * weights[symbol]) ** 2 for symbol in symbols))
            sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0
            
            return {
                'weights': weights,
                'expected_return': expected_return,
                'expected_risk': expected_risk,
                'sharpe_ratio': sharpe_ratio,
                'method': 'local'
            }
            
        except Exception as e:
            logger.error(f"Local optimization failed: {str(e)}")
            # Ultimate fallback - equal weights
            symbols = list(returns_data.keys())
            if symbols:
                equal_weight = 1.0 / len(symbols)
                return {
                    'weights': {symbol: equal_weight for symbol in symbols},
                    'expected_return': 0.05,  # Assume 5% return
                    'expected_risk': 0.15,    # Assume 15% risk
                    'sharpe_ratio': 0.33,
                    'method': 'equal_weight'
                }
            
            return {'weights': {}, 'expected_return': 0, 'expected_risk': 0}
    
    def get_risk_metrics(self, portfolio_weights: Dict[str, float], 
                        returns_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics."""
        try:
            # Calculate portfolio returns
            portfolio_returns = []
            min_length = min(len(returns) for returns in returns_data.values() if returns)
            
            for i in range(min_length):
                portfolio_return = sum(
                    portfolio_weights.get(symbol, 0) * returns_data[symbol][i]
                    for symbol in returns_data.keys()
                )
                portfolio_returns.append(portfolio_return)
            
            if not portfolio_returns:
                return {}
            
            # Calculate metrics
            portfolio_returns = np.array(portfolio_returns)
            
            return {
                'var_95': np.percentile(portfolio_returns, 5),  # 95% VaR
                'cvar_95': np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]),
                'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
                'volatility': np.std(portfolio_returns),
                'skewness': float(np.mean(((portfolio_returns - np.mean(portfolio_returns)) / np.std(portfolio_returns)) ** 3)),
                'kurtosis': float(np.mean(((portfolio_returns - np.mean(portfolio_returns)) / np.std(portfolio_returns)) ** 4))
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return float(np.min(drawdown))
        except:
            return 0.0