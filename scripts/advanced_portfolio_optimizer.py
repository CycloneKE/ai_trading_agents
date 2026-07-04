#!/usr/bin/env python3
"""
Advanced Portfolio Optimization System

Sophisticated portfolio optimization using multiple approaches including
Black-Litterman, risk parity, and machine learning-based optimization.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

try:
    from scipy.optimize import minimize
    from scipy import linalg
    import cvxpy as cp
    SCIPY_AVAILABLE = True
    CVXPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    CVXPY_AVAILABLE = False

try:
    from sklearn.covariance import LedoitWolf
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdvancedPortfolioOptimizer:
    """Advanced portfolio optimization with multiple methodologies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.lookback_period = config.get('lookback_period', 252)
        self.rebalance_frequency = config.get('rebalance_frequency', 21)  # days
        
        # Optimization parameters
        self.max_weight = config.get('max_weight', 0.4)
        self.min_weight = config.get('min_weight', 0.0)
        self.target_volatility = config.get('target_volatility', 0.15)
        self.transaction_cost = config.get('transaction_cost', 0.001)
        
        # Data storage
        self.price_history = defaultdict(list)
        self.return_history = defaultdict(list)
        self.covariance_matrix = None
        self.expected_returns = None
        
        # Current portfolio
        self.current_weights = {}
        self.last_optimization = datetime.now() - timedelta(days=999)
        self.optimization_history = []
        
        # Black-Litterman parameters
        self.bl_tau = config.get('bl_tau', 0.025)
        self.bl_confidence = config.get('bl_confidence', 0.5)
        
        logger.info("Advanced Portfolio Optimizer initialized")
    
    def update_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """Update market data for portfolio optimization"""
        try:
            price = market_data.get('close', 0)
            if price <= 0:
                return
            
            self.price_history[symbol].append(price)
            
            # Keep only recent data
            if len(self.price_history[symbol]) > self.lookback_period * 2:
                self.price_history[symbol] = self.price_history[symbol][-self.lookback_period * 2:]
            
            # Calculate returns
            if len(self.price_history[symbol]) >= 2:
                returns = np.diff(self.price_history[symbol]) / self.price_history[symbol][:-1]
                self.return_history[symbol] = returns.tolist()
            
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
    
    def should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced"""
        days_since_rebalance = (datetime.now() - self.last_optimization).days
        return days_since_rebalance >= self.rebalance_frequency
    
    def optimize_portfolio(self, method: str = 'mean_variance', 
                          views: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Optimize portfolio using specified method"""
        try:
            if not self._has_sufficient_data():
                return {'error': 'Insufficient data for optimization'}
            
            # Prepare data
            symbols = list(self.return_history.keys())
            returns_matrix = self._prepare_returns_matrix(symbols)
            
            if returns_matrix is None:
                return {'error': 'Failed to prepare returns matrix'}
            
            # Calculate expected returns and covariance
            self.expected_returns = self._calculate_expected_returns(returns_matrix)
            self.covariance_matrix = self._calculate_covariance_matrix(returns_matrix)
            
            # Optimize based on method
            if method == 'mean_variance':
                weights = self._mean_variance_optimization(symbols)
            elif method == 'black_litterman':
                weights = self._black_litterman_optimization(symbols, views)
            elif method == 'risk_parity':
                weights = self._risk_parity_optimization(symbols)
            elif method == 'min_variance':
                weights = self._minimum_variance_optimization(symbols)
            elif method == 'max_sharpe':
                weights = self._maximum_sharpe_optimization(symbols)
            elif method == 'ml_enhanced':
                weights = self._ml_enhanced_optimization(symbols)
            else:
                weights = self._mean_variance_optimization(symbols)
            
            if weights is None:
                return {'error': f'Optimization failed for method: {method}'}
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(weights, symbols)
            
            # Store results
            optimization_result = {
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'weights': weights,
                'symbols': symbols,
                'metrics': portfolio_metrics,
                'expected_return': float(portfolio_metrics['expected_return']),
                'expected_volatility': float(portfolio_metrics['volatility']),
                'sharpe_ratio': float(portfolio_metrics['sharpe_ratio'])
            }
            
            self.optimization_history.append(optimization_result)
            self.current_weights = weights
            self.last_optimization = datetime.now()
            
            logger.info(f"Portfolio optimized using {method}: Sharpe={portfolio_metrics['sharpe_ratio']:.3f}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {'error': str(e)}
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data for optimization"""
        if len(self.return_history) < 2:
            return False
        
        min_observations = max(30, len(self.return_history) * 2)  # At least 2x number of assets
        
        for symbol, returns in self.return_history.items():
            if len(returns) < min_observations:
                return False
        
        return True
    
    def _prepare_returns_matrix(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Prepare returns matrix for optimization"""
        try:
            # Find common time period
            min_length = min(len(self.return_history[symbol]) for symbol in symbols)
            
            if min_length < 30:
                return None
            
            # Create returns matrix
            returns_matrix = np.zeros((min_length, len(symbols)))
            
            for i, symbol in enumerate(symbols):
                returns = self.return_history[symbol][-min_length:]
                returns_matrix[:, i] = returns
            
            # Remove any rows with NaN or infinite values
            valid_rows = np.isfinite(returns_matrix).all(axis=1)
            returns_matrix = returns_matrix[valid_rows]
            
            if len(returns_matrix) < 20:
                return None
            
            return returns_matrix
            
        except Exception as e:
            logger.error(f"Error preparing returns matrix: {e}")
            return None
    
    def _calculate_expected_returns(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Calculate expected returns using multiple methods"""
        try:
            # Simple historical mean
            historical_mean = np.mean(returns_matrix, axis=0)
            
            # Exponentially weighted mean (more weight on recent data)
            weights = np.exp(np.linspace(-1, 0, len(returns_matrix)))
            weights = weights / np.sum(weights)
            ew_mean = np.average(returns_matrix, axis=0, weights=weights)
            
            # Combine methods
            expected_returns = 0.7 * historical_mean + 0.3 * ew_mean
            
            # Annualize
            expected_returns = expected_returns * 252
            
            return expected_returns
            
        except Exception as e:
            logger.error(f"Error calculating expected returns: {e}")
            return np.zeros(returns_matrix.shape[1])
    
    def _calculate_covariance_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Calculate covariance matrix with shrinkage"""
        try:
            if SKLEARN_AVAILABLE:
                # Use Ledoit-Wolf shrinkage estimator
                lw = LedoitWolf()
                cov_matrix = lw.fit(returns_matrix).covariance_
            else:
                # Simple sample covariance
                cov_matrix = np.cov(returns_matrix.T)
            
            # Annualize
            cov_matrix = cov_matrix * 252
            
            # Ensure positive definite
            eigenvals, eigenvecs = linalg.eigh(cov_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)
            cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            return cov_matrix
            
        except Exception as e:
            logger.error(f"Error calculating covariance matrix: {e}")
            return np.eye(returns_matrix.shape[1]) * 0.04  # Default 20% volatility
    
    def _mean_variance_optimization(self, symbols: List[str]) -> Optional[Dict[str, float]]:
        """Mean-variance optimization (Markowitz)"""
        try:
            if not SCIPY_AVAILABLE:
                return self._equal_weight_fallback(symbols)
            
            n_assets = len(symbols)
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights, np.dot(self.covariance_matrix, weights))
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights_dict = {symbol: float(weight) for symbol, weight in zip(symbols, result.x)}
                return weights_dict
            else:
                logger.warning("Mean-variance optimization failed, using equal weights")
                return self._equal_weight_fallback(symbols)
                
        except Exception as e:
            logger.error(f"Mean-variance optimization error: {e}")
            return self._equal_weight_fallback(symbols)
    
    def _black_litterman_optimization(self, symbols: List[str], 
                                    views: Optional[Dict[str, float]] = None) -> Optional[Dict[str, float]]:
        """Black-Litterman optimization with investor views"""
        try:
            if not SCIPY_AVAILABLE:
                return self._mean_variance_optimization(symbols)
            
            n_assets = len(symbols)
            
            # Market capitalization weights (proxy using equal weights)
            market_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Implied equilibrium returns
            risk_aversion = 3.0  # Typical value
            pi = risk_aversion * np.dot(self.covariance_matrix, market_weights)
            
            # Incorporate views if provided
            if views:
                # Create views matrix
                view_symbols = [s for s in symbols if s in views]
                if view_symbols:
                    P = np.zeros((len(view_symbols), n_assets))
                    Q = np.zeros(len(view_symbols))
                    
                    for i, symbol in enumerate(view_symbols):
                        symbol_idx = symbols.index(symbol)
                        P[i, symbol_idx] = 1.0
                        Q[i] = views[symbol]
                    
                    # View uncertainty (Omega)
                    Omega = np.eye(len(view_symbols)) * self.bl_confidence
                    
                    # Black-Litterman formula
                    tau_cov = self.bl_tau * self.covariance_matrix
                    M1 = linalg.inv(tau_cov)
                    M2 = np.dot(P.T, np.dot(linalg.inv(Omega), P))
                    M3 = np.dot(linalg.inv(tau_cov), pi)
                    M4 = np.dot(P.T, np.dot(linalg.inv(Omega), Q))
                    
                    # New expected returns
                    mu_bl = np.dot(linalg.inv(M1 + M2), M3 + M4)
                    
                    # New covariance matrix
                    cov_bl = linalg.inv(M1 + M2)
                else:
                    mu_bl = pi
                    cov_bl = self.covariance_matrix
            else:
                mu_bl = pi
                cov_bl = self.covariance_matrix
            
            # Optimize with Black-Litterman inputs
            def objective(weights):
                portfolio_return = np.dot(weights, mu_bl)
                portfolio_variance = np.dot(weights, np.dot(cov_bl, weights))
                return -portfolio_return + 0.5 * risk_aversion * portfolio_variance
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            x0 = market_weights
            
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights_dict = {symbol: float(weight) for symbol, weight in zip(symbols, result.x)}
                return weights_dict
            else:
                return self._mean_variance_optimization(symbols)
                
        except Exception as e:
            logger.error(f"Black-Litterman optimization error: {e}")
            return self._mean_variance_optimization(symbols)
    
    def _risk_parity_optimization(self, symbols: List[str]) -> Optional[Dict[str, float]]:
        """Risk parity optimization (equal risk contribution)"""
        try:
            if not SCIPY_AVAILABLE:
                return self._equal_weight_fallback(symbols)
            
            n_assets = len(symbols)
            
            def risk_budget_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
                marginal_contrib = np.dot(self.covariance_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            result = minimize(risk_budget_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights_dict = {symbol: float(weight) for symbol, weight in zip(symbols, result.x)}
                return weights_dict
            else:
                return self._equal_weight_fallback(symbols)
                
        except Exception as e:
            logger.error(f"Risk parity optimization error: {e}")
            return self._equal_weight_fallback(symbols)
    
    def _minimum_variance_optimization(self, symbols: List[str]) -> Optional[Dict[str, float]]:
        """Minimum variance optimization"""
        try:
            if not SCIPY_AVAILABLE:
                return self._equal_weight_fallback(symbols)
            
            n_assets = len(symbols)
            
            def objective(weights):
                return np.dot(weights, np.dot(self.covariance_matrix, weights))
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights_dict = {symbol: float(weight) for symbol, weight in zip(symbols, result.x)}
                return weights_dict
            else:
                return self._equal_weight_fallback(symbols)
                
        except Exception as e:
            logger.error(f"Minimum variance optimization error: {e}")
            return self._equal_weight_fallback(symbols)
    
    def _maximum_sharpe_optimization(self, symbols: List[str]) -> Optional[Dict[str, float]]:
        """Maximum Sharpe ratio optimization"""
        try:
            if not SCIPY_AVAILABLE:
                return self._mean_variance_optimization(symbols)
            
            n_assets = len(symbols)
            
            def negative_sharpe(weights):
                portfolio_return = np.dot(weights, self.expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
                return -(portfolio_return - self.risk_free_rate) / portfolio_vol
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights_dict = {symbol: float(weight) for symbol, weight in zip(symbols, result.x)}
                return weights_dict
            else:
                return self._mean_variance_optimization(symbols)
                
        except Exception as e:
            logger.error(f"Maximum Sharpe optimization error: {e}")
            return self._mean_variance_optimization(symbols)
    
    def _ml_enhanced_optimization(self, symbols: List[str]) -> Optional[Dict[str, float]]:
        """ML-enhanced optimization using return predictions"""
        try:
            if not SKLEARN_AVAILABLE:
                return self._mean_variance_optimization(symbols)
            
            # Use ML to predict future returns
            ml_returns = self._predict_returns_ml(symbols)
            
            if ml_returns is not None:
                # Temporarily replace expected returns
                original_returns = self.expected_returns.copy()
                self.expected_returns = ml_returns
                
                # Optimize with ML predictions
                weights = self._maximum_sharpe_optimization(symbols)
                
                # Restore original returns
                self.expected_returns = original_returns
                
                return weights
            else:
                return self._mean_variance_optimization(symbols)
                
        except Exception as e:
            logger.error(f"ML-enhanced optimization error: {e}")
            return self._mean_variance_optimization(symbols)
    
    def _predict_returns_ml(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Predict returns using machine learning"""
        try:
            if not SKLEARN_AVAILABLE:
                return None
            
            predictions = []
            
            for symbol in symbols:
                returns = self.return_history[symbol]
                if len(returns) < 60:
                    predictions.append(0.0)
                    continue
                
                # Create features (lagged returns, moving averages, etc.)
                features = []
                targets = []
                
                for i in range(20, len(returns) - 1):
                    # Features: last 20 returns, moving averages
                    feature_vector = returns[i-20:i]
                    feature_vector.extend([
                        np.mean(returns[i-5:i]),   # 5-day MA
                        np.mean(returns[i-20:i]),  # 20-day MA
                        np.std(returns[i-20:i])    # 20-day volatility
                    ])
                    
                    features.append(feature_vector)
                    targets.append(returns[i])
                
                if len(features) < 30:
                    predictions.append(0.0)
                    continue
                
                # Train model
                X = np.array(features)
                y = np.array(targets)
                
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X, y)
                
                # Predict next return
                last_features = returns[-20:]
                last_features.extend([
                    np.mean(returns[-5:]),
                    np.mean(returns[-20:]),
                    np.std(returns[-20:])
                ])
                
                prediction = model.predict([last_features])[0]
                predictions.append(prediction)
            
            # Annualize predictions
            ml_returns = np.array(predictions) * 252
            
            return ml_returns
            
        except Exception as e:
            logger.error(f"ML return prediction error: {e}")
            return None
    
    def _equal_weight_fallback(self, symbols: List[str]) -> Dict[str, float]:
        """Fallback to equal weights"""
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}
    
    def _calculate_portfolio_metrics(self, weights: Dict[str, float], 
                                   symbols: List[str]) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        try:
            weight_array = np.array([weights[symbol] for symbol in symbols])
            
            # Expected return
            expected_return = np.dot(weight_array, self.expected_returns)
            
            # Volatility
            volatility = np.sqrt(np.dot(weight_array, np.dot(self.covariance_matrix, weight_array)))
            
            # Sharpe ratio
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Diversification ratio
            individual_vols = np.sqrt(np.diag(self.covariance_matrix))
            weighted_avg_vol = np.dot(weight_array, individual_vols)
            diversification_ratio = weighted_avg_vol / volatility if volatility > 0 else 1
            
            return {
                'expected_return': float(expected_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'diversification_ratio': float(diversification_ratio),
                'max_weight': float(max(weights.values())),
                'min_weight': float(min(weights.values())),
                'effective_assets': float(1 / np.sum(weight_array ** 2))  # Inverse Herfindahl index
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def get_current_allocation(self) -> Dict[str, Any]:
        """Get current portfolio allocation"""
        return {
            'weights': self.current_weights.copy(),
            'last_optimization': self.last_optimization.isoformat(),
            'next_rebalance': (self.last_optimization + timedelta(days=self.rebalance_frequency)).isoformat(),
            'optimization_count': len(self.optimization_history)
        }
    
    def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent optimization history"""
        return self.optimization_history[-limit:] if self.optimization_history else []
    
    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status"""
        return {
            'status': 'active',
            'scipy_available': SCIPY_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE,
            'cvxpy_available': CVXPY_AVAILABLE,
            'assets_tracked': len(self.return_history),
            'sufficient_data': self._has_sufficient_data(),
            'last_optimization': self.last_optimization.isoformat()
        }