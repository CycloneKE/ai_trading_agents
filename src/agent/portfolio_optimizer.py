"""
Portfolio Optimizer for intelligent rebalancing and diversification.
Implements modern portfolio theory and advanced optimization techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_sector_weight: float = 0.3
    max_single_position: float = 0.1
    min_positions: int = 5
    max_positions: int = 50
    target_volatility: Optional[float] = None
    target_return: Optional[float] = None


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    weights: Dict[str, float]
    sector_allocation: Dict[str, float]
    risk_contribution: Dict[str, float]


class PortfolioOptimizer:
    """
    Advanced portfolio optimizer with multiple optimization objectives.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk-free rate for Sharpe ratio calculation
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        
        # Optimization settings
        self.lookback_period = config.get('lookback_period', 252)  # 1 year
        self.rebalance_frequency = config.get('rebalance_frequency', 'monthly')
        
        # Covariance estimation method
        self.covariance_method = config.get('covariance_method', 'ledoit_wolf')
        
        # Transaction costs
        self.transaction_cost = config.get('transaction_cost', 0.001)  # 0.1%
        
        logger.info("Portfolio optimizer initialized")
    
    def calculate_expected_returns(self, price_data: pd.DataFrame, 
                                 method: str = 'historical') -> pd.Series:
        """
        Calculate expected returns for assets.
        
        Args:
            price_data: Historical price data
            method: Method for calculating expected returns
            
        Returns:
            Expected returns for each asset
        """
        try:
            if method == 'historical':
                # Simple historical mean
                returns = price_data.pct_change().dropna()
                expected_returns = returns.mean() * 252  # Annualized
                
            elif method == 'ewm':
                # Exponentially weighted moving average
                returns = price_data.pct_change().dropna()
                expected_returns = returns.ewm(span=60).mean().iloc[-1] * 252
                
            elif method == 'capm':
                # Capital Asset Pricing Model (simplified)
                returns = price_data.pct_change().dropna()
                market_return = returns.mean(axis=1)  # Equal-weighted market
                
                expected_returns = pd.Series(index=price_data.columns)
                for asset in price_data.columns:
                    asset_returns = returns[asset].dropna()
                    market_aligned = market_return.reindex(asset_returns.index)
                    
                    # Calculate beta
                    covariance = np.cov(asset_returns, market_aligned)[0, 1]
                    market_variance = np.var(market_aligned)
                    beta = covariance / market_variance if market_variance > 0 else 1.0
                    
                    # CAPM expected return
                    market_premium = market_return.mean() * 252 - self.risk_free_rate
                    expected_returns[asset] = self.risk_free_rate + beta * market_premium
            
            else:
                raise ValueError(f"Unknown expected returns method: {method}")
            
            return expected_returns.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating expected returns: {str(e)}")
            return pd.Series()
    
    def calculate_covariance_matrix(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate covariance matrix of asset returns.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Covariance matrix
        """
        try:
            returns = price_data.pct_change().dropna()
            
            if self.covariance_method == 'sample':
                # Sample covariance matrix
                cov_matrix = returns.cov() * 252  # Annualized
                
            elif self.covariance_method == 'ledoit_wolf':
                # Ledoit-Wolf shrinkage estimator
                lw = LedoitWolf()
                cov_matrix = pd.DataFrame(
                    lw.fit(returns).covariance_ * 252,
                    index=returns.columns,
                    columns=returns.columns
                )
                
            elif self.covariance_method == 'exponential':
                # Exponentially weighted covariance
                cov_matrix = returns.ewm(span=60).cov().iloc[-len(returns.columns):] * 252
                
            else:
                raise ValueError(f"Unknown covariance method: {self.covariance_method}")
            
            return cov_matrix
            
        except Exception as e:
            logger.error(f"Error calculating covariance matrix: {str(e)}")
            return pd.DataFrame()
    
    def optimize_mean_variance(self, expected_returns: pd.Series, 
                             cov_matrix: pd.DataFrame,
                             constraints: OptimizationConstraints,
                             target_return: Optional[float] = None) -> Dict[str, float]:
        """
        Mean-variance optimization (Markowitz).
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            constraints: Portfolio constraints
            target_return: Target return (if None, maximize Sharpe ratio)
            
        Returns:
            Optimal portfolio weights
        """
        try:
            n_assets = len(expected_returns)
            assets = expected_returns.index.tolist()
            
            # Decision variables
            weights = cp.Variable(n_assets)
            
            # Objective function
            portfolio_return = expected_returns.values @ weights
            portfolio_variance = cp.quad_form(weights, cov_matrix.values)
            
            if target_return is not None:
                # Minimize variance for target return
                objective = cp.Minimize(portfolio_variance)
                constraints_list = [
                    cp.sum(weights) == 1,  # Fully invested
                    portfolio_return >= target_return,  # Target return constraint
                    weights >= constraints.min_weight,  # Minimum weight
                    weights <= constraints.max_weight   # Maximum weight
                ]
            else:
                # Maximize Sharpe ratio (approximate)
                risk_free_return = self.risk_free_rate
                objective = cp.Maximize(portfolio_return - risk_free_return)
                constraints_list = [
                    cp.sum(weights) == 1,  # Fully invested
                    portfolio_variance <= 1,  # Variance constraint (will be adjusted)
                    weights >= constraints.min_weight,  # Minimum weight
                    weights <= constraints.max_weight   # Maximum weight
                ]
            
            # Additional constraints
            if constraints.max_single_position < 1.0:
                constraints_list.append(weights <= constraints.max_single_position)
            
            # Solve optimization problem
            problem = cp.Problem(objective, constraints_list)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                
                # Create weights dictionary
                weight_dict = {}
                for i, asset in enumerate(assets):
                    weight = float(optimal_weights[i])
                    if abs(weight) > 1e-6:  # Filter out very small weights
                        weight_dict[asset] = weight
                
                return weight_dict
            else:
                logger.error(f"Optimization failed with status: {problem.status}")
                return {}
                
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {str(e)}")
            return {}
    
    def optimize_risk_parity(self, cov_matrix: pd.DataFrame,
                           constraints: OptimizationConstraints) -> Dict[str, float]:
        """
        Risk parity optimization - equal risk contribution.
        
        Args:
            cov_matrix: Covariance matrix
            constraints: Portfolio constraints
            
        Returns:
            Risk parity portfolio weights
        """
        try:
            n_assets = len(cov_matrix)
            assets = cov_matrix.index.tolist()
            
            def risk_parity_objective(weights):
                """Objective function for risk parity."""
                weights = np.array(weights)
                portfolio_variance = weights.T @ cov_matrix.values @ weights
                
                # Risk contributions
                marginal_risk = cov_matrix.values @ weights
                risk_contributions = weights * marginal_risk / portfolio_variance
                
                # Target equal risk contribution
                target_risk = 1.0 / n_assets
                
                # Sum of squared deviations from target
                return np.sum((risk_contributions - target_risk) ** 2)
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
            ]
            
            # Bounds
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Create weights dictionary
                weight_dict = {}
                for i, asset in enumerate(assets):
                    weight = float(optimal_weights[i])
                    if abs(weight) > 1e-6:
                        weight_dict[asset] = weight
                
                return weight_dict
            else:
                logger.error(f"Risk parity optimization failed: {result.message}")
                return {}
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {str(e)}")
            return {}
    
    def optimize_black_litterman(self, price_data: pd.DataFrame,
                               views: Dict[str, float],
                               view_confidence: Dict[str, float],
                               constraints: OptimizationConstraints) -> Dict[str, float]:
        """
        Black-Litterman optimization with investor views.
        
        Args:
            price_data: Historical price data
            views: Investor views on expected returns
            view_confidence: Confidence in each view (0-1)
            constraints: Portfolio constraints
            
        Returns:
            Black-Litterman optimal weights
        """
        try:
            # Calculate market-implied returns (simplified)
            returns = price_data.pct_change().dropna()
            cov_matrix = self.calculate_covariance_matrix(price_data)
            
            # Market capitalization weights (simplified - equal weights)
            market_weights = pd.Series(1.0 / len(price_data.columns), index=price_data.columns)
            
            # Risk aversion parameter
            risk_aversion = 3.0
            
            # Implied equilibrium returns
            implied_returns = risk_aversion * cov_matrix @ market_weights
            
            # Black-Litterman calculation
            assets = price_data.columns.tolist()
            n_assets = len(assets)
            
            # Create picking matrix P and views vector Q
            P = np.zeros((len(views), n_assets))
            Q = np.zeros(len(views))
            
            for i, (asset, view_return) in enumerate(views.items()):
                if asset in assets:
                    asset_idx = assets.index(asset)
                    P[i, asset_idx] = 1.0
                    Q[i] = view_return
            
            # Uncertainty matrix Omega
            Omega = np.diag([1.0 / view_confidence.get(asset, 0.5) for asset in views.keys()])
            
            # Tau parameter (uncertainty in prior)
            tau = 1.0 / len(returns)
            
            # Black-Litterman formula
            M1 = np.linalg.inv(tau * cov_matrix.values)
            M2 = P.T @ np.linalg.inv(Omega) @ P
            M3 = np.linalg.inv(tau * cov_matrix.values) @ implied_returns.values
            M4 = P.T @ np.linalg.inv(Omega) @ Q
            
            # New expected returns
            bl_returns = np.linalg.inv(M1 + M2) @ (M3 + M4)
            bl_returns = pd.Series(bl_returns, index=assets)
            
            # Optimize with Black-Litterman returns
            optimal_weights = self.optimize_mean_variance(
                bl_returns, cov_matrix, constraints
            )
            
            return optimal_weights
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            return {}
    
    def calculate_portfolio_metrics(self, weights: Dict[str, float],
                                  expected_returns: pd.Series,
                                  cov_matrix: pd.DataFrame,
                                  price_data: pd.DataFrame) -> PortfolioMetrics:
        """
        Calculate portfolio performance metrics.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            price_data: Historical price data
            
        Returns:
            Portfolio metrics
        """
        try:
            # Convert weights to array
            assets = list(weights.keys())
            weight_array = np.array([weights[asset] for asset in assets])
            
            # Align data
            aligned_returns = expected_returns.reindex(assets).fillna(0)
            aligned_cov = cov_matrix.reindex(assets, columns=assets).fillna(0)
            
            # Portfolio metrics
            portfolio_return = np.sum(weight_array * aligned_returns.values)
            portfolio_variance = weight_array.T @ aligned_cov.values @ weight_array
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Calculate historical performance for additional metrics
            returns = price_data[assets].pct_change().dropna()
            portfolio_returns = (returns * weight_array).sum(axis=1)
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Value at Risk (95%)
            var_95 = np.percentile(portfolio_returns, 5)
            
            # Risk contribution
            marginal_risk = aligned_cov.values @ weight_array
            risk_contribution = {}
            for i, asset in enumerate(assets):
                contrib = weight_array[i] * marginal_risk[i] / portfolio_variance
                risk_contribution[asset] = float(contrib)
            
            # Sector allocation (simplified - assume first letter indicates sector)
            sector_allocation = {}
            for asset, weight in weights.items():
                sector = asset[0]  # Simplified sector classification
                sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
            
            return PortfolioMetrics(
                expected_return=float(portfolio_return),
                volatility=float(portfolio_volatility),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown=float(max_drawdown),
                var_95=float(var_95),
                weights=weights,
                sector_allocation=sector_allocation,
                risk_contribution=risk_contribution
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return PortfolioMetrics(
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                weights={},
                sector_allocation={},
                risk_contribution={}
            )
    
    def rebalance_portfolio(self, current_weights: Dict[str, float],
                          target_weights: Dict[str, float],
                          price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate rebalancing trades with transaction costs.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            price_data: Current price data
            
        Returns:
            Rebalancing information
        """
        try:
            # Calculate weight differences
            all_assets = set(current_weights.keys()) | set(target_weights.keys())
            
            trades = {}
            total_turnover = 0.0
            
            for asset in all_assets:
                current_weight = current_weights.get(asset, 0.0)
                target_weight = target_weights.get(asset, 0.0)
                
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 1e-6:  # Minimum trade threshold
                    trades[asset] = {
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'weight_change': weight_diff,
                        'trade_direction': 'buy' if weight_diff > 0 else 'sell'
                    }
                    
                    total_turnover += abs(weight_diff)
            
            # Calculate transaction costs
            total_transaction_cost = total_turnover * self.transaction_cost
            
            # Rebalancing metrics
            rebalancing_info = {
                'trades': trades,
                'total_turnover': total_turnover,
                'transaction_cost': total_transaction_cost,
                'n_trades': len(trades),
                'rebalancing_date': datetime.now().isoformat()
            }
            
            return rebalancing_info
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing: {str(e)}")
            return {}
    
    def optimize_portfolio(self, price_data: pd.DataFrame,
                         optimization_method: str = 'mean_variance',
                         constraints: Optional[OptimizationConstraints] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Main portfolio optimization function.
        
        Args:
            price_data: Historical price data
            optimization_method: Optimization method to use
            constraints: Portfolio constraints
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Optimization results
        """
        try:
            if constraints is None:
                constraints = OptimizationConstraints()
            
            # Calculate inputs
            expected_returns = self.calculate_expected_returns(
                price_data, 
                kwargs.get('return_method', 'historical')
            )
            cov_matrix = self.calculate_covariance_matrix(price_data)
            
            # Optimize based on method
            if optimization_method == 'mean_variance':
                optimal_weights = self.optimize_mean_variance(
                    expected_returns, cov_matrix, constraints,
                    kwargs.get('target_return')
                )
                
            elif optimization_method == 'risk_parity':
                optimal_weights = self.optimize_risk_parity(cov_matrix, constraints)
                
            elif optimization_method == 'black_litterman':
                optimal_weights = self.optimize_black_litterman(
                    price_data,
                    kwargs.get('views', {}),
                    kwargs.get('view_confidence', {}),
                    constraints
                )
                
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
            
            if not optimal_weights:
                return {}
            
            # Calculate portfolio metrics
            portfolio_metrics = self.calculate_portfolio_metrics(
                optimal_weights, expected_returns, cov_matrix, price_data
            )
            
            return {
                'optimal_weights': optimal_weights,
                'portfolio_metrics': portfolio_metrics,
                'optimization_method': optimization_method,
                'optimization_date': datetime.now().isoformat(),
                'constraints': constraints,
                'inputs': {
                    'expected_returns': expected_returns.to_dict(),
                    'assets': list(price_data.columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {}

