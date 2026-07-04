"""
Risk Calculator for portfolio risk metrics and real-time risk assessment.
Implements various risk measures including VaR, CVaR, Sharpe ratio, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RiskCalculator:
    """
    Calculator for various portfolio risk metrics and measures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk parameters
        self.confidence_levels = config.get('confidence_levels', [0.95, 0.99])
        self.lookback_periods = config.get('lookback_periods', [30, 60, 252])
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # Annual risk-free rate
        
        # Portfolio limits
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)  # 2% daily VaR
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.max_sector_exposure = config.get('max_sector_exposure', 0.3)  # 30% per sector
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.5)
        
        # Risk monitoring
        self.risk_alerts = []
        self.risk_history = []
        
        logger.info("Risk calculator initialized")
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95, 
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level for VaR calculation
            method: Method to use ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            VaR value
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            returns = np.array(returns)
            returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                return 0.0
            
            if method == 'historical':
                # Historical simulation method
                var = np.percentile(returns, (1 - confidence_level) * 100)
                
            elif method == 'parametric':
                # Parametric method (assumes normal distribution)
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                z_score = stats.norm.ppf(1 - confidence_level)
                var = mean_return + z_score * std_return
                
            elif method == 'monte_carlo':
                # Monte Carlo simulation
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                
                # Generate random scenarios
                n_simulations = 10000
                simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
                var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
                
            else:
                logger.warning(f"Unknown VaR method: {method}, using historical")
                var = np.percentile(returns, (1 - confidence_level) * 100)
            
            return float(var)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level for CVaR calculation
            
        Returns:
            CVaR value
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            returns = np.array(returns)
            returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                return 0.0
            
            # Calculate VaR first
            var = self.calculate_var(returns, confidence_level, method='historical')
            
            # Calculate CVaR as the mean of returns below VaR
            tail_returns = returns[returns <= var]
            
            if len(tail_returns) == 0:
                return var
            
            cvar = np.mean(tail_returns)
            return float(cvar)
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {str(e)}")
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            returns = np.array(returns)
            returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                return 0.0
            
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            # Convert annual risk-free rate to period rate
            daily_rf_rate = risk_free_rate / 252
            
            excess_returns = returns - daily_rf_rate
            
            if np.std(excess_returns) == 0:
                return 0.0
            
            sharpe = np.mean(excess_returns) / np.std(excess_returns)
            
            # Annualize
            sharpe_annualized = sharpe * np.sqrt(252)
            
            return float(sharpe_annualized)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sortino ratio
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            returns = np.array(returns)
            returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                return 0.0
            
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            daily_rf_rate = risk_free_rate / 252
            excess_returns = returns - daily_rf_rate
            
            # Calculate downside deviation
            negative_returns = excess_returns[excess_returns < 0]
            
            if len(negative_returns) == 0:
                return float('inf')  # No downside risk
            
            downside_deviation = np.std(negative_returns)
            
            if downside_deviation == 0:
                return 0.0
            
            sortino = np.mean(excess_returns) / downside_deviation
            
            # Annualize
            sortino_annualized = sortino * np.sqrt(252)
            
            return float(sortino_annualized)
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
    
    def calculate_max_drawdown(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            returns: Array of returns
            
        Returns:
            Dict with drawdown metrics
        """
        try:
            if len(returns) == 0:
                return {'max_drawdown': 0.0, 'drawdown_duration': 0, 'current_drawdown': 0.0}
            
            returns = np.array(returns)
            returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                return {'max_drawdown': 0.0, 'drawdown_duration': 0, 'current_drawdown': 0.0}
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + returns)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Maximum drawdown
            max_drawdown = np.min(drawdown)
            
            # Current drawdown
            current_drawdown = drawdown[-1]
            
            # Drawdown duration (periods in current drawdown)
            drawdown_duration = 0
            for i in range(len(drawdown) - 1, -1, -1):
                if drawdown[i] < 0:
                    drawdown_duration += 1
                else:
                    break
            
            return {
                'max_drawdown': float(max_drawdown),
                'drawdown_duration': drawdown_duration,
                'current_drawdown': float(current_drawdown)
            }
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return {'max_drawdown': 0.0, 'drawdown_duration': 0, 'current_drawdown': 0.0}
    
    def calculate_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """
        Calculate beta (systematic risk).
        
        Args:
            asset_returns: Asset returns
            market_returns: Market benchmark returns
            
        Returns:
            Beta value
        """
        try:
            if len(asset_returns) == 0 or len(market_returns) == 0:
                return 1.0
            
            # Align arrays
            min_length = min(len(asset_returns), len(market_returns))
            asset_returns = np.array(asset_returns[-min_length:])
            market_returns = np.array(market_returns[-min_length:])
            
            # Remove NaN values
            valid_mask = ~(np.isnan(asset_returns) | np.isnan(market_returns))
            asset_returns = asset_returns[valid_mask]
            market_returns = market_returns[valid_mask]
            
            if len(asset_returns) < 2:
                return 1.0
            
            # Calculate beta using covariance
            covariance = np.cov(asset_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            
            return float(beta)
            
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return 1.0
    
    def calculate_portfolio_risk(self, positions: Dict[str, Dict[str, Any]], 
                               price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            positions: Dict of positions with weights and quantities
            price_data: Historical price data for each asset
            
        Returns:
            Dict with portfolio risk metrics
        """
        try:
            if not positions or not price_data:
                return self._empty_portfolio_risk()
            
            # Calculate returns for each asset
            asset_returns = {}
            portfolio_weights = {}
            
            total_value = sum(pos.get('market_value', 0) for pos in positions.values())
            
            for symbol, position in positions.items():
                if symbol in price_data and not price_data[symbol].empty:
                    # Calculate returns
                    prices = price_data[symbol]['close'].dropna()
                    if len(prices) > 1:
                        returns = prices.pct_change().dropna()
                        asset_returns[symbol] = returns.values
                        
                        # Calculate weight
                        market_value = position.get('market_value', 0)
                        weight = market_value / total_value if total_value > 0 else 0
                        portfolio_weights[symbol] = weight
            
            if not asset_returns:
                return self._empty_portfolio_risk()
            
            # Align returns data
            min_length = min(len(returns) for returns in asset_returns.values())
            if min_length < 2:
                return self._empty_portfolio_risk()
            
            # Create returns matrix
            symbols = list(asset_returns.keys())
            returns_matrix = np.array([asset_returns[symbol][-min_length:] for symbol in symbols]).T
            weights = np.array([portfolio_weights[symbol] for symbol in symbols])
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(returns_matrix, weights)
            
            # Calculate risk metrics
            portfolio_risk = {
                'portfolio_return': {
                    'daily_mean': float(np.mean(portfolio_returns)),
                    'daily_std': float(np.std(portfolio_returns)),
                    'annualized_return': float(np.mean(portfolio_returns) * 252),
                    'annualized_volatility': float(np.std(portfolio_returns) * np.sqrt(252))
                },
                'var_metrics': {},
                'risk_ratios': {},
                'drawdown_metrics': {},
                'correlation_metrics': {},
                'position_metrics': {}
            }
            
            # VaR calculations
            for confidence_level in self.confidence_levels:
                var = self.calculate_var(portfolio_returns, confidence_level)
                cvar = self.calculate_cvar(portfolio_returns, confidence_level)
                
                portfolio_risk['var_metrics'][f'var_{int(confidence_level*100)}'] = float(var)
                portfolio_risk['var_metrics'][f'cvar_{int(confidence_level*100)}'] = float(cvar)
            
            # Risk ratios
            portfolio_risk['risk_ratios']['sharpe_ratio'] = self.calculate_sharpe_ratio(portfolio_returns)
            portfolio_risk['risk_ratios']['sortino_ratio'] = self.calculate_sortino_ratio(portfolio_returns)
            
            # Drawdown metrics
            portfolio_risk['drawdown_metrics'] = self.calculate_max_drawdown(portfolio_returns)
            
            # Correlation analysis
            if len(symbols) > 1:
                correlation_matrix = np.corrcoef(returns_matrix.T)
                avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
                max_correlation = np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
                
                portfolio_risk['correlation_metrics'] = {
                    'average_correlation': float(avg_correlation),
                    'maximum_correlation': float(max_correlation),
                    'correlation_matrix': correlation_matrix.tolist()
                }
            
            # Position-level metrics
            for i, symbol in enumerate(symbols):
                asset_return = asset_returns[symbol][-min_length:]
                weight = portfolio_weights[symbol]
                
                portfolio_risk['position_metrics'][symbol] = {
                    'weight': float(weight),
                    'daily_volatility': float(np.std(asset_return)),
                    'annualized_volatility': float(np.std(asset_return) * np.sqrt(252)),
                    'var_95': float(self.calculate_var(asset_return, 0.95)),
                    'contribution_to_risk': float(weight * np.std(asset_return))
                }
            
            # Risk alerts
            self._check_risk_alerts(portfolio_risk, positions)
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            return self._empty_portfolio_risk()
    
    def _check_risk_alerts(self, portfolio_risk: Dict[str, Any], positions: Dict[str, Dict[str, Any]]):
        """
        Check for risk limit breaches and generate alerts.
        
        Args:
            portfolio_risk: Portfolio risk metrics
            positions: Current positions
        """
        try:
            alerts = []
            
            # Check portfolio VaR
            var_95 = portfolio_risk.get('var_metrics', {}).get('var_95', 0)
            if abs(var_95) > self.max_portfolio_risk:
                alerts.append({
                    'type': 'portfolio_var_breach',
                    'message': f"Portfolio VaR ({abs(var_95):.3f}) exceeds limit ({self.max_portfolio_risk:.3f})",
                    'severity': 'high',
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Check position sizes
            total_value = sum(pos.get('market_value', 0) for pos in positions.values())
            for symbol, position in positions.items():
                weight = position.get('market_value', 0) / total_value if total_value > 0 else 0
                if weight > self.max_position_size:
                    alerts.append({
                        'type': 'position_size_breach',
                        'message': f"Position {symbol} ({weight:.3f}) exceeds size limit ({self.max_position_size:.3f})",
                        'severity': 'medium',
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
            # Check correlation exposure
            max_correlation = portfolio_risk.get('correlation_metrics', {}).get('maximum_correlation', 0)
            if max_correlation > self.max_correlation_exposure:
                alerts.append({
                    'type': 'correlation_breach',
                    'message': f"Maximum correlation ({max_correlation:.3f}) exceeds limit ({self.max_correlation_exposure:.3f})",
                    'severity': 'medium',
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Store alerts
            self.risk_alerts.extend(alerts)
            
            # Keep only recent alerts
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.risk_alerts = [
                alert for alert in self.risk_alerts
                if datetime.fromisoformat(alert['timestamp']) > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error checking risk alerts: {str(e)}")
    
    def calculate_position_sizing(self, expected_return: float, volatility: float, 
                                confidence: float, max_risk_per_trade: float = 0.01) -> float:
        """
        Calculate optimal position size using Kelly criterion and risk management.
        
        Args:
            expected_return: Expected return of the trade
            volatility: Volatility of the asset
            confidence: Confidence in the trade (0-1)
            max_risk_per_trade: Maximum risk per trade as fraction of portfolio
            
        Returns:
            Optimal position size as fraction of portfolio
        """
        try:
            if volatility <= 0 or confidence <= 0:
                return 0.0
            
            # Kelly criterion
            kelly_fraction = expected_return / (volatility ** 2)
            
            # Adjust by confidence
            adjusted_kelly = kelly_fraction * confidence
            
            # Apply risk limits
            position_size = min(adjusted_kelly, max_risk_per_trade, self.max_position_size)
            position_size = max(position_size, 0.0)  # No negative positions
            
            return float(position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position sizing: {str(e)}")
            return 0.0
    
    def _empty_portfolio_risk(self) -> Dict[str, Any]:
        """
        Return empty portfolio risk metrics.
        
        Returns:
            Empty portfolio risk structure
        """
        return {
            'portfolio_return': {
                'daily_mean': 0.0,
                'daily_std': 0.0,
                'annualized_return': 0.0,
                'annualized_volatility': 0.0
            },
            'var_metrics': {f'var_{int(cl*100)}': 0.0 for cl in self.confidence_levels} | 
                          {f'cvar_{int(cl*100)}': 0.0 for cl in self.confidence_levels},
            'risk_ratios': {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0
            },
            'drawdown_metrics': {
                'max_drawdown': 0.0,
                'drawdown_duration': 0,
                'current_drawdown': 0.0
            },
            'correlation_metrics': {
                'average_correlation': 0.0,
                'maximum_correlation': 0.0,
                'correlation_matrix': []
            },
            'position_metrics': {}
        }
    
    def get_risk_alerts(self) -> List[Dict[str, Any]]:
        """
        Get current risk alerts.
        
        Returns:
            List of risk alerts
        """
        return self.risk_alerts.copy()
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get risk calculator summary and status.
        
        Returns:
            Risk summary
        """
        return {
            'active_alerts': len(self.risk_alerts),
            'alert_severities': {
                'high': len([a for a in self.risk_alerts if a.get('severity') == 'high']),
                'medium': len([a for a in self.risk_alerts if a.get('severity') == 'medium']),
                'low': len([a for a in self.risk_alerts if a.get('severity') == 'low'])
            },
            'risk_limits': {
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_position_size': self.max_position_size,
                'max_sector_exposure': self.max_sector_exposure,
                'max_correlation_exposure': self.max_correlation_exposure
            },
            'confidence_levels': self.confidence_levels,
            'lookback_periods': self.lookback_periods
        }

