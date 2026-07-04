"""
Scenario Simulator for stress testing trading strategies.
Simulates various market conditions including historical crises and synthetic scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of market scenarios."""
    HISTORICAL_CRISIS = "historical_crisis"
    SYNTHETIC_STRESS = "synthetic_stress"
    MONTE_CARLO = "monte_carlo"
    REGIME_CHANGE = "regime_change"
    BLACK_SWAN = "black_swan"


@dataclass
class ScenarioParameters:
    """Parameters for scenario generation."""
    scenario_type: ScenarioType
    duration_days: int
    volatility_multiplier: float = 1.0
    correlation_shift: float = 0.0
    trend_direction: str = "neutral"  # "bull", "bear", "neutral"
    shock_magnitude: float = 0.0
    shock_probability: float = 0.0
    regime_persistence: float = 0.95


@dataclass
class ScenarioResult:
    """Result of scenario simulation."""
    scenario_name: str
    scenario_type: ScenarioType
    simulated_data: pd.DataFrame
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    drawdown_analysis: Dict[str, float]
    stress_test_results: Dict[str, Any]


class ScenarioSimulator:
    """
    Advanced scenario simulator for stress testing trading strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Simulation settings
        self.n_simulations = config.get('n_simulations', 1000)
        self.confidence_levels = config.get('confidence_levels', [0.95, 0.99])
        
        # Historical crisis periods
        self.crisis_periods = {
            'dot_com_crash': {
                'start': '2000-03-01',
                'end': '2002-10-01',
                'description': 'Dot-com bubble burst',
                'characteristics': {
                    'volatility_multiplier': 2.5,
                    'correlation_increase': 0.3,
                    'trend': 'bear'
                }
            },
            'financial_crisis_2008': {
                'start': '2007-10-01',
                'end': '2009-03-01',
                'description': 'Global Financial Crisis',
                'characteristics': {
                    'volatility_multiplier': 3.0,
                    'correlation_increase': 0.5,
                    'trend': 'bear'
                }
            },
            'covid_crash_2020': {
                'start': '2020-02-01',
                'end': '2020-04-01',
                'description': 'COVID-19 Market Crash',
                'characteristics': {
                    'volatility_multiplier': 4.0,
                    'correlation_increase': 0.7,
                    'trend': 'bear'
                }
            },
            'flash_crash_2010': {
                'start': '2010-05-06',
                'end': '2010-05-06',
                'description': 'Flash Crash',
                'characteristics': {
                    'volatility_multiplier': 10.0,
                    'correlation_increase': 0.9,
                    'trend': 'bear'
                }
            }
        }
        
        logger.info("Scenario simulator initialized")
    
    def generate_historical_crisis_scenario(self, base_data: pd.DataFrame,
                                          crisis_name: str) -> pd.DataFrame:
        """
        Generate scenario based on historical crisis characteristics.
        
        Args:
            base_data: Base market data
            crisis_name: Name of historical crisis
            
        Returns:
            Simulated crisis scenario data
        """
        try:
            if crisis_name not in self.crisis_periods:
                raise ValueError(f"Unknown crisis: {crisis_name}")
            
            crisis_info = self.crisis_periods[crisis_name]
            characteristics = crisis_info['characteristics']
            
            # Calculate base statistics
            returns = base_data.pct_change().dropna()
            base_volatility = returns.std()
            base_correlation = returns.corr()
            
            # Apply crisis characteristics
            crisis_volatility = base_volatility * characteristics['volatility_multiplier']
            
            # Modify correlation matrix
            correlation_increase = characteristics['correlation_increase']
            crisis_correlation = base_correlation.copy()
            
            # Increase correlations (flight to quality effect)
            for i in range(len(crisis_correlation)):
                for j in range(len(crisis_correlation)):
                    if i != j:
                        current_corr = crisis_correlation.iloc[i, j]
                        new_corr = current_corr + (1 - current_corr) * correlation_increase
                        crisis_correlation.iloc[i, j] = new_corr
                        crisis_correlation.iloc[j, i] = new_corr
            
            # Generate scenario data
            n_days = len(base_data)
            n_assets = len(base_data.columns)
            
            # Create covariance matrix
            volatility_matrix = np.outer(crisis_volatility, crisis_volatility)
            covariance_matrix = volatility_matrix * crisis_correlation.values
            
            # Generate correlated random returns
            random_returns = np.random.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=covariance_matrix,
                size=n_days
            )
            
            # Apply trend
            trend_factor = self._get_trend_factor(characteristics['trend'], n_days)
            random_returns += trend_factor.reshape(-1, 1)
            
            # Convert to price data
            scenario_data = base_data.copy()
            scenario_data.iloc[0] = base_data.iloc[0]  # Start with base prices
            
            for i in range(1, n_days):
                scenario_data.iloc[i] = scenario_data.iloc[i-1] * (1 + random_returns[i])
            
            return scenario_data
            
        except Exception as e:
            logger.error(f"Error generating historical crisis scenario: {str(e)}")
            return pd.DataFrame()
    
    def generate_synthetic_stress_scenario(self, base_data: pd.DataFrame,
                                         parameters: ScenarioParameters) -> pd.DataFrame:
        """
        Generate synthetic stress scenario.
        
        Args:
            base_data: Base market data
            parameters: Scenario parameters
            
        Returns:
            Simulated stress scenario data
        """
        try:
            returns = base_data.pct_change().dropna()
            base_volatility = returns.std()
            base_correlation = returns.corr()
            
            n_days = parameters.duration_days
            n_assets = len(base_data.columns)
            
            # Modify volatility
            stress_volatility = base_volatility * parameters.volatility_multiplier
            
            # Modify correlation matrix
            stress_correlation = base_correlation.copy()
            if parameters.correlation_shift != 0:
                # Shift all correlations
                for i in range(len(stress_correlation)):
                    for j in range(len(stress_correlation)):
                        if i != j:
                            current_corr = stress_correlation.iloc[i, j]
                            new_corr = np.clip(
                                current_corr + parameters.correlation_shift,
                                -0.99, 0.99
                            )
                            stress_correlation.iloc[i, j] = new_corr
                            stress_correlation.iloc[j, i] = new_corr
            
            # Create covariance matrix
            volatility_matrix = np.outer(stress_volatility, stress_volatility)
            covariance_matrix = volatility_matrix * stress_correlation.values
            
            # Generate returns with potential shocks
            random_returns = np.random.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=covariance_matrix,
                size=n_days
            )
            
            # Add shock events
            if parameters.shock_probability > 0:
                shock_days = np.random.binomial(1, parameters.shock_probability, n_days)
                shock_magnitude = parameters.shock_magnitude
                
                for day in range(n_days):
                    if shock_days[day]:
                        # Apply shock to all assets (systemic risk)
                        shock_direction = np.random.choice([-1, 1])
                        random_returns[day] += shock_direction * shock_magnitude
            
            # Apply trend
            trend_factor = self._get_trend_factor(parameters.trend_direction, n_days)
            random_returns += trend_factor.reshape(-1, 1)
            
            # Convert to price data
            scenario_data = pd.DataFrame(
                index=pd.date_range(start=base_data.index[-1], periods=n_days, freq='D'),
                columns=base_data.columns
            )
            
            # Start with last prices from base data
            scenario_data.iloc[0] = base_data.iloc[-1]
            
            for i in range(1, n_days):
                scenario_data.iloc[i] = scenario_data.iloc[i-1] * (1 + random_returns[i])
            
            return scenario_data
            
        except Exception as e:
            logger.error(f"Error generating synthetic stress scenario: {str(e)}")
            return pd.DataFrame()
    
    def generate_monte_carlo_scenarios(self, base_data: pd.DataFrame,
                                     n_scenarios: int = 1000,
                                     time_horizon: int = 252) -> List[pd.DataFrame]:
        """
        Generate multiple Monte Carlo scenarios.
        
        Args:
            base_data: Base market data
            n_scenarios: Number of scenarios to generate
            time_horizon: Time horizon in days
            
        Returns:
            List of scenario DataFrames
        """
        try:
            returns = base_data.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            scenarios = []
            
            for scenario_idx in range(n_scenarios):
                # Generate random returns
                random_returns = np.random.multivariate_normal(
                    mean=mean_returns.values,
                    cov=cov_matrix.values,
                    size=time_horizon
                )
                
                # Convert to price data
                scenario_data = pd.DataFrame(
                    index=pd.date_range(
                        start=base_data.index[-1], 
                        periods=time_horizon, 
                        freq='D'
                    ),
                    columns=base_data.columns
                )
                
                scenario_data.iloc[0] = base_data.iloc[-1]
                
                for i in range(1, time_horizon):
                    scenario_data.iloc[i] = scenario_data.iloc[i-1] * (1 + random_returns[i])
                
                scenarios.append(scenario_data)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating Monte Carlo scenarios: {str(e)}")
            return []
    
    def generate_regime_change_scenario(self, base_data: pd.DataFrame,
                                      parameters: ScenarioParameters) -> pd.DataFrame:
        """
        Generate scenario with regime changes.
        
        Args:
            base_data: Base market data
            parameters: Scenario parameters
            
        Returns:
            Simulated regime change scenario data
        """
        try:
            returns = base_data.pct_change().dropna()
            
            # Define two regimes: normal and stress
            normal_volatility = returns.std()
            stress_volatility = normal_volatility * parameters.volatility_multiplier
            
            normal_correlation = returns.corr()
            stress_correlation = normal_correlation.copy()
            
            # Increase correlations in stress regime
            for i in range(len(stress_correlation)):
                for j in range(len(stress_correlation)):
                    if i != j:
                        current_corr = stress_correlation.iloc[i, j]
                        new_corr = current_corr + (1 - current_corr) * 0.3
                        stress_correlation.iloc[i, j] = new_corr
                        stress_correlation.iloc[j, i] = new_corr
            
            n_days = parameters.duration_days
            n_assets = len(base_data.columns)
            
            # Generate regime states (0 = normal, 1 = stress)
            regime_states = self._generate_regime_states(n_days, parameters.regime_persistence)
            
            # Generate returns based on regime
            scenario_returns = np.zeros((n_days, n_assets))
            
            for day in range(n_days):
                if regime_states[day] == 0:  # Normal regime
                    volatility_matrix = np.outer(normal_volatility, normal_volatility)
                    covariance_matrix = volatility_matrix * normal_correlation.values
                else:  # Stress regime
                    volatility_matrix = np.outer(stress_volatility, stress_volatility)
                    covariance_matrix = volatility_matrix * stress_correlation.values
                
                scenario_returns[day] = np.random.multivariate_normal(
                    mean=np.zeros(n_assets),
                    cov=covariance_matrix
                )
            
            # Convert to price data
            scenario_data = pd.DataFrame(
                index=pd.date_range(start=base_data.index[-1], periods=n_days, freq='D'),
                columns=base_data.columns
            )
            
            scenario_data.iloc[0] = base_data.iloc[-1]
            
            for i in range(1, n_days):
                scenario_data.iloc[i] = scenario_data.iloc[i-1] * (1 + scenario_returns[i])
            
            return scenario_data
            
        except Exception as e:
            logger.error(f"Error generating regime change scenario: {str(e)}")
            return pd.DataFrame()
    
    def _get_trend_factor(self, trend_direction: str, n_days: int) -> np.ndarray:
        """
        Generate trend factor for scenario.
        
        Args:
            trend_direction: Direction of trend
            n_days: Number of days
            
        Returns:
            Trend factor array
        """
        if trend_direction == "bull":
            # Positive trend with some noise
            base_trend = np.linspace(0, 0.001, n_days)  # 0.1% daily positive trend
            noise = np.random.normal(0, 0.0005, n_days)
            return base_trend + noise
            
        elif trend_direction == "bear":
            # Negative trend with some noise
            base_trend = np.linspace(0, -0.002, n_days)  # 0.2% daily negative trend
            noise = np.random.normal(0, 0.0005, n_days)
            return base_trend + noise
            
        else:  # neutral
            # No trend, just noise
            return np.random.normal(0, 0.0001, n_days)
    
    def _generate_regime_states(self, n_days: int, persistence: float) -> np.ndarray:
        """
        Generate regime states using Markov chain.
        
        Args:
            n_days: Number of days
            persistence: Regime persistence probability
            
        Returns:
            Array of regime states
        """
        # Transition matrix
        # P(stay in same regime) = persistence
        # P(switch regime) = 1 - persistence
        transition_matrix = np.array([
            [persistence, 1 - persistence],      # From normal to normal/stress
            [1 - persistence, persistence]       # From stress to normal/stress
        ])
        
        states = np.zeros(n_days, dtype=int)
        states[0] = 0  # Start in normal regime
        
        for day in range(1, n_days):
            current_state = states[day - 1]
            # Sample next state based on transition probabilities
            states[day] = np.random.choice(2, p=transition_matrix[current_state])
        
        return states
    
    def run_stress_test(self, strategy_func, base_data: pd.DataFrame,
                       scenario_configs: List[Dict[str, Any]]) -> Dict[str, ScenarioResult]:
        """
        Run comprehensive stress test across multiple scenarios.
        
        Args:
            strategy_func: Trading strategy function to test
            base_data: Base market data
            scenario_configs: List of scenario configurations
            
        Returns:
            Dictionary of scenario results
        """
        try:
            stress_test_results = {}
            
            for config in scenario_configs:
                scenario_name = config['name']
                scenario_type = ScenarioType(config['type'])
                
                logger.info(f"Running stress test for scenario: {scenario_name}")
                
                # Generate scenario data
                if scenario_type == ScenarioType.HISTORICAL_CRISIS:
                    scenario_data = self.generate_historical_crisis_scenario(
                        base_data, config['crisis_name']
                    )
                    
                elif scenario_type == ScenarioType.SYNTHETIC_STRESS:
                    parameters = ScenarioParameters(
                        scenario_type=scenario_type,
                        duration_days=config.get('duration_days', 252),
                        volatility_multiplier=config.get('volatility_multiplier', 2.0),
                        correlation_shift=config.get('correlation_shift', 0.3),
                        trend_direction=config.get('trend_direction', 'bear'),
                        shock_magnitude=config.get('shock_magnitude', 0.05),
                        shock_probability=config.get('shock_probability', 0.1)
                    )
                    scenario_data = self.generate_synthetic_stress_scenario(base_data, parameters)
                    
                elif scenario_type == ScenarioType.REGIME_CHANGE:
                    parameters = ScenarioParameters(
                        scenario_type=scenario_type,
                        duration_days=config.get('duration_days', 252),
                        volatility_multiplier=config.get('volatility_multiplier', 2.0),
                        regime_persistence=config.get('regime_persistence', 0.95)
                    )
                    scenario_data = self.generate_regime_change_scenario(base_data, parameters)
                    
                else:
                    logger.warning(f"Unsupported scenario type: {scenario_type}")
                    continue
                
                if scenario_data.empty:
                    logger.error(f"Failed to generate scenario data for {scenario_name}")
                    continue
                
                # Run strategy on scenario data
                strategy_performance = strategy_func(scenario_data)
                
                # Calculate performance metrics
                performance_metrics = self._calculate_performance_metrics(strategy_performance)
                risk_metrics = self._calculate_risk_metrics(strategy_performance)
                drawdown_analysis = self._calculate_drawdown_analysis(strategy_performance)
                
                # Create scenario result
                scenario_result = ScenarioResult(
                    scenario_name=scenario_name,
                    scenario_type=scenario_type,
                    simulated_data=scenario_data,
                    performance_metrics=performance_metrics,
                    risk_metrics=risk_metrics,
                    drawdown_analysis=drawdown_analysis,
                    stress_test_results={
                        'scenario_config': config,
                        'data_points': len(scenario_data),
                        'assets': list(scenario_data.columns)
                    }
                )
                
                stress_test_results[scenario_name] = scenario_result
                
                logger.info(f"Completed stress test for {scenario_name}")
            
            return stress_test_results
            
        except Exception as e:
            logger.error(f"Error running stress test: {str(e)}")
            return {}
    
    def _calculate_performance_metrics(self, performance_data: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics from strategy returns."""
        try:
            returns = performance_data.pct_change().dropna()
            
            metrics = {
                'total_return': float((performance_data.iloc[-1] / performance_data.iloc[0]) - 1),
                'annualized_return': float(returns.mean() * 252),
                'volatility': float(returns.std() * np.sqrt(252)),
                'sharpe_ratio': float((returns.mean() * 252) / (returns.std() * np.sqrt(252))),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'best_day': float(returns.max()),
                'worst_day': float(returns.min())
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def _calculate_risk_metrics(self, performance_data: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics from strategy returns."""
        try:
            returns = performance_data.pct_change().dropna()
            
            # Value at Risk
            var_95 = float(np.percentile(returns, 5))
            var_99 = float(np.percentile(returns, 1))
            
            # Conditional Value at Risk (Expected Shortfall)
            cvar_95 = float(returns[returns <= var_95].mean())
            cvar_99 = float(returns[returns <= var_99].mean())
            
            metrics = {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'downside_deviation': float(returns[returns < 0].std() * np.sqrt(252)),
                'sortino_ratio': float((returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252)))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def _calculate_drawdown_analysis(self, performance_data: pd.Series) -> Dict[str, float]:
        """Calculate drawdown analysis from strategy performance."""
        try:
            cumulative_returns = performance_data / performance_data.iloc[0]
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            
            metrics = {
                'max_drawdown': float(drawdowns.min()),
                'avg_drawdown': float(drawdowns[drawdowns < 0].mean()),
                'drawdown_duration': int((drawdowns < -0.05).sum()),  # Days with >5% drawdown
                'recovery_time': int(len(drawdowns) - drawdowns.idxmin())  # Days to recover from max DD
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating drawdown analysis: {str(e)}")
            return {}
    
    def generate_stress_test_report(self, stress_test_results: Dict[str, ScenarioResult]) -> Dict[str, Any]:
        """
        Generate comprehensive stress test report.
        
        Args:
            stress_test_results: Results from stress testing
            
        Returns:
            Comprehensive stress test report
        """
        try:
            report = {
                'summary': {
                    'total_scenarios': len(stress_test_results),
                    'test_date': datetime.now().isoformat(),
                    'scenarios_tested': list(stress_test_results.keys())
                },
                'performance_summary': {},
                'risk_summary': {},
                'worst_case_scenarios': {},
                'recommendations': []
            }
            
            # Aggregate performance metrics
            all_returns = []
            all_sharpe_ratios = []
            all_max_drawdowns = []
            
            for scenario_name, result in stress_test_results.items():
                perf_metrics = result.performance_metrics
                risk_metrics = result.risk_metrics
                
                all_returns.append(perf_metrics.get('total_return', 0))
                all_sharpe_ratios.append(perf_metrics.get('sharpe_ratio', 0))
                all_max_drawdowns.append(result.drawdown_analysis.get('max_drawdown', 0))
            
            # Performance summary
            report['performance_summary'] = {
                'avg_return': float(np.mean(all_returns)),
                'worst_return': float(np.min(all_returns)),
                'best_return': float(np.max(all_returns)),
                'avg_sharpe_ratio': float(np.mean(all_sharpe_ratios)),
                'worst_sharpe_ratio': float(np.min(all_sharpe_ratios))
            }
            
            # Risk summary
            report['risk_summary'] = {
                'avg_max_drawdown': float(np.mean(all_max_drawdowns)),
                'worst_max_drawdown': float(np.min(all_max_drawdowns)),
                'scenarios_with_large_losses': sum(1 for r in all_returns if r < -0.2),
                'stress_test_pass_rate': sum(1 for r in all_returns if r > -0.1) / len(all_returns)
            }
            
            # Identify worst-case scenarios
            worst_scenarios = sorted(
                stress_test_results.items(),
                key=lambda x: x[1].performance_metrics.get('total_return', 0)
            )[:3]
            
            report['worst_case_scenarios'] = {
                name: {
                    'total_return': result.performance_metrics.get('total_return', 0),
                    'max_drawdown': result.drawdown_analysis.get('max_drawdown', 0),
                    'scenario_type': result.scenario_type.value
                }
                for name, result in worst_scenarios
            }
            
            # Generate recommendations
            recommendations = []
            
            if report['risk_summary']['worst_max_drawdown'] < -0.3:
                recommendations.append("Consider implementing stronger risk controls to limit maximum drawdown")
            
            if report['performance_summary']['worst_return'] < -0.5:
                recommendations.append("Strategy shows vulnerability to extreme market conditions")
            
            if report['risk_summary']['stress_test_pass_rate'] < 0.7:
                recommendations.append("Strategy fails stress tests in majority of scenarios - review risk management")
            
            report['recommendations'] = recommendations
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating stress test report: {str(e)}")
            return {}

