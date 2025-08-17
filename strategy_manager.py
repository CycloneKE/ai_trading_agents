"""
Strategy Manager for coordinating multiple trading strategies.
Handles strategy selection, ensemble methods, and performance optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

# Fix relative imports
try:
    from base_strategy import BaseStrategy
    from reinforcement_learning import DQNStrategy
    from supervised_learning import SupervisedLearningStrategy
except ImportError as e:
    logger.warning(f"Strategy import error: {e}")
    BaseStrategy = None
    DQNStrategy = None
    SupervisedLearningStrategy = None

class MockStrategy:
    """Simple mock strategy for testing."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'action': 'hold',
            'confidence': 0.5,
            'position_size': 0.0
        }
    
    def update_model(self, data: Dict[str, Any], feedback: Dict[str, Any] = None):
        pass
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        return {
            'sharpe_ratio': 1.0,
            'win_rate': 0.6,
            'max_drawdown': 0.05,
            'total_return': 0.1
        }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'enabled': self.enabled,
            'type': 'mock'
        }
    
    def save_model(self, filepath: str):
        pass
    
    def load_model(self, filepath: str):
        pass


class StrategyManager:
    """
    Manager for multiple trading strategies with ensemble capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategies = {}
        self.strategy_weights = {}
        self.ensemble_method = config.get('ensemble_method', 'weighted_average')
        self.performance_window = config.get('performance_window', 30)  # Days
        self.rebalance_frequency = config.get('rebalance_frequency', 7)  # Days
        
        # Performance tracking
        self.strategy_performance = {}
        self.ensemble_performance = {}
        self.last_rebalance = None
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        
        # Initialize strategies
        self._initialize_strategies()
        
        logger.info(f"Strategy manager initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self):
        """
        Initialize all configured strategies.
        """
        try:
            strategies_config = self.config.get('strategies', {})
            
            for strategy_name, strategy_config in strategies_config.items():
                strategy_type = strategy_config.get('type', 'supervised_learning')
                
                try:
                    if strategy_type == 'dqn' and DQNStrategy:
                        strategy = DQNStrategy(strategy_name, strategy_config)
                    elif strategy_type == 'supervised_learning' and SupervisedLearningStrategy:
                        strategy = SupervisedLearningStrategy(strategy_name, strategy_config)
                    else:
                        # Create a simple mock strategy for now
                        strategy = MockStrategy(strategy_name, strategy_config)
                        logger.info(f"Using mock strategy for {strategy_name} ({strategy_type})")
                    
                    self.strategies[strategy_name] = strategy
                    self.strategy_weights[strategy_name] = strategy_config.get('weight', 1.0)
                    self.strategy_performance[strategy_name] = {
                        'returns': [],
                        'sharpe_ratio': 0.0,
                        'win_rate': 0.0,
                        'max_drawdown': 0.0,
                        'last_update': None
                    }
                    
                    logger.info(f"Initialized strategy: {strategy_name} ({strategy_type})")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize strategy {strategy_name}: {str(e)}")
            
            # Normalize weights
            total_weight = sum(self.strategy_weights.values())
            if total_weight > 0:
                for name in self.strategy_weights:
                    self.strategy_weights[name] /= total_weight
            
        except Exception as e:
            logger.error(f"Error initializing strategies: {str(e)}")
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ensemble trading signals from all strategies.
        
        Args:
            data: Market data and features
            
        Returns:
            Dict containing ensemble signals
        """
        try:
            if not self.strategies:
                logger.warning("No strategies available")
                return {
                    'symbol': data.get('symbol', 'UNKNOWN'),
                    'action': 'hold',
                    'confidence': 0.0,
                    'position_size': 0.0,
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Get signals from all strategies in parallel
            strategy_signals = {}
            futures = {}
            
            for name, strategy in self.strategies.items():
                future = self.executor.submit(strategy.generate_signals, data)
                futures[name] = future
            
            # Collect results
            for name, future in futures.items():
                try:
                    signals = future.result(timeout=30)
                    strategy_signals[name] = signals
                except Exception as e:
                    logger.error(f"Error getting signals from {name}: {str(e)}")
                    strategy_signals[name] = {
                        'action': 'hold',
                        'confidence': 0.0,
                        'position_size': 0.0
                    }
            
            # Combine signals using ensemble method
            ensemble_signals = self._combine_signals(strategy_signals, data)
            
            return ensemble_signals
            
        except Exception as e:
            logger.error(f"Error generating ensemble signals: {str(e)}")
            return {
                'symbol': data.get('symbol', 'UNKNOWN'),
                'action': 'hold',
                'confidence': 0.0,
                'position_size': 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _combine_signals(self, strategy_signals: Dict[str, Dict[str, Any]], 
                        data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine signals from multiple strategies using ensemble method.
        
        Args:
            strategy_signals: Signals from individual strategies
            data: Original market data
            
        Returns:
            Combined ensemble signals
        """
        try:
            if not strategy_signals:
                return {
                    'symbol': data.get('symbol', 'UNKNOWN'),
                    'action': 'hold',
                    'confidence': 0.0,
                    'position_size': 0.0,
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            symbol = data.get('symbol', 'UNKNOWN')
            
            if self.ensemble_method == 'weighted_average':
                return self._weighted_average_ensemble(strategy_signals, symbol)
            elif self.ensemble_method == 'majority_vote':
                return self._majority_vote_ensemble(strategy_signals, symbol)
            elif self.ensemble_method == 'confidence_weighted':
                return self._confidence_weighted_ensemble(strategy_signals, symbol)
            elif self.ensemble_method == 'performance_weighted':
                return self._performance_weighted_ensemble(strategy_signals, symbol)
            else:
                logger.warning(f"Unknown ensemble method: {self.ensemble_method}")
                return self._weighted_average_ensemble(strategy_signals, symbol)
            
        except Exception as e:
            logger.error(f"Error combining signals: {str(e)}")
            return {
                'symbol': data.get('symbol', 'UNKNOWN'),
                'action': 'hold',
                'confidence': 0.0,
                'position_size': 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _weighted_average_ensemble(self, strategy_signals: Dict[str, Dict[str, Any]], 
                                 symbol: str) -> Dict[str, Any]:
        """
        Combine signals using weighted average of confidences.
        """
        action_scores = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        total_weight = 0.0
        total_confidence = 0.0
        total_position_size = 0.0
        
        for name, signals in strategy_signals.items():
            weight = self.strategy_weights.get(name, 1.0)
            action = signals.get('action', 'hold')
            confidence = signals.get('confidence', 0.0)
            position_size = signals.get('position_size', 0.0)
            
            action_scores[action] += weight * confidence
            total_weight += weight
            total_confidence += weight * confidence
            total_position_size += weight * position_size
        
        # Normalize
        if total_weight > 0:
            for action in action_scores:
                action_scores[action] /= total_weight
            total_confidence /= total_weight
            total_position_size /= total_weight
        
        # Select action with highest score
        final_action = max(action_scores, key=action_scores.get)
        final_confidence = action_scores[final_action]
        
        return {
            'symbol': symbol,
            'action': final_action,
            'confidence': final_confidence,
            'position_size': total_position_size,
            'action_scores': action_scores,
            'strategy_signals': strategy_signals,
            'ensemble_method': self.ensemble_method,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _majority_vote_ensemble(self, strategy_signals: Dict[str, Dict[str, Any]], 
                              symbol: str) -> Dict[str, Any]:
        """
        Combine signals using majority voting.
        """
        action_votes = {'buy': 0, 'sell': 0, 'hold': 0}
        confidences = []
        position_sizes = []
        
        for name, signals in strategy_signals.items():
            action = signals.get('action', 'hold')
            confidence = signals.get('confidence', 0.0)
            position_size = signals.get('position_size', 0.0)
            
            action_votes[action] += 1
            confidences.append(confidence)
            position_sizes.append(position_size)
        
        # Select action with most votes
        final_action = max(action_votes, key=action_votes.get)
        final_confidence = np.mean(confidences) if confidences else 0.0
        final_position_size = np.mean(position_sizes) if position_sizes else 0.0
        
        return {
            'symbol': symbol,
            'action': final_action,
            'confidence': final_confidence,
            'position_size': final_position_size,
            'action_votes': action_votes,
            'strategy_signals': strategy_signals,
            'ensemble_method': self.ensemble_method,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _confidence_weighted_ensemble(self, strategy_signals: Dict[str, Dict[str, Any]], 
                                    symbol: str) -> Dict[str, Any]:
        """
        Combine signals weighted by individual strategy confidence.
        """
        action_scores = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        total_confidence = 0.0
        total_position_size = 0.0
        
        for name, signals in strategy_signals.items():
            action = signals.get('action', 'hold')
            confidence = signals.get('confidence', 0.0)
            position_size = signals.get('position_size', 0.0)
            
            action_scores[action] += confidence
            total_confidence += confidence
            total_position_size += confidence * position_size
        
        # Normalize
        if total_confidence > 0:
            for action in action_scores:
                action_scores[action] /= total_confidence
            total_position_size /= total_confidence
        
        # Select action with highest score
        final_action = max(action_scores, key=action_scores.get)
        final_confidence = action_scores[final_action]
        
        return {
            'symbol': symbol,
            'action': final_action,
            'confidence': final_confidence,
            'position_size': total_position_size,
            'action_scores': action_scores,
            'strategy_signals': strategy_signals,
            'ensemble_method': self.ensemble_method,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _performance_weighted_ensemble(self, strategy_signals: Dict[str, Dict[str, Any]], 
                                     symbol: str) -> Dict[str, Any]:
        """
        Combine signals weighted by recent strategy performance.
        """
        action_scores = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        total_weight = 0.0
        total_confidence = 0.0
        total_position_size = 0.0
        
        for name, signals in strategy_signals.items():
            # Get performance-based weight
            performance = self.strategy_performance.get(name, {})
            sharpe_ratio = performance.get('sharpe_ratio', 0.0)
            win_rate = performance.get('win_rate', 0.5)
            
            # Calculate performance weight (higher is better)
            perf_weight = max(0.1, sharpe_ratio * win_rate)
            
            action = signals.get('action', 'hold')
            confidence = signals.get('confidence', 0.0)
            position_size = signals.get('position_size', 0.0)
            
            action_scores[action] += perf_weight * confidence
            total_weight += perf_weight
            total_confidence += perf_weight * confidence
            total_position_size += perf_weight * position_size
        
        # Normalize
        if total_weight > 0:
            for action in action_scores:
                action_scores[action] /= total_weight
            total_confidence /= total_weight
            total_position_size /= total_weight
        
        # Select action with highest score
        final_action = max(action_scores, key=action_scores.get)
        final_confidence = action_scores[final_action]
        
        return {
            'symbol': symbol,
            'action': final_action,
            'confidence': final_confidence,
            'position_size': total_position_size,
            'action_scores': action_scores,
            'strategy_signals': strategy_signals,
            'ensemble_method': self.ensemble_method,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def update_strategies(self, data: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None):
        """
        Update all strategies with new data and feedback.
        
        Args:
            data: New market data
            feedback: Performance feedback
        """
        try:
            # Update strategies in parallel
            futures = {}
            
            for name, strategy in self.strategies.items():
                future = self.executor.submit(strategy.update_model, data, feedback)
                futures[name] = future
            
            # Wait for completion
            for name, future in futures.items():
                try:
                    future.result(timeout=60)
                except Exception as e:
                    logger.error(f"Error updating strategy {name}: {str(e)}")
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Rebalance weights if needed
            if self._should_rebalance():
                self._rebalance_weights()
            
        except Exception as e:
            logger.error(f"Error updating strategies: {str(e)}")
    
    def _update_performance_metrics(self):
        """
        Update performance metrics for all strategies.
        """
        try:
            for name, strategy in self.strategies.items():
                metrics = strategy.calculate_performance_metrics()
                
                if metrics:
                    self.strategy_performance[name].update({
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                        'win_rate': metrics.get('win_rate', 0.0),
                        'max_drawdown': metrics.get('max_drawdown', 0.0),
                        'total_return': metrics.get('total_return', 0.0),
                        'last_update': datetime.utcnow().isoformat()
                    })
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _should_rebalance(self) -> bool:
        """
        Check if strategy weights should be rebalanced.
        
        Returns:
            bool: True if rebalancing is needed
        """
        if self.last_rebalance is None:
            return True
        
        days_since_rebalance = (datetime.now() - self.last_rebalance).days
        return days_since_rebalance >= self.rebalance_frequency
    
    def _rebalance_weights(self):
        """
        Rebalance strategy weights based on recent performance.
        """
        try:
            if self.ensemble_method != 'performance_weighted':
                return  # Only rebalance for performance-weighted ensemble
            
            # Calculate new weights based on performance
            new_weights = {}
            total_score = 0.0
            
            for name in self.strategies:
                performance = self.strategy_performance.get(name, {})
                sharpe_ratio = performance.get('sharpe_ratio', 0.0)
                win_rate = performance.get('win_rate', 0.5)
                
                # Calculate performance score
                score = max(0.1, sharpe_ratio * win_rate)
                new_weights[name] = score
                total_score += score
            
            # Normalize weights
            if total_score > 0:
                for name in new_weights:
                    new_weights[name] /= total_score
                
                # Update weights with smoothing
                alpha = 0.3  # Smoothing factor
                for name in self.strategy_weights:
                    old_weight = self.strategy_weights[name]
                    new_weight = new_weights.get(name, 0.1)
                    self.strategy_weights[name] = alpha * new_weight + (1 - alpha) * old_weight
                
                self.last_rebalance = datetime.now()
                logger.info(f"Rebalanced strategy weights: {self.strategy_weights}")
        
        except Exception as e:
            logger.error(f"Error rebalancing weights: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all strategies and ensemble performance.
        
        Returns:
            Dict containing strategy status
        """
        status = {
            'ensemble_method': self.ensemble_method,
            'strategy_count': len(self.strategies),
            'strategy_weights': self.strategy_weights,
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'strategies': {}
        }
        
        for name, strategy in self.strategies.items():
            strategy_status = strategy.get_status()
            strategy_status['performance'] = self.strategy_performance.get(name, {})
            status['strategies'][name] = strategy_status
        
        return status
    
    def start(self):
        """Start the strategy manager."""
        logger.info("Strategy manager started")
    
    def stop(self):
        """Stop the strategy manager."""
        logger.info("Strategy manager stopped")
    
    def save_strategies(self, directory: str) -> bool:
        """
        Save all strategies to files.
        
        Args:
            directory: Directory to save strategies
            
        Returns:
            bool: True if successful
        """
        try:
            import os
            os.makedirs(directory, exist_ok=True)
            
            # Save individual strategies
            for name, strategy in self.strategies.items():
                filepath = os.path.join(directory, f"{name}_strategy.pkl")
                strategy.save_model(filepath)
            
            # Save manager state
            manager_state = {
                'strategy_weights': self.strategy_weights,
                'strategy_performance': self.strategy_performance,
                'ensemble_method': self.ensemble_method,
                'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
                'config': self.config
            }
            
            manager_filepath = os.path.join(directory, "strategy_manager.json")
            with open(manager_filepath, 'w') as f:
                json.dump(manager_state, f, indent=2, default=str)
            
            logger.info(f"Strategies saved to {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save strategies: {str(e)}")
            return False
    
    def load_strategies(self, directory: str) -> bool:
        """
        Load strategies from files.
        
        Args:
            directory: Directory to load strategies from
            
        Returns:
            bool: True if successful
        """
        try:
            import os
            
            # Load manager state
            manager_filepath = os.path.join(directory, "strategy_manager.json")
            if os.path.exists(manager_filepath):
                with open(manager_filepath, 'r') as f:
                    manager_state = json.load(f)
                
                self.strategy_weights = manager_state.get('strategy_weights', {})
                self.strategy_performance = manager_state.get('strategy_performance', {})
                self.ensemble_method = manager_state.get('ensemble_method', self.ensemble_method)
                
                last_rebalance_str = manager_state.get('last_rebalance')
                if last_rebalance_str:
                    self.last_rebalance = datetime.fromisoformat(last_rebalance_str)
            
            # Load individual strategies
            for name, strategy in self.strategies.items():
                filepath = os.path.join(directory, f"{name}_strategy.pkl")
                if os.path.exists(filepath):
                    strategy.load_model(filepath)
            
            logger.info(f"Strategies loaded from {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load strategies: {str(e)}")
            return False