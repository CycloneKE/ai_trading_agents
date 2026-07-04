#!/usr/bin/env python3
"""
Ensemble Strategy Manager

Advanced ensemble learning system that combines multiple trading strategies
with dynamic weighting and meta-learning capabilities.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque

try:
    from sklearn.ensemble import VotingClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class StrategyPerformanceTracker:
    """Track individual strategy performance for ensemble weighting"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history = defaultdict(lambda: deque(maxlen=window_size))
        self.prediction_history = defaultdict(lambda: deque(maxlen=window_size))
        self.actual_returns = deque(maxlen=window_size)
        
    def update_performance(self, strategy_name: str, prediction: float, actual_return: float):
        """Update strategy performance metrics"""
        self.performance_history[strategy_name].append(actual_return)
        self.prediction_history[strategy_name].append(prediction)
        self.actual_returns.append(actual_return)
    
    def get_strategy_metrics(self, strategy_name: str) -> Dict[str, float]:
        """Get comprehensive metrics for a strategy"""
        if strategy_name not in self.performance_history:
            return {'accuracy': 0.0, 'sharpe': 0.0, 'correlation': 0.0, 'weight': 0.0}
        
        predictions = list(self.prediction_history[strategy_name])
        actuals = list(self.actual_returns)[-len(predictions):]
        
        if len(predictions) < 10:
            return {'accuracy': 0.0, 'sharpe': 0.0, 'correlation': 0.0, 'weight': 0.0}
        
        # Calculate accuracy (directional)
        pred_directions = [1 if p > 0 else 0 for p in predictions]
        actual_directions = [1 if a > 0 else 0 for a in actuals]
        accuracy = accuracy_score(actual_directions, pred_directions) if len(pred_directions) > 0 else 0.0
        
        # Calculate Sharpe ratio
        returns = list(self.performance_history[strategy_name])
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Calculate correlation
        correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0.0
        correlation = 0.0 if np.isnan(correlation) else correlation
        
        # Calculate dynamic weight
        weight = self._calculate_dynamic_weight(accuracy, sharpe, correlation)
        
        return {
            'accuracy': float(accuracy),
            'sharpe': float(sharpe),
            'correlation': float(correlation),
            'weight': float(weight)
        }
    
    def _calculate_dynamic_weight(self, accuracy: float, sharpe: float, correlation: float) -> float:
        """Calculate dynamic weight based on multiple performance metrics"""
        # Normalize metrics
        accuracy_score = max(0, (accuracy - 0.5) * 2)  # 0.5 = random, scale to 0-1
        sharpe_score = max(0, min(1, sharpe / 2))  # Cap at 2.0 Sharpe
        correlation_score = max(0, correlation)  # Only positive correlation
        
        # Weighted combination
        weight = (0.4 * accuracy_score + 0.4 * sharpe_score + 0.2 * correlation_score)
        return max(0.01, min(1.0, weight))  # Ensure minimum weight


class MetaLearner:
    """Meta-learning system to optimize ensemble combinations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.meta_model = None
        self.feature_history = deque(maxlen=1000)
        self.target_history = deque(maxlen=1000)
        
        if SKLEARN_AVAILABLE:
            self.meta_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
    
    def prepare_meta_features(self, strategy_signals: Dict[str, Dict[str, Any]], 
                            market_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for meta-learning"""
        features = []
        
        # Strategy predictions and confidences
        for strategy_name, signal in strategy_signals.items():
            features.extend([
                signal.get('confidence', 0),
                1 if signal.get('action') == 'buy' else (-1 if signal.get('action') == 'sell' else 0),
                signal.get('position_size', 0)
            ])
        
        # Market regime features
        features.extend([
            market_data.get('volatility', 0),
            market_data.get('volume_ratio', 1),
            market_data.get('price_momentum', 0),
            market_data.get('market_sentiment', 0)
        ])
        
        # Pad or truncate to fixed size
        target_size = 50
        if len(features) > target_size:
            features = features[:target_size]
        else:
            features.extend([0] * (target_size - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def update_meta_model(self, features: np.ndarray, target: float):
        """Update meta-learning model"""
        self.feature_history.append(features)
        self.target_history.append(target)
        
        if len(self.feature_history) >= 50 and SKLEARN_AVAILABLE:
            try:
                X = np.array(list(self.feature_history))
                y = np.array(list(self.target_history))
                self.meta_model.fit(X, y)
            except Exception as e:
                logger.warning(f"Meta-model update failed: {e}")
    
    def predict_ensemble_weight(self, features: np.ndarray) -> float:
        """Predict optimal ensemble weight using meta-model"""
        if self.meta_model is None or len(self.feature_history) < 50:
            return 1.0
        
        try:
            prediction = self.meta_model.predict([features])[0]
            return max(0.1, min(2.0, prediction))  # Reasonable bounds
        except Exception as e:
            logger.warning(f"Meta-prediction failed: {e}")
            return 1.0


class EnsembleStrategyManager:
    """Advanced ensemble strategy manager with meta-learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategies = {}
        self.performance_tracker = StrategyPerformanceTracker(
            window_size=config.get('performance_window', 100)
        )
        self.meta_learner = MetaLearner(config)
        
        # Ensemble parameters
        self.min_strategies = config.get('min_strategies', 2)
        self.max_strategies = config.get('max_strategies', 10)
        self.rebalance_frequency = config.get('rebalance_frequency', 24)  # hours
        self.last_rebalance = datetime.now()
        
        # Strategy weights
        self.strategy_weights = {}
        self.base_weights = {}
        
        # Performance tracking
        self.ensemble_history = deque(maxlen=1000)
        self.signal_history = deque(maxlen=100)
        
        logger.info("Ensemble Strategy Manager initialized")
    
    def register_strategy(self, name: str, strategy: Any, base_weight: float = 1.0):
        """Register a strategy with the ensemble"""
        self.strategies[name] = strategy
        self.base_weights[name] = base_weight
        self.strategy_weights[name] = base_weight
        logger.info(f"Strategy '{name}' registered with base weight {base_weight}")
    
    def generate_ensemble_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ensemble trading signals"""
        try:
            if len(self.strategies) < self.min_strategies:
                logger.warning(f"Not enough strategies ({len(self.strategies)} < {self.min_strategies})")
                return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0}
            
            # Get signals from all strategies
            strategy_signals = {}
            for name, strategy in self.strategies.items():
                try:
                    if hasattr(strategy, 'generate_signals'):
                        signal = strategy.generate_signals(market_data)
                        strategy_signals[name] = signal
                except Exception as e:
                    logger.warning(f"Strategy {name} failed: {e}")
                    continue
            
            if not strategy_signals:
                return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0}
            
            # Update strategy weights if needed
            if self._should_rebalance():
                self._rebalance_weights(strategy_signals, market_data)
            
            # Generate ensemble signal
            ensemble_signal = self._combine_signals(strategy_signals, market_data)
            
            # Store for performance tracking
            self.signal_history.append({
                'timestamp': datetime.now(),
                'individual_signals': strategy_signals,
                'ensemble_signal': ensemble_signal,
                'weights': self.strategy_weights.copy()
            })
            
            return ensemble_signal
            
        except Exception as e:
            logger.error(f"Ensemble signal generation failed: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0, 'error': str(e)}
    
    def _should_rebalance(self) -> bool:
        """Check if weights should be rebalanced"""
        time_since_rebalance = datetime.now() - self.last_rebalance
        return time_since_rebalance.total_seconds() > (self.rebalance_frequency * 3600)
    
    def _rebalance_weights(self, strategy_signals: Dict[str, Dict[str, Any]], 
                          market_data: Dict[str, Any]):
        """Rebalance strategy weights based on performance"""
        try:
            new_weights = {}
            total_weight = 0
            
            for strategy_name in strategy_signals.keys():
                metrics = self.performance_tracker.get_strategy_metrics(strategy_name)
                performance_weight = metrics['weight']
                
                # Combine with base weight
                base_weight = self.base_weights.get(strategy_name, 1.0)
                combined_weight = 0.7 * performance_weight + 0.3 * base_weight
                
                new_weights[strategy_name] = combined_weight
                total_weight += combined_weight
            
            # Normalize weights
            if total_weight > 0:
                for name in new_weights:
                    new_weights[name] /= total_weight
                
                self.strategy_weights = new_weights
                self.last_rebalance = datetime.now()
                
                logger.info(f"Weights rebalanced: {self.strategy_weights}")
            
        except Exception as e:
            logger.error(f"Weight rebalancing failed: {e}")
    
    def _combine_signals(self, strategy_signals: Dict[str, Dict[str, Any]], 
                        market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine individual strategy signals into ensemble signal"""
        try:
            # Weighted voting for actions
            action_votes = {'buy': 0, 'sell': 0, 'hold': 0}
            confidence_sum = 0
            position_size_sum = 0
            total_weight = 0
            
            for strategy_name, signal in strategy_signals.items():
                weight = self.strategy_weights.get(strategy_name, 0)
                if weight <= 0:
                    continue
                
                action = signal.get('action', 'hold')
                confidence = signal.get('confidence', 0)
                position_size = signal.get('position_size', 0)
                
                # Weight the votes
                action_votes[action] += weight * confidence
                confidence_sum += weight * confidence
                position_size_sum += weight * position_size
                total_weight += weight
            
            if total_weight == 0:
                return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0}
            
            # Determine ensemble action
            ensemble_action = max(action_votes, key=action_votes.get)
            ensemble_confidence = confidence_sum / total_weight
            ensemble_position_size = position_size_sum / total_weight
            
            # Apply meta-learning adjustment
            meta_features = self.meta_learner.prepare_meta_features(strategy_signals, market_data)
            meta_weight = self.meta_learner.predict_ensemble_weight(meta_features)
            
            # Adjust confidence and position size
            ensemble_confidence *= meta_weight
            ensemble_position_size *= meta_weight
            
            # Apply ensemble-specific filters
            ensemble_signal = self._apply_ensemble_filters({
                'action': ensemble_action,
                'confidence': float(np.clip(ensemble_confidence, 0, 1)),
                'position_size': float(np.clip(ensemble_position_size, 0, 0.2)),
                'strategy_votes': action_votes,
                'meta_weight': float(meta_weight),
                'contributing_strategies': len(strategy_signals),
                'strategy': 'ensemble'
            }, market_data)
            
            return ensemble_signal
            
        except Exception as e:
            logger.error(f"Signal combination failed: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0}
    
    def _apply_ensemble_filters(self, signal: Dict[str, Any], 
                               market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ensemble-specific filters and risk controls"""
        try:
            # Minimum confidence threshold
            min_confidence = self.config.get('min_ensemble_confidence', 0.3)
            if signal['confidence'] < min_confidence:
                signal['action'] = 'hold'
                signal['position_size'] = 0.0
            
            # Market volatility filter
            volatility = market_data.get('volatility', 0)
            if volatility > self.config.get('max_volatility_threshold', 0.05):
                signal['position_size'] *= 0.5  # Reduce position in high volatility
            
            # Diversification requirement
            if signal['contributing_strategies'] < self.min_strategies:
                signal['confidence'] *= 0.5
                signal['position_size'] *= 0.5
            
            return signal
            
        except Exception as e:
            logger.error(f"Ensemble filter application failed: {e}")
            return signal
    
    def update_performance(self, market_data: Dict[str, Any], actual_return: float):
        """Update performance tracking for all strategies"""
        try:
            if not self.signal_history:
                return
            
            # Get the most recent signals
            recent_signal = self.signal_history[-1]
            individual_signals = recent_signal.get('individual_signals', {})
            
            # Update individual strategy performance
            for strategy_name, signal in individual_signals.items():
                prediction = signal.get('confidence', 0)
                if signal.get('action') == 'sell':
                    prediction *= -1
                elif signal.get('action') == 'hold':
                    prediction = 0
                
                self.performance_tracker.update_performance(
                    strategy_name, prediction, actual_return
                )
            
            # Update meta-learner
            if len(individual_signals) > 0:
                meta_features = self.meta_learner.prepare_meta_features(
                    individual_signals, market_data
                )
                self.meta_learner.update_meta_model(meta_features, actual_return)
            
            # Store ensemble performance
            self.ensemble_history.append({
                'timestamp': datetime.now(),
                'return': actual_return,
                'signal': recent_signal.get('ensemble_signal', {}),
                'weights': recent_signal.get('weights', {})
            })
            
        except Exception as e:
            logger.error(f"Performance update failed: {e}")
    
    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble performance metrics"""
        try:
            metrics = {
                'total_strategies': len(self.strategies),
                'active_strategies': sum(1 for w in self.strategy_weights.values() if w > 0.01),
                'current_weights': self.strategy_weights.copy(),
                'last_rebalance': self.last_rebalance.isoformat(),
                'ensemble_signals_generated': len(self.signal_history)
            }
            
            # Individual strategy metrics
            strategy_metrics = {}
            for name in self.strategies.keys():
                strategy_metrics[name] = self.performance_tracker.get_strategy_metrics(name)
            metrics['strategy_performance'] = strategy_metrics
            
            # Ensemble performance
            if self.ensemble_history:
                returns = [h['return'] for h in self.ensemble_history]
                metrics['ensemble_performance'] = {
                    'total_signals': len(returns),
                    'avg_return': float(np.mean(returns)),
                    'volatility': float(np.std(returns)),
                    'sharpe_ratio': float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)),
                    'win_rate': float(sum(1 for r in returns if r > 0) / len(returns))
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting ensemble metrics: {e}")
            return {'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get ensemble manager status"""
        return {
            'status': 'active',
            'strategies_registered': len(self.strategies),
            'sklearn_available': SKLEARN_AVAILABLE,
            'last_rebalance': self.last_rebalance.isoformat(),
            'meta_learning_samples': len(self.meta_learner.feature_history)
        }