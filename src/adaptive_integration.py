"""
Integration layer for Self-Adaptive Agent with existing trading system

Connects the adaptive agent with supervised learning strategies and provides
seamless integration with the existing trading infrastructure.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from adaptive_agent import SelfAdaptiveAgent, GoalType
from goal_manager import GoalManager

logger = logging.getLogger(__name__)

class AdaptiveStrategyIntegration:
    """
    Integrates self-adaptive capabilities with existing trading strategies.
    Modifies strategy parameters based on agent goals and performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adaptive_agent = SelfAdaptiveAgent(config)
        self.goal_manager = GoalManager(config)
        self.strategy_adaptations: Dict[str, Dict[str, Any]] = {}
        self.performance_buffer: List[Dict[str, Any]] = []
        
    def update_with_performance(self, performance_data: Dict[str, Any], market_data: Dict[str, Any]):
        """Update adaptive agent with performance and market data"""
        # Update performance buffer
        self.performance_buffer.append({
            'timestamp': datetime.utcnow(),
            'performance': performance_data,
            'market': market_data
        })
        
        # Keep only recent data
        if len(self.performance_buffer) > 100:
            self.performance_buffer = self.performance_buffer[-100:]
        
        # Update agent and goals
        self.adaptive_agent.update_performance(performance_data)
        self.goal_manager.update_goals(market_data, performance_data)
        
        # Generate strategy adaptations
        self._generate_strategy_adaptations()
    
    def _generate_strategy_adaptations(self):
        """Generate adaptations for trading strategies based on current goals"""
        priority_goals = self.goal_manager.get_priority_goals(3)
        
        for goal in priority_goals:
            if goal.progress < 0.5:  # Goal is struggling
                adaptation = self._create_adaptation_for_goal(goal)
                if adaptation:
                    self.strategy_adaptations[goal.goal_type.value] = adaptation
    
    def _create_adaptation_for_goal(self, goal) -> Optional[Dict[str, Any]]:
        """Create strategy adaptation for a specific goal"""
        if goal.goal_type == GoalType.PROFIT_TARGET:
            return {
                'increase_position_size': True,
                'position_multiplier': 1.1,
                'focus_momentum': True,
                'momentum_weight': 0.6
            }
        
        elif goal.goal_type == GoalType.SHARPE_OPTIMIZATION:
            return {
                'reduce_volatility_exposure': True,
                'volatility_threshold': 0.15,
                'increase_diversification': True,
                'correlation_limit': 0.7
            }
        
        elif goal.goal_type == GoalType.DRAWDOWN_CONTROL:
            return {
                'tighten_stop_losses': True,
                'stop_loss_multiplier': 0.8,
                'reduce_position_size': True,
                'position_multiplier': 0.9
            }
        
        elif goal.goal_type == GoalType.RISK_REDUCTION:
            return {
                'lower_var_target': True,
                'var_multiplier': 0.8,
                'increase_cash_allocation': True,
                'cash_buffer': 0.1
            }
        
        return None
    
    def get_strategy_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Get adapted parameters for a specific strategy"""
        base_params = self.config.get('strategies', {}).get(strategy_name, {})
        
        # Apply adaptations
        adapted_params = base_params.copy()
        
        for goal_type, adaptation in self.strategy_adaptations.items():
            adapted_params.update(self._apply_adaptation(adapted_params, adaptation))
        
        return adapted_params
    
    def _apply_adaptation(self, params: Dict[str, Any], adaptation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptation to strategy parameters"""
        updated_params = {}
        
        if adaptation.get('increase_position_size'):
            current_size = params.get('position_size', 0.1)
            multiplier = adaptation.get('position_multiplier', 1.1)
            updated_params['position_size'] = min(current_size * multiplier, 0.2)  # Cap at 20%
        
        if adaptation.get('reduce_position_size'):
            current_size = params.get('position_size', 0.1)
            multiplier = adaptation.get('position_multiplier', 0.9)
            updated_params['position_size'] = max(current_size * multiplier, 0.01)  # Floor at 1%
        
        if adaptation.get('tighten_stop_losses'):
            current_stop = params.get('stop_loss', 0.05)
            multiplier = adaptation.get('stop_loss_multiplier', 0.8)
            updated_params['stop_loss'] = current_stop * multiplier
        
        if adaptation.get('focus_momentum'):
            updated_params['momentum_weight'] = adaptation.get('momentum_weight', 0.6)
            updated_params['mean_reversion_weight'] = 1.0 - updated_params['momentum_weight']
        
        return updated_params
    
    def should_trade(self, symbol: str, signal_strength: float) -> bool:
        """Determine if trade should be executed based on adaptive criteria"""
        # Get current risk tolerance from adaptive agent
        risk_tolerance = self.adaptive_agent.config.get('risk_tolerance', 0.02)
        
        # Adjust signal threshold based on current goals
        priority_goals = self.goal_manager.get_priority_goals(1)
        
        if priority_goals:
            primary_goal = priority_goals[0]
            
            # If drawdown control is priority, be more conservative
            if primary_goal.goal_type == GoalType.DRAWDOWN_CONTROL:
                signal_threshold = 0.7  # Higher threshold
            # If profit target is priority, be more aggressive
            elif primary_goal.goal_type == GoalType.PROFIT_TARGET:
                signal_threshold = 0.3  # Lower threshold
            else:
                signal_threshold = 0.5  # Default threshold
        else:
            signal_threshold = 0.5
        
        return abs(signal_strength) > signal_threshold
    
    def get_position_size(self, symbol: str, signal_strength: float, account_value: float) -> float:
        """Calculate position size based on adaptive parameters"""
        base_size = self.adaptive_agent.config.get('position_size_multiplier', 1.0)
        risk_tolerance = self.adaptive_agent.config.get('risk_tolerance', 0.02)
        
        # Adjust based on signal strength
        size_multiplier = min(abs(signal_strength) * 2, 1.5)  # Cap at 1.5x
        
        # Adjust based on current market regime
        if self.adaptive_agent.market_regime == "high_volatility":
            size_multiplier *= 0.7  # Reduce size in volatile markets
        elif self.adaptive_agent.market_regime == "low_volatility":
            size_multiplier *= 1.2  # Increase size in stable markets
        
        # Calculate final position size
        position_value = account_value * risk_tolerance * base_size * size_multiplier
        
        return min(position_value, account_value * 0.1)  # Cap at 10% of account
    
    def get_adaptive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of adaptive system"""
        return {
            'agent_state': self.adaptive_agent.get_agent_state(),
            'goal_status': self.goal_manager.get_goal_status(),
            'active_adaptations': len(self.strategy_adaptations),
            'adaptation_details': self.strategy_adaptations,
            'recommendations': self.adaptive_agent.get_adaptation_recommendations(),
            'goal_recommendations': self.goal_manager.get_goal_recommendations(),
            'performance_buffer_size': len(self.performance_buffer)
        }
    
    def reset_adaptations(self):
        """Reset all adaptations to baseline"""
        self.strategy_adaptations.clear()
        self.adaptive_agent.goals = []
        self.adaptive_agent._initialize_default_goals()
        logger.info("Reset all adaptations to baseline")

class SupervisedLearningAdapter:
    """
    Adapter that modifies supervised learning strategy behavior based on
    adaptive agent recommendations.
    """
    
    def __init__(self, supervised_strategy, adaptive_integration: AdaptiveStrategyIntegration):
        self.supervised_strategy = supervised_strategy
        self.adaptive_integration = adaptive_integration
        self.original_threshold = supervised_strategy.threshold
        
    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal with adaptive modifications"""
        # Get base signal from supervised strategy
        base_signal = self.supervised_strategy.generate_signal(symbol, data)
        
        if not base_signal:
            return base_signal
        
        # Apply adaptive modifications
        adapted_params = self.adaptive_integration.get_strategy_parameters('supervised_learning')
        
        # Modify signal strength based on adaptive parameters
        signal_strength = base_signal.get('strength', 0.0)
        
        # Adjust threshold based on current goals
        if 'volatility_threshold' in adapted_params:
            volatility = data.get('volatility', 0.2)
            if volatility > adapted_params['volatility_threshold']:
                signal_strength *= 0.8  # Reduce signal in high volatility
        
        # Check if trade should be executed
        should_trade = self.adaptive_integration.should_trade(symbol, signal_strength)
        
        if not should_trade:
            return None
        
        # Modify position size
        account_value = data.get('account_value', 100000)
        position_size = self.adaptive_integration.get_position_size(symbol, signal_strength, account_value)
        
        # Return adapted signal
        adapted_signal = base_signal.copy()
        adapted_signal['strength'] = signal_strength
        adapted_signal['position_size'] = position_size
        adapted_signal['adaptive_modified'] = True
        
        return adapted_signal
    
    def update_model(self, symbol: str, X: List[List[float]], y: List[float], feedback: Dict[str, Any]) -> bool:
        """Update model with adaptive feedback integration"""
        # Add adaptive performance metrics to feedback
        adaptive_status = self.adaptive_integration.get_adaptive_status()
        
        enhanced_feedback = feedback.copy()
        enhanced_feedback.update({
            'goal_progress': adaptive_status['goal_status']['average_progress'],
            'adaptation_count': adaptive_status['active_adaptations'],
            'market_regime': self.adaptive_integration.adaptive_agent.market_regime
        })
        
        # Update the underlying supervised strategy
        return self.supervised_strategy.update_model(symbol, X, y, enhanced_feedback)