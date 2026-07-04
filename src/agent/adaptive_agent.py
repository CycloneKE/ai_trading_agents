"""
Self-Adaptive Goal-Oriented Trading Agent

This module implements a trading agent that can:
1. Set and adapt its own goals based on market conditions
2. Self-modify strategies and parameters
3. Learn from performance feedback
4. Evolve its behavior over time
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import json

logger = logging.getLogger(__name__)

class GoalType(Enum):
    PROFIT_TARGET = "profit_target"
    RISK_REDUCTION = "risk_reduction"
    SHARPE_OPTIMIZATION = "sharpe_optimization"
    DRAWDOWN_CONTROL = "drawdown_control"
    MARKET_ADAPTATION = "market_adaptation"

class AdaptationTrigger(Enum):
    PERFORMANCE_DECLINE = "performance_decline"
    MARKET_REGIME_CHANGE = "market_regime_change"
    GOAL_ACHIEVEMENT = "goal_achievement"
    TIME_BASED = "time_based"
    RISK_THRESHOLD = "risk_threshold"

@dataclass
class Goal:
    """Represents a trading goal with success criteria"""
    goal_type: GoalType
    target_value: float
    current_value: float = 0.0
    priority: float = 1.0
    deadline: Optional[datetime] = None
    achieved: bool = False
    progress: float = 0.0
    
    def update_progress(self):
        """Update goal progress"""
        if self.target_value != 0:
            self.progress = min(self.current_value / self.target_value, 1.0)
            self.achieved = self.progress >= 1.0

@dataclass
class AdaptationAction:
    """Represents an adaptation action the agent can take"""
    action_type: str
    parameters: Dict[str, Any]
    expected_impact: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

class SelfAdaptiveAgent:
    """
    Self-adaptive, goal-oriented trading agent that can modify its behavior
    based on performance feedback and market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.goals: List[Goal] = []
        self.adaptation_history: List[AdaptationAction] = []
        self.performance_metrics: Dict[str, float] = {}
        self.market_regime: str = "normal"
        self.adaptation_threshold = config.get('adaptation_threshold', 0.1)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.last_adaptation = datetime.utcnow()
        self.adaptation_cooldown = timedelta(hours=config.get('adaptation_cooldown_hours', 24))
        
        # Initialize default goals
        self._initialize_default_goals()
        
    def _initialize_default_goals(self):
        """Initialize default trading goals"""
        self.goals = [
            Goal(GoalType.PROFIT_TARGET, target_value=0.15, priority=1.0),  # 15% annual return
            Goal(GoalType.SHARPE_OPTIMIZATION, target_value=1.5, priority=0.8),  # Sharpe > 1.5
            Goal(GoalType.DRAWDOWN_CONTROL, target_value=0.1, priority=0.9),  # Max 10% drawdown
            Goal(GoalType.RISK_REDUCTION, target_value=0.02, priority=0.7)  # 2% daily VaR
        ]
    
    def update_performance(self, metrics: Dict[str, float]):
        """Update performance metrics and goal progress"""
        self.performance_metrics.update(metrics)
        
        # Update goal progress
        for goal in self.goals:
            if goal.goal_type == GoalType.PROFIT_TARGET:
                goal.current_value = metrics.get('annual_return', 0.0)
            elif goal.goal_type == GoalType.SHARPE_OPTIMIZATION:
                goal.current_value = metrics.get('sharpe_ratio', 0.0)
            elif goal.goal_type == GoalType.DRAWDOWN_CONTROL:
                goal.current_value = 1.0 - metrics.get('max_drawdown', 0.0)  # Invert for progress
            elif goal.goal_type == GoalType.RISK_REDUCTION:
                goal.current_value = 1.0 - metrics.get('daily_var', 0.0)  # Invert for progress
            
            goal.update_progress()
        
        # Check if adaptation is needed
        if self._should_adapt():
            self._trigger_adaptation()
    
    def _should_adapt(self) -> bool:
        """Determine if agent should adapt based on current state"""
        # Cooldown check
        if datetime.utcnow() - self.last_adaptation < self.adaptation_cooldown:
            return False
        
        # Performance-based triggers
        underperforming_goals = [g for g in self.goals if g.progress < 0.5 and g.priority > 0.5]
        if len(underperforming_goals) >= 2:
            return True
        
        # Market regime change
        if self._detect_regime_change():
            return True
        
        # Risk threshold breach
        if self.performance_metrics.get('daily_var', 0) > 0.05:  # 5% VaR threshold
            return True
        
        return False
    
    def _detect_regime_change(self) -> bool:
        """Detect if market regime has changed"""
        volatility = self.performance_metrics.get('volatility', 0.0)
        correlation = self.performance_metrics.get('market_correlation', 0.0)
        
        # Simple regime detection based on volatility and correlation
        if volatility > 0.3 and self.market_regime != "high_volatility":
            self.market_regime = "high_volatility"
            return True
        elif volatility < 0.1 and correlation > 0.8 and self.market_regime != "low_volatility":
            self.market_regime = "low_volatility"
            return True
        elif 0.1 <= volatility <= 0.3 and self.market_regime != "normal":
            self.market_regime = "normal"
            return True
        
        return False
    
    def _trigger_adaptation(self):
        """Trigger adaptation process"""
        logger.info("Triggering agent adaptation")
        
        # Analyze current state
        adaptation_actions = self._generate_adaptation_actions()
        
        # Select best action
        if adaptation_actions:
            best_action = max(adaptation_actions, key=lambda x: x.expected_impact * x.confidence)
            self._execute_adaptation(best_action)
            self.adaptation_history.append(best_action)
            self.last_adaptation = datetime.utcnow()
    
    def _generate_adaptation_actions(self) -> List[AdaptationAction]:
        """Generate possible adaptation actions"""
        actions = []
        
        # Strategy parameter adjustments
        if self.performance_metrics.get('sharpe_ratio', 0) < 1.0:
            actions.append(AdaptationAction(
                action_type="adjust_risk_tolerance",
                parameters={"new_risk_tolerance": max(0.01, self.config.get('risk_tolerance', 0.02) * 0.8)},
                expected_impact=0.3,
                confidence=0.7
            ))
        
        # Position sizing adjustments
        if self.performance_metrics.get('max_drawdown', 0) > 0.15:
            actions.append(AdaptationAction(
                action_type="reduce_position_size",
                parameters={"size_multiplier": 0.7},
                expected_impact=0.4,
                confidence=0.8
            ))
        
        # Strategy switching based on market regime
        if self.market_regime == "high_volatility":
            actions.append(AdaptationAction(
                action_type="switch_strategy_weights",
                parameters={"momentum_weight": 0.3, "mean_reversion_weight": 0.7},
                expected_impact=0.5,
                confidence=0.6
            ))
        
        # Goal adjustment
        underperforming_goals = [g for g in self.goals if g.progress < 0.3]
        if underperforming_goals:
            actions.append(AdaptationAction(
                action_type="adjust_goals",
                parameters={"reduce_targets": True, "focus_on_achievable": True},
                expected_impact=0.2,
                confidence=0.9
            ))
        
        return actions
    
    def _execute_adaptation(self, action: AdaptationAction):
        """Execute an adaptation action"""
        logger.info(f"Executing adaptation: {action.action_type}")
        
        if action.action_type == "adjust_risk_tolerance":
            self.config['risk_tolerance'] = action.parameters['new_risk_tolerance']
            
        elif action.action_type == "reduce_position_size":
            current_size = self.config.get('position_size_multiplier', 1.0)
            self.config['position_size_multiplier'] = current_size * action.parameters['size_multiplier']
            
        elif action.action_type == "switch_strategy_weights":
            self.config['strategy_weights'] = {
                'momentum': action.parameters['momentum_weight'],
                'mean_reversion': action.parameters['mean_reversion_weight']
            }
            
        elif action.action_type == "adjust_goals":
            if action.parameters.get('reduce_targets'):
                for goal in self.goals:
                    if goal.progress < 0.3:
                        goal.target_value *= 0.8  # Reduce target by 20%
    
    def get_current_goals(self) -> List[Dict[str, Any]]:
        """Get current goals and their progress"""
        return [
            {
                'type': goal.goal_type.value,
                'target': goal.target_value,
                'current': goal.current_value,
                'progress': goal.progress,
                'achieved': goal.achieved,
                'priority': goal.priority
            }
            for goal in self.goals
        ]
    
    def add_goal(self, goal_type: GoalType, target_value: float, priority: float = 1.0, deadline: Optional[datetime] = None):
        """Add a new goal"""
        goal = Goal(goal_type, target_value, priority=priority, deadline=deadline)
        self.goals.append(goal)
        logger.info(f"Added new goal: {goal_type.value} with target {target_value}")
    
    def get_adaptation_recommendations(self) -> List[Dict[str, Any]]:
        """Get current adaptation recommendations"""
        if not self._should_adapt():
            return []
        
        actions = self._generate_adaptation_actions()
        return [
            {
                'action': action.action_type,
                'parameters': action.parameters,
                'expected_impact': action.expected_impact,
                'confidence': action.confidence
            }
            for action in actions
        ]
    
    def get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            'goals': self.get_current_goals(),
            'market_regime': self.market_regime,
            'performance_metrics': self.performance_metrics,
            'last_adaptation': self.last_adaptation.isoformat(),
            'adaptation_history_count': len(self.adaptation_history),
            'config': self.config
        }