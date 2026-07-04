"""
Goal Management System for Self-Adaptive Trading Agent

Manages dynamic goal setting, tracking, and adaptation based on market conditions
and performance feedback.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from .adaptive_agent import Goal, GoalType, SelfAdaptiveAgent

logger = logging.getLogger(__name__)

@dataclass
class GoalTemplate:
    """Template for creating goals based on market conditions"""
    goal_type: GoalType
    base_target: float
    volatility_adjustment: float = 0.0
    regime_multipliers: Dict[str, float] = None
    
    def create_goal(self, market_regime: str, volatility: float) -> Goal:
        """Create a goal instance based on current market conditions"""
        target = self.base_target
        
        # Adjust for volatility
        target += self.volatility_adjustment * volatility
        
        # Adjust for market regime
        if self.regime_multipliers and market_regime in self.regime_multipliers:
            target *= self.regime_multipliers[market_regime]
        
        return Goal(self.goal_type, target)

class GoalManager:
    """
    Manages goals for the self-adaptive trading agent.
    Dynamically creates, adjusts, and removes goals based on market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.goal_templates = self._create_goal_templates()
        self.active_goals: List[Goal] = []
        self.completed_goals: List[Goal] = []
        self.goal_history: List[Dict[str, Any]] = []
        
    def _create_goal_templates(self) -> List[GoalTemplate]:
        """Create goal templates for different market conditions"""
        return [
            GoalTemplate(
                goal_type=GoalType.PROFIT_TARGET,
                base_target=0.12,  # 12% base return
                volatility_adjustment=-0.5,  # Reduce target in high volatility
                regime_multipliers={
                    "bull_market": 1.3,
                    "bear_market": 0.7,
                    "sideways": 0.9,
                    "high_volatility": 0.8
                }
            ),
            GoalTemplate(
                goal_type=GoalType.SHARPE_OPTIMIZATION,
                base_target=1.2,
                volatility_adjustment=-1.0,
                regime_multipliers={
                    "bull_market": 1.1,
                    "bear_market": 1.4,  # Focus more on risk-adjusted returns in bear markets
                    "high_volatility": 1.5
                }
            ),
            GoalTemplate(
                goal_type=GoalType.DRAWDOWN_CONTROL,
                base_target=0.08,  # 8% max drawdown
                volatility_adjustment=0.3,  # Allow higher drawdown in volatile markets
                regime_multipliers={
                    "bear_market": 0.6,  # Stricter drawdown control in bear markets
                    "high_volatility": 1.2
                }
            )
        ]
    
    def update_goals(self, market_data: Dict[str, Any], performance_metrics: Dict[str, Any]):
        """Update goals based on current market conditions and performance"""
        market_regime = self._detect_market_regime(market_data)
        volatility = market_data.get('volatility', 0.2)
        
        # Check if goals need updating
        if self._should_update_goals(market_regime, volatility, performance_metrics):
            self._refresh_goals(market_regime, volatility)
        
        # Update progress on active goals
        self._update_goal_progress(performance_metrics)
        
        # Handle completed goals
        self._process_completed_goals()
    
    def _detect_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Detect current market regime"""
        returns = market_data.get('returns', [])
        volatility = market_data.get('volatility', 0.2)
        trend = market_data.get('trend', 0.0)
        
        if not returns:
            return "normal"
        
        # Simple regime detection
        if volatility > 0.3:
            return "high_volatility"
        elif trend > 0.1:
            return "bull_market"
        elif trend < -0.1:
            return "bear_market"
        else:
            return "sideways"
    
    def _should_update_goals(self, market_regime: str, volatility: float, performance: Dict[str, Any]) -> bool:
        """Determine if goals should be updated"""
        # Update if no active goals
        if not self.active_goals:
            return True
        
        # Update if significant market regime change
        last_regime = getattr(self, '_last_regime', None)
        if last_regime and last_regime != market_regime:
            return True
        
        # Update if performance significantly deviates from goals
        underperforming_count = sum(1 for goal in self.active_goals if goal.progress < 0.3)
        if underperforming_count > len(self.active_goals) / 2:
            return True
        
        return False
    
    def _refresh_goals(self, market_regime: str, volatility: float):
        """Refresh goals based on current market conditions"""
        # Archive current goals
        for goal in self.active_goals:
            self.goal_history.append({
                'goal': goal,
                'archived_at': datetime.utcnow(),
                'reason': f'Market regime change to {market_regime}'
            })
        
        # Create new goals
        self.active_goals = []
        for template in self.goal_templates:
            goal = template.create_goal(market_regime, volatility)
            self.active_goals.append(goal)
        
        self._last_regime = market_regime
        logger.info(f"Refreshed goals for market regime: {market_regime}")
    
    def _update_goal_progress(self, performance_metrics: Dict[str, Any]):
        """Update progress on active goals"""
        for goal in self.active_goals:
            if goal.goal_type == GoalType.PROFIT_TARGET:
                goal.current_value = performance_metrics.get('total_return', 0.0)
            elif goal.goal_type == GoalType.SHARPE_OPTIMIZATION:
                goal.current_value = performance_metrics.get('sharpe_ratio', 0.0)
            elif goal.goal_type == GoalType.DRAWDOWN_CONTROL:
                # Invert drawdown for progress calculation
                drawdown = performance_metrics.get('max_drawdown', 0.0)
                goal.current_value = max(0, goal.target_value - drawdown)
            elif goal.goal_type == GoalType.RISK_REDUCTION:
                var = performance_metrics.get('daily_var', 0.0)
                goal.current_value = max(0, goal.target_value - var)
            
            goal.update_progress()
    
    def _process_completed_goals(self):
        """Process completed goals and create new ones if needed"""
        completed = [goal for goal in self.active_goals if goal.achieved]
        
        for goal in completed:
            self.completed_goals.append(goal)
            self.active_goals.remove(goal)
            logger.info(f"Goal completed: {goal.goal_type.value} - {goal.target_value}")
            
            # Create stretch goal
            self._create_stretch_goal(goal)
    
    def _create_stretch_goal(self, completed_goal: Goal):
        """Create a stretch goal based on completed goal"""
        stretch_multiplier = 1.2  # 20% increase
        
        stretch_goal = Goal(
            goal_type=completed_goal.goal_type,
            target_value=completed_goal.target_value * stretch_multiplier,
            priority=completed_goal.priority * 0.8  # Lower priority for stretch goals
        )
        
        self.active_goals.append(stretch_goal)
        logger.info(f"Created stretch goal: {stretch_goal.goal_type.value} - {stretch_goal.target_value}")
    
    def get_priority_goals(self, top_n: int = 3) -> List[Goal]:
        """Get top priority goals"""
        return sorted(self.active_goals, key=lambda g: g.priority, reverse=True)[:top_n]
    
    def get_goal_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for goal adjustments"""
        recommendations = []
        
        # Identify struggling goals
        struggling_goals = [g for g in self.active_goals if g.progress < 0.2]
        
        for goal in struggling_goals:
            if goal.goal_type == GoalType.PROFIT_TARGET:
                recommendations.append({
                    'type': 'reduce_target',
                    'goal': goal.goal_type.value,
                    'current_target': goal.target_value,
                    'suggested_target': goal.target_value * 0.8,
                    'reason': 'Low progress on profit target'
                })
            elif goal.goal_type == GoalType.SHARPE_OPTIMIZATION:
                recommendations.append({
                    'type': 'focus_risk_management',
                    'goal': goal.goal_type.value,
                    'suggestion': 'Increase focus on risk management strategies',
                    'reason': 'Poor risk-adjusted returns'
                })
        
        return recommendations
    
    def get_goal_status(self) -> Dict[str, Any]:
        """Get comprehensive goal status"""
        return {
            'active_goals': len(self.active_goals),
            'completed_goals': len(self.completed_goals),
            'average_progress': np.mean([g.progress for g in self.active_goals]) if self.active_goals else 0,
            'goals_on_track': len([g for g in self.active_goals if g.progress > 0.5]),
            'struggling_goals': len([g for g in self.active_goals if g.progress < 0.3]),
            'goal_details': [
                {
                    'type': goal.goal_type.value,
                    'target': goal.target_value,
                    'current': goal.current_value,
                    'progress': goal.progress,
                    'priority': goal.priority
                }
                for goal in self.active_goals
            ]
        }