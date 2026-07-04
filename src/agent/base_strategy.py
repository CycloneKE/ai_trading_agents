"""
Base Strategy class for all trading strategies.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import abc

logger = logging.getLogger(__name__)

class BaseStrategy(abc.ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.weight = config.get('weight', 1.0)
        self.last_update = None
        self.performance_history = []
        
    @abc.abstractmethod
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals from market data.
        
        Args:
            data: Market data and features
            
        Returns:
            Dict containing trading signals
        """
        pass
    
    @abc.abstractmethod
    def update_model(self, data: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the strategy model with new data.
        
        Args:
            data: New market data
            feedback: Optional performance feedback
            
        Returns:
            bool: True if update successful
        """
        pass
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.
        
        Returns:
            Dict containing performance metrics
        """
        if not self.performance_history:
            return {
                'sharpe_ratio': 0.0,
                'win_rate': 0.5,
                'max_drawdown': 0.0,
                'total_return': 0.0
            }
        
        # Simple implementation - should be overridden by subclasses
        wins = sum(1 for p in self.performance_history if p.get('return', 0) > 0)
        win_rate = wins / len(self.performance_history) if self.performance_history else 0.5
        
        returns = [p.get('return', 0) for p in self.performance_history]
        total_return = sum(returns)
        
        # Simple Sharpe calculation (should be improved in real implementation)
        import numpy as np
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Calculate max drawdown
        max_drawdown = 0.0
        peak = 0
        for r in returns:
            peak = max(peak, r)
            drawdown = peak - r
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'sharpe_ratio': float(sharpe),
            'win_rate': win_rate,
            'max_drawdown': float(max_drawdown),
            'total_return': float(total_return)
        }
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the strategy model to a file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            bool: True if save successful
        """
        # Default implementation - should be overridden by subclasses
        return True
    
    def load_model(self, filepath: str) -> bool:
        """
        Load the strategy model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            bool: True if load successful
        """
        # Default implementation - should be overridden by subclasses
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get strategy status.
        
        Returns:
            Dict containing status information
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'enabled': self.enabled,
            'weight': self.weight,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'config': self.config
        }