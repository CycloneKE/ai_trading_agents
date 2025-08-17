"""
Live trading risk management.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LiveRiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.max_position_size = config.get('max_position_size', 0.05)  # 5% max per position
        self.max_daily_loss = config.get('max_daily_loss', 0.02)  # 2% max daily loss
        self.max_total_exposure = config.get('max_total_exposure', 0.8)  # 80% max exposure
        
    def check_position_size(self, symbol: str, quantity: int, price: float, portfolio_value: float) -> bool:
        """Check if position size is within limits."""
        position_value = quantity * price
        position_ratio = position_value / portfolio_value
        
        if position_ratio > self.max_position_size:
            logger.warning(f"Position size too large: {position_ratio:.2%} > {self.max_position_size:.2%}")
            return False
        return True
    
    def check_daily_loss(self, current_pnl: float, portfolio_value: float) -> bool:
        """Check if daily loss exceeds limit."""
        loss_ratio = abs(current_pnl) / portfolio_value if current_pnl < 0 else 0
        
        if loss_ratio > self.max_daily_loss:
            logger.warning(f"Daily loss limit exceeded: {loss_ratio:.2%} > {self.max_daily_loss:.2%}")
            return False
        return True