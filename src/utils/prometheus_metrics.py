"""
Prometheus metrics for the AI Trading Agent.
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PrometheusMetrics:
    """
    Prometheus metrics handler for the AI Trading Agent.
    """
    
    def __init__(self):
        """Initialize the Prometheus metrics handler."""
        self.metrics = {}
        self.last_update = {}
        
        # Initialize metrics
        self._init_metrics()
        
        logger.info("Prometheus metrics initialized")
    
    def _init_metrics(self):
        """Initialize metrics dictionary."""
        # Portfolio metrics
        self.metrics['trading_portfolio_value'] = 0.0
        self.metrics['trading_cash_balance'] = 0.0
        self.metrics['trading_equity'] = 0.0
        self.metrics['trading_total_pnl'] = 0.0
        self.metrics['trading_daily_pnl'] = 0.0
        self.metrics['trading_win_rate'] = 0.0
        self.metrics['trading_trade_count'] = 0
        self.metrics['trading_position_count'] = 0
        
        # Risk metrics
        self.metrics['trading_max_drawdown'] = 0.0
        self.metrics['trading_sharpe_ratio'] = 0.0
        self.metrics['trading_portfolio_var'] = 0.0
        self.metrics['trading_volatility'] = 0.0
        
        # Strategy metrics
        self.metrics['trading_strategy_returns'] = {}
        
        # System metrics
        self.metrics['system_cpu_usage'] = 0.0
        self.metrics['system_memory_usage'] = 0.0
        self.metrics['system_disk_usage'] = 0.0
    
    def update_portfolio_metrics(self, portfolio_value: float, cash: float, equity: float, 
                               total_pnl: float, daily_pnl: float, positions: List[Dict[str, Any]]):
        """
        Update portfolio metrics.
        
        Args:
            portfolio_value: Total portfolio value
            cash: Available cash
            equity: Total equity
            total_pnl: Total profit and loss
            daily_pnl: Daily profit and loss
            positions: List of positions
        """
        self.metrics['trading_portfolio_value'] = portfolio_value
        self.metrics['trading_cash_balance'] = cash
        self.metrics['trading_equity'] = equity
        self.metrics['trading_total_pnl'] = total_pnl
        self.metrics['trading_daily_pnl'] = daily_pnl
        self.metrics['trading_position_count'] = len(positions)
        
        self.last_update['portfolio'] = datetime.utcnow()
    
    def update_risk_metrics(self, max_drawdown: float, sharpe_ratio: float, 
                          portfolio_var: float, volatility: float):
        """
        Update risk metrics.
        
        Args:
            max_drawdown: Maximum drawdown
            sharpe_ratio: Sharpe ratio
            portfolio_var: Portfolio Value at Risk
            volatility: Portfolio volatility
        """
        self.metrics['trading_max_drawdown'] = max_drawdown
        self.metrics['trading_sharpe_ratio'] = sharpe_ratio
        self.metrics['trading_portfolio_var'] = portfolio_var
        self.metrics['trading_volatility'] = volatility
        
        self.last_update['risk'] = datetime.utcnow()
    
    def update_trade_metrics(self, trade_count: int, win_count: int):
        """
        Update trade metrics.
        
        Args:
            trade_count: Total number of trades
            win_count: Number of winning trades
        """
        self.metrics['trading_trade_count'] = trade_count
        
        # Calculate win rate
        if trade_count > 0:
            self.metrics['trading_win_rate'] = win_count / trade_count
        else:
            self.metrics['trading_win_rate'] = 0.0
        
        self.last_update['trades'] = datetime.utcnow()
    
    def update_strategy_returns(self, strategy_name: str, returns: float):
        """
        Update strategy returns.
        
        Args:
            strategy_name: Strategy name
            returns: Strategy returns
        """
        if 'trading_strategy_returns' not in self.metrics:
            self.metrics['trading_strategy_returns'] = {}
        
        self.metrics['trading_strategy_returns'][strategy_name] = returns
        
        if 'strategy' not in self.last_update:
            self.last_update['strategy'] = {}
        
        self.last_update['strategy'][strategy_name] = datetime.utcnow()
    
    def update_system_metrics(self, cpu_usage: float, memory_usage: float, disk_usage: float):
        """
        Update system metrics.
        
        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            disk_usage: Disk usage percentage
        """
        self.metrics['system_cpu_usage'] = cpu_usage
        self.metrics['system_memory_usage'] = memory_usage
        self.metrics['system_disk_usage'] = disk_usage
        
        self.last_update['system'] = datetime.utcnow()
    
    def get_prometheus_metrics(self) -> str:
        """
        Get metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus format
        """
        lines = []
        
        # Add metric types and help
        lines.append("# HELP trading_portfolio_value Total portfolio value in USD")
        lines.append("# TYPE trading_portfolio_value gauge")
        lines.append(f"trading_portfolio_value {self.metrics['trading_portfolio_value']}")
        
        lines.append("# HELP trading_cash_balance Available cash in USD")
        lines.append("# TYPE trading_cash_balance gauge")
        lines.append(f"trading_cash_balance {self.metrics['trading_cash_balance']}")
        
        lines.append("# HELP trading_equity Total equity in USD")
        lines.append("# TYPE trading_equity gauge")
        lines.append(f"trading_equity {self.metrics['trading_equity']}")
        
        lines.append("# HELP trading_total_pnl Total profit and loss in USD")
        lines.append("# TYPE trading_total_pnl gauge")
        lines.append(f"trading_total_pnl {self.metrics['trading_total_pnl']}")
        
        lines.append("# HELP trading_daily_pnl Daily profit and loss in USD")
        lines.append("# TYPE trading_daily_pnl gauge")
        lines.append(f"trading_daily_pnl {self.metrics['trading_daily_pnl']}")
        
        lines.append("# HELP trading_win_rate Percentage of winning trades")
        lines.append("# TYPE trading_win_rate gauge")
        lines.append(f"trading_win_rate {self.metrics['trading_win_rate']}")
        
        lines.append("# HELP trading_trade_count Total number of trades")
        lines.append("# TYPE trading_trade_count counter")
        lines.append(f"trading_trade_count {self.metrics['trading_trade_count']}")
        
        lines.append("# HELP trading_position_count Number of open positions")
        lines.append("# TYPE trading_position_count gauge")
        lines.append(f"trading_position_count {self.metrics['trading_position_count']}")
        
        lines.append("# HELP trading_max_drawdown Maximum drawdown percentage")
        lines.append("# TYPE trading_max_drawdown gauge")
        lines.append(f"trading_max_drawdown {self.metrics['trading_max_drawdown']}")
        
        lines.append("# HELP trading_sharpe_ratio Sharpe ratio")
        lines.append("# TYPE trading_sharpe_ratio gauge")
        lines.append(f"trading_sharpe_ratio {self.metrics['trading_sharpe_ratio']}")
        
        lines.append("# HELP trading_portfolio_var Portfolio Value at Risk")
        lines.append("# TYPE trading_portfolio_var gauge")
        lines.append(f"trading_portfolio_var {self.metrics['trading_portfolio_var']}")
        
        lines.append("# HELP trading_volatility Portfolio volatility")
        lines.append("# TYPE trading_volatility gauge")
        lines.append(f"trading_volatility {self.metrics['trading_volatility']}")
        
        # Strategy returns
        if 'trading_strategy_returns' in self.metrics:
            lines.append("# HELP trading_strategy_return Strategy return percentage")
            lines.append("# TYPE trading_strategy_return gauge")
            
            for strategy, returns in self.metrics['trading_strategy_returns'].items():
                lines.append(f'trading_strategy_return{{strategy="{strategy}"}} {returns}')
        
        # System metrics
        lines.append("# HELP system_cpu_usage CPU usage percentage")
        lines.append("# TYPE system_cpu_usage gauge")
        lines.append(f"system_cpu_usage {self.metrics['system_cpu_usage']}")
        
        lines.append("# HELP system_memory_usage Memory usage percentage")
        lines.append("# TYPE system_memory_usage gauge")
        lines.append(f"system_memory_usage {self.metrics['system_memory_usage']}")
        
        lines.append("# HELP system_disk_usage Disk usage percentage")
        lines.append("# TYPE system_disk_usage gauge")
        lines.append(f"system_disk_usage {self.metrics['system_disk_usage']}")
        
        return "\n".join(lines)

# Singleton instance
_prometheus_metrics = None

def get_prometheus_metrics() -> PrometheusMetrics:
    """
    Get the Prometheus metrics instance.
    
    Returns:
        PrometheusMetrics instance
    """
    global _prometheus_metrics
    
    if _prometheus_metrics is None:
        _prometheus_metrics = PrometheusMetrics()
    
    return _prometheus_metrics