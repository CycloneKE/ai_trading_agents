"""
Base Broker class for all broker implementations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import abc
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OrderRequest:
    """Order request data."""
    symbol: str
    quantity: float
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop', 'stop_limit'
    time_in_force: str  # 'day', 'gtc', 'ioc'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    extended_hours: bool = False

@dataclass
class OrderResponse:
    """Order response data."""
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    quantity: float
    filled_quantity: float
    side: str
    order_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_avg_price: Optional[float] = None
    broker_name: str = ""

@dataclass
class Position:
    """Position data."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_percent: float
    cost_basis: float
    broker_name: str = ""

@dataclass
class AccountInfo:
    """Account information."""
    account_id: str
    cash: float
    equity: float
    buying_power: float
    initial_margin: float
    maintenance_margin: float
    day_trade_count: int
    last_updated: datetime
    broker_name: str = ""

class BaseBroker(abc.ABC):
    """
    Abstract base class for all broker implementations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the broker.
        
        Args:
            config: Broker configuration
        """
        self.config = config
        self.broker_name = "base"
        self.is_connected = False
        self.is_paper_trading = True
        self.last_connection_time = None
        self.account_info = None
    
    @abc.abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker API.
        
        Returns:
            bool: True if connection successful
        """
        pass
    
    @abc.abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the broker API.
        
        Returns:
            bool: True if disconnection successful
        """
        pass
    
    @abc.abstractmethod
    def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        """
        Place an order.
        
        Args:
            order: Order request
            
        Returns:
            OrderResponse or None if error
        """
        pass
    
    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if cancellation successful
        """
        pass
    
    @abc.abstractmethod
    def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get account information.
        
        Returns:
            AccountInfo or None if error
        """
        pass
    
    @abc.abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get current positions.
        
        Returns:
            List of positions
        """
        pass
    
    @abc.abstractmethod
    def get_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """
        Get open orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of orders
        """
        pass
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market data dict or None if error
        """
        # Default implementation - should be overridden by subclasses
        return None
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary.
        
        Returns:
            Dict containing portfolio summary
        """
        try:
            account = self.get_account_info()
            positions = self.get_positions()
            
            total_positions_value = sum(p.market_value for p in positions)
            total_unrealized_pl = sum(p.unrealized_pl for p in positions)
            
            return {
                'broker_name': self.broker_name,
                'cash': account.cash if account else 0.0,
                'equity': account.equity if account else 0.0,
                'positions_count': len(positions),
                'positions_value': total_positions_value,
                'unrealized_pl': total_unrealized_pl,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {str(e)}")
            return {
                'broker_name': self.broker_name,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Dict containing health status
        """
        try:
            # Check connection
            connection_ok = self.is_connected
            
            # Check account info
            account_info = self.get_account_info()
            account_ok = account_info is not None
            
            # Check market data
            market_data_ok = False
            try:
                test_symbol = self.config.get('test_symbol', 'AAPL')
                market_data = self.get_market_data(test_symbol)
                market_data_ok = market_data is not None
            except:
                market_data_ok = False
            
            # Overall health
            overall_health = connection_ok and account_ok
            
            return {
                'broker_name': self.broker_name,
                'overall_health': overall_health,
                'connection_ok': connection_ok,
                'account_ok': account_ok,
                'market_data_ok': market_data_ok,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error performing health check: {str(e)}")
            return {
                'broker_name': self.broker_name,
                'overall_health': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }