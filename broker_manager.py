"""
Broker Manager for coordinating multiple broker connections.
Provides unified interface for trading across different platforms.
"""

from typing import Dict, Any, List, Optional, Type
import logging
from datetime import datetime
import threading

from base_broker import BaseBroker, OrderRequest, OrderResponse, Position, AccountInfo
from paper_trading import PaperTradingBroker

try:
    from alpaca_broker import AlpacaBroker
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    AlpacaBroker = None

logger = logging.getLogger(__name__)


class BrokerManager:
    """
    Manager for multiple broker connections.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.brokers = {}
        self.primary_broker = None
        self.lock = threading.Lock()
        
        # Available broker types
        self.broker_types = {
            'paper': PaperTradingBroker
        }
        
        if ALPACA_AVAILABLE:
            self.broker_types['alpaca'] = AlpacaBroker
        
        # Initialize brokers from config
        self._initialize_brokers()
        
        logger.info("Broker manager initialized")
    
    def _initialize_brokers(self):
        """
        Initialize brokers from configuration.
        """
        try:
            brokers_config = self.config.get('brokers', {})
            
            for broker_name, broker_config in brokers_config.items():
                broker_type = broker_config.get('type', '').lower()
                
                if broker_type in self.broker_types:
                    broker_class = self.broker_types[broker_type]
                    broker = broker_class(broker_config)
                    
                    self.brokers[broker_name] = broker
                    
                    # Set primary broker
                    if broker_config.get('primary', False) or self.primary_broker is None:
                        self.primary_broker = broker_name
                    
                    logger.info(f"Initialized {broker_type} broker: {broker_name}")
                else:
                    logger.warning(f"Unknown broker type: {broker_type}")
            
            if not self.brokers:
                # Create default paper trading broker
                default_config = {
                    'type': 'paper',
                    'initial_cash': 100000,
                    'commission_per_trade': 0.0
                }
                
                broker = PaperTradingBroker(default_config)
                self.brokers['default_paper'] = broker
                self.primary_broker = 'default_paper'
                
                logger.info("Created default paper trading broker")
                
        except Exception as e:
            logger.error(f"Error initializing brokers: {str(e)}")
    
    def add_broker(self, name: str, broker_type: str, config: Dict[str, Any]) -> bool:
        """
        Add a new broker.
        
        Args:
            name: Broker name
            broker_type: Type of broker
            config: Broker configuration
            
        Returns:
            bool: True if broker added successfully
        """
        try:
            if broker_type.lower() not in self.broker_types:
                logger.error(f"Unknown broker type: {broker_type}")
                return False
            
            broker_class = self.broker_types[broker_type.lower()]
            broker = broker_class(config)
            
            with self.lock:
                self.brokers[name] = broker
                
                # Set as primary if it's the first broker
                if self.primary_broker is None:
                    self.primary_broker = name
            
            logger.info(f"Added {broker_type} broker: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding broker {name}: {str(e)}")
            return False
    
    def remove_broker(self, name: str) -> bool:
        """
        Remove a broker.
        
        Args:
            name: Broker name
            
        Returns:
            bool: True if broker removed successfully
        """
        try:
            with self.lock:
                if name in self.brokers:
                    # Disconnect broker
                    self.brokers[name].disconnect()
                    del self.brokers[name]
                    
                    # Update primary broker if needed
                    if self.primary_broker == name:
                        self.primary_broker = next(iter(self.brokers.keys())) if self.brokers else None
                    
                    logger.info(f"Removed broker: {name}")
                    return True
                else:
                    logger.warning(f"Broker not found: {name}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error removing broker {name}: {str(e)}")
            return False
    
    def get_broker(self, name: Optional[str] = None) -> Optional[BaseBroker]:
        """
        Get a broker by name.
        
        Args:
            name: Broker name (uses primary if None)
            
        Returns:
            BaseBroker or None if not found
        """
        try:
            if name is None:
                name = self.primary_broker
            
            return self.brokers.get(name)
            
        except Exception as e:
            logger.error(f"Error getting broker {name}: {str(e)}")
            return None
    
    def set_primary_broker(self, name: str) -> bool:
        """
        Set the primary broker.
        
        Args:
            name: Broker name
            
        Returns:
            bool: True if set successfully
        """
        try:
            if name in self.brokers:
                with self.lock:
                    self.primary_broker = name
                logger.info(f"Set primary broker: {name}")
                return True
            else:
                logger.error(f"Broker not found: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting primary broker: {str(e)}")
            return False
    
    def connect_all(self) -> Dict[str, bool]:
        """
        Connect all brokers.
        
        Returns:
            Dict of broker_name -> connection_success
        """
        results = {}
        
        for name, broker in self.brokers.items():
            try:
                success = broker.connect()
                results[name] = success
                
                if success:
                    logger.info(f"Connected to broker: {name}")
                else:
                    logger.error(f"Failed to connect to broker: {name}")
                    
            except Exception as e:
                logger.error(f"Error connecting to broker {name}: {str(e)}")
                results[name] = False
        
        return results
    
    def disconnect_all(self):
        """
        Disconnect all brokers.
        """
        for name, broker in self.brokers.items():
            try:
                broker.disconnect()
                logger.info(f"Disconnected from broker: {name}")
                
            except Exception as e:
                logger.error(f"Error disconnecting from broker {name}: {str(e)}")
    
    def place_order(self, order: OrderRequest, broker_name: Optional[str] = None) -> Optional[OrderResponse]:
        """
        Place an order using specified or primary broker.
        
        Args:
            order: Order request
            broker_name: Broker to use (uses primary if None)
            
        Returns:
            OrderResponse or None if error
        """
        try:
            broker = self.get_broker(broker_name)
            
            if not broker:
                logger.error(f"Broker not found: {broker_name or self.primary_broker}")
                return None
            
            if not broker.is_connected:
                logger.error(f"Broker not connected: {broker_name or self.primary_broker}")
                return None
            
            return broker.place_order(order)
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str, broker_name: Optional[str] = None) -> bool:
        """
        Cancel an order using specified or primary broker.
        
        Args:
            order_id: Order ID to cancel
            broker_name: Broker to use (uses primary if None)
            
        Returns:
            bool: True if cancellation successful
        """
        try:
            broker = self.get_broker(broker_name)
            
            if not broker:
                logger.error(f"Broker not found: {broker_name or self.primary_broker}")
                return False
            
            return broker.cancel_order(order_id)
            
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    def get_account_info(self, broker_name: Optional[str] = None) -> Optional[AccountInfo]:
        """
        Get account information from specified or primary broker.
        
        Args:
            broker_name: Broker to use (uses primary if None)
            
        Returns:
            AccountInfo or None if error
        """
        try:
            broker = self.get_broker(broker_name)
            
            if not broker:
                return None
            
            return broker.get_account_info()
            
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None
    
    def get_positions(self, broker_name: Optional[str] = None) -> List[Position]:
        """
        Get positions from specified or primary broker.
        
        Args:
            broker_name: Broker to use (uses primary if None)
            
        Returns:
            List of positions
        """
        try:
            broker = self.get_broker(broker_name)
            
            if not broker:
                return []
            
            return broker.get_positions()
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_orders(self, symbol: Optional[str] = None, broker_name: Optional[str] = None) -> List[OrderResponse]:
        """
        Get orders from specified or primary broker.
        
        Args:
            symbol: Optional symbol filter
            broker_name: Broker to use (uses primary if None)
            
        Returns:
            List of orders
        """
        try:
            broker = self.get_broker(broker_name)
            
            if not broker:
                return []
            
            return broker.get_orders(symbol)
            
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def get_market_data(self, symbol: str, broker_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get market data from specified or primary broker.
        
        Args:
            symbol: Trading symbol
            broker_name: Broker to use (uses primary if None)
            
        Returns:
            Market data dict or None if error
        """
        try:
            broker = self.get_broker(broker_name)
            
            if not broker:
                return None
            
            return broker.get_market_data(symbol)
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return None
    
    def get_all_portfolios(self) -> Dict[str, Dict[str, Any]]:
        """
        Get portfolio summaries from all connected brokers.
        
        Returns:
            Dict of broker_name -> portfolio_summary
        """
        portfolios = {}
        
        for name, broker in self.brokers.items():
            try:
                if broker.is_connected:
                    portfolio = broker.get_portfolio_summary()
                    portfolios[name] = portfolio
                    
            except Exception as e:
                logger.error(f"Error getting portfolio from {name}: {str(e)}")
                portfolios[name] = {'error': str(e)}
        
        return portfolios
    
    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health check on all brokers.
        
        Returns:
            Dict of broker_name -> health_status
        """
        health_results = {}
        
        for name, broker in self.brokers.items():
            try:
                health_status = broker.health_check()
                health_results[name] = health_status
                
            except Exception as e:
                logger.error(f"Error performing health check on {name}: {str(e)}")
                health_results[name] = {
                    'broker_name': name,
                    'overall_health': False,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        return health_results
    
    def get_broker_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all brokers with their status.
        
        Returns:
            List of broker information
        """
        broker_list = []
        
        for name, broker in self.brokers.items():
            broker_info = {
                'name': name,
                'type': broker.broker_name,
                'is_connected': broker.is_connected,
                'is_paper_trading': broker.is_paper_trading,
                'is_primary': name == self.primary_broker
            }
            broker_list.append(broker_info)
        
        return broker_list
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get broker manager status.
        
        Returns:
            Status information
        """
        try:
            connected_brokers = sum(1 for broker in self.brokers.values() if broker.is_connected)
            
            return {
                'total_brokers': len(self.brokers),
                'connected_brokers': connected_brokers,
                'primary_broker': self.primary_broker,
                'available_broker_types': list(self.broker_types.keys()),
                'brokers': self.get_broker_list(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting broker manager status: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

