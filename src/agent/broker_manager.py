"""
Broker Manager for coordinating multiple broker connections.
Provides unified interface for trading across different platforms.
"""

from typing import Dict, Any, List, Optional, Type, Any as TypingAny
import logging
from datetime import datetime
import threading

from src.connectors.base_broker import BaseBroker, OrderRequest, OrderResponse, Position, AccountInfo

from .paper_trading import PaperTradingBroker
import logging
try:
    from src.connectors.alpaca_broker import AlpacaBroker
    ALPACA_AVAILABLE = True
except Exception as e:
    ALPACA_AVAILABLE = False
    AlpacaBroker = None
    logging.error(f"AlpacaBroker import failed: {e}")
try:
    from src.connectors.coinbase_broker import CoinbaseBroker
    COINBASE_AVAILABLE = True
except Exception as e:
    COINBASE_AVAILABLE = False
    CoinbaseBroker = None
    logging.error(f"CoinbaseBroker import failed: {e}")
try:
    from src.connectors.oanda_broker import OandaBroker
    OANDA_AVAILABLE = True
except Exception as e:
    OANDA_AVAILABLE = False
    OandaBroker = None
    logging.error(f"OandaBroker import failed: {e}")

logger = logging.getLogger(__name__)

_SENSITIVE_CONFIG_KEYS = {'api_key', 'api_secret', 'secret', 'passphrase', 'password'}


def _redact_broker_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mask credential fields before a broker config touches the logs."""
    return {
        k: ('***REDACTED***' if k.lower() in _SENSITIVE_CONFIG_KEYS and v else v)
        for k, v in config.items()
    }


class BrokerConfigurationError(RuntimeError):
    """The configured brokers cannot be honoured as written."""


def available_broker_types() -> Dict[str, TypingAny]:
    """Broker types this process can actually construct.

    A type is missing here only because its connector failed to import,
    which the guarded imports above log as an error. Exposed as a function so
    src/agent/run_readiness.py can ask the question without building a
    manager, and so the gate's answer cannot drift from the runtime's.
    """
    types: Dict[str, TypingAny] = {'paper': PaperTradingBroker}
    if ALPACA_AVAILABLE:
        types['alpaca'] = AlpacaBroker
    if COINBASE_AVAILABLE:
        types['coinbase'] = CoinbaseBroker
        types['coinbase_broker'] = CoinbaseBroker
    if OANDA_AVAILABLE:
        types['oanda'] = OandaBroker
        types['oanda_broker'] = OandaBroker
    return types


def broker_is_enabled(config: Dict[str, Any]) -> bool:
    """Whether a broker entry in the config should be created.

    Absent means enabled, which keeps configs written before the flag
    existed working. `enabled: false` used to mean nothing at all: the
    startup loop never read it, so the Coinbase broker that
    config/config.json disables was constructed and connected on every run.

    run_readiness.py imports this so the gate and the runtime agree on which
    brokers count. They did not agree: the gate treated any broker other
    than one literally named 'paper_broker' with no `enabled` key as
    disabled, and so skipped it in the paper-mode check. That would have
    waved through a live-money broker.
    """
    return bool(config.get('enabled', True))


class BrokerManager:
    """
    Manager for multiple broker connections.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.brokers = {}
        self.primary_broker = None
        self.lock = threading.Lock()
        
        self.broker_types: Dict[str, TypingAny] = available_broker_types()
        # Brokers the config names but this process did not build, and why.
        # The readiness gate reads these: it used to certify the config's
        # intent without ever checking what actually got created.
        self.disabled_brokers: List[str] = []
        self.unavailable_brokers: Dict[str, str] = {}
        logger.info(f"Available broker types: {list(self.broker_types.keys())}")
        # Initialize brokers from config
        self._initialize_brokers()
        logger.info("Broker manager initialized")
    
    def _initialize_brokers(self):
        """Create the brokers the config asks for, or refuse to run.

        Three things used to fail silently here. `enabled: false` was never
        read. A broker whose type was unavailable produced a single warning
        and was dropped. And because the primary was whichever broker
        happened to get built, losing the one marked `primary` quietly
        promoted another: config/config.json marks Alpaca primary, its
        connector could not import, and every order went to the internal
        paper simulator instead. Nothing said the venue had changed.
        """
        brokers_config = self.config.get('brokers', {}) or {}
        logger.info("BrokerManager received brokers config: %s",
                    {n: _redact_broker_config(c) for n, c in brokers_config.items()})
        logger.info("BrokerManager available broker types: %s",
                    sorted(self.broker_types))

        for broker_name, broker_config in brokers_config.items():
            broker_type = (broker_config.get('type') or '').lower()

            if not broker_is_enabled(broker_config):
                self.disabled_brokers.append(broker_name)
                logger.info("Skipping broker %s (type %s): disabled in config",
                            broker_name, broker_type or '<none>')
                continue

            if broker_type not in self.broker_types:
                self.unavailable_brokers[broker_name] = broker_type
                logger.error(
                    "Broker %s is enabled but its type %r is not available. "
                    "Known types: %s. Its connector failed to import; the "
                    "reason was logged at import time.",
                    broker_name, broker_type, sorted(self.broker_types))
                continue

            try:
                broker = self.broker_types[broker_type](broker_config)
            except Exception as e:
                self.unavailable_brokers[broker_name] = broker_type
                logger.error("Broker %s (type %s) could not be created: %s",
                             broker_name, broker_type, e)
                continue

            self.brokers[broker_name] = broker
            if broker_config.get('primary', False) or self.primary_broker is None:
                self.primary_broker = broker_name
            logger.info("Initialized %s broker: %s", broker_type, broker_name)

        self._verify_configured_brokers(brokers_config)

    def _verify_configured_brokers(self, brokers_config: Dict[str, Any]) -> None:
        """Refuse to trade against a venue the operator did not choose.

        Raising stops the agent. That is the point: an order routed to the
        wrong broker is worse than an agent that does not start, and the
        failure is visible in the logs on the first line rather than
        inferred from a P&L that does not match anybody's expectations.
        """
        if not brokers_config:
            # Nothing configured at all. Long-standing behaviour, kept.
            self.brokers['default_paper'] = PaperTradingBroker({
                'type': 'paper',
                'initial_cash': 100000,
                'commission_per_trade': 0.0,
            })
            self.primary_broker = 'default_paper'
            logger.info("No brokers configured; created default paper trading broker")
            return

        missing_primary = {
            n: t for n, t in sorted(self.unavailable_brokers.items())
            if brokers_config.get(n, {}).get('primary', False)
        }
        if missing_primary:
            raise BrokerConfigurationError(
                f"Broker(s) marked primary could not be created: {missing_primary}. "
                f"Available types: {sorted(self.broker_types)}. Refusing to start, "
                "because orders would be routed to a venue other than the "
                "configured one. Fix the broker config or the connector import.")

        disabled_primary = sorted(
            n for n, c in brokers_config.items()
            if c.get('primary', False) and not broker_is_enabled(c))
        if disabled_primary:
            raise BrokerConfigurationError(
                f"Broker(s) {disabled_primary} are marked primary but disabled. "
                "Refusing to start; mark an enabled broker as primary.")

        if not self.brokers:
            raise BrokerConfigurationError(
                f"{len(brokers_config)} broker(s) configured, none created. "
                f"Disabled: {sorted(self.disabled_brokers) or 'none'}. "
                f"Unavailable: {self.unavailable_brokers or 'none'}. Refusing to "
                "start rather than substituting a paper broker nobody configured.")

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

