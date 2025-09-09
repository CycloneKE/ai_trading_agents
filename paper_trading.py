"""
Paper Trading Broker implementation for simulated trading.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import uuid
import json
import os
import threading
import time
import random

from base_broker import BaseBroker, OrderRequest, OrderResponse, Position, AccountInfo

logger = logging.getLogger(__name__)

class PaperTradingBroker(BaseBroker):
    def get_status(self) -> Dict[str, Any]:
        """Return health/status info for PaperTradingBroker."""
        return {
            'broker_name': self.broker_name,
            'is_connected': self.is_connected,
            'is_paper_trading': self.is_paper_trading,
            'cash': self.cash,
            'timestamp': datetime.now().isoformat(),
            'status': 'ok' if self.is_connected else 'not_connected'
        }
    """
    Paper trading broker for simulated trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the paper trading broker.
        
        Args:
            config: Broker configuration
        """
        super().__init__(config)
        self.broker_name = "paper"
        self.is_paper_trading = True
        
        # Paper trading specific configuration
        self.initial_cash = config.get('initial_cash', 100000.0)
        self.commission_per_trade = config.get('commission_per_trade', 0.0)
        self.slippage_percent = config.get('slippage_percent', 0.0)
        self.data_source = config.get('data_source', None)
        
        # Internal state
        self.cash = self.initial_cash
        self.positions = {}  # symbol -> Position
        self.orders = {}  # order_id -> OrderResponse
        self.order_history = []
        self.trade_history = []
        self.market_prices = {}  # symbol -> price
        
        # Threading
        self.lock = threading.Lock()
        self.order_processor_thread = None
        self.is_running = False
        
        # Load state if available
        self._load_state()
    
    def connect(self) -> bool:
        """
        Connect to the paper trading system.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.is_connected = True
            self.last_connection_time = datetime.utcnow()
            
            # Start order processor thread
            if not self.is_running:
                self.is_running = True
                self.order_processor_thread = threading.Thread(
                    target=self._order_processor_loop,
                    daemon=True
                )
                self.order_processor_thread.start()
            
            logger.info("Connected to paper trading broker")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to paper trading broker: {str(e)}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the paper trading system.
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            self.is_running = False
            
            # Wait for order processor thread to finish
            if self.order_processor_thread and self.order_processor_thread.is_alive():
                self.order_processor_thread.join(timeout=2)
            
            self.is_connected = False
            
            # Save state
            self._save_state()
            
            logger.info("Disconnected from paper trading broker")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from paper trading broker: {str(e)}")
            return False
    
    def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        """
        Place a paper trading order.
        
        Args:
            order: Order request
            
        Returns:
            OrderResponse or None if error
        """
        try:
            if not self.is_connected:
                logger.error("Cannot place order: not connected")
                return None
            
            # Generate order ID
            order_id = str(uuid.uuid4())
            client_order_id = order.client_order_id or f"paper_{int(time.time())}"
            
            # Get current price for the symbol
            current_price = self._get_current_price(order.symbol)
            if current_price is None:
                logger.error(f"Cannot place order: no price data for {order.symbol}")
                return None
            
            # Create order response
            now = datetime.utcnow()
            order_response = OrderResponse(
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=order.symbol,
                quantity=order.quantity,
                filled_quantity=0.0,
                side=order.side,
                order_type=order.order_type,
                status="new",
                created_at=now,
                updated_at=now,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                filled_avg_price=None,
                broker_name=self.broker_name
            )
            
            # Store order
            with self.lock:
                self.orders[order_id] = order_response
            
            logger.info(f"Placed paper trading order: {order_id} ({order.side} {order.quantity} {order.symbol})")
            return order_response
            
        except Exception as e:
            logger.error(f"Error placing paper trading order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a paper trading order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if cancellation successful
        """
        try:
            if not self.is_connected:
                logger.error("Cannot cancel order: not connected")
                return False
            
            with self.lock:
                if order_id in self.orders:
                    order = self.orders[order_id]
                    
                    # Only cancel if not already filled
                    if order.status not in ["filled", "canceled", "rejected"]:
                        order.status = "canceled"
                        order.updated_at = datetime.utcnow()
                        logger.info(f"Canceled paper trading order: {order_id}")
                        return True
                    else:
                        logger.warning(f"Cannot cancel order {order_id}: status is {order.status}")
                        return False
                else:
                    logger.warning(f"Order not found: {order_id}")
                    return False
            
        except Exception as e:
            logger.error(f"Error canceling paper trading order: {str(e)}")
            return False
    
    def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get paper trading account information.
        
        Returns:
            AccountInfo or None if error
        """
        try:
            if not self.is_connected:
                logger.error("Cannot get account info: not connected")
                return None
            
            # Calculate equity (cash + positions value)
            positions_value = sum(
                pos.quantity * self._get_current_price(pos.symbol)
                for symbol, pos in self.positions.items()
                if self._get_current_price(symbol) is not None
            )
            
            equity = self.cash + positions_value
            
            # Create account info
            account_info = AccountInfo(
                account_id="paper_account",
                cash=self.cash,
                equity=equity,
                buying_power=self.cash * 2.0,  # Simplified margin calculation
                initial_margin=0.0,
                maintenance_margin=0.0,
                day_trade_count=0,
                last_updated=datetime.utcnow(),
                broker_name=self.broker_name
            )
            
            self.account_info = account_info
            return account_info
            
        except Exception as e:
            logger.error(f"Error getting paper trading account info: {str(e)}")
            return None
    
    def get_positions(self) -> List[Position]:
        """
        Get paper trading positions.
        
        Returns:
            List of positions
        """
        try:
            if not self.is_connected:
                logger.error("Cannot get positions: not connected")
                return []
            
            positions = []
            
            with self.lock:
                for symbol, position in self.positions.items():
                    # Update current price and calculations
                    current_price = self._get_current_price(symbol)
                    if current_price is not None:
                        market_value = position.quantity * current_price
                        unrealized_pl = market_value - position.cost_basis
                        unrealized_pl_percent = (unrealized_pl / position.cost_basis) if position.cost_basis > 0 else 0.0
                        
                        updated_position = Position(
                            symbol=symbol,
                            quantity=position.quantity,
                            avg_entry_price=position.avg_entry_price,
                            current_price=current_price,
                            market_value=market_value,
                            unrealized_pl=unrealized_pl,
                            unrealized_pl_percent=unrealized_pl_percent,
                            cost_basis=position.cost_basis,
                            broker_name=self.broker_name
                        )
                        
                        positions.append(updated_position)
                        # Update stored position
                        self.positions[symbol] = updated_position
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting paper trading positions: {str(e)}")
            return []
    
    def get_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """
        Get paper trading orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of orders
        """
        try:
            if not self.is_connected:
                logger.error("Cannot get orders: not connected")
                return []
            
            orders = []
            
            with self.lock:
                for order_id, order in self.orders.items():
                    if symbol is None or order.symbol == symbol:
                        orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Error getting paper trading orders: {str(e)}")
            return []
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market data dict or None if error
        """
        try:
            price = self._get_current_price(symbol)
            
            if price is None:
                return None
            
            # Create simple market data
            bid = price * 0.999  # Simulated bid (0.1% lower)
            ask = price * 1.001  # Simulated ask (0.1% higher)
            
            return {
                'symbol': symbol,
                'price': price,
                'bid': bid,
                'ask': ask,
                'volume': random.randint(1000, 100000),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting paper trading market data: {str(e)}")
            return None
    
    def _order_processor_loop(self):
        """
        Background thread to process paper trading orders.
        """
        while self.is_running:
            try:
                # Process all open orders
                with self.lock:
                    for order_id, order in list(self.orders.items()):
                        if order.status in ["new", "partially_filled"]:
                            self._process_order(order)
                
                # Sleep to avoid high CPU usage
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in paper trading order processor: {str(e)}")
                time.sleep(5)  # Longer sleep on error
    
    def _process_order(self, order: OrderResponse):
        """
        Process a single paper trading order.
        
        Args:
            order: Order to process
        """
        try:
            # Get current price
            current_price = self._get_current_price(order.symbol)
            if current_price is None:
                return
            
            # Check if order can be executed
            can_execute = False
            execution_price = current_price
            
            if order.order_type == "market":
                can_execute = True
            elif order.order_type == "limit":
                if order.side == "buy" and current_price <= order.limit_price:
                    can_execute = True
                    execution_price = order.limit_price
                elif order.side == "sell" and current_price >= order.limit_price:
                    can_execute = True
                    execution_price = order.limit_price
            elif order.order_type == "stop":
                if order.side == "buy" and current_price >= order.stop_price:
                    can_execute = True
                elif order.side == "sell" and current_price <= order.stop_price:
                    can_execute = True
            elif order.order_type == "stop_limit":
                if order.side == "buy" and current_price >= order.stop_price:
                    if current_price <= order.limit_price:
                        can_execute = True
                        execution_price = order.limit_price
                elif order.side == "sell" and current_price <= order.stop_price:
                    if current_price >= order.limit_price:
                        can_execute = True
                        execution_price = order.limit_price
            
            if can_execute:
                # Apply slippage
                if order.side == "buy":
                    execution_price *= (1.0 + self.slippage_percent / 100.0)
                else:
                    execution_price *= (1.0 - self.slippage_percent / 100.0)
                
                # Execute order
                self._execute_order(order, execution_price)
        
        except Exception as e:
            logger.error(f"Error processing paper trading order {order.order_id}: {str(e)}")
    
    def _execute_order(self, order: OrderResponse, execution_price: float):
        """
        Execute a paper trading order.
        
        Args:
            order: Order to execute
            execution_price: Price to execute at
        """
        try:
            # Calculate order value
            order_value = order.quantity * execution_price
            commission = self.commission_per_trade
            
            # Check if we have enough cash for buy orders
            if order.side == "buy":
                total_cost = order_value + commission
                if total_cost > self.cash:
                    # Partial fill if possible
                    max_quantity = (self.cash - commission) / execution_price
                    if max_quantity > 0:
                        order.quantity = max_quantity
                        order_value = order.quantity * execution_price
                    else:
                        order.status = "rejected"
                        order.updated_at = datetime.utcnow()
                        logger.warning(f"Order {order.order_id} rejected: insufficient funds")
                        return
            
            # Update order
            order.filled_quantity = order.quantity
            order.filled_avg_price = execution_price
            order.status = "filled"
            order.updated_at = datetime.utcnow()
            
            # Update cash
            if order.side == "buy":
                self.cash -= (order_value + commission)
            else:
                self.cash += (order_value - commission)
            
            # Update position
            self._update_position(order.symbol, order.side, order.quantity, execution_price)
            
            # Add to trade history
            trade = {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'price': execution_price,
                'commission': commission,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.trade_history.append(trade)
            
            logger.info(f"Executed paper trading order: {order.order_id} ({order.side} {order.quantity} {order.symbol} @ {execution_price:.2f})")
            
            # Save state periodically
            if len(self.trade_history) % 10 == 0:
                self._save_state()
        
        except Exception as e:
            logger.error(f"Error executing paper trading order {order.order_id}: {str(e)}")
    
    def _update_position(self, symbol: str, side: str, quantity: float, price: float):
        """
        Update position after order execution.
        
        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Execution price
        """
        with self.lock:
            if symbol in self.positions:
                position = self.positions[symbol]
                
                if side == "buy":
                    # Add to position
                    new_quantity = position.quantity + quantity
                    new_cost_basis = position.cost_basis + (quantity * price)
                    new_avg_price = new_cost_basis / new_quantity if new_quantity > 0 else 0.0
                    
                    position.quantity = new_quantity
                    position.avg_entry_price = new_avg_price
                    position.cost_basis = new_cost_basis
                    
                else:  # sell
                    # Reduce position
                    new_quantity = position.quantity - quantity
                    
                    if new_quantity > 0:
                        # Partial sell
                        position.quantity = new_quantity
                        # Cost basis is reduced proportionally
                        position.cost_basis = position.cost_basis * (new_quantity / position.quantity)
                    elif new_quantity == 0:
                        # Position closed
                        del self.positions[symbol]
                    else:
                        # Short position
                        position.quantity = new_quantity
                        position.avg_entry_price = price
                        position.cost_basis = -new_quantity * price
            
            else:
                # New position
                if side == "buy":
                    # Long position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_entry_price=price,
                        current_price=price,
                        market_value=quantity * price,
                        unrealized_pl=0.0,
                        unrealized_pl_percent=0.0,
                        cost_basis=quantity * price,
                        broker_name=self.broker_name
                    )
                else:
                    # Short position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=-quantity,
                        avg_entry_price=price,
                        current_price=price,
                        market_value=-quantity * price,
                        unrealized_pl=0.0,
                        unrealized_pl_percent=0.0,
                        cost_basis=quantity * price,
                        broker_name=self.broker_name
                    )
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if not available
        """
        # Check if we have a cached price
        if symbol in self.market_prices:
            # Add small random price movement (0.1% in either direction)
            last_price = self.market_prices[symbol]
            movement = random.uniform(-0.001, 0.001)
            new_price = last_price * (1.0 + movement)
            self.market_prices[symbol] = new_price
            return new_price
        
        # If no data source is configured, use dummy prices
        if self.data_source is None:
            # Generate random initial price between $10 and $1000
            price = random.uniform(10.0, 1000.0)
            self.market_prices[symbol] = price
            return price
        
        # Try to get price from data source (not implemented in this example)
        return 100.0  # Default price
    
    def _save_state(self):
        """Save paper trading state to file."""
        try:
            state = {
                'cash': self.cash,
                'positions': {
                    symbol: {
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'avg_entry_price': pos.avg_entry_price,
                        'cost_basis': pos.cost_basis
                    }
                    for symbol, pos in self.positions.items()
                },
                'market_prices': self.market_prices,
                'trade_history': self.trade_history[-100:],  # Keep last 100 trades
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Create directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            with open('data/paper_trading_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving paper trading state: {str(e)}")
    
    def _load_state(self):
        """Load paper trading state from file."""
        try:
            if os.path.exists('data/paper_trading_state.json'):
                with open('data/paper_trading_state.json', 'r') as f:
                    state = json.load(f)
                
                self.cash = state.get('cash', self.initial_cash)
                self.market_prices = state.get('market_prices', {})
                self.trade_history = state.get('trade_history', [])
                
                # Load positions
                positions = {}
                for symbol, pos_data in state.get('positions', {}).items():
                    current_price = self._get_current_price(symbol)
                    if current_price is None:
                        current_price = pos_data.get('avg_entry_price', 0.0)
                    
                    quantity = pos_data.get('quantity', 0.0)
                    avg_entry_price = pos_data.get('avg_entry_price', 0.0)
                    cost_basis = pos_data.get('cost_basis', quantity * avg_entry_price)
                    
                    market_value = quantity * current_price
                    unrealized_pl = market_value - cost_basis
                    unrealized_pl_percent = (unrealized_pl / cost_basis) if cost_basis > 0 else 0.0
                    
                    positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_entry_price=avg_entry_price,
                        current_price=current_price,
                        market_value=market_value,
                        unrealized_pl=unrealized_pl,
                        unrealized_pl_percent=unrealized_pl_percent,
                        cost_basis=cost_basis,
                        broker_name=self.broker_name
                    )
                
                self.positions = positions
                logger.info("Loaded paper trading state")
                
        except Exception as e:
            logger.error(f"Error loading paper trading state: {str(e)}")
            # Use default values