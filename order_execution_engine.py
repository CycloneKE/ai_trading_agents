"""
Order Execution Engine with Smart Order Routing
Handles intelligent order execution with slippage control and risk management
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class OrderRequest:
    symbol: str
    side: str  # buy/sell
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    strategy: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ExecutionReport:
    order_id: str
    symbol: str
    side: str
    quantity: float
    filled_quantity: float
    avg_fill_price: float
    status: OrderStatus
    timestamp: datetime
    slippage: float = 0.0
    commission: float = 0.0

class OrderExecutionEngine:
    """Smart order execution engine with risk controls"""
    
    def __init__(self, config: Dict[str, Any], broker_manager):
        self.config = config
        self.broker_manager = broker_manager
        self.pending_orders = {}
        self.execution_history = []
        
        # Execution parameters
        self.max_slippage = config.get('max_slippage', 0.005)  # 0.5%
        self.max_order_size = config.get('max_order_size', 10000)  # $10k
        self.min_order_size = config.get('min_order_size', 100)   # $100
        self.execution_timeout = config.get('execution_timeout', 300)  # 5 minutes
        
        # Smart routing
        self.use_twap = config.get('use_twap', True)
        self.twap_duration = config.get('twap_duration', 300)  # 5 minutes
        self.slice_size = config.get('slice_size', 0.1)  # 10% of order
        
    def execute_order(self, order_request: OrderRequest) -> Optional[str]:
        """Execute order with smart routing and risk controls"""
        try:
            # Pre-execution validation
            if not self._validate_order(order_request):
                return None
            
            # Calculate optimal execution strategy
            execution_plan = self._create_execution_plan(order_request)
            
            # Execute based on plan
            if execution_plan['strategy'] == 'immediate':
                return self._execute_immediate(order_request)
            elif execution_plan['strategy'] == 'twap':
                return self._execute_twap(order_request, execution_plan)
            elif execution_plan['strategy'] == 'iceberg':
                return self._execute_iceberg(order_request, execution_plan)
            
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return None
    
    def _validate_order(self, order: OrderRequest) -> bool:
        """Validate order before execution"""
        # Size validation
        order_value = order.quantity * (order.price or 100)  # Estimate
        if order_value > self.max_order_size:
            logger.warning(f"Order size too large: ${order_value}")
            return False
        
        if order_value < self.min_order_size:
            logger.warning(f"Order size too small: ${order_value}")
            return False
        
        # Market hours validation
        if not self._is_market_open():
            logger.warning("Market is closed")
            return False
        
        # Account validation
        account_info = self.broker_manager.get_account_info()
        if not account_info:
            logger.warning("Cannot get account info")
            return False
        
        # Buying power check for buy orders
        if order.side == 'buy':
            required_cash = order_value * 1.1  # 10% buffer
            if account_info.buying_power < required_cash:
                logger.warning(f"Insufficient buying power: {account_info.buying_power} < {required_cash}")
                return False
        
        return True
    
    def _create_execution_plan(self, order: OrderRequest) -> Dict[str, Any]:
        """Create optimal execution plan based on order characteristics"""
        order_value = order.quantity * (order.price or 100)
        
        # Small orders - execute immediately
        if order_value < 1000:
            return {'strategy': 'immediate'}
        
        # Large orders - use TWAP
        elif order_value > 5000 and self.use_twap:
            slices = max(2, min(10, int(order_value / 1000)))
            return {
                'strategy': 'twap',
                'slices': slices,
                'duration': self.twap_duration
            }
        
        # Medium orders - use iceberg
        else:
            return {
                'strategy': 'iceberg',
                'slice_size': self.slice_size
            }
    
    def _execute_immediate(self, order: OrderRequest) -> Optional[str]:
        """Execute order immediately"""
        try:
            # Get current market data for slippage estimation
            market_data = self.broker_manager.get_market_data(order.symbol)
            if not market_data:
                logger.warning(f"No market data for {order.symbol}")
                return None
            
            # Adjust price for market orders to limit slippage
            if order.order_type == OrderType.MARKET:
                current_price = market_data.get('last_price', order.price)
                if order.side == 'buy':
                    # Use ask price + small buffer
                    order.price = current_price * (1 + self.max_slippage)
                    order.order_type = OrderType.LIMIT
                else:
                    # Use bid price - small buffer
                    order.price = current_price * (1 - self.max_slippage)
                    order.order_type = OrderType.LIMIT
            
            # Submit order
            response = self.broker_manager.place_order(order)
            if response:
                self.pending_orders[response.order_id] = {
                    'order': order,
                    'response': response,
                    'timestamp': datetime.utcnow(),
                    'strategy': 'immediate'
                }
                return response.order_id
            
        except Exception as e:
            logger.error(f"Immediate execution error: {e}")
            return None
    
    def _execute_twap(self, order: OrderRequest, plan: Dict[str, Any]) -> Optional[str]:
        """Execute order using Time-Weighted Average Price strategy"""
        try:
            slices = plan['slices']
            duration = plan['duration']
            slice_quantity = order.quantity / slices
            interval = duration / slices
            
            parent_order_id = f"twap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Schedule slice executions
            for i in range(slices):
                slice_order = OrderRequest(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_quantity,
                    order_type=OrderType.LIMIT,
                    price=order.price,
                    strategy=f"twap_slice_{i+1}",
                    metadata={'parent_order': parent_order_id}
                )
                
                # Schedule execution
                execution_time = datetime.utcnow() + timedelta(seconds=i * interval)
                self._schedule_order(slice_order, execution_time)
            
            return parent_order_id
            
        except Exception as e:
            logger.error(f"TWAP execution error: {e}")
            return None
    
    def _execute_iceberg(self, order: OrderRequest, plan: Dict[str, Any]) -> Optional[str]:
        """Execute order using Iceberg strategy"""
        try:
            slice_size = plan['slice_size']
            visible_quantity = order.quantity * slice_size
            
            # Create initial visible slice
            slice_order = OrderRequest(
                symbol=order.symbol,
                side=order.side,
                quantity=visible_quantity,
                order_type=order.order_type,
                price=order.price,
                strategy="iceberg_slice",
                metadata={'remaining_quantity': order.quantity - visible_quantity}
            )
            
            response = self.broker_manager.place_order(slice_order)
            if response:
                self.pending_orders[response.order_id] = {
                    'order': order,
                    'response': response,
                    'timestamp': datetime.utcnow(),
                    'strategy': 'iceberg',
                    'remaining_quantity': order.quantity - visible_quantity
                }
                return response.order_id
            
        except Exception as e:
            logger.error(f"Iceberg execution error: {e}")
            return None
    
    def _schedule_order(self, order: OrderRequest, execution_time: datetime):
        """Schedule order for future execution"""
        # In a real implementation, this would use a task scheduler
        # For now, we'll store it and check in update_orders
        scheduled_id = f"scheduled_{len(self.pending_orders)}"
        self.pending_orders[scheduled_id] = {
            'order': order,
            'execution_time': execution_time,
            'strategy': 'scheduled',
            'status': 'scheduled'
        }
    
    def update_orders(self):
        """Update order status and handle fills"""
        try:
            current_time = datetime.utcnow()
            completed_orders = []
            
            for order_id, order_info in self.pending_orders.items():
                # Handle scheduled orders
                if order_info.get('status') == 'scheduled':
                    if current_time >= order_info['execution_time']:
                        # Execute scheduled order
                        response = self.broker_manager.place_order(order_info['order'])
                        if response:
                            order_info['response'] = response
                            order_info['status'] = 'submitted'
                    continue
                
                # Check order status
                if 'response' in order_info:
                    # Get updated order status from broker
                    orders = self.broker_manager.get_orders(order_info['order'].symbol)
                    current_order = next((o for o in orders if o.order_id == order_id), None)
                    
                    if current_order:
                        if current_order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                            # Order completed
                            execution_report = ExecutionReport(
                                order_id=order_id,
                                symbol=order_info['order'].symbol,
                                side=order_info['order'].side,
                                quantity=order_info['order'].quantity,
                                filled_quantity=current_order.filled_quantity,
                                avg_fill_price=current_order.avg_fill_price,
                                status=current_order.status,
                                timestamp=current_time,
                                slippage=self._calculate_slippage(order_info['order'], current_order),
                                commission=current_order.commission or 0.0
                            )
                            
                            self.execution_history.append(execution_report)
                            completed_orders.append(order_id)
                            
                            # Handle iceberg continuation
                            if (order_info['strategy'] == 'iceberg' and 
                                current_order.status == OrderStatus.FILLED and
                                order_info.get('remaining_quantity', 0) > 0):
                                self._continue_iceberg(order_info)
            
            # Remove completed orders
            for order_id in completed_orders:
                del self.pending_orders[order_id]
                
        except Exception as e:
            logger.error(f"Order update error: {e}")
    
    def _continue_iceberg(self, order_info: Dict[str, Any]):
        """Continue iceberg order with next slice"""
        try:
            remaining = order_info['remaining_quantity']
            if remaining <= 0:
                return
            
            original_order = order_info['order']
            slice_quantity = min(remaining, original_order.quantity * self.slice_size)
            
            next_slice = OrderRequest(
                symbol=original_order.symbol,
                side=original_order.side,
                quantity=slice_quantity,
                order_type=original_order.order_type,
                price=original_order.price,
                strategy="iceberg_slice",
                metadata={'remaining_quantity': remaining - slice_quantity}
            )
            
            response = self.broker_manager.place_order(next_slice)
            if response:
                self.pending_orders[response.order_id] = {
                    'order': original_order,
                    'response': response,
                    'timestamp': datetime.utcnow(),
                    'strategy': 'iceberg',
                    'remaining_quantity': remaining - slice_quantity
                }
                
        except Exception as e:
            logger.error(f"Iceberg continuation error: {e}")
    
    def _calculate_slippage(self, original_order: OrderRequest, executed_order) -> float:
        """Calculate slippage for executed order"""
        if not original_order.price or not executed_order.avg_fill_price:
            return 0.0
        
        expected_price = original_order.price
        actual_price = executed_order.avg_fill_price
        
        if original_order.side == 'buy':
            slippage = (actual_price - expected_price) / expected_price
        else:
            slippage = (expected_price - actual_price) / expected_price
        
        return slippage
    
    def _is_market_open(self) -> bool:
        """Check if market is open"""
        # Simplified check - in production, use proper market calendar
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour
        
        # Monday to Friday, 9:30 AM to 4:00 PM EST (simplified)
        return weekday < 5 and 9 <= hour < 16
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            if order_id in self.pending_orders:
                success = self.broker_manager.cancel_order(order_id)
                if success:
                    del self.pending_orders[order_id]
                return success
            return False
            
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return False
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics"""
        if not self.execution_history:
            return {}
        
        filled_orders = [r for r in self.execution_history if r.status == OrderStatus.FILLED]
        
        if not filled_orders:
            return {}
        
        total_slippage = sum(abs(r.slippage) for r in filled_orders)
        avg_slippage = total_slippage / len(filled_orders)
        
        total_commission = sum(r.commission for r in filled_orders)
        
        return {
            'total_orders': len(self.execution_history),
            'filled_orders': len(filled_orders),
            'fill_rate': len(filled_orders) / len(self.execution_history),
            'avg_slippage': avg_slippage,
            'total_slippage_cost': total_slippage,
            'total_commission': total_commission,
            'pending_orders': len(self.pending_orders)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get execution engine status"""
        return {
            'pending_orders': len(self.pending_orders),
            'execution_history': len(self.execution_history),
            'metrics': self.get_execution_metrics(),
            'config': {
                'max_slippage': self.max_slippage,
                'max_order_size': self.max_order_size,
                'use_twap': self.use_twap
            }
        }