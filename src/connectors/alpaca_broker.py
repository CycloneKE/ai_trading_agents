"""
Alpaca broker integration for live and paper trading.
Ensures compliance with the BaseBroker interface.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import alpaca_trade_api as tradeapi

from .base_broker import BaseBroker, OrderRequest, OrderResponse, Position, AccountInfo

logger = logging.getLogger(__name__)

class AlpacaBroker(BaseBroker):
    """
    Alpaca broker implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Alpaca broker.
        
        Args:
            config: Broker configuration
        """
        super().__init__(config)
        self.broker_name = "alpaca"
        
        # Alpaca specific configuration
        self.api_key = config.get('api_key') or os.getenv('TRADING_ALPACA_API_KEY')
        self.api_secret = config.get('api_secret') or os.getenv('TRADING_ALPACA_API_SECRET')
        self.is_paper_trading = config.get('paper', True)
        
        if self.is_paper_trading:
            self.base_url = 'https://paper-api.alpaca.markets'
        else:
            self.base_url = 'https://api.alpaca.markets'
        
        self.api = None
    
    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            if not self.api_key or not self.api_secret:
                logger.error("Alpaca API keys not provided")
                return False
                
            self.api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version='v2'
            )
            # Test connection
            self.api.get_account()
            self.is_connected = True
            self.last_connection_time = datetime.utcnow()
            logger.info("Successfully connected to Alpaca")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {str(e)}")
            self.is_connected = False
            return False
            
    def disconnect(self) -> bool:
        """Disconnect from Alpaca API."""
        self.is_connected = False
        self.api = None
        return True
        
    def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        """Place an order through Alpaca."""
        try:
            if not self.is_connected or not self.api:
                if not self.connect():
                    return None
            
            alpaca_order = self.api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side,
                type=order.order_type,
                time_in_force=order.time_in_force,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                client_order_id=order.client_order_id,
                # Alpaca only fills pre/after-market orders when this is set
                # (requires a DAY limit order per their extended-hours rules)
                extended_hours=order.extended_hours
            )
            
            return OrderResponse(
                order_id=alpaca_order.id,
                client_order_id=alpaca_order.client_order_id,
                symbol=alpaca_order.symbol,
                quantity=float(alpaca_order.qty),
                filled_quantity=float(alpaca_order.filled_qty),
                side=alpaca_order.side,
                order_type=alpaca_order.type,
                status=alpaca_order.status,
                created_at=alpaca_order.created_at,
                updated_at=alpaca_order.updated_at,
                limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                filled_avg_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                broker_name=self.broker_name
            )
        except Exception as e:
            logger.error(f"Error placing Alpaca order: {str(e)}")
            return None
            
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an Alpaca order."""
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"Error canceling Alpaca order: {str(e)}")
            return False
            
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get Alpaca account details."""
        try:
            acc = self.api.get_account()
            return AccountInfo(
                account_id=acc.id,
                cash=float(acc.cash),
                equity=float(acc.equity),
                buying_power=float(acc.buying_power),
                initial_margin=float(getattr(acc, 'initial_margin', 0) or 0),
                maintenance_margin=float(getattr(acc, 'maintenance_margin', 0) or 0),
                day_trade_count=int(acc.daytrade_count),
                last_updated=datetime.utcnow(),
                broker_name=self.broker_name
            )
        except Exception as e:
            logger.error(f"Error getting Alpaca account info: {str(e)}")
            return None
            
    def get_positions(self) -> List[Position]:
        """Get current Alpaca positions."""
        try:
            alpaca_positions = self.api.list_positions()
            positions = []
            for p in alpaca_positions:
                positions.append(Position(
                    symbol=p.symbol,
                    quantity=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_pl_percent=float(p.unrealized_plpc),
                    cost_basis=float(p.cost_basis),
                    broker_name=self.broker_name
                ))
            return positions
        except Exception as e:
            logger.error(f"Error getting Alpaca positions: {str(e)}")
            return []
            
    def get_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """Get Alpaca open orders."""
        try:
            params = {'status': 'open'}
            if symbol:
                params['symbols'] = [symbol]
            alpaca_orders = self.api.list_orders(**params)
            
            orders = []
            for o in alpaca_orders:
                orders.append(OrderResponse(
                    order_id=o.id,
                    client_order_id=o.client_order_id,
                    symbol=o.symbol,
                    quantity=float(o.qty),
                    filled_quantity=float(o.filled_qty),
                    side=o.side,
                    order_type=o.type,
                    status=o.status,
                    created_at=o.created_at,
                    updated_at=o.updated_at,
                    limit_price=float(o.limit_price) if o.limit_price else None,
                    stop_price=float(o.stop_price) if o.stop_price else None,
                    filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                    broker_name=self.broker_name
                ))
            return orders
        except Exception as e:
            logger.error(f"Error getting Alpaca orders: {str(e)}")
            return []

    def get_portfolio_history(self, period: str = '1D', timeframe: str = '1Min') -> Dict[str, Any]:
        """Get Alpaca portfolio history."""
        try:
            history = self.api.get_portfolio_history(
                period=period,
                timeframe=timeframe
            )
            return {
                'timestamp': history.timestamp,
                'equity': [float(e) for e in history.equity],
                'profit_loss': [float(pl) for pl in history.profit_loss],
                'profit_loss_pct': [float(plp) for plp in history.profit_loss_pct]
            }
        except Exception as e:
            logger.error(f"Error getting Alpaca portfolio history: {str(e)}")
            return {'timestamp': [], 'equity': [], 'profit_loss': [], 'profit_loss_pct': []}