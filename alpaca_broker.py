"""
Alpaca broker integration for live trading.
"""

import os
import logging
from typing import Dict, Any, List
import alpaca_trade_api as tradeapi

logger = logging.getLogger(__name__)

class AlpacaBroker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.getenv('TRADING_ALPACA_API_KEY')
        self.api_secret = os.getenv('TRADING_ALPACA_API_SECRET')
        self.base_url = 'https://paper-api.alpaca.markets'  # Change to https://api.alpaca.markets for live
        
        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            self.base_url,
            api_version='v2'
        )
    
    def place_order(self, symbol: str, qty: int, side: str, order_type: str = 'market'):
        """Place an order."""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force='gtc'
            )
            logger.info(f"Order placed: {side} {qty} {symbol}")
            return order
        except Exception as e:
            logger.error(f"Order failed: {str(e)}")
            return None
    
    def get_positions(self):
        """Get current positions."""
        return self.api.list_positions()
    
    def get_account(self):
        """Get account info."""
        return self.api.get_account()