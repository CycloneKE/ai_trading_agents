"""
Real-time Data Feed Manager
Handles live market data from multiple sources
"""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, List
from threading import Thread
import time

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    websocket = None

logger = logging.getLogger(__name__)

class RealTimeDataFeed:
    """Real-time market data feed manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connections = {}
        self.subscribers = {}
        self.running = False
        self.data_buffer = {}
        self.enabled = WEBSOCKET_AVAILABLE
        
        if not self.enabled:
            logger.warning("WebSocket not available. Install websocket-client to enable real-time data feeds.")
        
    def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to symbol updates"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
        
    def start(self):
        """Start real-time data feeds"""
        if not self.enabled:
            logger.info("Real-time data feeds disabled (websocket-client not available)")
            return
            
        self.running = True
        
        # Start WebSocket connections in separate threads
        for source in self.config.get('data_sources', []):
            if source['enabled']:
                thread = Thread(target=self._start_websocket, args=(source,))
                thread.daemon = True
                thread.start()
                
        logger.info("Real-time data feeds started")
    
    def stop(self):
        """Stop all data feeds"""
        self.running = False
        for ws in self.connections.values():
            if ws:
                ws.close()
        logger.info("Real-time data feeds stopped")
    
    def _start_websocket(self, source_config: Dict[str, Any]):
        """Start WebSocket connection for a data source"""
        try:
            if source_config['type'] == 'alpaca':
                self._connect_alpaca(source_config)
            elif source_config['type'] == 'polygon':
                self._connect_polygon(source_config)
        except Exception as e:
            logger.error(f"Failed to start WebSocket for {source_config['type']}: {e}")
    
    def _connect_alpaca(self, config: Dict[str, Any]):
        """Connect to Alpaca WebSocket"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._process_alpaca_message(data)
            except Exception as e:
                logger.error(f"Error processing Alpaca message: {e}")
        
        def on_error(ws, error):
            logger.error(f"Alpaca WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("Alpaca WebSocket closed")
        
        def on_open(ws):
            # Subscribe to symbols
            symbols = config.get('symbols', [])
            subscribe_msg = {
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols
            }
            ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to Alpaca feed for {symbols}")
        
        ws_url = config.get('websocket_url', 'wss://stream.data.alpaca.markets/v2/iex')
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.connections['alpaca'] = ws
        ws.run_forever()
    
    def _connect_polygon(self, config: Dict[str, Any]):
        """Connect to Polygon WebSocket"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._process_polygon_message(data)
            except Exception as e:
                logger.error(f"Error processing Polygon message: {e}")
        
        api_key = config.get('api_key')
        ws_url = f"wss://socket.polygon.io/stocks?apikey={api_key}"
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message
        )
        
        self.connections['polygon'] = ws
        ws.run_forever()
    
    def _process_alpaca_message(self, data: List[Dict[str, Any]]):
        """Process Alpaca WebSocket message"""
        for item in data:
            if item.get('T') == 't':  # Trade
                symbol = item.get('S')
                trade_data = {
                    'symbol': symbol,
                    'price': item.get('p'),
                    'size': item.get('s'),
                    'timestamp': item.get('t'),
                    'type': 'trade'
                }
                self._notify_subscribers(symbol, trade_data)
            
            elif item.get('T') == 'q':  # Quote
                symbol = item.get('S')
                quote_data = {
                    'symbol': symbol,
                    'bid': item.get('bp'),
                    'ask': item.get('ap'),
                    'bid_size': item.get('bs'),
                    'ask_size': item.get('as'),
                    'timestamp': item.get('t'),
                    'type': 'quote'
                }
                self._notify_subscribers(symbol, quote_data)
    
    def _process_polygon_message(self, data: List[Dict[str, Any]]):
        """Process Polygon WebSocket message"""
        for item in data:
            if item.get('ev') == 'T':  # Trade
                symbol = item.get('sym')
                trade_data = {
                    'symbol': symbol,
                    'price': item.get('p'),
                    'size': item.get('s'),
                    'timestamp': item.get('t'),
                    'type': 'trade'
                }
                self._notify_subscribers(symbol, trade_data)
    
    def _notify_subscribers(self, symbol: str, data: Dict[str, Any]):
        """Notify all subscribers for a symbol"""
        if symbol in self.subscribers:
            for callback in self.subscribers[symbol]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
        
        # Store in buffer for recent access
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
        
        self.data_buffer[symbol].append(data)
        
        # Keep only last 1000 data points
        if len(self.data_buffer[symbol]) > 1000:
            self.data_buffer[symbol] = self.data_buffer[symbol][-1000:]
    
    def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        """Get latest data for symbol"""
        if symbol in self.data_buffer and self.data_buffer[symbol]:
            return self.data_buffer[symbol][-1]
        return {}
    
    def get_recent_data(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent data for symbol"""
        if symbol in self.data_buffer:
            return self.data_buffer[symbol][-count:]
        return []