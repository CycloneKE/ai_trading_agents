"""
Polygon.io API connector for real-time and historical market data.
"""

import os
import logging
import requests
from typing import Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PolygonConnector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.getenv('TRADING_POLYGON_API_KEY', config.get('api_key', ''))
        self.base_url = "https://api.polygon.io"
        self.is_connected = False
        
    def connect(self) -> bool:
        """Connect to Polygon API."""
        if not self.api_key:
            logger.error("Polygon API key not provided")
            return False
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/v2/aggs/ticker/AAPL/prev?apikey={self.api_key}")
            if response.status_code == 200:
                self.is_connected = True
                logger.info("Connected to Polygon.io")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to Polygon: {str(e)}")
        
        return False
    
    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol."""
        try:
            url = f"{self.base_url}/v2/last/trade/{symbol}?apikey={self.api_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': symbol,
                    'price': data['results']['p'],
                    'size': data['results']['s'],
                    'timestamp': data['results']['t'],
                    'exchange': data['results']['x']
                }
        except Exception as e:
            logger.error(f"Error getting real-time quote for {symbol}: {str(e)}")
        
        return {}
    
    def get_daily_bars(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily price bars."""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?apikey={self.api_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                bars = []
                
                for result in data.get('results', []):
                    bars.append({
                        'timestamp': result['t'],
                        'open': result['o'],
                        'high': result['h'],
                        'low': result['l'],
                        'close': result['c'],
                        'volume': result['v']
                    })
                
                return bars
        except Exception as e:
            logger.error(f"Error getting daily bars for {symbol}: {str(e)}")
        
        return []
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status."""
        try:
            url = f"{self.base_url}/v1/marketstatus/now?apikey={self.api_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error getting market status: {str(e)}")
        
        return {}