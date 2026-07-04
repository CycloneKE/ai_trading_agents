"""
Market data connectors for fetching real-time and historical market data.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import requests

logger = logging.getLogger(__name__)

class BaseConnector:
    """Base class for all data connectors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_connected = False
        self.name = "base"
    
    def connect(self) -> bool:
        """Connect to the data source."""
        self.is_connected = True
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from the data source."""
        self.is_connected = False
        return True
    
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time data for symbols."""
        raise NotImplementedError
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get historical data for a symbol."""
        raise NotImplementedError
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status."""
        return {
            'name': self.name,
            'is_connected': self.is_connected,
            'config': {k: v for k, v in self.config.items() if k != 'api_key'}
        }

class YahooFinanceConnector(BaseConnector):
    # Circuit breaker state
    _failure_count = 0
    _failure_threshold = 5
    _circuit_open = False
    _circuit_reset_time = 300  # seconds
    _last_failure_time = None

    def _check_circuit_breaker(self):
        import time
        if self._circuit_open:
            if self._last_failure_time and (time.time() - self._last_failure_time > self._circuit_reset_time):
                self._failure_count = 0
                self._circuit_open = False
                logger.info("Circuit breaker reset for YahooFinanceConnector.")
            else:
                logger.warning("Circuit breaker is open. Using fallback data.")
                return True
        return False

    def _record_failure(self):
        import time
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self._failure_threshold:
            self._circuit_open = True
            logger.error("Circuit breaker triggered for YahooFinanceConnector. Too many failures.")

    def _retry_api_call(self, func, *args, **kwargs):
        import time
        max_retries = self.config.get('max_retries', 3)
        delay = 2
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(delay)
        self._record_failure()
        return None
    """Connector for Yahoo Finance data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "yahoo_finance"
        self.rate_limit = config.get('rate_limit', 2000)  # Requests per hour
        self.last_request_time = 0
    
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time data from Yahoo Finance."""
        try:
            # Implement rate limiting
            current_time = time.time()
            if current_time - self.last_request_time < 3600 / self.rate_limit:
                time.sleep(3600 / self.rate_limit - (current_time - self.last_request_time))
            
            self.last_request_time = time.time()
            
            import yfinance as yf
            results = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        results[symbol] = {
                            'price': float(hist['Close'].iloc[-1]),
                            'volume': int(hist['Volume'].iloc[-1]),
                            'timestamp': datetime.utcnow().isoformat()
                        }
                    else:
                        logger.warning(f"yfinance returned empty data for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to fetch yfinance data for {symbol}: {e}")
            return results
            
        except Exception as e:
            logger.error(f"Error getting real-time data from Yahoo Finance: {str(e)}")
            return {}
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get historical data from Yahoo Finance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"yfinance historical returned empty for {symbol}")
                return {}
            
            data = {
                'dates': [d.strftime("%Y-%m-%d") for d in hist.index],
                'prices': hist['Close'].tolist(),
                'volumes': hist['Volume'].tolist()
            }
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data from Yahoo Finance: {str(e)}")
            return {}

class AlphaVantageConnector(BaseConnector):
    """Connector for Alpha Vantage data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "alpha_vantage"
        import os
        self.api_key = os.getenv('TRADING_ALPHA_VANTAGE_API_KEY', config.get('api_key', ''))
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit = config.get('rate_limit', 5)  # Requests per minute
        self.last_request_time = 0
    
    def connect(self) -> bool:
        """Connect to Alpha Vantage."""
        if not self.api_key:
            logger.error("Alpha Vantage API key not provided")
            return False
        
        self.is_connected = True
        return True
    
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time data from Alpha Vantage."""
        try:
            # Implement rate limiting
            current_time = time.time()
            if current_time - self.last_request_time < 60 / self.rate_limit:
                time.sleep(60 / self.rate_limit - (current_time - self.last_request_time))
            
            self.last_request_time = time.time()
            
            import requests
            results = {}
            for symbol in symbols:
                try:
                    url = f"{self.base_url}?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.api_key}"
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    res_data = resp.json()
                    
                    if 'Global Quote' in res_data and res_data['Global Quote']:
                        quote = res_data['Global Quote']
                        results[symbol] = {
                            'price': float(quote.get('05. price', 0)),
                            'volume': int(quote.get('06. volume', 0)),
                            'timestamp': datetime.utcnow().isoformat()
                        }
                    else:
                        logger.warning(f"AlphaVantage returned no quote for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to fetch AlphaVantage data for {symbol}: {e}")
            return results
            
        except Exception as e:
            logger.error(f"Error getting real-time data from Alpha Vantage: {str(e)}")
            return {}
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get historical data from Alpha Vantage."""
        try:
            import requests
            url = f"{self.base_url}?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_key}&outputsize=compact"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            res_data = resp.json()
            
            if 'Time Series (Daily)' in res_data:
                ts = res_data['Time Series (Daily)']
                # Filter dates
                filtered_dates = [d for d in ts.keys() if start_date <= d <= end_date]
                filtered_dates.sort()
                
                prices = [float(ts[d]['4. close']) for d in filtered_dates]
                volumes = [int(ts[d]['5. volume']) for d in filtered_dates]
                
                return {
                    'dates': filtered_dates,
                    'prices': prices,
                    'volumes': volumes
                }
            logger.warning(f"AlphaVantage historical returned no valid series for {symbol}")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting historical data from Alpha Vantage: {str(e)}")
            return {}

class FinnhubConnector(BaseConnector):
    """Connector for Finnhub data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "finnhub"
        import os
        self.api_key = os.getenv('TRADING_FINNHUB_API_KEY', config.get('api_key', ''))
        self.base_url = 'https://finnhub.io/api/v1/'
    
    def connect(self) -> bool:
        """Connect to Finnhub."""
        if not self.api_key:
            logger.error("Finnhub API key not provided")
            return False
        
        self.is_connected = True
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from Finnhub."""
        self.is_connected = False
        return True

    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time data from Finnhub, normalized to the shape the
        strategies consume (price/close/open/high/low/change_percent)."""
        try:
            results = {}
            for symbol in symbols:
                url = f"{self.base_url}quote?symbol={symbol}&token={self.api_key}"
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    continue
                q = resp.json()
                # Finnhub returns c=current, o=open, h=high, l=low, dp=change%.
                # c == 0 means no data for the symbol.
                if not q or not q.get('c'):
                    continue
                results[symbol] = {
                    'symbol': symbol,
                    'price': float(q['c']),
                    'close': float(q['c']),
                    'open': float(q.get('o', 0)),
                    'high': float(q.get('h', 0)),
                    'low': float(q.get('l', 0)),
                    'volume': 0,  # not provided by the quote endpoint
                    'change_percent': float(q.get('dp') or 0),
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': 'finnhub',
                }

            return results

        except Exception as e:
            logger.error(f"Error getting real-time data from Finnhub: {str(e)}")
            return {}
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get historical data from Finnhub."""
        try:
            # Finnhub uses UNIX timestamps for historical data
            import time
            from datetime import datetime
            
            start_ts = int(time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
            end_ts = int(time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple()))
            url = f"{self.base_url}stock/candle?symbol={symbol}&resolution=D&from={start_ts}&to={end_ts}&token={self.api_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            return None
        
        except Exception as e:
            logger.error(f"Error getting historical data from Finnhub: {str(e)}")
            return {}