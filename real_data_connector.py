"""
Real Market Data Connector
Integrates multiple data sources for live market data
"""

import os
import time
import logging
import requests
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class RealDataConnector:
    # Circuit breaker state
    _failure_count = 0
    _failure_threshold = 5
    _circuit_open = False
    _circuit_reset_time = 300  # seconds
    _last_failure_time = None

    def _check_circuit_breaker(self):
        if self._circuit_open:
            if self._last_failure_time and (time.time() - self._last_failure_time > self._circuit_reset_time):
                self._failure_count = 0
                self._circuit_open = False
                logger.info("Circuit breaker reset for RealDataConnector.")
            else:
                logger.warning("Circuit breaker is open. Using mock data.")
                return True
        return False

    def _record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self._failure_threshold:
            self._circuit_open = True
            logger.error("Circuit breaker triggered for RealDataConnector. Too many failures.")

    def _retry_api_call(self, func, *args, **kwargs):
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
    def get_status(self) -> Dict[str, Any]:
        """Return health/status info for RealDataConnector."""
        return {
            'alpha_vantage_key': bool(self.alpha_vantage_key),
            'fmp_key': bool(self.fmp_key),
            'finnhub_key': bool(self.finnhub_key),
            'last_update': self.last_update,
            'cache_size': len(self.cache),
            'timestamp': datetime.now().isoformat(),
            'status': 'ok' if (self.alpha_vantage_key or self.fmp_key or self.finnhub_key) else 'mock'
        }
    """Real-time market data connector supporting multiple APIs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha_vantage_key = os.getenv('TRADING_ALPHA_VANTAGE_API_KEY')
        self.fmp_key = os.getenv('TRADING_FMP_API_KEY')
        self.finnhub_key = os.getenv('TRADING_FINNHUB_API_KEY')
        self.cache = {}
        self.last_update = {}
        
    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data for symbol"""
        try:
            # Try Alpha Vantage first
            if self.alpha_vantage_key:
                data = self._get_alpha_vantage_data(symbol)
                if self._check_circuit_breaker():
                    return self._generate_mock_data(symbol)
                # Try Alpha Vantage first
                if self.alpha_vantage_key:
                    data = self._retry_api_call(self._get_alpha_vantage_data, symbol)
                    if data:
                        self._failure_count = 0
                        return data
                # Fallback to FMP
                if self.fmp_key:
                    data = self._retry_api_call(self._get_fmp_data, symbol)
                    if data:
                        self._failure_count = 0
                        return data
                # Fallback to Finnhub
                if self.finnhub_key:
                    data = self._retry_api_call(self._get_finnhub_data, symbol)
                    if data:
                        self._failure_count = 0
                        return data
                logger.warning(f"No real data available for {symbol}, using mock data")
                self._record_failure()
                return self._generate_mock_data(symbol)
    
    def _get_alpha_vantage_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data from Alpha Vantage API"""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'symbol': symbol,
                    'price': float(quote.get('05. price', 0)),
                    'open': float(quote.get('02. open', 0)),
                    'high': float(quote.get('03. high', 0)),
                    'low': float(quote.get('04. low', 0)),
                    'volume': int(quote.get('06. volume', 0)),
                    'change_percent': float(quote.get('10. change percent', '0%').replace('%', '')),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'alpha_vantage'
                }
        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
        return None
    
    def _get_fmp_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data from Financial Modeling Prep API"""
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
            params = {'apikey': self.fmp_key}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data and len(data) > 0:
                quote = data[0]
                return {
                    'symbol': symbol,
                    'price': float(quote.get('price', 0)),
                    'open': float(quote.get('open', 0)),
                    'high': float(quote.get('dayHigh', 0)),
                    'low': float(quote.get('dayLow', 0)),
                    'volume': int(quote.get('volume', 0)),
                    'change_percent': float(quote.get('changesPercentage', 0)),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'fmp'
                }
        except Exception as e:
            logger.error(f"FMP error for {symbol}: {e}")
        return None
    
    def _get_finnhub_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data from Finnhub API"""
        try:
            url = f"https://finnhub.io/api/v1/quote"
            params = {'symbol': symbol, 'token': self.finnhub_key}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'c' in data:  # Current price
                return {
                    'symbol': symbol,
                    'price': float(data.get('c', 0)),
                    'open': float(data.get('o', 0)),
                    'high': float(data.get('h', 0)),
                    'low': float(data.get('l', 0)),
                    'volume': 0,  # Not provided in quote endpoint
                    'change_percent': float(data.get('dp', 0)),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'finnhub'
                }
        except Exception as e:
            logger.error(f"Finnhub error for {symbol}: {e}")
        return None
    
    def _generate_mock_data(self, symbol: str) -> Dict[str, Any]:
        """Generate realistic mock data as fallback"""
        import random
        base_price = 100 + hash(symbol) % 400  # Deterministic base price
        
        return {
            'symbol': symbol,
            'price': base_price + random.uniform(-5, 5),
            'open': base_price + random.uniform(-3, 3),
            'high': base_price + random.uniform(0, 8),
            'low': base_price + random.uniform(-8, 0),
            'volume': random.randint(100000, 10000000),
            'change_percent': random.uniform(-5, 5),
            'timestamp': datetime.now().isoformat(),
            'source': 'mock'
        }
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data for backtesting"""
        if self._check_circuit_breaker():
            return self._generate_mock_historical(symbol, days)
        # Try Alpha Vantage first
        if self.alpha_vantage_key:
            df = self._retry_api_call(self._get_alpha_vantage_historical, symbol, days)
            if df is not None:
                self._failure_count = 0
                return df
        elif self.fmp_key:
            df = self._retry_api_call(self._get_fmp_historical, symbol, days)
            if df is not None:
                self._failure_count = 0
                return df
        self._record_failure()
        return self._generate_mock_historical(symbol, days)
    
    def _get_alpha_vantage_historical(self, symbol: str, days: int) -> pd.DataFrame:
        """Get historical data from Alpha Vantage"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df = df.astype(float)
                return df.tail(days)
        except Exception as e:
            logger.error(f"Alpha Vantage historical error: {e}")
        
        return self._generate_mock_historical(symbol, days)
    
    def _get_fmp_historical(self, symbol: str, days: int) -> pd.DataFrame:
        """Get historical data from FMP"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10)  # Extra buffer
            
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
            params = {
                'apikey': self.fmp_key,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'historical' in data:
                df = pd.DataFrame(data['historical'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()
                return df[['open', 'high', 'low', 'close', 'volume']].tail(days)
        except Exception as e:
            logger.error(f"FMP historical error: {e}")
        
        return self._generate_mock_historical(symbol, days)
    
    def _generate_mock_historical(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate mock historical data"""
        import random
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        base_price = 100 + hash(symbol) % 400
        
        data = []
        price = base_price
        for _ in range(days):
            change = random.uniform(-0.05, 0.05)
            price *= (1 + change)
            
            open_price = price * random.uniform(0.98, 1.02)
            high_price = max(price, open_price) * random.uniform(1.0, 1.03)
            low_price = min(price, open_price) * random.uniform(0.97, 1.0)
            volume = random.randint(100000, 10000000)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df