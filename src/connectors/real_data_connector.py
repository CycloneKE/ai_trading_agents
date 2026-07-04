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

# Prometheus connector error counter (optional)
try:
    from prometheus_client import Counter as _Counter
    connector_error_counter = _Counter('connector_errors_total', 'Total connector errors encountered')
except Exception:
    class _NoOpC:
        def inc(self, *a, **k):
            return None
    connector_error_counter = _NoOpC()

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
                logger.warning("Circuit breaker is open. Unable to fetch live data (mock data is disabled).")
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
            'status': 'ok' if (self.alpha_vantage_key or self.fmp_key or self.finnhub_key) else 'no_api_keys'
        }
    """Real-time market data connector supporting multiple APIs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha_vantage_key = os.getenv('TRADING_ALPHA_VANTAGE_API_KEY')
        self.fmp_key = os.getenv('TRADING_FMP_API_KEY')
        self.finnhub_key = os.getenv('TRADING_FINNHUB_API_KEY')
        self.cache = {}
        self.last_update = {}
        
    # Quotes younger than this are served from self.cache instead of
    # re-fetching, so overlapping callers (trading loop, API endpoints)
    # don't multiply vendor requests for the same symbol.
    CACHE_TTL_SECONDS = 30

    def get_real_time_data(self, symbols) -> Dict[str, Any]:
        """Get real-time market data.

        Accepts a single symbol string (returns one quote dict) or a list of
        symbols (returns {symbol: quote} — the shape data_manager and
        _extract_symbol_data expect).
        """
        if isinstance(symbols, (list, tuple, set)):
            results = {}
            for sym in symbols:
                quote = self.get_real_time_data(sym)
                if quote:
                    results[sym] = quote
            return results
        return self._get_single_quote(symbols)

    def _get_single_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data for one symbol"""
        try:
            cached = self.cache.get(symbol)
            if cached and time.time() - cached['fetched_at'] < self.CACHE_TTL_SECONDS:
                return cached['quote']

            if self._check_circuit_breaker():
                return None  # mock data is disabled; fail cleanly

            # Try Finnhub first (working API)
            if self.finnhub_key:
                data = self._retry_api_call(self._get_finnhub_data, symbol)
                if data:
                    self._failure_count = 0
                    self.cache[symbol] = {'quote': data, 'fetched_at': time.time()}
                    return data

            # Fallback to Alpha Vantage (rate limited)
            if self.alpha_vantage_key:
                data = self._retry_api_call(self._get_alpha_vantage_data, symbol)
                if data:
                    self._failure_count = 0
                    self.cache[symbol] = {'quote': data, 'fetched_at': time.time()}
                    return data
                    
            # Skip FMP (403 error)
            # if self.fmp_key:
            #     data = self._retry_api_call(self._get_fmp_data, symbol)
            #     if data:
            #         self._failure_count = 0
            #         return data
                        
            logger.warning(f"No real data available for {symbol}. Ensure API keys are valid.")
            self._record_failure()
            return None
            
        except Exception as e:
            logger.error(f"Error in get_real_time_data for {symbol}: {e}")
            try:
                connector_error_counter.inc()
            except Exception:
                pass
            self._record_failure()
            return None
    
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
            response.raise_for_status()
            data = response.json()
            
            # Check for API error messages
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error for {symbol}: {data['Error Message']}")
                return None
                
            if 'Note' in data:
                logger.warning(f"Alpha Vantage rate limit for {symbol}: {data['Note']}")
                return None
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'symbol': symbol,
                    'price': float(quote.get('05. price', 0)),
                    'close': float(quote.get('05. price', 0)),
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
    
    def get_market_news(self, category: str = 'general') -> List[Dict[str, Any]]:
        """Get latest market news from Finnhub."""
        try:
            if not self.finnhub_key:
                return []
                
            url = "https://finnhub.io/api/v1/news"
            params = {'category': category, 'token': self.finnhub_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            news_items = response.json()
            
            formatted_news = []
            for item in news_items[:10]: # Top 10
                formatted_news.append({
                    'id': item.get('id'),
                    'title': item.get('headline'),
                    'summary': item.get('summary'),
                    'source': item.get('source'),
                    'url': item.get('url'),
                    'time': datetime.fromtimestamp(item.get('datetime', 0)).isoformat(),
                    'sentiment': 0 # Placeholder for sentiment analysis
                })
            return formatted_news
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []

    def get_sector_performance(self) -> List[Dict[str, Any]]:
        """Get sector performance using major sector ETFs."""
        sectors = {
            'Technology': 'XLK',
            'Energy': 'XLE',
            'Financials': 'XLF',
            'Healthcare': 'XLV',
            'Utilities': 'XLU',
            'Consumer Staples': 'XLP',
            'Real Estate': 'XLRE',
            'Materials': 'XLB'
        }
        
        results = []
        for name, symbol in sectors.items():
            try:
                data = self.get_real_time_data(symbol)
                if data and data['source'] != 'mock':
                    results.append({
                        'sector': name,
                        'symbol': symbol,
                        'change': data.get('change_percent', 0),
                        'price': data.get('price', 0),
                        'volume': data.get('volume', 0)
                    })
            except Exception as e:
                logger.error(f"Error getting sector data for {name}: {e}")
                
        # If no real data, return empty list instead of mocking.
        # User requested to fail cleanly without mocks.
        if not results:
            logger.warning("Failed to collect sector performance. Returning empty list.")
        return results

    def _get_finnhub_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data from Finnhub API"""
        try:
            url = f"https://finnhub.io/api/v1/quote"
            params = {'symbol': symbol, 'token': self.finnhub_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Check for API error
            if 'error' in data:
                logger.error(f"Finnhub API error for {symbol}: {data['error']}")
                return None
            
            if 'c' in data and data['c'] != 0:  # Current price exists and is not zero
                return {
                    'symbol': symbol,
                    'price': float(data.get('c', 0)),
                    'close': float(data.get('c', 0)),
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
            try:
                connector_error_counter.inc()
            except Exception:
                pass
        return None
    

    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data for backtesting"""
        if self._check_circuit_breaker():
            logger.warning(f"Circuit breaker blocks historical data for {symbol}.")
            return None
        # Try Alpha Vantage for historical data (Finnhub doesn't have good historical endpoint)
        if self.alpha_vantage_key:
            df = self._retry_api_call(self._get_alpha_vantage_historical, symbol, days)
            if df is not None:
                self._failure_count = 0
                return df
        self._record_failure()
        logger.warning(f"Failed to fetch historical data for {symbol}.")
        return None
    
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
        
        return None
    
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
        
        return None