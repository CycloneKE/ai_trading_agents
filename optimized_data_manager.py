import time
import threading
from concurrent.futures import ThreadPoolExecutor
from performance_optimization import DataCache, ConnectionPool

class OptimizedDataManager:
    def __init__(self):
        self.cache = DataCache(max_size=1000, ttl=300)
        self.connection_pool = ConnectionPool(max_connections=10)
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    def get_market_data(self, symbol):
        """Get market data with caching"""
        cache_key = f"market_{symbol}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        # Fetch new data
        data = self._fetch_market_data(symbol)
        if data:
            self.cache.set(cache_key, data)
        
        return data
    
    def _fetch_market_data(self, symbol):
        """Fetch data from API (optimized)"""
        import requests
        import os
        
        fmp_key = os.getenv('TRADING_FMP_API_KEY')
        if not fmp_key:
            return self._generate_mock_data(symbol)
        
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={fmp_key}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
        except:
            pass
        
        return self._generate_mock_data(symbol)
    
    def _generate_mock_data(self, symbol):
        """Generate mock data for testing"""
        import random
        base_prices = {'AAPL': 175, 'GOOGL': 140, 'MSFT': 415, 'TSLA': 250, 'NVDA': 875}
        base_price = base_prices.get(symbol, 100)
        
        return {
            'symbol': symbol,
            'price': base_price + random.uniform(-5, 5),
            'change': random.uniform(-2, 2),
            'volume': random.randint(1000000, 10000000),
            'timestamp': time.time()
        }
    
    def get_multiple_symbols(self, symbols):
        """Fetch multiple symbols concurrently"""
        futures = []
        for symbol in symbols:
            future = self.executor.submit(self.get_market_data, symbol)
            futures.append((symbol, future))
        
        results = {}
        for symbol, future in futures:
            try:
                results[symbol] = future.result(timeout=10)
            except:
                results[symbol] = None
        
        return results
